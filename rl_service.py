import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import csv
from datetime import datetime
from pymongo import MongoClient

app = FastAPI()

# ----------------------------------------------------------------
# MongoDB Setup
# ----------------------------------------------------------------
MONGO_CONNECTION_STRING = "mongodb+srv://cmtong123:20020430@cluster0.d6vff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["rl_database"]
logs_collection = db["action_logs"]
q_table_collection = db["q_table"]

# ----------------------------------------------------------------
# 1. LOAD PROMPTS FROM CSV
# ----------------------------------------------------------------
# The CSV should have headers: persuasive_type,system_prompt
PROMPTS_BY_TONE = {}

def load_prompts_from_csv(csv_file_path="actions.csv"):
    global PROMPTS_BY_TONE
    PROMPTS_BY_TONE.clear()
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tone = row["persuasive_type"].strip().lower()
            prompt = row["system_prompt"]
            PROMPTS_BY_TONE[tone] = prompt

load_prompts_from_csv()  # Call at startup

# ----------------------------------------------------------------
# 2. LOGGING TO MONGODB
# ----------------------------------------------------------------
def log_interaction(session_id, state, action, reward, new_state):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "state": str(state),
        "action": action,
        "reward": reward,
        "new_state": str(new_state)
    }
    logs_collection.insert_one(log_entry)

# ----------------------------------------------------------------
# 2B. Q-TABLE PERSISTENCE FUNCTIONS (per session)
# ----------------------------------------------------------------
def save_q_table_to_mongo(session_id: str):
    q_table = Q_by_session.get(session_id, {})
    q_table_collection.delete_many({"session_id": session_id})
    docs = []
    for (state, action), value in q_table.items():
        docs.append({
            "session_id": session_id,
            "state": str(state),
            "action": action,
            "value": value
        })
    if docs:
        q_table_collection.insert_many(docs)

def load_q_table_from_mongo():
    global Q_by_session
    Q_by_session.clear()
    for doc in q_table_collection.find():
        session_id = doc.get("session_id", "unknown")
        state = doc.get("state", "")
        action = doc.get("action", "")
        value = doc.get("value", 0.0)
        if session_id not in Q_by_session:
            Q_by_session[session_id] = {}
        Q_by_session[session_id][(state, action)] = value

# ----------------------------------------------------------------
# 3. DATA MODELS
# ----------------------------------------------------------------
class SelectActionRequest(BaseModel):
    session_id: Optional[str] = "unknown"
    user_intent: str  # May be used for logging or state tracking
    last_message: str
    conversation_history: List[str]
    user_profile: Dict[str, str]
    persuasive_type: Optional[str] = None  # Tone preference: supportive, motivational, or informative

class SelectActionResponse(BaseModel):
    action: str
    system_prompt: str

class UpdateRewardRequest(BaseModel):
    session_id: Optional[str] = "unknown"
    action: str
    reward: float
    feedback_type: str
    new_user_message: Optional[str] = ""

class UpdateRewardResponse(BaseModel):
    status: str

class FlowiseInteractionRequest(BaseModel):
    session_id: str
    user_intent: str
    feedback_label: Optional[str] = None  # "positive", "negative", or None

class FlowiseInteractionResponse(BaseModel):
    status: str
    selected_prompt: str

# ----------------------------------------------------------------
# 4. Q-LEARNING LOGIC (per session)
# ----------------------------------------------------------------
Q_by_session: Dict[str, Dict] = {}
last_interaction_by_session: Dict[str, Dict] = {}

def get_q_table(session_id: str) -> dict:
    if session_id not in Q_by_session:
        Q_by_session[session_id] = {}
    return Q_by_session[session_id]

def get_Q_value(q_table: dict, state: str, action: str) -> float:
    return q_table.get((state, action), 0.0)

def set_Q_value(q_table: dict, state: str, action: str, value: float):
    q_table[(state, action)] = value

def choose_action(session_id: str, persuasive_type: Optional[str] = None, epsilon=0.1) -> dict:
    # If a persuasive type is specified and exists, use its corresponding prompt.
    if persuasive_type:
        tone = persuasive_type.strip().lower()
        if tone in PROMPTS_BY_TONE:
            return {"action": tone, "system_prompt": PROMPTS_BY_TONE[tone]}
    # Otherwise, select one at random.
    if PROMPTS_BY_TONE:
        tone, prompt = random.choice(list(PROMPTS_BY_TONE.items()))
        return {"action": tone, "system_prompt": prompt}
    return {"action": "NO_ACTION", "system_prompt": "No persuasive prompts available."}

def update_q_learning(session_id: str, state: str, action: str, reward: float,
                      next_state: str, alpha=0.1, gamma=0.9):
    q_table = get_q_table(session_id)
    old_value = get_Q_value(q_table, state, action)
    # For tone selection, we consider all available tones as potential future actions.
    available_actions = list(PROMPTS_BY_TONE.keys())
    if available_actions:
        future_qs = [get_Q_value(q_table, next_state, a) for a in available_actions]
        best_future_val = max(future_qs) if future_qs else 0.0
    else:
        best_future_val = 0.0
    new_value = old_value + alpha * (reward + gamma * best_future_val - old_value)
    set_Q_value(q_table, state, action, new_value)
    save_q_table_to_mongo(session_id)

# ----------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_prompts_from_csv()
    load_q_table_from_mongo()

@app.post("/select_action", response_model=SelectActionResponse)
def select_action(request: SelectActionRequest) -> SelectActionResponse:
    session_id = request.session_id
    persuasive_type = request.persuasive_type  # Tone preference from the client
    chosen = choose_action(session_id, persuasive_type=persuasive_type)
    action = chosen["action"]
    system_prompt = chosen["system_prompt"]

    # Log the interaction; here we use the persuasive_type (or "random" if none provided) as the state.
    last_interaction_by_session[session_id] = {
        "state": persuasive_type if persuasive_type else "random",
        "action": action,
        "session_id": session_id
    }
    return SelectActionResponse(action=action, system_prompt=system_prompt)
    
@app.post("/update_reward", response_model=UpdateRewardResponse)
def update_reward(request: UpdateRewardRequest) -> UpdateRewardResponse:
    session_id = request.session_id
    session_data = last_interaction_by_session.get(session_id)
    if not session_data:
        return UpdateRewardResponse(status="No previous state/action to update.")
    
    state = session_data.get("state", "unknown")
    action = session_data.get("action", "unknown")
    next_state = "feedback"
    update_q_learning(session_id, state, action, request.reward, next_state)
    log_interaction(
        session_id=session_id,
        state=state,
        action=action,
        reward=request.reward,
        new_state=next_state
    )
    return UpdateRewardResponse(status="Q-table updated for session: " + session_id)

@app.get("/q_table_debug")
def get_q_table_debug() -> dict:
    result = {}
    for sess_id, q_table in Q_by_session.items():
        result[sess_id] = [
            {
                "state": key[0],
                "action": key[1],
                "value": value
            }
            for key, value in q_table.items()
        ]
    return {"Q": result}

@app.post("/flowise_interaction", response_model=FlowiseInteractionResponse)
def flowise_interaction(request: FlowiseInteractionRequest) -> FlowiseInteractionResponse:
    session_id = request.session_id

    # If feedback is provided, update Q-learning for the previous interaction.
    if session_id in last_interaction_by_session and request.user_intent in ["positive_feedback", "needs_clarification", "intent_ended"]:
        if request.user_intent == "positive_feedback":
            reward = 1.0
        elif request.user_intent == "needs_clarification":
            reward = -0.5  # Encourage clarity
        elif request.feedback_label == "positive":
            reward = 1.0
        elif request.feedback_label == "negative":
            reward = -1.0
        else:
            reward = 0.0

        previous = last_interaction_by_session[session_id]
        state = previous.get("state", "unknown")
        action = previous.get("action", "unknown")
        next_state = "feedback"
        update_q_learning(session_id, state, action, reward, next_state)
        log_interaction(
            session_id=session_id,
            state=state,
            action=action,
            reward=reward,
            new_state=next_state
        )

    # For a new interaction, use the provided user_intent as the state.
    new_state = request.user_intent
    chosen_action = choose_action(session_id)
    
    last_interaction_by_session[session_id] = {
        "state": new_state,
        "action": chosen_action["action"],
        "session_id": session_id
    }
    
    return FlowiseInteractionResponse(
         status="Action selected for session: " + session_id,
         selected_prompt=chosen_action["system_prompt"]
     )

# ----------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
