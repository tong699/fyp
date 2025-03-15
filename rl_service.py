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
# 1. LOAD ACTIONS FROM CSV
# ----------------------------------------------------------------
ACTIONS_BY_INTENT = {}

def load_actions_from_csv(csv_file_path="actions.csv"):
    global ACTIONS_BY_INTENT
    ACTIONS_BY_INTENT.clear()
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row["user_intent"]
            action = row["action"]
            bot_response = row["bot_response"]
            if intent not in ACTIONS_BY_INTENT:
                ACTIONS_BY_INTENT[intent] = []
            ACTIONS_BY_INTENT[intent].append({
                "action": action,
                "bot_response": bot_response
            })

load_actions_from_csv()  # Call at startup

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
    # Remove previous entries for this session
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
        session_id = doc["session_id"]
        state = doc["state"]
        action = doc["action"]
        value = doc["value"]
        if session_id not in Q_by_session:
            Q_by_session[session_id] = {}
        Q_by_session[session_id][(state, action)] = value

# ----------------------------------------------------------------
# 3. DATA MODELS
# ----------------------------------------------------------------
class SelectActionRequest(BaseModel):
    # For backward compatibility, you might use user_id as session_id.
    session_id: Optional[str] = "unknown"
    user_intent: str
    last_message: str
    conversation_history: List[str]
    user_profile: Dict[str, str]

class SelectActionResponse(BaseModel):
    action: str
    bot_response: str

class UpdateRewardRequest(BaseModel):
    session_id: Optional[str] = "unknown"
    action: str
    reward: float
    feedback_type: str
    new_user_message: Optional[str] = ""

class UpdateRewardResponse(BaseModel):
    status: str

# NEW: Simplified Flowise Interaction Request
class FlowiseInteractionRequest(BaseModel):
    session_id: str
    user_intent: str
    feedback_label: Optional[str] = None  # "positive", "negative", or None

class FlowiseInteractionResponse(BaseModel):
    status: str
    selected_prompt: str  # Returns the short system prompt

# ----------------------------------------------------------------
# 4. Q-LEARNING LOGIC (per session)
# ----------------------------------------------------------------
# Global dictionaries for per-session Q-tables and last interaction.
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

def choose_action(session_id: str, state: str, intent: str, epsilon=0.1) -> dict:
    available_actions = ACTIONS_BY_INTENT.get(intent, [])
    if not available_actions:
        return {"action": "NO_ACTIONS_FOR_INTENT", "bot_response": "Hmm, I'm not sure."}
    q_table = get_q_table(session_id)
    if random.random() < epsilon:
        return random.choice(available_actions)
    best_action = None
    best_q = float("-inf")
    for act in available_actions:
        a = act["action"]
        q_val = get_Q_value(q_table, state, a)
        if q_val > best_q:
            best_q = q_val
            best_action = act
    if best_action is None:
        best_action = random.choice(available_actions)
    return best_action

def update_q_learning(session_id: str, state: str, action: str, reward: float,
                      next_state: str, alpha=0.1, gamma=0.9):
    q_table = get_q_table(session_id)
    old_value = get_Q_value(q_table, state, action)
    # Compute future rewards over all actions from any intent
    future_qs = [get_Q_value(q_table, next_state, a["action"]) for acts in ACTIONS_BY_INTENT.values() for a in acts]
    best_future_val = max(future_qs) if future_qs else 0.0
    new_value = old_value + alpha * (reward + gamma * best_future_val - old_value)
    set_Q_value(q_table, state, action, new_value)
    save_q_table_to_mongo(session_id)

# ----------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_actions_from_csv()
    load_q_table_from_mongo()

@app.post("/select_action", response_model=SelectActionResponse)
def select_action(request: SelectActionRequest) -> SelectActionResponse:
    session_id = request.session_id  # now using session_id instead of user_id
    state = request.user_intent  # you might choose a more complex state
    chosen = choose_action(session_id, state, intent=request.user_intent)
    action = chosen["action"]
    bot_response = chosen["bot_response"]

    # Save last interaction per session
    last_interaction_by_session[session_id] = {
        "state": state,
        "action": action,
        "session_id": session_id
    }
    return SelectActionResponse(action=action, bot_response=bot_response)

@app.post("/update_reward", response_model=UpdateRewardResponse)
def update_reward(request: UpdateRewardRequest) -> UpdateRewardResponse:
    session_id = request.session_id
    session_data = last_interaction_by_session.get(session_id)
    if not session_data:
        return UpdateRewardResponse(status="No previous state/action to update.")
    
    state = session_data["state"]
    action = session_data["action"]
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
    # Return the Q-table for all sessions.
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

# NEW: Flowise Interaction Endpoint
@app.post("/flowise_interaction", response_model=FlowiseInteractionResponse)
def flowise_interaction(request: FlowiseInteractionRequest) -> FlowiseInteractionResponse:
    session_id = request.session_id

    # If this is a feedback or followup, update the Q-table using the previous interaction
    if (session_id in last_interaction_by_session and request.feedback_label is not None and 
        (request.user_intent.lower() == "feedback" or request.user_intent.lower().startswith("followup_"))):
        if request.feedback_label.lower() == "positive":
            reward = 1.0
        elif request.feedback_label.lower() == "negative":
            reward = -1.0
        else:
            reward = 0.0

        previous = last_interaction_by_session[session_id]
        state = previous["state"]
        action = previous["action"]
        next_state = "feedback"
        update_q_learning(session_id, state, action, reward, next_state)
        log_interaction(
            session_id=session_id,
            state=state,
            action=action,
            reward=reward,
            new_state=next_state
        )

    # Use the provided user_intent as the new state for this session
    new_state = request.user_intent
    chosen_action = choose_action(session_id, new_state, intent=request.user_intent)
    
    # Update last interaction for this session
    last_interaction_by_session[session_id] = {
        "state": new_state,
        "action": chosen_action["action"],
        "session_id": session_id
    }
    
    return FlowiseInteractionResponse(
         status="Action selected for session: " + session_id,
         selected_prompt=chosen_action["bot_response"]
     )

# ----------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
