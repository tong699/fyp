# rl_service.py
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
# Global dictionary: keys are user_intent, values are lists of dictionaries.
# Each dictionary contains an "action" and "bot_response" (short system prompt).
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
def log_interaction(user_id, state, action, reward, new_state):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "state": str(state),
        "action": action,
        "reward": reward,
        "new_state": str(new_state)
    }
    logs_collection.insert_one(log_entry)

# ----------------------------------------------------------------
# 2B. Q-TABLE PERSISTENCE FUNCTIONS
# ----------------------------------------------------------------
def save_q_table_to_mongo():
    q_table_collection.delete_many({})
    docs = []
    for (state, action), value in Q.items():
        docs.append({
            "state": str(state),
            "action": action,
            "value": value
        })
    if docs:
        q_table_collection.insert_many(docs)

def load_q_table_from_mongo():
    global Q
    Q.clear()
    for doc in q_table_collection.find():
        state = doc["state"]
        action = doc["action"]
        value = doc["value"]
        Q[(state, action)] = value

# ----------------------------------------------------------------
# 3. DATA MODELS
# ----------------------------------------------------------------
class SelectActionRequest(BaseModel):
    user_id: Optional[str] = "unknown"
    user_intent: str
    last_message: str
    conversation_history: List[str]
    user_profile: Dict[str, str]

class SelectActionResponse(BaseModel):
    action: str
    bot_response: str

class UpdateRewardRequest(BaseModel):
    user_id: Optional[str] = "unknown"
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
# 4. Q-LEARNING LOGIC
# ----------------------------------------------------------------
Q = {}
last_state = None
last_action = None
last_user_id = None

def get_Q_value(state: str, action: str) -> float:
    return Q.get((state, action), 0.0)

def set_Q_value(state: str, action: str, value: float):
    Q[(state, action)] = value

def choose_action(state: str, intent: str, epsilon=0.1) -> dict:
    available_actions = ACTIONS_BY_INTENT.get(intent, [])
    if not available_actions:
        return {"action": "NO_ACTIONS_FOR_INTENT", "bot_response": "Hmm, I'm not sure."}
    if random.random() < epsilon:
        return random.choice(available_actions)
    best_action = None
    best_q = float("-inf")
    for act in available_actions:
        a = act["action"]
        q_val = get_Q_value(state, a)
        if q_val > best_q:
            best_q = q_val
            best_action = act
    if best_action is None:
        best_action = random.choice(available_actions)
    return best_action

def update_q_learning(state: str, action: str, reward: float,
                      next_state: str, alpha=0.1, gamma=0.9):
    old_value = get_Q_value(state, action)
    future_qs = [get_Q_value(next_state, a["action"]) for acts in ACTIONS_BY_INTENT.values() for a in acts]
    best_future_val = max(future_qs) if future_qs else 0.0
    new_value = old_value + alpha * (reward + gamma * best_future_val - old_value)
    set_Q_value(state, action, new_value)
    save_q_table_to_mongo()

# ----------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_actions_from_csv()
    load_q_table_from_mongo()

@app.post("/select_action", response_model=SelectActionResponse)
def select_action(request: SelectActionRequest) -> SelectActionResponse:
    global last_state, last_action, last_user_id
    # Directly use user_intent as state
    state = request.user_intent
    chosen = choose_action(state, intent=request.user_intent)
    action = chosen["action"]
    bot_response = chosen["bot_response"]

    last_state = state
    last_action = action
    last_user_id = request.user_id

    return SelectActionResponse(action=action, bot_response=bot_response)

@app.post("/update_reward", response_model=UpdateRewardResponse)
def update_reward(request: UpdateRewardRequest) -> UpdateRewardResponse:
    global last_state, last_action, last_user_id
    if last_state is None or last_action is None:
        return UpdateRewardResponse(status="No previous state/action to update.")

    # Here we use "feedback" as state when updating reward
    next_state = "feedback"
    update_q_learning(last_state, last_action, request.reward, next_state)

    log_interaction(
        user_id=last_user_id,
        state=last_state,
        action=last_action,
        reward=request.reward,
        new_state=next_state
    )

    return UpdateRewardResponse(status="Q-table updated.")

@app.get("/q_table_debug")
def get_q_table() -> dict:
    return {
        "Q": [
            {
                "state": key[0],
                "action": key[1],
                "value": value
            }
            for key, value in Q.items()
        ]
    }

# NEW: Flowise Interaction Endpoint
@app.post("/flowise_interaction", response_model=FlowiseInteractionResponse)
def flowise_interaction(request: FlowiseInteractionRequest) -> FlowiseInteractionResponse:
    global last_state, last_action, last_user_id

    # Update Q-table if applicable (for feedback or followup intents)
    if (last_state is not None 
        and last_action is not None
        and request.feedback_label is not None
        and (
            request.user_intent.lower() == "feedback" 
            or request.user_intent.lower().startswith("followup_")
        )
    ):
        if request.feedback_label.lower() == "positive":
            reward = 1.0
        elif request.feedback_label.lower() == "negative":
            reward = -1.0
        else:
            reward = 0.0

        next_state = "feedback"
        update_q_learning(last_state, last_action, reward, next_state)
        log_interaction(
            user_id=last_user_id,
            state=last_state,
            action=last_action,
            reward=reward,
            new_state=next_state
        )

    # Use the provided user_intent directly as the new state
    new_state = request.user_intent
    chosen_action = choose_action(new_state, intent=request.user_intent)

    # Update globals for subsequent interactions
    last_state = new_state
    last_action = chosen_action["action"]
    last_user_id = request.session_id

    # Return only the selected prompt (the bot_response) from the chosen action.
    return FlowiseInteractionResponse(
        status="Action selected.",
        selected_prompt=chosen_action["bot_response"]
    )

# ----------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
