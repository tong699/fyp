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
# ACTIONS CSV LOADING
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
            ACTIONS_BY_INTENT.setdefault(intent, []).append({
                "action": action,
                "bot_response": bot_response
            })

# ----------------------------------------------------------------
# LOGGING AND Q-TABLE MANAGEMENT
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

def save_q_table_to_mongo():
    q_table_collection.delete_many({})
    docs = [{"state": str(state), "action": action, "value": value} for (state, action), value in Q.items()]
    if docs:
        q_table_collection.insert_many(docs)

def load_q_table_from_mongo():
    global Q
    Q.clear()
    for doc in q_table_collection.find():
        state = eval(doc["state"])
        Q[(state, doc["action"])] = doc["value"]

# ----------------------------------------------------------------
# DATA MODELS
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

class FlowiseInteractionRequest(BaseModel):
    user_intent: str
    feedback_label: Optional[str] = None

class FlowiseInteractionResponse(BaseModel):
    status: str
    selected_action: dict

# ----------------------------------------------------------------
# Q-LEARNING LOGIC
# ----------------------------------------------------------------
Q = {}
last_state = None
last_action = None
last_user_id = None

def map_intent_to_number(intent: str) -> int:
    return {"motivation_request": 0, "feedback": 1, "general_query": 2}.get(intent, 99)

def get_sentiment_score(message: str) -> float:
    text = message.lower()
    return -1.0 if any(w in text for w in ["unmotivated", "bad", "frustrated"]) else 1.0 if any(w in text for w in ["good", "great", "thanks"]) else 0.0

def make_state(user_intent: str, last_message: str) -> tuple:
    return (map_intent_to_number(user_intent), get_sentiment_score(last_message))

def get_Q_value(state: tuple, action: str) -> float:
    return Q.get((state, action), 0.0)

def set_Q_value(state: tuple, action: str, value: float):
    Q[(state, action)] = value

def choose_action(state: tuple, intent: str, epsilon=0.1) -> dict:
    available_actions = ACTIONS_BY_INTENT.get(intent, [])
    if random.random() < epsilon or not available_actions:
        return random.choice(available_actions) if available_actions else {"action": "NO_ACTION", "bot_response": "No actions."}
    return max(available_actions, key=lambda act: get_Q_value(state, act["action"]))

def update_q_learning(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    best_future_val = max([get_Q_value(next_state, a["action"]) for acts in ACTIONS_BY_INTENT.values() for a in acts], default=0.0)
    updated_q = get_Q_value(state, action) + alpha * (reward + gamma * best_future_val - get_Q_value(state, action))
    set_Q_value(state, action, updated_q)
    save_q_table_to_mongo()

# ----------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_actions_from_csv()
    load_q_table_from_mongo()

@app.post("/select_action", response_model=SelectActionResponse)
def select_action(request: SelectActionRequest):
    global last_state, last_action, last_user_id
    state = make_state(request.user_intent, request.last_message)
    chosen = choose_action(state, request.user_intent)
    last_state, last_action, last_user_id = state, chosen["action"], request.user_id
    return SelectActionResponse(**chosen)

@app.post("/update_reward", response_model=UpdateRewardResponse)
def update_reward(request: UpdateRewardRequest):
    global last_state, last_action
    if last_state and last_action:
        next_state = make_state("feedback", request.new_user_message or "")
        update_q_learning(last_state, last_action, request.reward, next_state)
        log_interaction(request.user_id, last_state, last_action, request.reward, next_state)
        return UpdateRewardResponse(status="Updated")
    return UpdateRewardResponse(status="No update")

# MAIN
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
