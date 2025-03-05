# rl_service.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import csv
import os
from datetime import datetime
from pymongo import MongoClient

app = FastAPI()

# ----------------------------------------------------------------
# MongoDB Setup
# ----------------------------------------------------------------
# Replace with your actual MongoDB Atlas connection string
MONGO_CONNECTION_STRING = "mongodb+srv://cmtong123:20020430@cluster0.d6vff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["rl_database"]  # use your preferred database name
logs_collection = db["action_logs"]
q_table_collection = db["q_table"]

# ----------------------------------------------------------------
# 1. LOAD ACTIONS FROM CSV
# ----------------------------------------------------------------
# Global dictionary: keys are user_intent, values are lists of dictionaries
# Each dictionary contains an "action" and "bot_response"
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
    """
    Save the current Q-table to MongoDB.
    This function clears the previous Q-table entries and inserts the updated ones.
    """
    q_table_collection.delete_many({})  # Remove old Q-table entries
    docs = []
    for (state, action), value in Q.items():
        docs.append({
            "state": str(state),  # Convert tuple to string
            "action": action,
            "value": value
        })
    if docs:
        q_table_collection.insert_many(docs)

def load_q_table_from_mongo():
    """
    Load the Q-table from MongoDB and update the global Q dictionary.
    """
    global Q
    Q.clear()
    for doc in q_table_collection.find():
        # Here we store state as a string; in production, parse it safely
        state = eval(doc["state"])  # Caution: eval can be unsafe in production.
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
    user_intent: str
    feedback_label: Optional[str] = None  # "positive", "negative", or None

class FlowiseInteractionResponse(BaseModel):
    status: str
    selected_action: dict  # Contains "action" and "bot_response"

# ----------------------------------------------------------------
# 4. Q-LEARNING LOGIC
# ----------------------------------------------------------------
Q = {}
last_state = None
last_action = None
last_user_id = None

def map_intent_to_number(intent: str) -> int:
    mapping = {"motivation_request": 0, "feedback": 1, "general_query": 2}
    return mapping.get(intent, 99)

def get_sentiment_score(message: str) -> float:
    text = message.lower()
    if "unmotivated" in text or "bad" in text or "frustrated" in text:
        return -1.0
    elif "good" in text or "great" in text or "thanks" in text:
        return 1.0
    return 0.0

def make_state(user_intent: str, last_message: str) -> tuple:
    return (map_intent_to_number(user_intent), get_sentiment_score(last_message))

def get_Q_value(state: tuple, action: str) -> float:
    return Q.get((state, action), 0.0)

def set_Q_value(state: tuple, action: str, value: float):
    Q[(state, action)] = value

def choose_action(state: tuple, intent: str, epsilon=0.1) -> dict:
    # Filter available actions by the current user_intent
    available_actions = ACTIONS_BY_INTENT.get(intent, [])
    if not available_actions:
        return {"action": "NO_ACTIONS_FOR_INTENT", "bot_response": "No action defined for this intent."}
    
    # Epsilon-greedy: explore with probability epsilon
    if random.random() < epsilon:
        return random.choice(available_actions)
    
    # Otherwise, select the action with highest Q-value
    best_action = None
    best_q = float("-inf")
    for act in available_actions:
        a = act["action"]
        q_val = get_Q_value(state, a)
        if q_val > best_q:
            best_q = q_val
            best_action = act
    # Fallback to random choice if no best action is found
    if best_action is None:
        best_action = random.choice(available_actions)
    return best_action

def update_q_learning(state: tuple, action: str, reward: float,
                      next_state: tuple, alpha=0.1, gamma=0.9):
    old_value = get_Q_value(state, action)
    # For updating Q, consider all actions regardless of intent
    future_qs = [get_Q_value(next_state, a["action"]) for acts in ACTIONS_BY_INTENT.values() for a in acts]
    best_future_val = max(future_qs) if future_qs else 0.0
    new_value = old_value + alpha * (reward + gamma * best_future_val - old_value)
    set_Q_value(state, action, new_value)
    # Save Q-table to MongoDB after each update
    save_q_table_to_mongo()

# ----------------------------------------------------------------
# 5. ENDPOINTS
# ----------------------------------------------------------------
@app.post("/select_action", response_model=SelectActionResponse)
def select_action(request: SelectActionRequest) -> SelectActionResponse:
    global last_state, last_action, last_user_id
    state = make_state(request.user_intent, request.last_message)
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

    # For reward updates, we use "feedback" as intent and new_user_message for sentiment
    next_state = make_state("feedback", request.new_user_message or "")
    update_q_learning(last_state, last_action, request.reward, next_state)

    # Log interaction in MongoDB
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
                "state": str(key[0]),
                "action": key[1],
                "value": value
            }
            for key, value in Q.items()
        ]
    }

# NEW: A simplified endpoint to handle both feedback and next action from Flowise.
# This endpoint now expects only 'user_intent' and 'feedback_label'.
@app.post("/flowise_interaction", response_model=FlowiseInteractionResponse)
def flowise_interaction(request: FlowiseInteractionRequest) -> FlowiseInteractionResponse:
    global last_state, last_action, last_user_id

    # 1) If we have a previous action and a feedback_label is provided, update the Q-table.
    if last_state is not None and last_action is not None and request.feedback_label is not None:
        if request.feedback_label.lower() == "positive":
            reward = 1.0
        elif request.feedback_label.lower() == "negative":
            reward = -1.0
        else:
            reward = 0.0

        # Use an empty string as last_message for state computation.
        next_state = make_state("feedback", "")
        update_q_learning(last_state, last_action, reward, next_state)
        log_interaction(
            user_id=last_user_id,
            state=last_state,
            action=last_action,
            reward=reward,
            new_state=next_state
        )

    # 2) Select a new action based on the provided user_intent.
    new_state = make_state(request.user_intent, "")
    chosen_action = choose_action(new_state, intent=request.user_intent)

    # Update global tracking variables for subsequent updates.
    last_state = new_state
    last_action = chosen_action["action"]
    # If you don't receive user_id here, you can leave last_user_id unchanged or set to None.
    last_user_id = None

    return FlowiseInteractionResponse(
        status="Q-table updated and action selected.",
        selected_action=chosen_action
    )

# ----------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
