import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pymongo import MongoClient
import gridfs
from collections import deque
import csv

app = FastAPI()

# ----------------------------------------------------------------
# Load CSV Mapping for Actions and Prompts
# ----------------------------------------------------------------
# The CSV file (actions.csv) should have two columns:
# persuasive_type,system_prompt
ACTIONS_PROMPTS = {}
try:
    with open("actions.csv", "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row["persuasive_type"].strip().lower()
            ACTIONS_PROMPTS[key] = row["system_prompt"]
except Exception as e:
    print("Error loading actions.csv:", e)

# ----------------------------------------------------------------
# MongoDB Setup (GridFS for Model Persistence)
# ----------------------------------------------------------------
MONGO_CONNECTION_STRING = "mongodb+srv://cmtong123:20020430@cluster0.d6vff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_CONNECTION_STRING)
db = client["rl_database"]
fs = gridfs.GridFS(db)  # Initialize GridFS for storing model

MODEL_FILENAME = "dqn_model.pth"

# ----------------------------------------------------------------
# DQN Model Definition
# ----------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# State and action dimensions
STATE_SIZE = 10  # Placeholder for real state representation
ACTION_SIZE = 3  # Number of possible persuasive types

# The order here should match the persuasive_type in your CSV if possible.
ACTIONS = ["supportive", "motivational", "informative"]

# Initialize model, optimizer, and memory for experience replay
model = DQN(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = deque(maxlen=1000)  # Experience replay buffer

# ----------------------------------------------------------------
# Model Persistence (Save/Load from MongoDB GridFS)
# ----------------------------------------------------------------
def save_model_to_db():
    """Save the trained model to MongoDB GridFS."""
    existing = fs.find_one({"filename": MODEL_FILENAME})
    if existing:
        fs.delete(existing._id)  # Remove old model

    with open(MODEL_FILENAME, "wb") as f:
        torch.save(model.state_dict(), f)

    with open(MODEL_FILENAME, "rb") as f:
        file_id = fs.put(f, filename=MODEL_FILENAME)
        print(f"Model saved to MongoDB with file ID: {file_id}")

def load_model_from_db():
    """Load the trained model from MongoDB GridFS."""
    file = fs.find_one({"filename": MODEL_FILENAME})
    if file:
        with open(MODEL_FILENAME, "wb") as f:
            f.write(file.read())  # Write model to disk

        model.load_state_dict(torch.load(MODEL_FILENAME))
        model.eval()
        print("Model loaded from MongoDB.")
    else:
        print("No model found in MongoDB. Training from scratch.")

# ----------------------------------------------------------------
# Data Models for Flowise Interaction
# ----------------------------------------------------------------
class FlowiseInteractionRequest(BaseModel):
    session_id: str
    user_intent: str
    feedback_label: Optional[str] = None

class FlowiseInteractionResponse(BaseModel):
    status: str
    selected_prompt: str

# ----------------------------------------------------------------
# Experience Replay and Q-learning Logic
# ----------------------------------------------------------------
EPSILON = 0.1
GAMMA = 0.9
ALPHA = 0.1

def get_state_representation(user_intent: str) -> np.array:
    """Convert user intent into a numerical state representation."""
    state_vector = np.random.rand(STATE_SIZE)  # Placeholder: replace with actual features
    return state_vector

def select_action(state):
    """Selects an action using an epsilon-greedy strategy."""
    if random.random() < EPSILON:
        action_index = random.randint(0, ACTION_SIZE - 1)  # Explore
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_index = torch.argmax(model(state_tensor)).item()  # Exploit
    return ACTIONS[action_index]

def replay():
    """Train the DQN using experience replay."""
    if len(memory) < 32:
        return
    batch = random.sample(memory, 32)
    states, actions, rewards, next_states = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor([ACTIONS.index(a) for a in actions], dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    max_next_q_values = model(next_states).max(1)[0].detach()
    target_q_values = rewards + (GAMMA * max_next_q_values)

    loss = nn.functional.mse_loss(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if len(memory) % 50 == 0:
        save_model_to_db()

# ----------------------------------------------------------------
# Endpoints (Only startup_event and flowise_interaction are used)
# ----------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    load_model_from_db()

@app.post("/flowise_interaction", response_model=FlowiseInteractionResponse)
def flowise_interaction(request: FlowiseInteractionRequest) -> FlowiseInteractionResponse:
    # Check if the user_intent is out_of_scope
    if request.user_intent.lower() == "out_of_scope":
        return FlowiseInteractionResponse(
            status="error",
            selected_prompt="Sorry, I cannot answer your question, please ask a health related question."
        )
    
    state = get_state_representation(request.user_intent)
    action = select_action(state)
    # Get the system prompt from the CSV mapping based on the selected persuasive type
    selected_prompt = ACTIONS_PROMPTS.get(action.lower(), f"Generated response in {action} tone.")

    # Assign rewards based on user feedback
    reward = 1.0 if request.feedback_label == "positive" else -1.0 if request.feedback_label == "negative" else 0.0
    next_state = get_state_representation(request.user_intent)

    memory.append((state, action, reward, next_state))
    replay()

    return FlowiseInteractionResponse(
        status="Action selected successfully",
        selected_prompt=selected_prompt
    )

# ----------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
