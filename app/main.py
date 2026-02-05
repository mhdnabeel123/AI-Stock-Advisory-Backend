import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from models.ml_engine import MLEngine
from agent.decision_agent import DecisionAgent


app = FastAPI(
    title="AI Stock Advisory API",
    version="1.0"
)

# ------------------------
# GLOBAL ML ENGINE (lazy-loaded)
# ------------------------
ml_engine = None


# ------------------------
# Schemas
# ------------------------
class ChatRequest(BaseModel):
    message: str
    capital: int = 10000
    risk: str = "medium"


class ChatResponse(BaseModel):
    reply: str
    action: str
    invest_amount: int


# ------------------------
# Startup event (IMPORTANT)
# ------------------------
@app.on_event("startup")
def load_ml_engine():
    global ml_engine
    print("🚀 Loading ML Engine...")
    ml_engine = MLEngine()
    print("✅ ML Engine loaded")


# ------------------------
# Health check
# ------------------------
@app.get("/")
def health():
    return {"status": "AI Stock Advisory API is running"}


# ------------------------
# Chat endpoint
# ------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest):
    if ml_engine is None:
        return {
            "reply": "System is warming up, please try again.",
            "action": "HOLD",
            "invest_amount": 0
        }

    # ML prediction
    prob_up = ml_engine.predict_probability()

    # Decision agent
    agent = DecisionAgent(
        capital=data.capital,
        risk=data.risk
    )
    decision = agent.decide(prob_up)

    return {
        "reply": decision["reply"],
        "action": decision["action"],
        "invest_amount": decision["invest_amount"]
    }
