"""
FastAPI backend for the restaurant review analyst.

Endpoints:
  GET  /           – health check
  POST /analyze    – run the full Collect → EDA → Hypothesize pipeline
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_agents import OrchestratorAgent

app = FastAPI(
    title="Restaurant Review Analyst API",
    description="Multi-agent system: CollectorAgent → AnalysisAgent → HypothesisAgent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiated once at startup to avoid re-loading VADER on every request
orchestrator = OrchestratorAgent()


class AnalyzeRequest(BaseModel):
    restaurant_name: str
    location: str = ""
    compare_restaurant: Optional[str] = None


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "restaurant-review-analyst"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if not req.restaurant_name.strip():
        raise HTTPException(status_code=400, detail="restaurant_name is required")
    try:
        return orchestrator.run(
            req.restaurant_name.strip(),
            req.location.strip(),
            req.compare_restaurant.strip() if req.compare_restaurant else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
