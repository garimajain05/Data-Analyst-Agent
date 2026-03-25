# Data-Analyst-Agent

A **multi-agent restaurant review analyst** that collects real Yelp reviews at runtime, runs exploratory data analysis (EDA), and generates data-backed hypotheses about customer experience.

## Architecture

```
User input (restaurant name)
        │
        ▼
OrchestratorAgent          ← coordinates the pipeline
  ├─ CollectorAgent         ← streams Yelp reviews from HuggingFace at runtime
  ├─ AnalysisAgent          ← VADER sentiment + keyword themes + pandas EDA
  └─ HypothesisAgent        ← Claude claude-sonnet-4-6 (or rule-based fallback)
        │
        ▼
FastAPI backend  ◄──►  Streamlit frontend
```

## Code → Pipeline Mapping

| Stage | File | Class / Function |
|-------|------|-----------------|
| **Collect** | `data_agents.py` | `CollectorAgent.collect_reviews` |
| **EDA** | `data_agents.py` | `AnalysisAgent.analyze` |
| **Hypothesize** | `data_agents.py` | `HypothesisAgent.generate_hypothesis` |
| Orchestration | `data_agents.py` | `OrchestratorAgent.run` |
| API | `app.py` | `POST /analyze` |
| UI | `streamlit_app.py` | — |

## How It Works

### Data Collection (`CollectorAgent`)

- Loads [`yelp_review_full`](https://huggingface.co/datasets/yelp_review_full) from HuggingFace in **streaming mode** at runtime — no local download required.
- Scans the first 5,000 rows and filters for reviews whose text mentions the queried restaurant name.
- If fewer than 30 matches are found, falls back to a 200-row random sample (flagged in the UI).
- Supports dynamic queries: different restaurant names, optional location strings, and optional comparison restaurants.
- Underlying dataset: **650,000+ Yelp reviews** (train split).

### Exploratory Data Analysis (`AnalysisAgent`)

All computation uses **pandas** and **VADER**:

- **Sentiment classification**: each review is scored with VADER's `compound` score; reviews ≥ 0.05 → positive, ≤ −0.05 → negative, else neutral.
- **Sentiment distribution**: `value_counts()` → counts + percentages.
- **Theme extraction**: keyword matching across 5 themes (food quality, service, wait time, ambience, price).
- **Theme frequency**: count of reviews mentioning each theme.
- **Theme × sentiment grouped stats**: for each theme, breakdown of positive / neutral / negative reviews.
- **Rolling sentiment trend**: 20-review rolling average of compound scores (pandas `.rolling()`).
- **Rating distribution**: `value_counts()` on star ratings.

### Hypothesis Generation (`HypothesisAgent`)

- Sends the full analysis summary to **Claude claude-sonnet-4-6** (Anthropic API) with a prompt asking for a 2–3 sentence data-backed hypothesis.
- References specific percentages and counts from the analysis.
- Falls back to a deterministic rule-based statement when `ANTHROPIC_API_KEY` is not set.

## Grab-Bag Features

- **Data visualisation**: Plotly charts — sentiment pie, theme bar, theme × sentiment heatmap, rating distribution, sentiment trend line.
- **Structured output**: all results returned as typed JSON (Pydantic-validated at the API layer).
- **Artifacts**: "Download analysis JSON" button in the UI saves the full result.
- **Comparison mode**: analyze two restaurants side-by-side with a head-to-head sentiment bar chart.

## Setup (Local)

### Prerequisites

- Python ≥ 3.11
- `uv` (recommended) or `pip`

### Steps

```bash
git clone <repo-url>
cd Data-Analyst-Agent

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install uv
uv pip install -e .

# Set your Anthropic API key (optional — falls back to rule-based if not set)
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY

# Terminal 1 – start backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 – start frontend
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

### Quick API test

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"restaurant_name":"Shake Shack","location":"New York"}'
```

## Deployment

The app is split into a FastAPI backend and a Streamlit frontend. Both can be deployed independently.

### Backend (Render / Railway / Fly)

```bash
# Render: set start command to:
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Frontend (Streamlit Community Cloud)

1. Push repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `streamlit_app.py`.
3. Add secret: `API_URL = https://your-backend.onrender.com`.

### Live URL

`https://<your-deployment>.onrender.com` ← replace after deploying

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | No | Enables Claude-powered hypothesis generation |
| `API_URL` | No | Backend URL for Streamlit (default: `http://localhost:8000`) |

## Project Structure

```
Data-Analyst-Agent/
├── data_agents.py      # All 4 agents (Collector, Analysis, Hypothesis, Orchestrator)
├── app.py              # FastAPI backend
├── streamlit_app.py    # Streamlit frontend
├── pyproject.toml      # uv/pip dependencies
├── .env.example        # Environment variable template
└── README.md
```
