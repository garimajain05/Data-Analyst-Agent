# Data-Analyst-Agent

A **multi-agent restaurant review analyst** that fetches real Google Places reviews at runtime, runs exploratory data analysis (EDA), and generates data-backed hypotheses about customer experience — **no LLM API key required**.

## Architecture

```
User input (restaurant name + optional location)
        │
        ▼
OrchestratorAgent              ← coordinates the pipeline
  ├─ CollectorAgent             ← Google Places API: Text Search → Place Details
  ├─ AnalysisAgent              ← VADER sentiment + keyword themes + pandas EDA
  └─ HypothesisAgent            ← deterministic rule-based hypothesis generator
        │
        ▼
FastAPI backend  ◄──►  Streamlit frontend
```

## Three-Step Implementation

### Step 1: Collect — `data_agents.py` → `CollectorAgent`

- **Method:** Google Places API (REST, two calls per restaurant)
  1. **Text Search** (`/textsearch/json`) — finds the best-matching `place_id` for the query string (name + optional location).
  2. **Place Details** (`/details/json`) — fetches up to **5 real user reviews** (Google's public limit) _and_ the **overall star rating** (aggregated across all ratings, which can be thousands).
- Data retrieved at runtime: different restaurant names → completely different API calls and results.
- Graceful fallback: if `GOOGLE_PLACES_API_KEY` is not set, clearly-labelled placeholder rows are used so the pipeline still runs end-to-end.
- **Why non-trivial?** The underlying Google ratings database covers millions of places and billions of reviews — the 5 fetched reviews are a live slice of that corpus, dynamically selected by Google's ranking algorithm.

### Step 2: Explore and Analyze — `data_agents.py` → `AnalysisAgent`

EDA is implemented as **five distinct tool methods** that each surface a specific finding:

| Tool Method | What it surfaces |
|---|---|
| `compute_sentiment_distribution` | % positive / neutral / negative (VADER compound score) |
| `compute_theme_frequency` | Which topics (food, service, wait, ambience, price) appear most |
| `compute_theme_by_sentiment` | Per-theme sentiment breakdown — which theme has the most negative mentions |
| `compute_rating_distribution` | Count of 1★–5★ ratings in the fetched sample |
| `compute_sentiment_trend` | Rolling average compound score across reviews |

Each tool is called sequentially by `AnalysisAgent.analyze()` and its output feeds directly into the hypothesis step. The EDA adapts to different inputs: a restaurant with uniformly positive reviews will produce different theme-by-sentiment findings than one with mixed reviews.

### Step 3: Hypothesize — `data_agents.py` → `HypothesisAgent`

- Generates a **2–3 sentence data-backed hypothesis** using deterministic rules — no LLM required.
- Three rule branches (positive-dominant, negative-dominant, mixed) each reference **specific percentages, counts, and theme names** drawn from the EDA output.
- Cites Google's overall rating (across potentially thousands of ratings) alongside the sample-level sentiment, giving a richer evidential basis.
- Example output: *"Based on 5 sampled reviews, Shake Shack (Google overall: 4.3/5 across 2,847 ratings) enjoys a strongly positive reception. 80% of reviews express positive sentiment with an average sample rating of 4.4/5. The most-discussed topic is food quality (80% of reviews), suggesting it is the primary driver of customer satisfaction."*

## Code → Pipeline Mapping

| Stage | File | Class / Method |
|---|---|---|
| **Collect** | `data_agents.py` | `CollectorAgent.collect_reviews` |
| **EDA** | `data_agents.py` | `AnalysisAgent.analyze` + 5 tool methods |
| **Hypothesize** | `data_agents.py` | `HypothesisAgent.generate_hypothesis` |
| Orchestration | `data_agents.py` | `OrchestratorAgent.run` |
| API | `app.py` | `POST /analyze` |
| UI | `streamlit_app.py` | — |

## Core Requirements

| Requirement | Implementation |
|---|---|
| **Frontend** | Streamlit (`streamlit_app.py`) with sidebar inputs, Plotly charts, inline review cards |
| **Agent framework** | Custom orchestrator-handoff pattern (`OrchestratorAgent` → sub-agents) |
| **Tool calling** | Five EDA tool methods on `AnalysisAgent`; Text Search + Place Details on `CollectorAgent` |
| **Non-trivial dataset** | Google Places (billions of reviews, live API, dynamic per query) |
| **Multi-agent pattern** | Orchestrator-handoff: `OrchestratorAgent` hands off to `CollectorAgent`, `AnalysisAgent`, `HypothesisAgent` in sequence; fan-out for comparison mode |
| **Deployed** | See deployment section below |
| **README** | This file |

## Grab-Bag Features (≥ 2 implemented)

### ✅ Structured Output
All agent outputs are typed Python dicts validated by Pydantic at the FastAPI layer (`AnalyzeRequest` model, typed return dict). The `AnalysisAgent` methods each return well-defined dict schemas consumed downstream. **File:** `app.py` (`AnalyzeRequest`), `data_agents.py` (`AnalysisAgent.analyze` return dict).

### ✅ Data Visualization
Five Plotly charts rendered inline in the Streamlit UI:
- Sentiment donut pie chart
- Theme frequency bar chart
- Theme × sentiment heatmap
- Sentiment trend line chart
- Star rating distribution bar chart
- Head-to-head sentiment comparison (comparison mode)

**File:** `streamlit_app.py` → `render_analysis()`

### ✅ Artifacts
"Download analysis JSON" button exports the full analysis dict as a `.json` file. **File:** `streamlit_app.py` → `st.download_button`.

## Setup (Local)

### Prerequisites

- Python ≥ 3.11
- A **Google Places API key** (free tier, enable "Places API" in Google Cloud Console)

### Steps

```bash
git clone <repo-url>
cd Data-Analyst-Agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install uv
uv pip install -e .

cp .env.example .env
# edit .env → set GOOGLE_PLACES_API_KEY=your-key-here

# Terminal 1 – backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 – frontend
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

### Backend (Render / Railway / Fly)

```bash
# start command:
uvicorn app:app --host 0.0.0.0 --port $PORT
# environment variables: GOOGLE_PLACES_API_KEY
```

### Frontend (Streamlit Community Cloud)

1. Push repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → select `streamlit_app.py`.
3. Add secrets: `API_URL = https://your-backend.onrender.com`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_PLACES_API_KEY` | **Yes** (for live data) | Google Places API key. Without it, placeholder reviews are used. |
| `API_URL` | No | Backend URL for Streamlit (default: `http://localhost:8000`) |

## Project Structure

```
Data-Analyst-Agent/
├── data_agents.py      # All 4 agents (Collector, Analysis, Hypothesis, Orchestrator)
├── app.py              # FastAPI backend
├── streamlit_app.py    # Streamlit frontend
├── pyproject.toml      # dependencies
├── .env.example        # Environment variable template
└── README.md
```

## Google Places API — Key Facts

- **Text Search**: finds place_id from a free-text query (name + optional city).
- **Place Details**: returns `name`, `rating` (aggregate), `user_ratings_total`, and up to **5 `reviews`** (each with `text`, `rating`, `author_name`, `relative_time_description`).
- **Free tier**: $200/month credit → ~40,000 Text Search calls or ~17,000 Place Details calls per month.
- Get a key: [console.cloud.google.com](https://console.cloud.google.com/) → Enable "Places API".