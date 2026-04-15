# Data-Analyst-Agent

A **multi-agent restaurant review analyst** that fetches real Google Places reviews at runtime, runs exploratory data analysis (EDA), and generates data-backed hypotheses about customer experience.

**Mandatory Fields**: Restaurant Name and Location

## App Link: 
https://data-analyst-frontend-v2-940502312023.us-central1.run.app

## Architecture

```
User input (restaurant name + optional location)
        │
        ▼
OrchestratorAgent              ← coordinates the pipeline
  ├─ CollectorAgent             ← Google Places API: Text Search → Place Details
  ├─ AnalysisAgent              ← VADER sentiment + keyword themes + pandas EDA
  └─ HypothesisAgent            ← PydanticAI agent (Claude) with tool-calling
        │
        ▼
FastAPI backend  ◄──►  Streamlit frontend
```

## Agent Framework — PydanticAI

`HypothesisAgent` is built on **[PydanticAI](https://ai.pydantic.dev/)** backed by **Claude claude-haiku-4-5** (Anthropic). It uses PydanticAI's typed dependency-injection system (`RunContext[HypothesisDeps]`) to expose three registered tools to the model:

| Tool | What it returns |
|---|---|
| `get_sentiment_data` | Positive / neutral / negative percentages and counts |
| `get_theme_data` | Theme frequency percentages and per-theme sentiment breakdown |
| `get_rating_data` | Overall rating, sample average, and star distribution |

The agent calls these tools autonomously, then synthesises a 2–3 sentence data-backed hypothesis. Falls back to deterministic rule-based logic when `ANTHROPIC_API_KEY` is not set, so the pipeline always produces output.

## Three-Step Implementation

### Step 1: Collect — `data_agents.py` → `CollectorAgent`

- **Method:** Google Places API (REST, two calls per restaurant)
  1. **Text Search** (`/textsearch/json`) — finds the best-matching `place_id` for the query string (name + optional location).
  2. **Place Details** (`/details/json`) — fetches up to **5 real user reviews** (Google's public limit) _and_ the **overall star rating** (aggregated across all ratings, which can be thousands).
- Data retrieved at runtime: different restaurant names → completely different API calls and results.
- Graceful fallback: if `GOOGLE_PLACES_API_KEY` is not set, clearly-labelled placeholder rows are used so the pipeline still runs end-to-end.

### Step 2: Explore and Analyze — `data_agents.py` → `AnalysisAgent`

EDA is implemented as **five distinct tool methods** that each surface a specific finding:

| Tool Method | What it surfaces |
|---|---|
| `compute_sentiment_distribution` | % positive / neutral / negative (VADER compound score) |
| `compute_theme_frequency` | Which topics (food, service, wait, ambience, price) appear most |
| `compute_theme_by_sentiment` | Per-theme sentiment breakdown — which theme has the most negative mentions |
| `compute_rating_distribution` | Count of 1★–5★ ratings in the fetched sample |
| `compute_sentiment_trend` | Rolling average compound score across reviews |

### Step 3: Hypothesize — `data_agents.py` → `HypothesisAgent`

- **Primary path** (requires `ANTHROPIC_API_KEY`): PydanticAI agent with Claude claude-haiku-4-5. The model calls the three registered tools to retrieve EDA results and synthesises a natural-language hypothesis.
- **Fallback path**: deterministic rules referencing specific percentages, counts, and theme names from the EDA output.
- Example output: *"Based on 5 sampled reviews, Shake Shack (Google overall: 4.3/5 across 2,847 ratings) enjoys a strongly positive reception. 80% of reviews express positive sentiment with an average sample rating of 4.4/5. Food quality is the most-discussed theme (80% of reviews), suggesting it is the primary driver of customer satisfaction."*

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
| **Agent framework** | PydanticAI — `HypothesisAgent` is a `pydantic_ai.Agent` with three `@agent.tool` functions and typed `RunContext` dependency injection |
| **Tool calling** | Three PydanticAI tools on `HypothesisAgent`; five EDA tool methods on `AnalysisAgent`; Text Search + Place Details on `CollectorAgent` |
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

---

## Setup (Local)

### Prerequisites

- Python ≥ 3.11
- A **Google Places API key** (for live review data)
- An **Anthropic API key** (for LLM-generated hypotheses — optional, falls back to rule-based)

### Steps

```bash
git clone <repo-url>
cd Data-Analyst-Agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install uv
uv pip install -e .

cp .env.example .env
# edit .env and set:
#   GOOGLE_PLACES_API_KEY=your-google-key
#   ANTHROPIC_API_KEY=your-anthropic-key   (optional)

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

---

## Deployment (Google Cloud Run)

Both services deploy via Cloud Build. All API keys are stored in **Secret Manager** — never hardcoded in any file.

### Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk) installed and authenticated
- A GCP project with billing enabled

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Step 1 — Enable Required APIs

```bash
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  containerregistry.googleapis.com \
  places-backend.googleapis.com \
  places.googleapis.com
```

### Step 2 — Get Your API Keys

**Google Places API key:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com) → **APIs & Services → Credentials**
2. Click **+ Create Credentials → API Key** and copy the key
3. Optional: click **Restrict Key** and limit it to the Places API only

**Anthropic API key:**
1. Go to [console.anthropic.com](https://console.anthropic.com) → **API Keys**
2. Click **Create Key** and copy it (`sk-ant-...`)

### Step 3 — Store Both Keys in Secret Manager

```bash
# Google Places API key
echo -n "YOUR_GOOGLE_PLACES_KEY" | \
  gcloud secrets create GOOGLE_PLACES_API_KEY \
    --data-file=- \
    --replication-policy=automatic

# Anthropic API key
echo -n "sk-ant-YOUR_ANTHROPIC_KEY" | \
  gcloud secrets create ANTHROPIC_API_KEY \
    --data-file=- \
    --replication-policy=automatic

# Verify both were stored
gcloud secrets list
```

To update a key later:
```bash
echo -n "NEW_KEY" | gcloud secrets versions add ANTHROPIC_API_KEY --data-file=-
echo -n "NEW_KEY" | gcloud secrets versions add GOOGLE_PLACES_API_KEY --data-file=-
```

### Step 4 — Grant Cloud Run Access to the Secrets

```bash
PROJECT_NUMBER=$(gcloud projects describe \
  $(gcloud config get-value project) \
  --format='value(projectNumber)')

# Grant access to both secrets
for SECRET in GOOGLE_PLACES_API_KEY ANTHROPIC_API_KEY; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

### Step 5 — Deploy the Backend

```bash
gcloud builds submit --config cloudbuild-backend.yaml .
```

Note the service URL printed at the end (e.g. `https://data-analyst-backend-v2-XXX-uc.a.run.app`).

Verify both keys are injected:

```bash
# Should return {"GOOGLE_PLACES_API_KEY_set": true, ...}
curl https://YOUR-BACKEND-URL/debug
```

### Step 6 — Deploy the Frontend

Open `cloudbuild-frontend.yaml` and update `API_URL` to your backend URL from Step 5:

```yaml
- '--set-env-vars=API_URL=https://YOUR-BACKEND-URL'
```

Then deploy:

```bash
gcloud builds submit --config cloudbuild-frontend.yaml .
```

The frontend URL will be printed when the build completes.

---

### Troubleshooting

**`Permission denied on secret` during deploy**

The compute service account doesn't have access to a secret. Run:

```bash
PROJECT_NUMBER=$(gcloud projects describe \
  $(gcloud config get-value project) \
  --format='value(projectNumber)')

gcloud secrets add-iam-policy-binding ANTHROPIC_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Then redeploy:
```bash
gcloud builds submit --config cloudbuild-backend.yaml .
```

**Still seeing placeholder reviews after deployment?**

```bash
# 1. Confirm the secret exists
gcloud secrets versions access latest --secret=GOOGLE_PLACES_API_KEY

# 2. Check the debug endpoint
curl https://YOUR-BACKEND-URL/debug

# 3. If a previous deploy used --set-env-vars, remove the collision first
gcloud run services update data-analyst-backend-v2 \
  --region=us-central1 \
  --remove-env-vars=GOOGLE_PLACES_API_KEY

# Then redeploy
gcloud builds submit --config cloudbuild-backend.yaml .
```

**`PROJECT_NUMBER` is empty?**

```bash
# Run separately to debug
gcloud config get-value project
gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)'
```

---

## Environment Variables

| Variable | Required | Where set | Description |
|---|---|---|---|
| `GOOGLE_PLACES_API_KEY` | Yes (for live data) | Secret Manager / `.env` | Enables real review fetching via Google Places API |
| `ANTHROPIC_API_KEY` | No (falls back to rule-based) | Secret Manager / `.env` | Enables LLM-generated hypotheses via PydanticAI + Claude |
| `API_URL` | No | Cloud Run env var / `.env` | Backend URL for Streamlit (default: `http://localhost:8000`) |

## Project Structure

```
Data-Analyst-Agent/
├── data_agents.py          # All 4 agents (Collector, Analysis, Hypothesis, Orchestrator)
│                           # + PydanticAI agent definition and tools
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── pyproject.toml          # Dependencies (includes pydantic-ai[anthropic])
├── Dockerfile.backend
├── Dockerfile.frontend
├── cloudbuild-backend.yaml
├── cloudbuild-frontend.yaml
├── .env.example            # Environment variable template
└── README.md
```

## Google Places API — Key Facts

- **Text Search**: finds place_id from a free-text query (name + optional city).
- **Place Details**: returns `name`, `rating` (aggregate), `user_ratings_total`, and up to **5 `reviews`** (each with `text`, `rating`, `author_name`, `relative_time_description`).
- **Free tier**: $200/month credit → ~40,000 Text Search calls or ~17,000 Place Details calls per month.
- Get a key: [console.cloud.google.com/apis](https://console.cloud.google.com/apis) → Enable "Places API".
