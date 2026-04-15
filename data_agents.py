"""
Multi-agent data analyst pipeline for restaurant reviews.

Agents:
  CollectorAgent    - fetches real reviews + star ratings via Google Places API
  AnalysisAgent     - VADER sentiment + keyword theme extraction + pandas EDA
  HypothesisAgent   - PydanticAI agent (Claude claude-haiku-4-5) with tool-calling;
                      falls back to rule-based logic when ANTHROPIC_API_KEY is not set
  OrchestratorAgent - coordinates Collect → EDA → Hypothesize

Agent framework: PydanticAI (pydantic-ai) — https://ai.pydantic.dev/
  HypothesisAgent is a PydanticAI Agent backed by Claude claude-haiku-4-5 via the
  Anthropic API.  It calls three @agent.tool-decorated functions
  (get_sentiment_data, get_theme_data, get_rating_data) that receive analysis
  results through PydanticAI's typed dependency-injection system (RunContext),
  then synthesises a data-backed hypothesis.
  Set ANTHROPIC_API_KEY to enable the LLM path.

Data source: Google Places API (Text Search + Place Details)
  - Fetches up to 5 reviews per restaurant (Google's free-tier limit)
  - Returns real star ratings (1-5) alongside review text
  - Dynamic: different restaurant names / locations → different API calls
"""

import os
import statistics
from dataclasses import dataclass
from typing import Optional

import httpx
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# PydanticAI setup
# ---------------------------------------------------------------------------
try:
    from pydantic_ai import Agent, RunContext
    _PYDANTIC_AI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYDANTIC_AI_AVAILABLE = False


@dataclass
class HypothesisDeps:
    """Typed dependency injected into every PydanticAI tool call."""
    analysis: dict


if _PYDANTIC_AI_AVAILABLE:
    # Tool functions are defined here at module level so they are importable
    # and inspectable.  The Agent itself is created lazily inside
    # HypothesisAgent._pydanticai_hypothesis() so that the Anthropic provider
    # does not attempt to read ANTHROPIC_API_KEY at import time.

    def get_sentiment_data(ctx: "RunContext[HypothesisDeps]") -> dict:
        """Return sentiment distribution percentages and counts from the reviews."""
        return {
            "sentiment_pct": ctx.deps.analysis.get("sentiment_pct", {}),
            "sentiment_counts": ctx.deps.analysis.get("sentiment_counts", {}),
            "total_reviews": ctx.deps.analysis.get("total_reviews", 0),
        }

    def get_theme_data(ctx: "RunContext[HypothesisDeps]") -> dict:
        """Return theme frequency percentages and per-theme sentiment breakdown."""
        return {
            "theme_pct": ctx.deps.analysis.get("theme_pct", {}),
            "theme_by_sentiment": ctx.deps.analysis.get("theme_by_sentiment", {}),
        }

    def get_rating_data(ctx: "RunContext[HypothesisDeps]") -> dict:
        """Return rating statistics: overall rating, sample average, and distribution."""
        return {
            "restaurant_name": ctx.deps.analysis.get("restaurant_name", ""),
            "overall_rating": ctx.deps.analysis.get("overall_rating"),
            "avg_rating_from_sample": ctx.deps.analysis.get("avg_rating_from_sample"),
            "rating_distribution": ctx.deps.analysis.get("rating_distribution", {}),
            "total_ratings_on_google": ctx.deps.analysis.get("total_ratings_on_google"),
        }

    _HYPOTHESIS_TOOLS = [get_sentiment_data, get_theme_data, get_rating_data]
    _HYPOTHESIS_SYSTEM_PROMPT = (
        "You are a restaurant data analyst. Use the three available tools "
        "(get_sentiment_data, get_theme_data, get_rating_data) to retrieve "
        "EDA results, then write a concise 2–3 sentence data-backed "
        "hypothesis about the restaurant's key strengths and main area for "
        "improvement. Always reference specific percentages and numbers."
    )

# ---------------------------------------------------------------------------
# Theme keyword map
# ---------------------------------------------------------------------------
THEME_KEYWORDS: dict[str, list[str]] = {
    "food_quality": [
        "food", "taste", "flavor", "flavour", "delicious", "fresh", "menu",
        "dish", "meal", "portion", "quality", "bland", "overcooked", "raw",
        "stale", "spicy", "salty", "sweet",
    ],
    "service": [
        "service", "staff", "waiter", "waitress", "server", "bartender",
        "friendly", "rude", "attentive", "ignored", "helpful", "manager",
        "polite", "unprofessional",
    ],
    "wait_time": [
        "wait", "slow", "quick", "fast", "time", "hour", "minute",
        "busy", "rush", "delay", "forever", "prompt", "speedy",
    ],
    "ambience": [
        "atmosphere", "ambiance", "ambience", "decor", "noise", "loud",
        "cozy", "vibe", "setting", "clean", "dirty", "crowded", "spacious",
        "lighting", "music",
    ],
    "price": [
        "price", "expensive", "cheap", "cost", "worth", "value",
        "overpriced", "affordable", "pricey", "reasonable", "deal",
    ],
}

GOOGLE_PLACES_BASE = "https://maps.googleapis.com/maps/api/place"


# ---------------------------------------------------------------------------
# CollectorAgent
# ---------------------------------------------------------------------------
class CollectorAgent:
    """
    Retrieves real restaurant reviews via the Google Places API at runtime.

    Strategy:
      1. Text Search → find the best-matching place_id for the query.
      2. Place Details → fetch up to 5 reviews (Google's public limit) and
         the overall star rating for that place.
      3. Returns a DataFrame with columns [text, stars, author, time_desc]
         plus a metadata dict (place name, overall rating, total ratings).

    If GOOGLE_PLACES_API_KEY is not set, a small set of clearly-labelled
    placeholder rows is returned so the rest of the pipeline can still run.
    """

    def __init__(self) -> None:
        pass  # api_key is read lazily per-request so .env changes take effect

    @property
    def api_key(self) -> str:
        return os.environ.get("GOOGLE_PLACES_API_KEY", "")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_search(self, query: str) -> Optional[str]:
        """Return the place_id for the top result of a text search."""
        url = f"{GOOGLE_PLACES_BASE}/textsearch/json"
        params = {"query": query, "type": "restaurant", "key": self.api_key}
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]["place_id"]

    def _place_details(self, place_id: str) -> dict:
        """Fetch place name, overall rating, user_ratings_total, and reviews."""
        url = f"{GOOGLE_PLACES_BASE}/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,rating,user_ratings_total,reviews",
            "key": self.api_key,
        }
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("result", {})

    def _placeholder_df(self, restaurant_name: str) -> tuple[pd.DataFrame, dict]:
        """Used when no API key is configured."""
        rows = [
            {
                "text": "The food here is absolutely amazing, best I have had in years!",
                "stars": 5,
                "author": "Sample Reviewer A",
                "time_desc": "a week ago",
            },
            {
                "text": "Service was a bit slow but the food quality made up for it.",
                "stars": 4,
                "author": "Sample Reviewer B",
                "time_desc": "2 weeks ago",
            },
            {
                "text": "Overpriced and the wait time was ridiculous. Disappointed.",
                "stars": 2,
                "author": "Sample Reviewer C",
                "time_desc": "a month ago",
            },
            {
                "text": "Cozy ambience and friendly staff. Will definitely come back.",
                "stars": 4,
                "author": "Sample Reviewer D",
                "time_desc": "a month ago",
            },
            {
                "text": "Decent meal but nothing special. Average experience overall.",
                "stars": 3,
                "author": "Sample Reviewer E",
                "time_desc": "2 months ago",
            },
        ]
        df = pd.DataFrame(rows)
        meta = {
            "place_name": restaurant_name + " (placeholder - add GOOGLE_PLACES_API_KEY)",
            "overall_rating": round(
                sum(r["stars"] for r in rows) / len(rows), 1
            ),
            "total_ratings": len(rows),
            "is_placeholder": True,
        }
        return df, meta

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def collect_reviews(
        self,
        restaurant_name: str,
        location: str = "",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Returns (dataframe, metadata).
        dataframe columns: text, stars, author, time_desc
        metadata keys: place_name, overall_rating, total_ratings, is_placeholder
        """
        if not self.api_key:
            return self._placeholder_df(restaurant_name)

        query = restaurant_name
        if location.strip():
            query = f"{restaurant_name} {location}"

        place_id = self._text_search(query)
        if place_id is None:
            return self._placeholder_df(restaurant_name)

        details = self._place_details(place_id)
        raw_reviews = details.get("reviews", [])

        rows = [
            {
                "text": r.get("text", ""),
                "stars": r.get("rating", 3),          # per-review star rating
                "author": r.get("author_name", ""),
                "time_desc": r.get("relative_time_description", ""),
            }
            for r in raw_reviews
            if r.get("text", "").strip()
        ]

        if not rows:
            return self._placeholder_df(restaurant_name)

        df = pd.DataFrame(rows)
        meta = {
            "place_name": details.get("name", restaurant_name),
            "overall_rating": details.get("rating"),          # Google's aggregate score
            "total_ratings": details.get("user_ratings_total"),
            "is_placeholder": False,
        }
        return df, meta

    def run(
        self, restaurant_name: str, location: str = ""
    ) -> tuple[pd.DataFrame, dict]:
        return self.collect_reviews(restaurant_name, location)


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------
class AnalysisAgent:
    """
    Performs EDA on the collected review DataFrame.

    Tool calls implemented as distinct methods so they can be wired as
    agent tools in any framework:
      - compute_sentiment_distribution
      - compute_theme_frequency
      - compute_theme_by_sentiment
      - compute_rating_distribution
      - compute_sentiment_trend
    """

    def __init__(self) -> None:
        self.sia = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Deterministic NLP helpers
    # ------------------------------------------------------------------

    def _classify_sentiment(self, text: str) -> str:
        score = self.sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    def _compound_score(self, text: str) -> float:
        return self.sia.polarity_scores(text)["compound"]

    def _extract_themes(self, text: str) -> list[str]:
        text_lower = text.lower()
        return [
            theme
            for theme, keywords in THEME_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]

    # ------------------------------------------------------------------
    # EDA tool methods (each surfaces a specific finding)
    # ------------------------------------------------------------------

    def compute_sentiment_distribution(self, df: pd.DataFrame) -> dict:
        """Tool: count and percentage of positive / neutral / negative reviews."""
        total = len(df)
        counts = df["sentiment"].value_counts().to_dict()
        for s in ("positive", "neutral", "negative"):
            counts.setdefault(s, 0)
        pct = {k: round(v / total * 100, 1) for k, v in counts.items()}
        return {"counts": counts, "pct": pct}

    def compute_theme_frequency(self, df: pd.DataFrame) -> dict:
        """Tool: how often each theme appears across reviews (%)."""
        total = len(df)
        theme_counts = {theme: 0 for theme in THEME_KEYWORDS}
        for themes in df["themes"]:
            for t in themes:
                theme_counts[t] += 1
        theme_pct = {k: round(v / total * 100, 1) for k, v in theme_counts.items()}
        return {"counts": theme_counts, "pct": theme_pct}

    def compute_theme_by_sentiment(self, df: pd.DataFrame) -> dict:
        """Tool: for each theme, breakdown of positive/neutral/negative reviews."""
        result = {}
        for theme in THEME_KEYWORDS:
            theme_df = df[df["themes"].apply(lambda x: theme in x)]
            if len(theme_df) > 0:
                result[theme] = theme_df["sentiment"].value_counts().to_dict()
        return result

    def compute_rating_distribution(self, df: pd.DataFrame) -> dict:
        """Tool: count of reviews per star rating."""
        dist = df["stars"].value_counts().sort_index().to_dict()
        return {int(k): int(v) for k, v in dist.items()}

    def compute_sentiment_trend(self, df: pd.DataFrame) -> list[float]:
        """Tool: rolling average compound score (window=3 for small datasets)."""
        window = min(3, max(1, len(df) // 2))
        rolling = df["compound"].rolling(window=window, min_periods=1).mean()
        return [round(v, 4) for v in rolling.tolist()]

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, restaurant_name: str, meta: dict) -> dict:
        df = df.copy()

        # Augment with derived columns
        df["sentiment"] = df["text"].apply(self._classify_sentiment)
        df["compound"] = df["text"].apply(self._compound_score)
        df["themes"] = df["text"].apply(self._extract_themes)

        total = len(df)

        # Run each EDA tool
        sentiment_result = self.compute_sentiment_distribution(df)
        theme_result = self.compute_theme_frequency(df)
        theme_by_sentiment = self.compute_theme_by_sentiment(df)
        rating_dist = self.compute_rating_distribution(df)
        trend = self.compute_sentiment_trend(df)

        # Top / bottom reviews by compound score
        top_idx = df["compound"].idxmax()
        bot_idx = df["compound"].idxmin()

        # Use Google's overall rating if available, else compute from reviews
        overall_rating = meta.get("overall_rating") or round(df["stars"].mean(), 1)

        return {
            "restaurant_name": meta.get("place_name", restaurant_name),
            "query_name": restaurant_name,
            "total_reviews": total,
            "overall_rating": overall_rating,        # Google aggregate (may cover 1000s of reviews)
            "total_ratings_on_google": meta.get("total_ratings"),
            "avg_rating_from_sample": round(df["stars"].mean(), 2),
            "rating_distribution": rating_dist,
            "sentiment_counts": sentiment_result["counts"],
            "sentiment_pct": sentiment_result["pct"],
            "theme_counts": theme_result["counts"],
            "theme_pct": theme_result["pct"],
            "theme_by_sentiment": theme_by_sentiment,
            "sentiment_trend": trend,
            "top_review": {
                "text": df.loc[top_idx, "text"],
                "stars": int(df.loc[top_idx, "stars"]),
                "author": df.loc[top_idx, "author"],
            },
            "bottom_review": {
                "text": df.loc[bot_idx, "text"],
                "stars": int(df.loc[bot_idx, "stars"]),
                "author": df.loc[bot_idx, "author"],
            },
            "all_reviews": df[["text", "stars", "author", "time_desc", "sentiment", "compound"]]
                .to_dict(orient="records"),
            "is_placeholder": meta.get("is_placeholder", False),
        }

    def run(self, df: pd.DataFrame, restaurant_name: str, meta: dict) -> dict:
        return self.analyze(df, restaurant_name, meta)


# ---------------------------------------------------------------------------
# HypothesisAgent  (Google ADK + Gemini, with rule-based fallback)
# ---------------------------------------------------------------------------
class HypothesisAgent:
    """
    Generates a data-backed hypothesis from the EDA output.

    Primary path (requires ANTHROPIC_API_KEY):
      Uses a PydanticAI Agent backed by Claude claude-haiku-4-5 (Anthropic).
      The agent calls three @agent.tool-decorated functions—get_sentiment_data,
      get_theme_data, get_rating_data—which receive typed analysis data through
      PydanticAI's RunContext dependency-injection system, then synthesises a
      concise hypothesis in natural language.

    Fallback path (no API key / pydantic-ai unavailable):
      Deterministic rule-based logic using the same EDA values so the
      pipeline always produces output.

    Rules (fallback, in priority order):
      1. Strong positive signal  (positive% ≥ 60)
      2. Strong negative signal  (negative% ≥ 40)
      3. Mixed / balanced signal
    """

    # ------------------------------------------------------------------
    # LLM path via PydanticAI
    # ------------------------------------------------------------------

    def _pydanticai_hypothesis(self, analysis: dict) -> str:
        """Build a PydanticAI Agent on demand and return its hypothesis text.

        The Agent is constructed here (not at import time) so that the
        Anthropic provider only reads ANTHROPIC_API_KEY when it is actually
        needed, avoiding import-time failures when the key is absent.
        """
        agent: Agent[HypothesisDeps, str] = Agent(
            "anthropic:claude-haiku-4-5-20251001",
            deps_type=HypothesisDeps,
            system_prompt=_HYPOTHESIS_SYSTEM_PROMPT,
            tools=_HYPOTHESIS_TOOLS,
        )
        deps = HypothesisDeps(analysis=analysis)
        name = analysis.get("restaurant_name", "Unknown")
        result = agent.run_sync(
            f"Generate a data-backed hypothesis for restaurant: {name}. "
            f"Use the available tools to retrieve the data first.",
            deps=deps,
        )
        return result.data

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_hypothesis(self, analysis: dict) -> str:
        sp = analysis["sentiment_pct"]
        tp = analysis["theme_pct"]
        pos = sp.get("positive", 0)
        neg = sp.get("negative", 0)
        neu = sp.get("neutral", 0)
        total = analysis["total_reviews"]
        name = analysis["restaurant_name"]
        avg = analysis["avg_rating_from_sample"]
        overall = analysis.get("overall_rating")
        total_google = analysis.get("total_ratings_on_google")

        top_theme = max(tp, key=lambda k: tp[k]) if tp else "food_quality"
        top_label = top_theme.replace("_", " ")
        top_pct = tp.get(top_theme, 0)

        tbs = analysis.get("theme_by_sentiment", {})
        worst_theme = top_theme
        worst_neg_ratio = 0.0
        for theme, counts in tbs.items():
            tot = sum(counts.values()) or 1
            ratio = counts.get("negative", 0) / tot
            if ratio > worst_neg_ratio:
                worst_theme, worst_neg_ratio = theme, ratio
        worst_label = worst_theme.replace("_", " ")

        google_ctx = ""
        if overall and total_google:
            google_ctx = (
                f" (Google overall: {overall}/5 across {total_google:,} ratings)"
            )
        elif overall:
            google_ctx = f" (Google overall rating: {overall}/5)"

        if pos >= 60:
            return (
                f"Based on {total} sampled reviews, {name} enjoys a "
                f"strongly positive reception{google_ctx}. "
                f"{pos:.0f}% of reviews express positive sentiment with an "
                f"average sample rating of {avg}/5. "
                f"The most-discussed topic is {top_label} ({top_pct:.0f}% of "
                f"reviews), suggesting it is the primary driver of customer "
                f"satisfaction."
            )

        if neg >= 40:
            return (
                f"Based on {total} sampled reviews, {name} faces notable "
                f"customer dissatisfaction{google_ctx}. "
                f"{neg:.0f}% of reviews are negative and only {pos:.0f}% are "
                f"positive, despite an average sample rating of {avg}/5. "
                f"The theme most associated with negative sentiment is "
                f"{worst_label}, which appears to be the primary pain point."
            )

        return (
            f"Reviews for {name} are mixed{google_ctx}: "
            f"{pos:.0f}% positive, {neg:.0f}% negative, {neu:.0f}% neutral "
            f"across {total} sampled reviews (avg {avg}/5). "
            f"{top_label.title()} is the most discussed theme "
            f"({top_pct:.0f}% of reviews), and {worst_label} shows the "
            f"highest proportion of negative mentions, suggesting it as the "
            f"key area for improvement."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_hypothesis(self, analysis: dict) -> str:
        if _PYDANTIC_AI_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                return self._pydanticai_hypothesis(analysis)
            except Exception as exc:  # noqa: BLE001
                print(f"[HypothesisAgent] PydanticAI call failed ({exc}); using rule-based fallback.")
        return self._rule_based_hypothesis(analysis)

    def run(self, analysis: dict) -> str:
        return self.generate_hypothesis(analysis)


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """
    Top-level orchestrator: Collect → EDA → Hypothesize.
    Supports optional comparison restaurant (fan-out pattern).
    """

    def __init__(self) -> None:
        self.collector = CollectorAgent()
        self.analyst = AnalysisAgent()
        self.hypothesis_agent = HypothesisAgent()

    def _process(self, restaurant_name: str, location: str) -> dict:
        # Step 1: Collect
        df, meta = self.collector.run(restaurant_name, location)

        # Step 2: EDA
        analysis = self.analyst.run(df, restaurant_name, meta)

        # Step 3: Hypothesize
        analysis["hypothesis"] = self.hypothesis_agent.run(analysis)

        return analysis

    def run(
        self,
        restaurant_name: str,
        location: str = "",
        compare_restaurant: Optional[str] = None,
    ) -> dict:
        result: dict = {"primary": self._process(restaurant_name, location)}
        if compare_restaurant:
            result["comparison"] = self._process(compare_restaurant, location)
        return result