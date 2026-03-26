"""
Multi-agent data analyst pipeline for restaurant reviews.

Agents:
  CollectorAgent   - retrieves reviews from HuggingFace Yelp dataset at runtime
  AnalysisAgent    - VADER sentiment + keyword theme extraction + pandas EDA
  HypothesisAgent - uses Claude claude-sonnet-4-6 (or rule-based fallback) to generate
                     a data-backed hypothesis
  OrchestratorAgent - coordinates Collect → EDA → Hypothesize
"""

import json
import os
from typing import Optional

import pandas as pd
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import anthropic

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


# ---------------------------------------------------------------------------
# CollectorAgent
# ---------------------------------------------------------------------------
class CollectorAgent:
    """
    Retrieves Yelp reviews from HuggingFace `yelp_review_full` at runtime.

    Strategy:
      1. Stream the first `scan_rows` rows from the training split.
      2. Filter rows whose text mentions any token from restaurant_name.
      3. If fewer than `min_matches` are found, fall back to a random sample
         of the scanned rows (so analysis still runs on real data).
    """

    def collect_reviews(
        self,
        restaurant_name: str,
        location: str = "",
        scan_rows: int = 5000,
        min_matches: int = 5,
        max_results: int = 400,
    ) -> tuple[pd.DataFrame, bool]:
        """
        Returns (dataframe, was_filtered).
        `was_filtered=True` means we found reviews mentioning the restaurant name.
        `was_filtered=False` means we fell back to a random sample.

        Each restaurant gets a deterministic but distinct slice of the dataset
        via a name-derived skip offset, so different queries return different data.
        """
        # Derive a skip offset from the restaurant name so different queries
        # read different portions of the 650k-row dataset.
        name_seed = abs(hash(restaurant_name.lower())) % 600_000
        # Keep offset within a range that still leaves room for scan_rows rows
        skip = min(name_seed, 650_000 - scan_rows - 1)

        dataset = load_dataset(
            "yelp_review_full",
            split="train",
            streaming=True,
        )

        query_tokens = [
            tok for tok in restaurant_name.lower().split() if len(tok) > 3
        ]

        rows: list[dict] = []
        matched: list[dict] = []

        for i, row in enumerate(dataset):
            if i < skip:
                continue
            if i >= skip + scan_rows:
                break
            record = {"text": row["text"], "stars": row["label"] + 1}
            rows.append(record)
            if query_tokens:
                text_lower = row["text"].lower()
                if any(tok in text_lower for tok in query_tokens):
                    matched.append(record)
                    if len(matched) >= max_results:
                        break

        if len(matched) >= min_matches:
            return pd.DataFrame(matched[:max_results]), True

        # Fallback: random sample using name-derived seed so results differ per restaurant
        name_random_seed = abs(hash(restaurant_name.lower())) % (2**31)
        df_all = pd.DataFrame(rows)
        sample = df_all.sample(min(200, len(df_all)), random_state=name_random_seed)
        return sample, False

    def run(
        self, restaurant_name: str, location: str = ""
    ) -> tuple[pd.DataFrame, bool]:
        return self.collect_reviews(restaurant_name, location)


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------
class AnalysisAgent:
    """
    Performs EDA on a review dataframe:
      - VADER sentiment classification (positive / neutral / negative)
      - Keyword-based theme detection
      - Pandas aggregations: counts, percentages, grouped stats, trends
    """

    def __init__(self) -> None:
        self.sia = SentimentIntensityAnalyzer()

    def _classify_sentiment(self, text: str) -> str:
        score = self.sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    def _extract_themes(self, text: str) -> list[str]:
        text_lower = text.lower()
        return [
            theme
            for theme, keywords in THEME_KEYWORDS.items()
            if any(kw in text_lower for kw in keywords)
        ]

    def analyze(self, df: pd.DataFrame, restaurant_name: str) -> dict:
        df = df.copy()

        # --- sentiment ---
        df["sentiment"] = df["text"].apply(self._classify_sentiment)
        df["compound"] = df["text"].apply(
            lambda t: self.sia.polarity_scores(t)["compound"]
        )

        # --- themes ---
        df["themes"] = df["text"].apply(self._extract_themes)

        total = len(df)

        # Sentiment distribution
        sentiment_counts: dict = df["sentiment"].value_counts().to_dict()
        for s in ("positive", "neutral", "negative"):
            sentiment_counts.setdefault(s, 0)
        sentiment_pct = {
            k: round(v / total * 100, 1) for k, v in sentiment_counts.items()
        }

        # Theme frequency
        theme_counts = {theme: 0 for theme in THEME_KEYWORDS}
        for themes in df["themes"]:
            for t in themes:
                theme_counts[t] += 1
        theme_pct = {k: round(v / total * 100, 1) for k, v in theme_counts.items()}

        # Theme × sentiment grouped stats
        theme_by_sentiment: dict = {}
        for theme in THEME_KEYWORDS:
            theme_df = df[df["themes"].apply(lambda x: theme in x)]
            if len(theme_df) > 0:
                theme_by_sentiment[theme] = (
                    theme_df["sentiment"].value_counts().to_dict()
                )

        # Rating stats
        avg_rating = round(df["stars"].mean(), 2)
        rating_dist = df["stars"].value_counts().sort_index().to_dict()

        # Trend: rolling avg compound score (window=20)
        rolling_avg = (
            df["compound"].rolling(window=20, min_periods=1).mean().tolist()
        )
        # Sample every 10th point to keep payload small
        trend_points = rolling_avg[::10]

        # Top and bottom reviews
        top_idx = df["compound"].idxmax()
        bot_idx = df["compound"].idxmin()
        top_review = df.loc[top_idx, "text"][:400]
        bottom_review = df.loc[bot_idx, "text"][:400]

        return {
            "restaurant_name": restaurant_name,
            "total_reviews": total,
            "avg_rating": avg_rating,
            "rating_distribution": {int(k): int(v) for k, v in rating_dist.items()},
            "sentiment_counts": sentiment_counts,
            "sentiment_pct": sentiment_pct,
            "theme_counts": theme_counts,
            "theme_pct": theme_pct,
            "theme_by_sentiment": theme_by_sentiment,
            "sentiment_trend": trend_points,
            "top_review": top_review,
            "bottom_review": bottom_review,
        }

    def run(self, df: pd.DataFrame, restaurant_name: str) -> dict:
        return self.analyze(df, restaurant_name)


# ---------------------------------------------------------------------------
# HypothesisAgent
# ---------------------------------------------------------------------------
class HypothesisAgent:
    """
    Generates a data-backed hypothesis using Claude claude-sonnet-4-6.
    Falls back to a rule-based statement when ANTHROPIC_API_KEY is not set.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def generate_hypothesis(self, analysis: dict) -> str:
        if self.client:
            return self._claude_hypothesis(analysis)
        return self._rule_based_hypothesis(analysis)

    def _claude_hypothesis(self, analysis: dict) -> str:
        summary = {
            "restaurant": analysis["restaurant_name"],
            "total_reviews": analysis["total_reviews"],
            "avg_rating": analysis["avg_rating"],
            "sentiment_pct": analysis["sentiment_pct"],
            "theme_pct": analysis["theme_pct"],
            "theme_by_sentiment": analysis["theme_by_sentiment"],
        }
        prompt = (
            "You are a data analyst reviewing restaurant feedback data. "
            "Generate ONE clear, concise hypothesis (2–3 sentences) about the "
            "customer experience at this restaurant. Reference specific "
            "percentages and counts from the data. Be direct and analytical.\n\n"
            f"Data:\n{json.dumps(summary, indent=2)}"
        )
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _rule_based_hypothesis(self, analysis: dict) -> str:
        sp = analysis.get("sentiment_pct", {})
        tp = analysis.get("theme_pct", {})
        pos = sp.get("positive", 0)
        neg = sp.get("negative", 0)
        top_theme = max(tp, key=lambda k: tp[k]) if tp else "food_quality"
        top_label = top_theme.replace("_", " ")
        top_pct = tp.get(top_theme, 0)

        tbs = analysis.get("theme_by_sentiment", {})

        # Find most negatively-associated theme
        worst_theme, worst_neg = top_theme, 0
        for theme, counts in tbs.items():
            n = counts.get("negative", 0)
            tot = sum(counts.values()) or 1
            if n / tot > worst_neg / max(sum(tbs.get(worst_theme, {}).values()), 1):
                worst_theme, worst_neg = theme, n

        if pos >= 60:
            return (
                f"Customer satisfaction at {analysis['restaurant_name']} is driven "
                f"primarily by {top_label} ({top_pct:.1f}% of reviews mention it), "
                f"with {pos:.1f}% of reviews expressing positive sentiment overall "
                f"and an average rating of {analysis['avg_rating']}/5."
            )
        if neg >= 40:
            worst_label = worst_theme.replace("_", " ")
            return (
                f"Negative reviews ({neg:.1f}%) at {analysis['restaurant_name']} "
                f"are concentrated around {worst_label}, suggesting it is the "
                f"primary pain point. Positive sentiment accounts for only "
                f"{pos:.1f}% despite an average rating of {analysis['avg_rating']}/5."
            )
        return (
            f"Reviews for {analysis['restaurant_name']} are mixed "
            f"({pos:.1f}% positive, {neg:.1f}% negative). "
            f"{top_label.title()} is the most discussed topic "
            f"({top_pct:.1f}% of reviews), pointing to it as the key driver "
            f"of overall customer experience."
        )

    def run(self, analysis: dict) -> str:
        return self.generate_hypothesis(analysis)


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------
class OrchestratorAgent:
    """
    Top-level orchestrator: Collect → EDA → Hypothesize.
    Supports optional comparison restaurant.
    """

    def __init__(self) -> None:
        self.collector = CollectorAgent()
        self.analyst = AnalysisAgent()
        self.hypothesis_agent = HypothesisAgent()

    def _process(self, restaurant_name: str, location: str) -> dict:
        df, was_filtered = self.collector.run(restaurant_name, location)
        analysis = self.analyst.run(df, restaurant_name)
        analysis["was_filtered"] = was_filtered
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
