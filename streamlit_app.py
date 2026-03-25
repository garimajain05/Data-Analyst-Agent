"""
Streamlit frontend for the restaurant review analyst.

Talks to the FastAPI backend at API_URL (default: http://localhost:8000).
Set API_URL env var to point at a deployed backend.
"""

import json
import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Restaurant Review Analyst",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Restaurant Review Analyst")
st.caption(
    "Multi-agent pipeline: **CollectorAgent** → **AnalysisAgent** → **HypothesisAgent**  "
    "· Data: HuggingFace Yelp dataset (650k+ reviews, fetched at runtime)"
)

# ---------------------------------------------------------------------------
# Sidebar – inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Query")
    restaurant = st.text_input("Restaurant name", placeholder="e.g. Shake Shack")
    location = st.text_input("Location (optional)", placeholder="e.g. New York")
    compare = st.text_input(
        "Compare with (optional)", placeholder="e.g. Five Guys"
    )
    run_btn = st.button(
        "Analyze", type="primary", disabled=not bool(restaurant.strip())
    )

    st.divider()
    st.caption(
        "Reviews are retrieved from the [yelp_review_full](https://huggingface.co/datasets/yelp_review_full) "
        "HuggingFace dataset in streaming mode. The first 5,000 rows are scanned and "
        "filtered for the restaurant name; if fewer than 30 matches are found a random "
        "200-row sample is used instead."
    )


# ---------------------------------------------------------------------------
# Helper: render one analysis result
# ---------------------------------------------------------------------------
def render_analysis(analysis: dict, label: str = "") -> None:
    name = analysis["restaurant_name"]
    filtered = analysis.get("was_filtered", False)

    st.subheader(f"{label}{name}")
    if not filtered:
        st.warning(
            "Fewer than 30 reviews mentioning this restaurant were found in the "
            "scanned sample. Showing a random sample of Yelp reviews instead – "
            "results reflect general Yelp distribution, not this specific restaurant."
        )

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analyzed", analysis["total_reviews"])
    c2.metric("Avg rating", f"{analysis['avg_rating']} / 5")
    c3.metric(
        "Positive sentiment",
        f"{analysis['sentiment_pct'].get('positive', 0):.1f}%",
    )
    c4.metric(
        "Negative sentiment",
        f"{analysis['sentiment_pct'].get('negative', 0):.1f}%",
    )

    # Hypothesis
    st.info(f"**Hypothesis:** {analysis['hypothesis']}")

    # Row 1: sentiment pie + theme bar
    col_l, col_r = st.columns(2)

    with col_l:
        sc = analysis["sentiment_counts"]
        colours = {
            "positive": "#2ecc71",
            "neutral": "#f39c12",
            "negative": "#e74c3c",
        }
        fig_pie = go.Figure(
            go.Pie(
                labels=list(sc.keys()),
                values=list(sc.values()),
                marker_colors=[colours.get(k, "#aaa") for k in sc.keys()],
                hole=0.35,
            )
        )
        fig_pie.update_layout(title="Sentiment Distribution", height=320, margin=dict(t=40, b=0))
        st.plotly_chart(fig_pie, width="stretch")

    with col_r:
        tp = analysis["theme_pct"]
        fig_bar = px.bar(
            x=[k.replace("_", " ").title() for k in tp.keys()],
            y=list(tp.values()),
            labels={"x": "Theme", "y": "% of Reviews"},
            title="Theme Frequency (% of reviews mentioning theme)",
            color=list(tp.values()),
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(height=320, showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_bar, width="stretch")

    # Row 2: theme × sentiment heatmap + sentiment trend
    col_a, col_b = st.columns(2)

    with col_a:
        tbs = analysis.get("theme_by_sentiment", {})
        if tbs:
            rows = [
                {
                    "Theme": theme.replace("_", " ").title(),
                    "Positive": sc.get("positive", 0),
                    "Neutral": sc.get("neutral", 0),
                    "Negative": sc.get("negative", 0),
                }
                for theme, sc in tbs.items()
            ]
            df_tbs = pd.DataFrame(rows).set_index("Theme")
            fig_heat = px.imshow(
                df_tbs,
                text_auto=True,
                color_continuous_scale="RdYlGn",
                title="Theme × Sentiment Heatmap",
                aspect="auto",
            )
            fig_heat.update_layout(height=320, margin=dict(t=40, b=0))
            st.plotly_chart(fig_heat, width="stretch")

    with col_b:
        trend = analysis.get("sentiment_trend", [])
        if trend:
            fig_trend = go.Figure(
                go.Scatter(
                    y=trend,
                    mode="lines",
                    line=dict(color="#3498db", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(52,152,219,0.15)",
                )
            )
            fig_trend.update_layout(
                title="Sentiment Trend (rolling avg compound score)",
                xaxis_title="Review index (sampled)",
                yaxis_title="Compound score",
                height=320,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig_trend, width="stretch")

    # Rating distribution
    rd = analysis.get("rating_distribution", {})
    if rd:
        fig_rd = px.bar(
            x=[f"{k}★" for k in sorted(rd.keys())],
            y=[rd[k] for k in sorted(rd.keys())],
            labels={"x": "Stars", "y": "Count"},
            title="Rating Distribution",
            color=[rd[k] for k in sorted(rd.keys())],
            color_continuous_scale="Greens",
        )
        fig_rd.update_layout(height=260, showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_rd, width="stretch")

    # Top / bottom reviews
    col_top, col_bot = st.columns(2)
    with col_top:
        with st.expander("Most positive review"):
            st.write(analysis.get("top_review", "—"))
    with col_bot:
        with st.expander("Most critical review"):
            st.write(analysis.get("bottom_review", "—"))

    # Download raw JSON artifact
    st.download_button(
        label="Download analysis JSON",
        data=json.dumps(analysis, default=str),
        file_name=f"{name.replace(' ', '_')}_analysis.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
if run_btn and restaurant.strip():
    with st.spinner(
        f"Collecting & analyzing reviews for **{restaurant}** … (this may take ~10–20 s)"
    ):
        try:
            resp = httpx.post(
                f"{API_URL}/analyze",
                json={
                    "restaurant_name": restaurant.strip(),
                    "location": location.strip(),
                    "compare_restaurant": compare.strip() or None,
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            st.error(f"Backend error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Could not reach backend: {e}")
            st.stop()

    render_analysis(data["primary"])

    if "comparison" in data:
        st.divider()
        render_analysis(data["comparison"], label="Comparison: ")

        # Side-by-side sentiment comparison
        st.subheader("Head-to-Head Sentiment Comparison")
        p_sp = data["primary"]["sentiment_pct"]
        c_sp = data["comparison"]["sentiment_pct"]
        sentiments = ["positive", "neutral", "negative"]
        fig_cmp = go.Figure(
            data=[
                go.Bar(
                    name=data["primary"]["restaurant_name"],
                    x=sentiments,
                    y=[p_sp.get(s, 0) for s in sentiments],
                    marker_color="#3498db",
                ),
                go.Bar(
                    name=data["comparison"]["restaurant_name"],
                    x=sentiments,
                    y=[c_sp.get(s, 0) for s in sentiments],
                    marker_color="#e67e22",
                ),
            ]
        )
        fig_cmp.update_layout(
            barmode="group",
            yaxis_title="% of reviews",
            height=360,
            margin=dict(t=20, b=0),
        )
        st.plotly_chart(fig_cmp, width="stretch")

elif not run_btn:
    st.info("Enter a restaurant name in the sidebar and click **Analyze** to begin.")
