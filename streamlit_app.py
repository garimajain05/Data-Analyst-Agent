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
    "· Data: Google Places API (real reviews + star ratings, fetched at runtime)"
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
        "Reviews are fetched live from **Google Places API**. "
        "Google returns up to **5 reviews** per place along with the "
        "overall star rating (aggregated across all user ratings). "
        "If `GOOGLE_PLACES_API_KEY` is not set, clearly-labelled placeholder "
        "reviews are used instead."
    )


# ---------------------------------------------------------------------------
# Helper: star display
# ---------------------------------------------------------------------------
def stars_html(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "1/2" * half + "☆" * empty


# ---------------------------------------------------------------------------
# Helper: render one analysis result
# ---------------------------------------------------------------------------
def render_analysis(analysis: dict, label: str = "") -> None:
    name = analysis["restaurant_name"]
    is_placeholder = analysis.get("is_placeholder", False)

    st.subheader(f"{label}{name}")

    if is_placeholder:
        st.warning(
            "⚠️ No `GOOGLE_PLACES_API_KEY` is configured. "
            "Showing **placeholder reviews** so you can see the pipeline in action. "
            "Add your key to `.env` for live Google Places data."
        )

    # ── Top-level metrics ────────────────────────────────────────────────────
    overall = analysis.get("overall_rating")
    total_google = analysis.get("total_ratings_on_google")
    avg_sample = analysis.get("avg_rating_from_sample")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reviews analyzed", analysis["total_reviews"])
    c2.metric(
        "Google overall ★",
        f"{overall}/5" if overall else "N/A",
        help="Aggregate rating across all Google reviewers",
    )
    c3.metric(
        "Sample avg ★",
        f"{avg_sample}/5" if avg_sample else "N/A",
        help="Mean star rating of the fetched reviews",
    )
    c4.metric(
        "Positive sentiment",
        f"{analysis['sentiment_pct'].get('positive', 0):.0f}%",
    )
    c5.metric(
        "Negative sentiment",
        f"{analysis['sentiment_pct'].get('negative', 0):.0f}%",
    )

    if total_google:
        st.caption(
            f"Google overall rating based on **{total_google:,}** total ratings. "
            f"Sentiment analysis performed on **{analysis['total_reviews']}** fetched reviews."
        )

    # ── Hypothesis ──────────────────────────────────────────────────────────
    st.info(f"**💡 Hypothesis:** {analysis['hypothesis']}")

    # ── Row 1: sentiment pie + theme bar ────────────────────────────────────
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
        fig_pie.update_layout(
            title="Sentiment Distribution",
            height=320,
            margin=dict(t=40, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

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
        fig_bar.update_layout(
            height=320,
            showlegend=False,
            margin=dict(t=40, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 2: theme × sentiment heatmap + sentiment trend ──────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        tbs = analysis.get("theme_by_sentiment", {})
        if tbs:
            rows = [
                {
                    "Theme": theme.replace("_", " ").title(),
                    "Positive": sc_inner.get("positive", 0),
                    "Neutral": sc_inner.get("neutral", 0),
                    "Negative": sc_inner.get("negative", 0),
                }
                for theme, sc_inner in tbs.items()
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
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        trend = analysis.get("sentiment_trend", [])
        if trend:
            fig_trend = go.Figure(
                go.Scatter(
                    y=trend,
                    mode="lines+markers",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=6),
                    fill="tozeroy",
                    fillcolor="rgba(52,152,219,0.15)",
                )
            )
            fig_trend.update_layout(
                title="Sentiment Trend (rolling avg compound score)",
                xaxis_title="Review index",
                yaxis_title="Compound score (−1 to +1)",
                yaxis=dict(range=[-1, 1]),
                height=320,
                margin=dict(t=40, b=0),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # ── Rating distribution ──────────────────────────────────────────────────
    rd = analysis.get("rating_distribution", {})
    if rd:
        fig_rd = px.bar(
            x=[f"{k}★" for k in sorted(rd.keys())],
            y=[rd[k] for k in sorted(rd.keys())],
            labels={"x": "Stars", "y": "Count"},
            title="Star Rating Distribution (fetched reviews)",
            color=[rd[k] for k in sorted(rd.keys())],
            color_continuous_scale="Greens",
        )
        fig_rd.update_layout(
            height=260,
            showlegend=False,
            margin=dict(t=40, b=0),
        )
        st.plotly_chart(fig_rd, use_container_width=True)

    # ── Individual reviews ───────────────────────────────────────────────────
    st.subheader("📝 Fetched Reviews")
    all_reviews = analysis.get("all_reviews", [])
    sentiment_colour = {
        "positive": "#d4edda",
        "neutral": "#fff3cd",
        "negative": "#f8d7da",
    }
    for rev in all_reviews:
        bg = sentiment_colour.get(rev.get("sentiment", "neutral"), "#f0f0f0")
        stars_display = "★" * int(rev.get("stars", 3)) + "☆" * (
            5 - int(rev.get("stars", 3))
        )
        with st.container():
            st.markdown(
                f"""
                <div style="background:{bg};border-radius:8px;padding:12px 16px;
                            margin-bottom:10px;border-left:4px solid #ccc;">
                  <strong>{rev.get('author','Anonymous')}</strong>
                  &nbsp;&nbsp;<span style="color:#f39c12;">{stars_display}</span>
                  &nbsp;&nbsp;<em style="color:#888;">{rev.get('time_desc','')}</em>
                  &nbsp;&nbsp;
                  <span style="font-size:0.8em;background:#ddd;border-radius:4px;
                               padding:2px 6px;">{rev.get('sentiment','').upper()}</span>
                  <p style="margin:8px 0 0;">{rev.get('text','')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Top / bottom reviews ─────────────────────────────────────────────────
    col_top, col_bot = st.columns(2)
    with col_top:
        top = analysis.get("top_review", {})
        with st.expander("🌟 Most positive review"):
            st.write(f"**{top.get('author','')}** — {'★' * top.get('stars',5)}")
            st.write(top.get("text", "—"))
    with col_bot:
        bot = analysis.get("bottom_review", {})
        with st.expander("⚠️ Most critical review"):
            st.write(f"**{bot.get('author','')}** — {'★' * bot.get('stars',1)}")
            st.write(bot.get("text", "—"))

    # ── Download artifact ────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download analysis JSON",
        data=json.dumps(analysis, default=str),
        file_name=f"{name.replace(' ', '_')}_analysis.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
if run_btn and restaurant.strip():
    with st.spinner(
        f"Fetching Google Places reviews for **{restaurant}** and running EDA…"
    ):
        try:
            resp = httpx.post(
                f"{API_URL}/analyze",
                json={
                    "restaurant_name": restaurant.strip(),
                    "location": location.strip(),
                    "compare_restaurant": compare.strip() or None,
                },
                timeout=60.0,
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

        # ── Head-to-head sentiment chart ─────────────────────────────────────
        st.subheader("📊 Head-to-Head Sentiment Comparison")
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
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Overall rating comparison ─────────────────────────────────────────
        p_overall = data["primary"].get("overall_rating")
        c_overall = data["comparison"].get("overall_rating")
        if p_overall and c_overall:
            st.subheader("⭐ Google Overall Rating Comparison")
            col1, col2 = st.columns(2)
            col1.metric(
                data["primary"]["restaurant_name"],
                f"{p_overall} / 5",
            )
            col2.metric(
                data["comparison"]["restaurant_name"],
                f"{c_overall} / 5",
                delta=f"{round(c_overall - p_overall, 1):+}",
            )

elif not run_btn:
    st.info("Enter a restaurant name in the sidebar and click **Analyze** to begin.")