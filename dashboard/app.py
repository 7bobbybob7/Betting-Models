"""
dashboard/app.py - Streamlit dashboard for the betting platform.

Consumes the FastAPI backend. Provides:
- Model comparison and performance overview
- Calibration plots
- CLV tracking over time
- Predictions browser
- Data health monitoring

Usage:
    # Start the API first:
    uvicorn api.app:app --port 8000
    # Then run the dashboard:
    streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np

API_BASE = "http://localhost:8000"


def api_get(endpoint, params=None):
    """Fetch data from the FastAPI backend."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Betting Models", layout="wide")
st.title("Multi-Sport Betting Platform")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Calibration", "CLV Tracker", "Predictions", "P&L", "Data Health"
])


# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab1:
    st.header("Model Performance Overview")

    models_data = api_get("/models")
    if models_data:
        df = pd.DataFrame(models_data)
        # Format for display
        display_df = df[["model_name", "total_predictions", "accuracy", "mean_clv", "brier_score"]].copy()
        display_df.columns = ["Model", "Predictions", "Accuracy", "Mean CLV", "Brier Score"]
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.1%}")
        display_df["Mean CLV"] = display_df["Mean CLV"].apply(lambda x: f"{x:+.4f}")
        display_df["Brier Score"] = display_df["Brier Score"].apply(lambda x: f"{x:.4f}")
        display_df["Predictions"] = display_df["Predictions"].apply(lambda x: f"{x:,}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # CLV by season
    st.subheader("CLV by Season")
    clv_data = api_get("/clv")
    if clv_data:
        clv_df = pd.DataFrame(clv_data)
        if not clv_df.empty:
            # Pivot for chart
            pivot = clv_df.pivot(index="season", columns="model_name", values="mean_clv")
            st.bar_chart(pivot)

            # Table
            display_clv = clv_df[["model_name", "season", "games", "mean_clv", "clv_positive_pct", "accuracy"]].copy()
            display_clv.columns = ["Model", "Season", "Games", "Mean CLV", "CLV > 0 %", "Accuracy"]
            display_clv["Mean CLV"] = display_clv["Mean CLV"].apply(lambda x: f"{x:+.4f}")
            display_clv["CLV > 0 %"] = display_clv["CLV > 0 %"].apply(lambda x: f"{x:.1%}")
            display_clv["Accuracy"] = display_clv["Accuracy"].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_clv, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Calibration
# ---------------------------------------------------------------------------
with tab2:
    st.header("Calibration Analysis")

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", ["mlb_logreg_v1", "mlb_xgb_v1"], key="cal_model")
    with col2:
        season = st.selectbox("Season", [None, 2024, 2023, 2022], format_func=lambda x: "All" if x is None else str(x), key="cal_season")

    params = {"model": model}
    if season:
        params["season"] = season

    cal_data = api_get("/calibration", params)
    if cal_data:
        cal_df = pd.DataFrame(cal_data)
        if not cal_df.empty:
            # Chart
            chart_df = cal_df[["predicted_mean", "actual_mean"]].copy()
            chart_df.columns = ["Predicted", "Actual"]
            chart_df["Perfect"] = chart_df["Predicted"]

            st.line_chart(chart_df.set_index("Predicted"))

            # Table
            display_cal = cal_df.copy()
            display_cal["bin"] = display_cal.apply(lambda r: f"{r['bin_low']:.0%}-{r['bin_high']:.0%}", axis=1)
            display_cal["predicted_mean"] = display_cal["predicted_mean"].apply(lambda x: f"{x:.3f}")
            display_cal["actual_mean"] = display_cal["actual_mean"].apply(lambda x: f"{x:.3f}")
            display_cal["diff"] = display_cal["diff"].apply(lambda x: f"{x:.3f}")
            st.dataframe(
                display_cal[["bin", "predicted_mean", "actual_mean", "count", "diff"]],
                use_container_width=True, hide_index=True
            )


# ---------------------------------------------------------------------------
# Tab 3: CLV Tracker
# ---------------------------------------------------------------------------
with tab3:
    st.header("CLV Over Time")

    model = st.selectbox("Model", ["mlb_logreg_v1", "mlb_xgb_v1"], key="clv_model")

    # Get predictions sorted by date
    preds = api_get("/predictions", {"model": model, "limit": 1000})
    if preds:
        pred_df = pd.DataFrame(preds)
        if not pred_df.empty and "edge" in pred_df.columns:
            pred_df = pred_df.dropna(subset=["edge"])
            pred_df = pred_df.sort_values("game_date")
            pred_df["rolling_clv"] = pred_df["edge"].rolling(100, min_periods=20).mean()
            pred_df["cumulative_clv"] = pred_df["edge"].cumsum()

            st.subheader("Rolling 100-Game CLV")
            chart_data = pred_df[["game_date", "rolling_clv"]].set_index("game_date")
            st.line_chart(chart_data)

            st.subheader("Cumulative CLV")
            cum_data = pred_df[["game_date", "cumulative_clv"]].set_index("game_date")
            st.line_chart(cum_data)


# ---------------------------------------------------------------------------
# Tab 4: Predictions
# ---------------------------------------------------------------------------
with tab4:
    st.header("Predictions Browser")

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", ["mlb_logreg_v1", "mlb_xgb_v1"], key="pred_model")
    with col2:
        season = st.selectbox("Season", [2024, 2023, 2022], key="pred_season")
    with col3:
        min_edge = st.slider("Min |Edge|", 0.0, 0.15, 0.0, 0.01, key="pred_edge")

    params = {"model": model, "season": season, "limit": 500}
    if min_edge > 0:
        params["min_edge"] = min_edge

    preds = api_get("/predictions", params)
    if preds:
        pred_df = pd.DataFrame(preds)
        if not pred_df.empty:
            display = pred_df[["game_date", "home_team", "away_team", "predicted_prob", "edge", "outcome", "home_score", "away_score"]].copy()
            display["predicted_prob"] = display["predicted_prob"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            display["edge"] = display["edge"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
            display.columns = ["Date", "Home", "Away", "Home Win Prob", "Edge", "Outcome", "H Score", "A Score"]

            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(display)} predictions")


# ---------------------------------------------------------------------------
# Tab 5: P&L (placeholder — needs bet tracking)
# ---------------------------------------------------------------------------
with tab5:
    st.header("P&L Tracker")
    st.info("P&L tracking requires bet placement data. Currently showing simulated results from backtesting.")

    st.subheader("Simulated ROI (2024, LogReg, Quarter-Kelly)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2%+ Edge", "-6.53% ROI", "1,465 bets")
    with col2:
        st.metric("5%+ Edge", "-8.84% ROI", "573 bets")
    with col3:
        st.metric("10%+ Edge", "+14.94% ROI", "56 bets", delta_color="normal")

    st.caption("Positive CLV confirms real signal. Negative ROI at lower thresholds is due to vig. Player-level models (Phase 2-3) needed for profitable betting.")


# ---------------------------------------------------------------------------
# Tab 6: Data Health
# ---------------------------------------------------------------------------
with tab6:
    st.header("Data Health")

    health = api_get("/health")
    if health:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Games")
            st.metric("MLB Games", f"{health.get('games_mlb', 0):,}")
            st.metric("CBB Games", f"{health.get('games_cbb', 0):,}")
            st.metric("Latest MLB Game", health.get("latest_mlb_game", "N/A"))

        with col2:
            st.subheader("MLB Data")
            st.metric("Batting Records", f"{health.get('mlb_batting_game', 0):,}")
            st.metric("Pitching Records", f"{health.get('mlb_pitching_game', 0):,}")
            st.metric("Statcast Pitches", f"{health.get('mlb_pitches', 0):,}")

        with col3:
            st.subheader("Odds & Predictions")
            st.metric("Odds Records", f"{health.get('odds', 0):,}")
            st.metric("Predictions", f"{health.get('predictions', 0):,}")
            st.metric("MLB Players", f"{health.get('players_mlb', 0):,}")
