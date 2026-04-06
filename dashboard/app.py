"""
dashboard/app.py - Streamlit dashboard.

Uses FastAPI backend when available, falls back to direct DB queries.

Usage:
    streamlit run dashboard/app.py
    # Or with API: uvicorn api.app:app --port 8000 & streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_BASE = "http://localhost:8000"
_api_available = None


def check_api():
    global _api_available
    if _api_available is None:
        try:
            resp = requests.get(f"{API_BASE}/health", timeout=2)
            _api_available = resp.status_code == 200
        except Exception:
            _api_available = False
    return _api_available


def api_get(endpoint, params=None):
    if check_api():
        try:
            resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass
    # Fallback to direct DB query
    return None


def db_query(sql, params=None):
    """Direct DB fallback."""
    try:
        from db.db import query
        return query(sql, params)
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()


st.set_page_config(page_title="Betting Models", layout="wide", page_icon="⚾")
st.title("Multi-Sport Betting Platform")

if check_api():
    st.caption("Connected to FastAPI backend")
else:
    st.caption("Using direct database connection (FastAPI not running)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Today's Bets", "Performance", "P&L Tracker", "Predictions", "Calibration", "Data Health"
])


# ---------------------------------------------------------------------------
# Tab 1: Today's Bets
# ---------------------------------------------------------------------------
with tab1:
    st.header("Today's Actionable Bets")

    data = api_get("/today")
    if data:
        df = pd.DataFrame(data)
    else:
        df = db_query("""
            SELECT p.model_name, p.market, p.predicted_value, p.edge,
                   p.bet_placed, p.outcome,
                   g.game_date, g.home_score, g.away_score, g.status,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.game_date = CURRENT_DATE AND p.model_name LIKE '%%_live'
            ORDER BY p.market, p.model_name
        """)
        if len(df) > 0:
            bets = df[df["bet_placed"] == True]
            if len(bets) > 0:
                st.subheader(f"Flagged Bets ({len(bets)})")
                for _, b in bets.iterrows():
                    side = "OVER" if b.get("edge") and b["edge"] > 0 else "UNDER"
                    st.success(
                        f"**{side} {b.get('predicted_value', '?')}** — "
                        f"{b['away_team']} @ {b['home_team']}"
                    )
            else:
                st.info("No +EV bets flagged for today")

            totals = df[df["market"] == "total"]
            if len(totals) > 0:
                st.subheader("All Totals Predictions")
                display = totals[["home_team", "away_team", "predicted_value", "edge", "bet_placed", "status"]].copy()
                display.columns = ["Home", "Away", "Pred Total", "Edge", "Bet?", "Status"]
                display["Pred Total"] = display["Pred Total"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                display["Edge"] = display["Edge"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
                st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions for today. Pipeline runs at 11:30 AM and 6:30 PM ET.")

    # Yesterday
    st.subheader("Yesterday's Results")
    import datetime
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    ydata = api_get("/today", {"date": yesterday})
    if ydata:
        ydf = pd.DataFrame(ydata)
    else:
        ydf = db_query("""
            SELECT p.model_name, p.market, p.predicted_value, p.edge,
                   p.bet_placed, p.outcome,
                   g.game_date, g.home_score, g.away_score, g.status,
                   ht.name as home_team, at.name as away_team
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.game_date = %s AND p.model_name LIKE '%%_live'
            ORDER BY p.market, p.model_name
        """, [yesterday])

    if len(ydf) > 0:
        ytotals = ydf[ydf["market"] == "total"]
    else:
        ytotals = pd.DataFrame()
        if len(ytotals) > 0:
            display_y = ytotals[["home_team", "away_team", "predicted_value", "edge",
                                 "bet_placed", "outcome", "home_score", "away_score"]].copy()
            display_y["Actual"] = display_y["home_score"].fillna(0).astype(int) + display_y["away_score"].fillna(0).astype(int)
            display_y = display_y.rename(columns={
                "home_team": "Home", "away_team": "Away", "predicted_value": "Pred",
                "edge": "Edge", "bet_placed": "Bet?", "outcome": "Result"
            })
            display_y["Pred"] = display_y["Pred"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            display_y["Edge"] = display_y["Edge"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
            st.dataframe(display_y[["Home", "Away", "Pred", "Actual", "Edge", "Bet?", "Result"]],
                         use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Performance
# ---------------------------------------------------------------------------
with tab2:
    st.header("Model Performance")

    models_data = api_get("/models")
    if models_data:
        mdf = pd.DataFrame(models_data)
    else:
        mdf = db_query("""
            SELECT p.model_name, COUNT(*) as total_predictions,
                   AVG(p.edge) as mean_clv,
                   AVG(CASE WHEN (p.predicted_prob > 0.5 AND p.outcome = 'win') OR
                       (p.predicted_prob < 0.5 AND p.outcome = 'loss')
                       THEN 1.0 ELSE 0.0 END) as accuracy
            FROM predictions p JOIN games g ON p.game_id = g.game_id
            WHERE p.outcome IS NOT NULL GROUP BY p.model_name ORDER BY p.model_name
        """)
        if len(mdf) > 0:
            display = mdf[["model_name", "total_predictions", "accuracy", "mean_clv"]].copy()
            display.columns = ["Model", "Predictions", "Accuracy", "Mean CLV"]
            display["Accuracy"] = display["Accuracy"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            display["Mean CLV"] = display["Mean CLV"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "—")
            display["Predictions"] = display["Predictions"].apply(lambda x: f"{x:,}")
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("CLV by Season")
    clv_data = api_get("/clv")
    if clv_data:
        cdf = pd.DataFrame(clv_data)
    else:
        cdf = db_query("""
            SELECT p.model_name, s.year as season, COUNT(*) as games,
                   AVG(p.edge) as mean_clv
            FROM predictions p JOIN games g ON p.game_id = g.game_id
            JOIN seasons s ON g.season_id = s.season_id
            WHERE p.edge IS NOT NULL AND p.market = 'moneyline'
            GROUP BY p.model_name, s.year ORDER BY p.model_name, s.year
        """)
        if len(cdf) > 0:
            pivot = cdf.pivot(index="season", columns="model_name", values="mean_clv")
            st.bar_chart(pivot)

    st.subheader("Key Finding: Line Shopping")
    st.markdown("""
    | Strategy | Bets/yr | Win Rate | ROI @ -110 | ROI @ Best Line |
    |----------|---------|---------|-----------|----------------|
    | All games | 2,076 | 51.6% | -1.6% | **+3.0%** |
    | ≥1% edge | 1,539 | 51.9% | -1.0% | **+3.8%** |
    | ≥3% edge | 733 | 53.3% | +1.8% | **+7.6%** |

    *Line shopping across sportsbooks turns a losing strategy into a winning one.*
    """)


# ---------------------------------------------------------------------------
# Tab 3: P&L Tracker
# ---------------------------------------------------------------------------
with tab3:
    st.header("P&L Tracker")

    bankroll_data = api_get("/bankroll")
    if not bankroll_data:
        # Deduped bankroll: one bet per game at best odds
        br = db_query("""
            WITH ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY game_id
                    ORDER BY COALESCE(bet_odds, 0) DESC
                ) as rn
                FROM predictions
                WHERE bet_placed = true
                  AND model_name IN ('mlb_totals_reg_live', 'mlb_totals_clf_live', 'mlb_totals_v1_live')
                  AND market = 'total'
            )
            SELECT COUNT(*) as total_bets,
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(SUM(bet_amount), 0) as total_wagered,
                   AVG(CASE WHEN edge IS NOT NULL THEN edge END) as avg_clv
            FROM ranked WHERE rn = 1
        """)
        if len(br) > 0:
            r = br.iloc[0]
            bankroll_data = {
                "starting_bankroll": 10000,
                "current_bankroll": round(10000 + float(r["total_pnl"] or 0), 2),
                "total_bets": int(r["total_bets"] or 0),
                "wins": int(r["wins"] or 0), "losses": int(r["losses"] or 0),
                "pending": int(r["pending"] or 0),
                "pnl": round(float(r["total_pnl"] or 0), 2),
                "wagered": round(float(r["total_wagered"] or 0), 2),
                "roi": round(float(r["total_pnl"] or 0) / max(float(r["total_wagered"] or 1), 1) * 100, 2),
                "win_rate": round(int(r["wins"] or 0) / max(int(r["wins"] or 0) + int(r["losses"] or 0), 1), 4),
                "avg_clv": round(float(r.get("avg_clv") or 0), 4),
            }
    if bankroll_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Bankroll",
                       f"${bankroll_data['current_bankroll']:,.0f}",
                       f"${bankroll_data['pnl']:+,.0f}")
        with col2:
            st.metric("Total Bets",
                       str(bankroll_data["total_bets"]),
                       f"{bankroll_data['wins']}W-{bankroll_data['losses']}L")
        with col3:
            st.metric("ROI", f"{bankroll_data['roi']:+.1f}%")
        with col4:
            st.metric("Win Rate",
                       f"{bankroll_data['win_rate']:.1%}",
                       f"{bankroll_data['pending']} pending")

        # Decomposed CLV
        clv_data = api_get("/clv")
        if not clv_data:
            clv_data = db_query("""
                SELECT model_name,
                    AVG(clv_model) as avg_clv_model,
                    AVG(clv_execution) as avg_clv_execution,
                    COUNT(clv_model) as n_bets
                FROM predictions
                WHERE bet_placed = true AND market = 'total'
                  AND model_name LIKE '%%totals%%live' AND clv_model IS NOT NULL
                GROUP BY model_name
            """)
            if len(clv_data) > 0:
                clv_data = clv_data.to_dict(orient="records")
            else:
                clv_data = []

        if clv_data:
            st.subheader("CLV Decomposition")
            clv_cols = st.columns(len(clv_data))
            for i, row in enumerate(clv_data):
                model = row.get("model_name", "unknown")
                m_clv = float(row.get("avg_clv_model") or 0)
                e_clv = float(row.get("avg_clv_execution") or 0)
                n = int(row.get("n_bets") or row.get("n_with_model_clv") or 0)
                with clv_cols[i]:
                    short_name = model.replace("mlb_totals_", "").replace("_live", "")
                    st.markdown(f"**{short_name}** ({n} bets)")
                    st.metric("Model CLV", f"{m_clv:+.2%}",
                              "Info edge" if m_clv > 0 else "No info edge")
                    st.metric("Execution CLV", f"{e_clv:+.2%}",
                              "Line shopping value")
                    st.metric("Total CLV", f"{m_clv + e_clv:+.2%}")

    st.subheader("Backtested Strategy (Reference)")
    st.markdown("""
    | Threshold | Bets/yr | Win% | ROI@-110 | ROI@Best |
    |-----------|---------|------|----------|----------|
    | ≥1% edge | 1,539 | 51.9% | -1.0% | **+3.8%** |
    | ≥3% edge | 733 | 53.3% | +1.8% | **+7.6%** |
    | ≥5% edge | 295 | 54.7% | +4.5% | **+12.4%** |
    """)


# ---------------------------------------------------------------------------
# Tab 4: Predictions Browser
# ---------------------------------------------------------------------------
with tab4:
    st.header("Predictions Browser")

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", [
            "mlb_logreg_v1_live", "mlb_totals_v1_live", "mlb_k_v1_live",
            "mlb_logreg_v1", "mlb_xgb_v1"
        ])
    with col2:
        season = st.selectbox("Season", [2026, 2025, 2024, 2023, 2022])
    with col3:
        limit = st.slider("Max rows", 50, 500, 200)

    preds = api_get("/predictions", {"model": model, "season": season, "limit": limit})
    if preds:
        pdf = pd.DataFrame(preds)
    else:
        pdf = db_query("""
            SELECT p.prediction_id, p.model_name, p.market,
                   p.predicted_prob, p.predicted_value, p.edge,
                   p.bet_placed, p.outcome, p.pnl,
                   g.game_date, g.home_score, g.away_score,
                   ht.name as home_team, at.name as away_team
            FROM predictions p JOIN games g ON p.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            JOIN seasons s ON g.season_id = s.season_id
            WHERE p.model_name = %s AND s.year = %s
            ORDER BY g.game_date DESC LIMIT %s
        """, [model, season, limit])

    if len(pdf) > 0:
        cols = ["game_date", "home_team", "away_team", "market",
                "predicted_prob", "predicted_value", "edge",
                "bet_placed", "outcome", "home_score", "away_score"]
        avail = [c for c in cols if c in pdf.columns]
        display = pdf[avail].copy()
        for c in ["predicted_prob", "predicted_value"]:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        if "edge" in display.columns:
            display["edge"] = display["edge"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(display)} predictions")
    else:
        st.info("No predictions found")


# ---------------------------------------------------------------------------
# Tab 5: Calibration
# ---------------------------------------------------------------------------
with tab5:
    st.header("Calibration Analysis")

    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", ["mlb_logreg_v1", "mlb_logreg_v1_live"], key="cal_m")
    with col2:
        season = st.selectbox("Season", [None, 2024, 2023, 2022],
                              format_func=lambda x: "All" if x is None else str(x), key="cal_s")

    params = {"model": model}
    if season:
        params["season"] = season

    cal = api_get("/calibration", params)
    if cal:
        cdf = pd.DataFrame(cal)
    else:
        sql = """SELECT p.predicted_prob, p.outcome FROM predictions p
                 JOIN games g ON p.game_id = g.game_id JOIN seasons s ON g.season_id = s.season_id
                 WHERE p.model_name = %s AND p.market = 'moneyline' AND p.outcome IS NOT NULL"""
        sql_params = [model]
        if season:
            sql += " AND s.year = %s"
            sql_params.append(season)
        raw = db_query(sql, sql_params)
        cdf = pd.DataFrame()
        if len(raw) > 0:
            raw["home_win"] = (raw["outcome"] == "win").astype(int)
            raw["predicted_prob"] = raw["predicted_prob"].astype(float)
            bins = np.linspace(0, 1, 11)
            cal_rows = []
            for i in range(10):
                mask = (raw["predicted_prob"] >= bins[i]) & (raw["predicted_prob"] < bins[i+1])
                subset = raw[mask]
                if len(subset) >= 5:
                    cal_rows.append({
                        "predicted_mean": round(subset["predicted_prob"].mean(), 4),
                        "actual_mean": round(subset["home_win"].mean(), 4),
                        "count": len(subset),
                        "diff": round(abs(subset["predicted_prob"].mean() - subset["home_win"].mean()), 4),
                        "bin_low": round(bins[i], 2), "bin_high": round(bins[i+1], 2),
                    })
            cdf = pd.DataFrame(cal_rows)
        if len(cdf) > 0:
            chart = cdf[["predicted_mean", "actual_mean"]].copy()
            chart["Perfect"] = chart["predicted_mean"]
            chart.columns = ["Predicted", "Actual", "Perfect"]
            st.line_chart(chart.set_index("Predicted"))

            cdf["Bin"] = cdf.apply(lambda r: f"{r['bin_low']:.0%}-{r['bin_high']:.0%}", axis=1)
            st.dataframe(cdf[["Bin", "predicted_mean", "actual_mean", "count", "diff"]],
                         use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 6: Data Health
# ---------------------------------------------------------------------------
with tab6:
    st.header("Data Health")

    health = api_get("/health")
    if not health:
        # Build health dict from direct DB queries
        health = {}
        for key, sql in [
            ("games_mlb", "SELECT COUNT(*) as cnt FROM games WHERE sport_id = 2"),
            ("games_wnba", "SELECT COUNT(*) as cnt FROM games WHERE sport_id = 3"),
            ("games_cbb", "SELECT COUNT(*) as cnt FROM games WHERE sport_id = 1"),
            ("mlb_batting_game", "SELECT COUNT(*) as cnt FROM mlb_batting_game"),
            ("mlb_pitching_game", "SELECT COUNT(*) as cnt FROM mlb_pitching_game"),
            ("mlb_pitches", "SELECT COUNT(*) as cnt FROM mlb_pitches"),
            ("odds", "SELECT COUNT(*) as cnt FROM odds"),
            ("predictions", "SELECT COUNT(*) as cnt FROM predictions"),
            ("predictions_live", "SELECT COUNT(*) as cnt FROM predictions WHERE model_name LIKE '%%_live'"),
            ("flagged_bets", "SELECT COUNT(*) as cnt FROM predictions WHERE bet_placed = true AND model_name LIKE '%%_live'"),
        ]:
            try:
                r = db_query(sql)
                health[key] = int(r.iloc[0]["cnt"])
            except:
                health[key] = 0
        try:
            r = db_query("SELECT MAX(game_date) as d FROM games WHERE sport_id = 2 AND status = 'final'")
            health["latest_mlb_game"] = str(r.iloc[0]["d"])
        except:
            health["latest_mlb_game"] = "N/A"
        try:
            pipeline = db_query("""
                SELECT model_name, MAX(g.game_date) as latest_prediction
                FROM predictions p JOIN games g ON p.game_id = g.game_id
                WHERE p.model_name LIKE '%%_live' GROUP BY p.model_name
            """)
            health["pipeline_status"] = pipeline.to_dict(orient="records")
        except:
            health["pipeline_status"] = []

    if health:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Games")
            st.metric("MLB Games", f"{health.get('games_mlb', 0):,}")
            st.metric("WNBA Games", f"{health.get('games_wnba', 0):,}")
            st.metric("CBB Games", f"{health.get('games_cbb', 0):,}")
            st.metric("Latest MLB Final", health.get("latest_mlb_game", "N/A"))

        with col2:
            st.subheader("MLB Data")
            st.metric("Batting Records", f"{health.get('mlb_batting_game', 0):,}")
            st.metric("Pitching Records", f"{health.get('mlb_pitching_game', 0):,}")
            st.metric("Statcast Pitches", f"{health.get('mlb_pitches', 0):,}")

        with col3:
            st.subheader("Odds & Predictions")
            st.metric("Odds Records", f"{health.get('odds', 0):,}")
            st.metric("Total Predictions", f"{health.get('predictions', 0):,}")
            st.metric("Live Predictions", f"{health.get('predictions_live', 0):,}")
            st.metric("Flagged Bets", f"{health.get('flagged_bets', 0):,}")

        st.subheader("Pipeline Status")
        pipeline = health.get("pipeline_status", [])
        if pipeline:
            st.dataframe(pd.DataFrame(pipeline), use_container_width=True, hide_index=True)
