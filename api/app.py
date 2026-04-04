"""
api/app.py - FastAPI backend for the betting platform.

Serves predictions, odds, CLV, calibration, and game data.
Streamlit (or any frontend) consumes this API.

Usage:
    uvicorn api.app:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
import numpy as np

from db.db import query

app = FastAPI(title="Betting Models API", version="1.0.0")


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
@app.get("/games")
def get_games(
    sport: str = "mlb",
    season: Optional[int] = None,
    date: Optional[str] = None,
    limit: int = Query(50, le=500),
):
    """Get game results. Filter by sport, season, or date."""
    sql = """
        SELECT g.game_id, g.game_date, g.home_score, g.away_score,
               g.status, g.venue, g.is_postseason,
               ht.name as home_team, at.name as away_team, s.year as season
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        JOIN seasons s ON g.season_id = s.season_id
        JOIN sports sp ON g.sport_id = sp.sport_id
        WHERE sp.name = %s
    """
    params = [sport]

    if season:
        sql += " AND s.year = %s"
        params.append(season)
    if date:
        sql += " AND g.game_date = %s"
        params.append(date)

    sql += " ORDER BY g.game_date DESC LIMIT %s"
    params.append(limit)

    df = query(sql, params)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@app.get("/predictions")
def get_predictions(
    model: Optional[str] = None,
    season: Optional[int] = None,
    min_edge: Optional[float] = None,
    limit: int = Query(100, le=1000),
):
    """Get model predictions with game info."""
    sql = """
        SELECT p.prediction_id, p.game_id, p.model_name, p.market,
               p.predicted_prob, p.edge, p.outcome, p.pnl,
               g.game_date, g.home_score, g.away_score,
               ht.name as home_team, at.name as away_team,
               s.year as season
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE 1=1
    """
    params = []

    if model:
        sql += " AND p.model_name = %s"
        params.append(model)
    if season:
        sql += " AND s.year = %s"
        params.append(season)
    if min_edge is not None:
        sql += " AND ABS(p.edge) >= %s"
        params.append(min_edge)

    sql += " ORDER BY g.game_date DESC LIMIT %s"
    params.append(limit)

    df = query(sql, params)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------
@app.get("/clv")
def get_clv(
    model: Optional[str] = None,
    season: Optional[int] = None,
):
    """Get CLV metrics by model and season."""
    sql = """
        SELECT p.model_name, s.year as season,
               COUNT(*) as games,
               AVG(p.edge) as mean_clv,
               AVG(CASE WHEN p.edge > 0 THEN 1.0 ELSE 0.0 END) as clv_positive_pct,
               AVG(CASE WHEN
                   (p.predicted_prob > 0.5 AND p.outcome = 'win') OR
                   (p.predicted_prob < 0.5 AND p.outcome = 'loss')
                   THEN 1.0 ELSE 0.0 END) as accuracy
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE p.edge IS NOT NULL AND p.market = 'moneyline'
    """
    params = []

    if model:
        sql += " AND p.model_name = %s"
        params.append(model)
    if season:
        sql += " AND s.year = %s"
        params.append(season)

    sql += " GROUP BY p.model_name, s.year ORDER BY p.model_name, s.year"

    df = query(sql, params)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@app.get("/calibration")
def get_calibration(
    model: str = "mlb_logreg_v1",
    season: Optional[int] = None,
    n_bins: int = 10,
):
    """Get calibration data (predicted vs actual by bin)."""
    sql = """
        SELECT p.predicted_prob, p.outcome
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE p.model_name = %s AND p.market = 'moneyline' AND p.outcome IS NOT NULL
    """
    params = [model]

    if season:
        sql += " AND s.year = %s"
        params.append(season)

    df = query(sql, params)
    if len(df) == 0:
        return []

    df["home_win"] = (df["outcome"] == "win").astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (df["predicted_prob"] >= bins[i]) & (df["predicted_prob"] < bins[i + 1])
        subset = df[mask]
        if len(subset) >= 5:
            result.append({
                "bin_low": round(float(bins[i]), 2),
                "bin_high": round(float(bins[i + 1]), 2),
                "predicted_mean": round(float(subset["predicted_prob"].mean()), 4),
                "actual_mean": round(float(subset["home_win"].mean()), 4),
                "count": int(len(subset)),
                "diff": round(abs(float(subset["predicted_prob"].mean()) - float(subset["home_win"].mean())), 4),
            })

    return result


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
@app.get("/odds")
def get_odds(
    game_id: Optional[int] = None,
    season: Optional[int] = None,
    market: str = "moneyline",
    limit: int = Query(50, le=500),
):
    """Get odds data."""
    sql = """
        SELECT o.game_id, o.sportsbook, o.market,
               o.home_line, o.away_line, o.total_line,
               o.home_implied, o.away_implied, o.is_closing,
               g.game_date, ht.name as home_team, at.name as away_team
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE o.market = %s
    """
    params = [market]

    if game_id:
        sql += " AND o.game_id = %s"
        params.append(game_id)
    if season:
        sql += " AND EXTRACT(YEAR FROM g.game_date) = %s"
        params.append(season)

    sql += " ORDER BY g.game_date DESC LIMIT %s"
    params.append(limit)

    df = query(sql, params)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Models metadata
# ---------------------------------------------------------------------------
@app.get("/models")
def get_models():
    """Get summary stats for each model."""
    df = query("""
        SELECT
            p.model_name,
            COUNT(*) as total_predictions,
            COUNT(p.edge) as predictions_with_odds,
            AVG(p.edge) as mean_clv,
            AVG(CASE WHEN
                (p.predicted_prob > 0.5 AND p.outcome = 'win') OR
                (p.predicted_prob < 0.5 AND p.outcome = 'loss')
                THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG((p.predicted_prob - CASE WHEN p.outcome = 'win' THEN 1.0 ELSE 0.0 END) ^ 2) as brier_score,
            MIN(g.game_date) as earliest_game,
            MAX(g.game_date) as latest_game
        FROM predictions p
        JOIN games g ON p.game_id = g.game_id
        WHERE p.outcome IS NOT NULL
        GROUP BY p.model_name
        ORDER BY p.model_name
    """)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Data health
# ---------------------------------------------------------------------------
@app.get("/health")
def get_health():
    """Get database row counts and data freshness."""
    tables = {
        "games_mlb": "SELECT COUNT(*) as cnt FROM games WHERE sport_id = 2",
        "games_cbb": "SELECT COUNT(*) as cnt FROM games WHERE sport_id = 1",
        "mlb_pitching_game": "SELECT COUNT(*) as cnt FROM mlb_pitching_game",
        "mlb_batting_game": "SELECT COUNT(*) as cnt FROM mlb_batting_game",
        "mlb_pitches": "SELECT COUNT(*) as cnt FROM mlb_pitches",
        "mlb_game_info": "SELECT COUNT(*) as cnt FROM mlb_game_info",
        "odds": "SELECT COUNT(*) as cnt FROM odds",
        "predictions": "SELECT COUNT(*) as cnt FROM predictions",
        "players_mlb": "SELECT COUNT(*) as cnt FROM players WHERE sport_id = 2",
    }

    result = {}
    for name, sql in tables.items():
        try:
            df = query(sql)
            result[name] = int(df.iloc[0]["cnt"])
        except Exception:
            result[name] = -1

    # Latest game date
    try:
        latest = query("SELECT MAX(game_date) as latest FROM games WHERE sport_id = 2 AND status = 'final'")
        result["latest_mlb_game"] = str(latest.iloc[0]["latest"])
    except Exception:
        result["latest_mlb_game"] = None

    return result
