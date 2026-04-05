"""
scripts/daily_pipeline.py - Full daily pipeline orchestrator.

Runs all steps in order:
1. Refresh data (new games, box scores)
2. Scrape today's odds from SBR
3. Build features for today's games
4. Generate predictions
5. Flag actionable bets (totals strategy)
6. Log everything to predictions table

Usage:
    python scripts/daily_pipeline.py
    python scripts/daily_pipeline.py --date 2026-04-04  # specific date
    python scripts/daily_pipeline.py --skip-scrape       # skip odds scrape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta

from db.db import query, execute, bulk_insert
from scripts.daily_refresh import main as refresh_main
from scrapers.odds.sbr_scraper import scrape_date as scrape_odds


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_todays_games(target_date):
    """Get today's scheduled/final games."""
    dt_str = target_date.strftime("%Y-%m-%d")
    games = query("""
        SELECT g.game_id, g.game_date, g.home_score, g.away_score,
               g.status, g.venue,
               g.home_team_id, g.away_team_id,
               ht.name as home_team, at.name as away_team,
               s.year as season,
               gi.weather_temp, gi.weather_wind
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        JOIN seasons s ON g.season_id = s.season_id
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = 2 AND g.game_date = %s
    """, [dt_str])
    return games


def get_todays_odds(game_ids):
    """Get odds for today's games."""
    if not game_ids:
        return {}

    odds = query("""
        SELECT game_id, sportsbook, market, home_line, away_line,
               total_line, home_implied, away_implied
        FROM odds
        WHERE game_id = ANY(%s) AND is_closing = true
    """, [list(game_ids)])

    result = {}
    for _, r in odds.iterrows():
        gid = int(r["game_id"])
        if gid not in result:
            result[gid] = {"moneyline": [], "total": []}

        if r["market"] == "moneyline":
            result[gid]["moneyline"].append({
                "sportsbook": r["sportsbook"],
                "home_implied": float(r["home_implied"]) if pd.notna(r["home_implied"]) else None,
                "home_line": float(r["home_line"]) if pd.notna(r["home_line"]) else None,
            })
        elif r["market"] == "total":
            result[gid]["total"].append({
                "sportsbook": r["sportsbook"],
                "total_line": float(r["total_line"]) if pd.notna(r["total_line"]) else None,
            })

    return result


def build_quick_features(games_df):
    """Build features for today's games using the full feature pipeline."""
    from models.mlb.features import build_feature_matrix

    # Build full matrix (loads all historical data, computes rolling features)
    # Filter to current season
    current_year = date.today().year
    df = build_feature_matrix(current_year, current_year)

    # Filter to just today's game IDs
    today_ids = set(games_df["game_id"].astype(int))
    today_df = df[df["game_id"].isin(today_ids)]

    return today_df


def generate_predictions(features_df, odds_data, target_date):
    """Generate and display predictions."""
    print(f"\n{'='*60}")
    print(f"  PREDICTIONS — {target_date}")
    print(f"{'='*60}")

    # Load models
    models = {}
    model_files = {
        "mlb_logreg_v1": "models/mlb/saved/lr_core.pkl",
        "mlb_totals_v1": "models/mlb/saved/totals_model.pkl",
        "mlb_k_v1": "models/mlb/saved/k_poisson.pkl",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = load_model(path)

    if not models:
        print("  No models found!")
        return

    # Load Statcast features for K model if available
    k_pitcher_feats = {}
    k_team_feats = {}
    if "mlb_k_v1" in models:
        try:
            from models.mlb.statcast_features import build_statcast_features
            k_pitcher_feats, k_team_feats = build_statcast_features()
        except Exception as e:
            print(f"  Warning: Statcast features failed: {e}")

    # Get starter mapping for K model
    starter_map = {}
    if "mlb_k_v1" in models:
        from db.db import query as db_query
        starters = db_query("""
            SELECT g.game_id, g.home_team_id, g.away_team_id,
                   hp.player_id as home_starter, ap.player_id as away_starter,
                   hp.so as home_k_history, ap.so as away_k_history
            FROM games g
            LEFT JOIN mlb_pitching_game hp ON g.game_id = hp.game_id
                AND hp.team_id = g.home_team_id AND hp.is_starter = true
            LEFT JOIN mlb_pitching_game ap ON g.game_id = ap.game_id
                AND ap.team_id = g.away_team_id AND ap.is_starter = true
            WHERE g.sport_id = 2
        """)
        for _, r in starters.iterrows():
            starter_map[int(r["game_id"])] = {
                "home_starter": int(r["home_starter"]) if pd.notna(r["home_starter"]) else None,
                "away_starter": int(r["away_starter"]) if pd.notna(r["away_starter"]) else None,
                "home_team_id": int(r["home_team_id"]),
                "away_team_id": int(r["away_team_id"]),
            }

    predictions = []

    for _, game in features_df.iterrows():
        gid = int(game["game_id"])
        game_odds = odds_data.get(gid, {})

        pred = {
            "game_id": gid,
            "game_date": game["game_date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
        }

        # Moneyline prediction
        if "mlb_logreg_v1" in models:
            bundle = models["mlb_logreg_v1"]
            avail = [c for c in bundle["features"] if c in features_df.columns]
            X = game[avail].to_frame().T.fillna(bundle["medians"])
            if "scaler" in bundle:
                X = bundle["scaler"].transform(X)
            prob = bundle["model"].predict_proba(X)[0][1]
            pred["ml_prob"] = round(float(prob), 4)

            # Compare with market
            ml_odds = game_odds.get("moneyline", [])
            if ml_odds:
                market_imp = ml_odds[0].get("home_implied")
                if market_imp:
                    pred["ml_edge"] = round(float(prob) - market_imp, 4)

        # Totals prediction (always generate, even without odds)
        if "mlb_totals_v1" in models:
            bundle = models["mlb_totals_v1"]
            avail = [c for c in bundle["features"] if c in features_df.columns]
            X = game[avail].to_frame().T.fillna(bundle["medians"])
            if "scaler" in bundle:
                X = bundle["scaler"].transform(X)
            pred_total = bundle["model"].predict(X)[0]
            pred["pred_total"] = round(float(pred_total), 2)

            # Compare with market total (if available)
            total_odds = game_odds.get("total", [])
            if total_odds:
                market_total = total_odds[0].get("total_line")
                if market_total:
                    pred["market_total"] = market_total
                    pred["total_edge"] = round(float(pred_total) - market_total, 2)

                    # Check if this qualifies for the conservative totals strategy
                    month = target_date.month
                    park_factor = game.get("park_factor", 1.0)
                    is_postseason = game.get("is_postseason", False)

                    qualifies = (
                        abs(pred["total_edge"]) >= 1.5 and
                        5 <= month <= 9 and
                        not is_postseason and
                        park_factor >= 1.0
                    )
                    pred["totals_bet"] = qualifies
                    if qualifies:
                        pred["totals_side"] = "OVER" if pred["total_edge"] > 0 else "UNDER"

        # K model predictions (for both starters)
        if "mlb_k_v1" in models and gid in starter_map:
            bundle = models["mlb_k_v1"]
            game_starters = starter_map[gid]

            for side, starter_key, opp_tid_key in [
                ("home", "home_starter", "away_team_id"),
                ("away", "away_starter", "home_team_id"),
            ]:
                pid = game_starters[starter_key]
                opp_tid = game_starters[opp_tid_key]
                if pid is None:
                    continue

                # Build K features for this pitcher
                sc = k_pitcher_feats.get((gid, pid), {})
                tk = k_team_feats.get((gid, opp_tid), {})

                # Also need rolling K features from box scores
                k_feat = {}
                k_feat.update(sc)
                k_feat.update(tk)

                # Get features the model expects
                avail = bundle["features"]
                feat_vals = {f: k_feat.get(f) for f in avail}
                X = pd.DataFrame([feat_vals])
                X = X.fillna(bundle["medians"])

                if "scaler" in bundle:
                    X = bundle["scaler"].transform(X)

                pred_k = bundle["model"].predict(X)[0]
                pred[f"{side}_pred_k"] = round(float(pred_k), 2)

        predictions.append(pred)

    # Display
    print(f"\n  {'Home':<25s} {'Away':<25s} {'ML Prob':>8s} {'ML Edge':>8s} {'Pred Tot':>9s} {'Mkt Tot':>8s} {'T Edge':>7s} {'H K':>5s} {'A K':>5s} {'BET':>8s}")
    print(f"  {'-'*120}")

    bets = []
    for p in predictions:
        ml_prob = f"{p.get('ml_prob', 0):.3f}" if "ml_prob" in p else "   —"
        ml_edge = f"{p.get('ml_edge', 0):+.3f}" if "ml_edge" in p else "   —"
        pred_t = f"{p.get('pred_total', 0):.1f}" if "pred_total" in p else "   —"
        mkt_t = f"{p.get('market_total', 0):.1f}" if "market_total" in p else "   —"
        t_edge = f"{p.get('total_edge', 0):+.1f}" if "total_edge" in p else "   —"
        h_k = f"{p.get('home_pred_k', 0):.1f}" if "home_pred_k" in p else "  —"
        a_k = f"{p.get('away_pred_k', 0):.1f}" if "away_pred_k" in p else "  —"
        bet = p.get("totals_side", "—") if p.get("totals_bet") else "—"

        print(f"  {p['home_team']:<25s} {p['away_team']:<25s} {ml_prob:>8s} {ml_edge:>8s} {pred_t:>9s} {mkt_t:>8s} {t_edge:>7s} {h_k:>5s} {a_k:>5s} {bet:>8s}")

        if p.get("totals_bet"):
            bets.append(p)

    if bets:
        print(f"\n  ACTIONABLE BETS ({len(bets)}):")
        for b in bets:
            print(f"    {b['totals_side']} {b['market_total']} — "
                  f"{b['away_team']} @ {b['home_team']} "
                  f"(model: {b['pred_total']:.1f}, edge: {b['total_edge']:+.1f} runs)")
    else:
        print(f"\n  No actionable bets today (filters: ≥1.5 edge, May-Sept, PF≥1.0)")

    # Log predictions to DB
    log_predictions(predictions)

    return predictions


def log_predictions(predictions):
    """Write predictions to the predictions table."""
    rows = []
    for p in predictions:
        # Log moneyline prediction
        if "ml_prob" in p:
            rows.append((
                p["game_id"], "mlb_logreg_v1_live", "moneyline",
                p["ml_prob"], None, p.get("ml_edge"),
                False, None, None, None, None,
            ))

        # Log totals prediction (always, even without odds)
        if "pred_total" in p:
            rows.append((
                p["game_id"], "mlb_totals_v1_live", "total",
                None, p["pred_total"], p.get("total_edge"),
                p.get("totals_bet", False), None, None, None, None,
            ))

        # Log K predictions (home and away starters)
        if "home_pred_k" in p:
            rows.append((
                p["game_id"], "mlb_k_v1_live", "pitcher_k_home",
                None, p["home_pred_k"], None,
                False, None, None, None, None,
            ))
        if "away_pred_k" in p:
            rows.append((
                p["game_id"], "mlb_k_v1_live", "pitcher_k_away",
                None, p["away_pred_k"], None,
                False, None, None, None, None,
            ))

    if rows:
        cols = [
            "game_id", "model_name", "market",
            "predicted_prob", "predicted_value", "edge",
            "bet_placed", "bet_amount", "bet_odds",
            "outcome", "pnl"
        ]
        try:
            bulk_insert("predictions", cols, rows)
            print(f"\n  Logged {len(rows)} predictions to DB")
        except Exception as e:
            print(f"\n  Warning: could not log predictions: {e}")


def main():
    parser = argparse.ArgumentParser(description="Daily prediction pipeline")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-scrape", action="store_true")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()

    print(f"\n{'='*60}")
    print(f"  DAILY PIPELINE — {target_date}")
    print(f"{'='*60}")

    # Step 1: Refresh data
    if not args.skip_refresh:
        print(f"\n  Step 1: Data refresh...")
        try:
            refresh_main()
        except Exception as e:
            print(f"    Warning: refresh failed: {e}")
    else:
        print(f"\n  Step 1: Skipped (--skip-refresh)")

    # Step 2: Scrape odds
    if not args.skip_scrape:
        print(f"\n  Step 2: Scraping odds from SBR...")
        try:
            scrape_odds("mlb", target_date)
        except Exception as e:
            print(f"    Warning: odds scrape failed: {e}")
    else:
        print(f"\n  Step 2: Skipped (--skip-scrape)")

    # Step 3: Get today's games
    print(f"\n  Step 3: Loading today's games...")
    games = get_todays_games(target_date)
    if len(games) == 0:
        print(f"    No games found for {target_date}")
        return
    print(f"    {len(games)} games found")

    # Step 4: Build features
    print(f"\n  Step 4: Building features...")
    try:
        features = build_quick_features(games)
        if len(features) == 0:
            print(f"    No features built (games may not have enough history)")
            return
    except Exception as e:
        print(f"    Feature building failed: {e}")
        return

    # Step 5: Get odds
    game_ids = set(games["game_id"].astype(int))
    odds = get_todays_odds(game_ids)
    print(f"    {len(odds)} games have odds")

    # Step 6: Generate predictions
    predictions = generate_predictions(features, odds, target_date)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
