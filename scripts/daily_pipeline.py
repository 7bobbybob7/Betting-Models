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
    """Get today's games that haven't started yet (scheduled only)."""
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
          AND g.status IN ('scheduled', 'pre_game')
    """, [dt_str])
    return games


def get_todays_odds(game_ids):
    """Get odds for today's games with full line shopping data."""
    if not game_ids:
        return {}

    odds = query("""
        SELECT game_id, sportsbook, market, home_line, away_line,
               total_line, over_odds, under_odds, home_implied, away_implied
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
                "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
            })

    return result


def build_quick_features(games_df):
    """Build features for today's games using the full feature pipeline."""
    from models.mlb.features import build_feature_matrix

    # Build full matrix including scheduled games (so today's games get features)
    current_year = date.today().year
    df = build_feature_matrix(current_year, current_year, include_scheduled=True)

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
        "mlb_totals_clf": "models/mlb/saved/totals_classifier.pkl",
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
    # Final games → actual starter from mlb_pitching_game
    # Scheduled games → probable starter from mlb_game_info
    starter_map = {}
    if "mlb_k_v1" in models:
        from db.db import query as db_query
        starters = db_query("""
            SELECT g.game_id, g.home_team_id, g.away_team_id,
                   COALESCE(hp.player_id, gi.home_starter_id) as home_starter,
                   COALESCE(ap.player_id, gi.away_starter_id) as away_starter
            FROM games g
            LEFT JOIN mlb_pitching_game hp ON g.game_id = hp.game_id
                AND hp.team_id = g.home_team_id AND hp.is_starter = true
            LEFT JOIN mlb_pitching_game ap ON g.game_id = ap.game_id
                AND ap.team_id = g.away_team_id AND ap.is_starter = true
            LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
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

        # Totals prediction — run both regression and classifier
        # Both use the same 2-step framework:
        #   Step 1: info_edge = model_p - devig_market_p (median book)
        #   Step 2: bet if info_edge >= threshold AND model_p > breakeven at best odds

        # Shared: collect market odds once
        # IMPORTANT: only use odds for the most common line. Books quoting alt
        # totals (e.g. 5.5 when consensus is 8.5) have wildly different odds
        # that would corrupt the median if mixed in.
        total_odds = game_odds.get("total", [])
        market_total = None
        over_odds_list = []
        under_odds_list = []
        total_odds_on_line = []
        if total_odds:
            from collections import Counter
            line_counts = Counter()
            for t in total_odds:
                line = t.get("total_line")
                if line is not None:
                    line_counts[line] += 1
            if line_counts:
                # Use the line that the most books quote
                market_total = line_counts.most_common(1)[0][0]
                for t in total_odds:
                    if t.get("total_line") != market_total:
                        continue
                    total_odds_on_line.append(t)
                    ov = t.get("over_odds")
                    un = t.get("under_odds")
                    if ov is not None:
                        over_odds_list.append(ov)
                    if un is not None:
                        under_odds_list.append(un)

        # De-vig once (shared by both models)
        devig_over = devig_under = None
        def _implied(am):
            return 100 / (am + 100) if am >= 0 else abs(am) / (abs(am) + 100)

        if market_total and over_odds_list and under_odds_list:
            med_over = np.median(over_odds_list)
            med_under = np.median(under_odds_list)
            raw_over_imp = _implied(med_over)
            raw_under_imp = _implied(med_under)
            total_imp = raw_over_imp + raw_under_imp
            devig_over = raw_over_imp / total_imp
            devig_under = raw_under_imp / total_imp

        # --- REGRESSION MODEL (mlb_totals_v1) ---
        if "mlb_totals_v1" in models:
            bundle = models["mlb_totals_v1"]
            avail = [c for c in bundle["features"] if c in features_df.columns]
            X = game[avail].to_frame().T.fillna(bundle["medians"])
            if "scaler" in bundle:
                X = bundle["scaler"].transform(X)
            pred_total = bundle["model"].predict(X)[0]
            pred["pred_total"] = round(float(pred_total), 2)

            if market_total:
                pred["market_total"] = market_total
                pred["total_edge"] = round(float(pred_total) - market_total, 2)

            if devig_over is not None:
                from scipy.stats import norm
                reg_p_over = 1 - norm.cdf(market_total, loc=pred_total, scale=4.5)
                reg_p_under = 1 - reg_p_over

                reg_over_edge = reg_p_over - devig_over
                reg_under_edge = reg_p_under - devig_under

                if reg_over_edge > reg_under_edge:
                    reg_side = "over"
                    reg_info_edge = reg_over_edge
                    reg_win_prob = reg_p_over
                else:
                    reg_side = "under"
                    reg_info_edge = reg_under_edge
                    reg_win_prob = reg_p_under

                if reg_side == "over":
                    reg_best = max(over_odds_list)
                    reg_book = "unknown"
                    for t in total_odds_on_line:
                        if t.get("over_odds") == reg_best:
                            reg_book = t.get("sportsbook", "unknown")
                            break
                else:
                    reg_best = max(under_odds_list)
                    reg_book = "unknown"
                    for t in total_odds_on_line:
                        if t.get("under_odds") == reg_best:
                            reg_book = t.get("sportsbook", "unknown")
                            break

                reg_breakeven = _implied(reg_best)
                reg_ev = reg_win_prob - reg_breakeven

                pred["best_odds"] = reg_best
                pred["best_book"] = reg_book
                pred["model_win_prob"] = round(float(reg_win_prob), 4)
                pred["breakeven"] = round(float(reg_breakeven), 4)
                pred["info_edge"] = round(float(reg_info_edge), 4)
                pred["devig_implied"] = round(float(devig_over if reg_side == "over" else devig_under), 4)

                REG_THRESHOLD = 0.01
                pred["totals_bet"] = reg_info_edge >= REG_THRESHOLD and reg_ev > 0
                if pred["totals_bet"]:
                    pred["totals_side"] = "OVER" if reg_side == "over" else "UNDER"

        # --- CLASSIFIER MODEL (mlb_totals_clf) ---
        if "mlb_totals_clf" in models and market_total and devig_over is not None:
            clf_bundle = models["mlb_totals_clf"]
            clf_avail = [c for c in clf_bundle["features"] if c in features_df.columns or c in ("market_line", "rpg_vs_line")]

            # Build classifier features (needs market_line and rpg_vs_line)
            clf_feat = {}
            for c in clf_avail:
                if c == "market_line":
                    clf_feat[c] = market_total
                elif c == "rpg_vs_line":
                    home_rpg = game.get("home_b_rpg_15", 0) or 0
                    away_rpg = game.get("away_b_rpg_15", 0) or 0
                    clf_feat[c] = home_rpg + away_rpg - market_total
                else:
                    clf_feat[c] = game.get(c)
            X_clf = pd.DataFrame([clf_feat])
            X_clf = X_clf[clf_bundle["features"]].fillna(clf_bundle["medians"])
            X_clf_s = clf_bundle["scaler"].transform(X_clf)

            clf_p_over = clf_bundle["model"].predict_proba(X_clf_s)[0][1]
            clf_p_under = 1 - clf_p_over

            clf_over_edge = clf_p_over - devig_over
            clf_under_edge = clf_p_under - devig_under

            if clf_over_edge > clf_under_edge:
                clf_side = "over"
                clf_info_edge = clf_over_edge
                clf_win_prob = clf_p_over
            else:
                clf_side = "under"
                clf_info_edge = clf_under_edge
                clf_win_prob = clf_p_under

            if clf_side == "over":
                clf_best = max(over_odds_list)
                clf_book = "unknown"
                for t in total_odds_on_line:
                    if t.get("over_odds") == clf_best:
                        clf_book = t.get("sportsbook", "unknown")
                        break
            else:
                clf_best = max(under_odds_list)
                clf_book = "unknown"
                for t in total_odds_on_line:
                    if t.get("under_odds") == clf_best:
                        clf_book = t.get("sportsbook", "unknown")
                        break

            clf_breakeven = _implied(clf_best)
            clf_ev = clf_win_prob - clf_breakeven

            CLF_THRESHOLD = 0.03
            pred["clf_info_edge"] = round(float(clf_info_edge), 4)
            pred["clf_win_prob"] = round(float(clf_win_prob), 4)
            pred["clf_best_odds"] = clf_best
            pred["clf_best_book"] = clf_book
            pred["clf_bet"] = clf_info_edge >= CLF_THRESHOLD and clf_ev > 0
            if pred["clf_bet"]:
                pred["clf_side"] = "OVER" if clf_side == "over" else "UNDER"

            # --- CLASSIFIER BEST-EDGE MODEL (mlb_totals_clf_be) ---
            # Same classifier probabilities, but compare model_p against best-odds
            # implied (no de-vig step). Rationale: line shopping IS edge. Betting
            # whenever model_p > 1/best_decimal_odds + threshold is simply pure +EV
            # with a minimum edge gate.
            # Note: best-odds implied is typically 2-4% lower than devig-median,
            # so this flags MORE bets than the devig classifier at the same threshold.
            over_best_am = max(over_odds_list)
            under_best_am = max(under_odds_list)
            over_best_implied = _implied(over_best_am)
            under_best_implied = _implied(under_best_am)

            clf_be_over_edge = clf_p_over - over_best_implied
            clf_be_under_edge = clf_p_under - under_best_implied

            if clf_be_over_edge > clf_be_under_edge:
                clf_be_side = "over"
                clf_be_edge = clf_be_over_edge
                clf_be_win_prob = clf_p_over
                clf_be_best = over_best_am
            else:
                clf_be_side = "under"
                clf_be_edge = clf_be_under_edge
                clf_be_win_prob = clf_p_under
                clf_be_best = under_best_am

            # Find book that has the best odds on our side
            clf_be_book = "unknown"
            for t in total_odds_on_line:
                key = "over_odds" if clf_be_side == "over" else "under_odds"
                if t.get(key) == clf_be_best:
                    clf_be_book = t.get("sportsbook", "unknown")
                    break

            CLF_BE_THRESHOLD = 0.01
            pred["clf_be_info_edge"] = round(float(clf_be_edge), 4)
            pred["clf_be_win_prob"] = round(float(clf_be_win_prob), 4)
            pred["clf_be_best_odds"] = clf_be_best
            pred["clf_be_best_book"] = clf_be_book
            pred["clf_be_bet"] = clf_be_edge >= CLF_BE_THRESHOLD
            if pred["clf_be_bet"]:
                pred["clf_be_side"] = "OVER" if clf_be_side == "over" else "UNDER"

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
    print(f"\n  {'Home':<22s} {'Away':<22s} {'Pred':>5s} {'Mkt':>5s} {'Reg%':>6s} {'Clf%':>6s} {'CBE%':>6s} {'REG':>5s} {'CLF':>5s} {'CBE':>5s}")
    print(f"  {'-'*110}")

    reg_bets = []
    clf_bets = []
    clf_be_bets = []
    for p in predictions:
        pred_t = f"{p.get('pred_total', 0):.1f}" if "pred_total" in p else "  —"
        mkt_t = f"{p.get('market_total', 0):.1f}" if "market_total" in p else "  —"
        reg_e = f"{p.get('info_edge', 0):+.1%}" if "info_edge" in p else "   —"
        clf_e = f"{p.get('clf_info_edge', 0):+.1%}" if "clf_info_edge" in p else "   —"
        cbe_e = f"{p.get('clf_be_info_edge', 0):+.1%}" if "clf_be_info_edge" in p else "   —"
        reg_bet = p.get("totals_side", "—") if p.get("totals_bet") else "—"
        clf_bet = p.get("clf_side", "—") if p.get("clf_bet") else "—"
        cbe_bet = p.get("clf_be_side", "—") if p.get("clf_be_bet") else "—"

        print(f"  {p['home_team']:<22s} {p['away_team']:<22s} {pred_t:>5s} {mkt_t:>5s} {reg_e:>6s} {clf_e:>6s} {cbe_e:>6s} {reg_bet:>5s} {clf_bet:>5s} {cbe_bet:>5s}")

        if p.get("totals_bet"):
            reg_bets.append(p)
        if p.get("clf_bet"):
            clf_bets.append(p)
        if p.get("clf_be_bet"):
            clf_be_bets.append(p)

    if reg_bets or clf_bets or clf_be_bets:
        print(f"\n  REGRESSION BETS ({len(reg_bets)}) — ≥1% info_edge vs devig + EV gate:")
        for b in reg_bets:
            side = b.get('totals_side', '?')
            total = b.get('market_total', '?')
            odds = b.get('best_odds', -110)
            book = b.get('best_book', '?')
            info = b.get('info_edge', 0)
            print(f"    {side} {total} at {odds:+.0f} ({book}) — "
                  f"{b['away_team']} @ {b['home_team']} (info edge: {info:+.1%})")

        print(f"\n  CLASSIFIER BETS ({len(clf_bets)}) — ≥3% info_edge vs devig + EV gate:")
        for b in clf_bets:
            side = b.get('clf_side', '?')
            total = b.get('market_total', '?')
            odds = b.get('clf_best_odds', -110)
            book = b.get('clf_best_book', '?')
            info = b.get('clf_info_edge', 0)
            print(f"    {side} {total} at {odds:+.0f} ({book}) — "
                  f"{b['away_team']} @ {b['home_team']} (info edge: {info:+.1%})")

        print(f"\n  CLASSIFIER BEST-EDGE BETS ({len(clf_be_bets)}) — ≥1% vs best-odds implied:")
        for b in clf_be_bets:
            side = b.get('clf_be_side', '?')
            total = b.get('market_total', '?')
            odds = b.get('clf_be_best_odds', -110)
            book = b.get('clf_be_best_book', '?')
            info = b.get('clf_be_info_edge', 0)
            print(f"    {side} {total} at {odds:+.0f} ({book}) — "
                  f"{b['away_team']} @ {b['home_team']} (edge: {info:+.1%})")
    else:
        print(f"\n  No +EV bets today")

    # Log predictions to DB (new predictions insert, existing ones update if bet status changed)
    log_predictions(predictions)

    return predictions


def log_predictions(predictions):
    """Write predictions to the predictions table.

    New predictions are inserted. If a totals prediction already exists
    without a bet flag but now has odds (bet_placed=true), update it.
    """
    new_rows = []
    updates = []

    def _to_decimal(american):
        if american >= 0:
            return round(1 + american / 100, 3)
        return round(1 + 100 / abs(american), 3)

    for p in predictions:
        # Moneyline prediction
        if "ml_prob" in p:
            new_rows.append((
                p["game_id"], "mlb_logreg_v1_live", "moneyline",
                p["ml_prob"], None, p.get("ml_edge"),
                False, None, None, None, None, None,
            ))

        # Regression totals prediction
        if "pred_total" in p:
            is_bet = bool(p.get("totals_bet", False))
            info_edge = p.get("info_edge")
            best_odds = p.get("best_odds")

            bet_odds_decimal = _to_decimal(best_odds) if is_bet and best_odds is not None else None
            bet_book = p.get("best_book") if is_bet else None

            new_rows.append((
                p["game_id"], "mlb_totals_reg_live", "total",
                p.get("model_win_prob"), float(p["pred_total"]), info_edge,
                is_bet, 100.0 if is_bet else None, bet_odds_decimal, None, None,
                bet_book,
            ))

            if is_bet and info_edge is not None:
                updates.append(("mlb_totals_reg_live", info_edge, is_bet, bet_book, bet_odds_decimal, p["game_id"]))

        # Classifier totals prediction (devig-median comparison)
        if "clf_info_edge" in p:
            is_clf_bet = bool(p.get("clf_bet", False))
            clf_edge = p.get("clf_info_edge")
            clf_odds = p.get("clf_best_odds")

            clf_odds_decimal = _to_decimal(clf_odds) if is_clf_bet and clf_odds is not None else None
            clf_book = p.get("clf_best_book") if is_clf_bet else None

            pred_total_val = float(p["pred_total"]) if "pred_total" in p else None
            new_rows.append((
                p["game_id"], "mlb_totals_clf_live", "total",
                p.get("clf_win_prob"), pred_total_val, clf_edge,
                is_clf_bet, 100.0 if is_clf_bet else None, clf_odds_decimal, None, None,
                clf_book,
            ))

            if is_clf_bet and clf_edge is not None:
                updates.append(("mlb_totals_clf_live", clf_edge, is_clf_bet, clf_book, clf_odds_decimal, p["game_id"]))

        # Classifier best-edge totals prediction (vs best-odds implied)
        if "clf_be_info_edge" in p:
            is_cbe_bet = bool(p.get("clf_be_bet", False))
            cbe_edge = p.get("clf_be_info_edge")
            cbe_odds = p.get("clf_be_best_odds")

            cbe_odds_decimal = _to_decimal(cbe_odds) if is_cbe_bet and cbe_odds is not None else None
            cbe_book = p.get("clf_be_best_book") if is_cbe_bet else None

            pred_total_val = float(p["pred_total"]) if "pred_total" in p else None
            new_rows.append((
                p["game_id"], "mlb_totals_clf_be_live", "total",
                p.get("clf_be_win_prob"), pred_total_val, cbe_edge,
                is_cbe_bet, 100.0 if is_cbe_bet else None, cbe_odds_decimal, None, None,
                cbe_book,
            ))

            if is_cbe_bet and cbe_edge is not None:
                updates.append(("mlb_totals_clf_be_live", cbe_edge, is_cbe_bet, cbe_book, cbe_odds_decimal, p["game_id"]))

        # K predictions
        if "home_pred_k" in p:
            new_rows.append((
                p["game_id"], "mlb_k_v1_live", "pitcher_k_home",
                None, p["home_pred_k"], None,
                False, None, None, None, None, None,
            ))
        if "away_pred_k" in p:
            new_rows.append((
                p["game_id"], "mlb_k_v1_live", "pitcher_k_away",
                None, p["away_pred_k"], None,
                False, None, None, None, None, None,
            ))

    if new_rows:
        cols = [
            "game_id", "model_name", "market",
            "predicted_prob", "predicted_value", "edge",
            "bet_placed", "bet_amount", "bet_odds",
            "outcome", "pnl", "bet_book",
        ]
        try:
            bulk_insert("predictions", cols, new_rows)
            print(f"\n  Logged {len(new_rows)} predictions to DB")
        except Exception as e:
            print(f"\n  Warning: could not log predictions: {e}")

    # Update existing totals predictions that now have odds/bet status
    if updates:
        for model_name, edge, is_bet, bet_book, bet_odds_dec, game_id in updates:
            try:
                execute("""
                    UPDATE predictions
                    SET edge = %s, bet_placed = %s, bet_book = %s,
                        bet_odds = %s, bet_amount = %s
                    WHERE game_id = %s AND model_name = %s
                      AND market = 'total' AND bet_placed = false AND outcome IS NULL
                """, [edge, is_bet, bet_book, bet_odds_dec,
                      100.0 if is_bet else None, game_id, model_name])
            except Exception:
                pass
        print(f"  Updated {len(updates)} existing predictions with new odds")


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
