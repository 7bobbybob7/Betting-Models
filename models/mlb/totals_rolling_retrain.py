"""
models/mlb/totals_rolling_retrain.py - Test rolling retraining strategy.

Instead of training once pre-season, retrain monthly/weekly
including current-season completed games.

Simulates: for each game, train on ALL data up to that point
(including earlier games from the same season).

Usage:
    python -m models.mlb.totals_rolling_retrain
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.stats import binomtest

from db.db import query


FEATURES = [
    "elo_diff",
    "home_p_fip_5", "away_p_fip_5",
    "home_p_era_szn", "away_p_era_szn",
    "home_p_kpct_5", "away_p_kpct_5",
    "home_bp_era_7d", "away_bp_era_7d",
    "home_b_rpg_15", "away_b_rpg_15",
    "home_b_ops_15", "away_b_ops_15",
    "home_b_woba_15", "away_b_woba_15",
    "home_b_iso_15", "away_b_iso_15",
    "park_factor",
    "weather_temp", "weather_wind",
]

LINE_FEATURES = ["market_line", "rpg_vs_line", "home_rpg_vs_half_line", "away_rpg_vs_half_line"]


def get_total_lines():
    odds = query("""
        SELECT o.game_id, o.total_line, o.over_odds, o.under_odds, o.sportsbook
        FROM odds o WHERE o.market = 'total' AND o.total_line IS NOT NULL AND o.is_closing = true
    """)
    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel", "betmgm", "caesars"]
    best = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        book = r["sportsbook"]
        if gid not in best:
            best[gid] = {"total_line": float(r["total_line"]),
                         "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                         "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                         "_sb": book}
        else:
            cur_p = book_priority.index(best[gid]["_sb"]) if best[gid]["_sb"] in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {"total_line": float(r["total_line"]),
                             "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                             "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                             "_sb": book}
    return best


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    return 1 + american / 100 if american >= 0 else 1 + 100 / abs(american)


def implied_from_odds(over_odds, under_odds):
    def to_imp(odds):
        return 100 / (odds + 100) if odds >= 0 else abs(odds) / (abs(odds) + 100)
    o, u = to_imp(over_odds), to_imp(under_odds)
    return o / (o + u)


def build_dataset(df, total_lines):
    """Build full dataset with line features."""
    rows = []
    for _, row in df.iterrows():
        gid = int(row["game_id"])
        if gid not in total_lines:
            continue
        market = total_lines[gid]
        actual = row["total_runs"]
        if actual == market["total_line"]:
            continue

        feat = {c: row[c] for c in FEATURES if c in row.index}
        feat["market_line"] = market["total_line"]
        home_rpg = row.get("home_b_rpg_15", 0) or 0
        away_rpg = row.get("away_b_rpg_15", 0) or 0
        feat["rpg_vs_line"] = home_rpg + away_rpg - market["total_line"]
        feat["home_rpg_vs_half_line"] = home_rpg - market["total_line"] / 2
        feat["away_rpg_vs_half_line"] = away_rpg - market["total_line"] / 2

        feat["target"] = 1 if actual > market["total_line"] else 0
        feat["game_id"] = gid
        feat["season"] = int(row["season"])
        feat["game_date"] = row["game_date"]
        feat["over_odds"] = market["over_odds"]
        feat["under_odds"] = market["under_odds"]
        rows.append(feat)

    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


def summarize(label, results, thresholds=[0.0, 0.01, 0.03, 0.05, 0.08]):
    rdf = pd.DataFrame(results)
    print(f"\n  {label}:")
    for t in thresholds:
        s = rdf[rdf["abs_edge"] >= t]
        if len(s) < 20:
            continue
        wins = int(s["correct"].sum())
        bets = len(s)
        profit = sum(100 * (american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"]) - 1)
                     if g["correct"] else -100 for _, g in s.iterrows())
        roi = profit / (bets * 100) * 100
        n_years = s["season"].nunique()
        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue
        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""
        print(f"    ≥{t:5.1%}: {bets:5d} bets ({bets/n_years:4.0f}/yr), {wins/bets:5.1%} win, ROI={roi:+5.1f}%, p={pval:.3f}{marker}")
    return rdf


def main():
    print(f"\n{'='*70}")
    print("  ROLLING RETRAIN vs STATIC MODEL")
    print(f"{'='*70}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    total_lines = get_total_lines()

    full = build_dataset(df, total_lines)
    clf_features = FEATURES + LINE_FEATURES
    avail = [c for c in clf_features if c in full.columns]

    print(f"\n  Total games: {len(full)}")
    print(f"  Testing on: 2019-2024")

    # ==========================================
    # APPROACH A: Static (pre-season training only)
    # ==========================================
    print(f"\n{'='*70}")
    print("  A) STATIC: Train pre-season, predict all season")
    print(f"{'='*70}")

    static_results = []
    for test_year in range(2019, 2025):
        train = full[full["season"] < test_year]
        test = full[full["season"] == test_year]
        if len(train) < 100 or len(test) < 20:
            continue

        X_tr = train[avail].fillna(train[avail].median())
        X_te = test[avail].fillna(train[avail].median())
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf.fit(X_tr_s, train["target"])
        probs = clf.predict_proba(X_te_s)[:, 1]

        for i, (_, row) in enumerate(test.iterrows()):
            market_p = implied_from_odds(row["over_odds"], row["under_odds"])
            edge = probs[i] - market_p
            static_results.append({
                "game_id": int(row["game_id"]), "season": int(row["season"]),
                "edge": edge, "abs_edge": abs(edge),
                "bet_over": edge > 0, "actual_over": row["target"] == 1,
                "correct": (edge > 0) == (row["target"] == 1),
                "over_odds": row["over_odds"], "under_odds": row["under_odds"],
            })

    summarize("Static (pre-season only)", static_results)

    # ==========================================
    # APPROACH B: Monthly retrain (include current season)
    # ==========================================
    print(f"\n{'='*70}")
    print("  B) MONTHLY RETRAIN: Retrain each month with current-season data")
    print(f"{'='*70}")

    monthly_results = []
    for test_year in range(2019, 2025):
        year_data = full[full["season"] == test_year].copy()
        year_data["month"] = pd.to_datetime(year_data["game_date"]).dt.month

        for month in sorted(year_data["month"].unique()):
            month_games = year_data[year_data["month"] == month]

            # Train on everything before this month (all prior seasons + earlier months this season)
            train = full[
                (full["season"] < test_year) |
                ((full["season"] == test_year) & (pd.to_datetime(full["game_date"]).dt.month < month))
            ]

            if len(train) < 100 or len(month_games) < 5:
                continue

            medians = train[avail].median()
            X_tr = train[avail].fillna(medians)
            X_te = month_games[avail].fillna(medians)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
            clf.fit(X_tr_s, train["target"])
            probs = clf.predict_proba(X_te_s)[:, 1]

            for i, (_, row) in enumerate(month_games.iterrows()):
                market_p = implied_from_odds(row["over_odds"], row["under_odds"])
                edge = probs[i] - market_p
                monthly_results.append({
                    "game_id": int(row["game_id"]), "season": int(row["season"]),
                    "month": month,
                    "edge": edge, "abs_edge": abs(edge),
                    "bet_over": edge > 0, "actual_over": row["target"] == 1,
                    "correct": (edge > 0) == (row["target"] == 1),
                    "over_odds": row["over_odds"], "under_odds": row["under_odds"],
                })

    summarize("Monthly retrain", monthly_results)

    # ==========================================
    # APPROACH C: Weekly retrain
    # ==========================================
    print(f"\n{'='*70}")
    print("  C) WEEKLY RETRAIN: Retrain every ~50 games with current-season data")
    print(f"{'='*70}")

    weekly_results = []
    for test_year in range(2019, 2025):
        year_data = full[full["season"] == test_year].sort_values("game_date").reset_index(drop=True)

        # Process in chunks of ~50 games (roughly weekly)
        chunk_size = 50
        for chunk_start in range(0, len(year_data), chunk_size):
            chunk = year_data.iloc[chunk_start:chunk_start + chunk_size]

            # Train on everything before this chunk
            if chunk_start == 0:
                train = full[full["season"] < test_year]
            else:
                train = pd.concat([
                    full[full["season"] < test_year],
                    year_data.iloc[:chunk_start]
                ])

            if len(train) < 100 or len(chunk) < 5:
                continue

            medians = train[avail].median()
            X_tr = train[avail].fillna(medians)
            X_te = chunk[avail].fillna(medians)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
            clf.fit(X_tr_s, train["target"])
            probs = clf.predict_proba(X_te_s)[:, 1]

            for i, (_, row) in enumerate(chunk.iterrows()):
                market_p = implied_from_odds(row["over_odds"], row["under_odds"])
                edge = probs[i] - market_p
                weekly_results.append({
                    "game_id": int(row["game_id"]), "season": int(row["season"]),
                    "edge": edge, "abs_edge": abs(edge),
                    "bet_over": edge > 0, "actual_over": row["target"] == 1,
                    "correct": (edge > 0) == (row["target"] == 1),
                    "over_odds": row["over_odds"], "under_odds": row["under_odds"],
                })

    summarize("Weekly retrain", weekly_results)

    # ==========================================
    # APPROACH D: Weekly retrain + XGBoost
    # ==========================================
    print(f"\n{'='*70}")
    print("  D) WEEKLY RETRAIN + XGBOOST")
    print(f"{'='*70}")

    weekly_xgb_results = []
    for test_year in range(2019, 2025):
        year_data = full[full["season"] == test_year].sort_values("game_date").reset_index(drop=True)

        chunk_size = 50
        for chunk_start in range(0, len(year_data), chunk_size):
            chunk = year_data.iloc[chunk_start:chunk_start + chunk_size]

            if chunk_start == 0:
                train = full[full["season"] < test_year]
            else:
                train = pd.concat([
                    full[full["season"] < test_year],
                    year_data.iloc[:chunk_start]
                ])

            if len(train) < 100 or len(chunk) < 5:
                continue

            medians = train[avail].median()
            X_tr = train[avail].fillna(medians)
            X_te = chunk[avail].fillna(medians)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            xgb = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                eval_metric="logloss",
            )
            xgb.fit(X_tr_s, train["target"])
            probs = xgb.predict_proba(X_te_s)[:, 1]

            for i, (_, row) in enumerate(chunk.iterrows()):
                market_p = implied_from_odds(row["over_odds"], row["under_odds"])
                edge = probs[i] - market_p
                weekly_xgb_results.append({
                    "game_id": int(row["game_id"]), "season": int(row["season"]),
                    "edge": edge, "abs_edge": abs(edge),
                    "bet_over": edge > 0, "actual_over": row["target"] == 1,
                    "correct": (edge > 0) == (row["target"] == 1),
                    "over_odds": row["over_odds"], "under_odds": row["under_odds"],
                })

    summarize("Weekly retrain + XGBoost", weekly_xgb_results)

    # ==========================================
    # APPROACH E: Ensemble (weekly LR + weekly XGB)
    # ==========================================
    print(f"\n{'='*70}")
    print("  E) WEEKLY RETRAIN ENSEMBLE (LR + XGB)")
    print(f"{'='*70}")

    ensemble_results = []
    for test_year in range(2019, 2025):
        year_data = full[full["season"] == test_year].sort_values("game_date").reset_index(drop=True)

        chunk_size = 50
        for chunk_start in range(0, len(year_data), chunk_size):
            chunk = year_data.iloc[chunk_start:chunk_start + chunk_size]

            if chunk_start == 0:
                train = full[full["season"] < test_year]
            else:
                train = pd.concat([
                    full[full["season"] < test_year],
                    year_data.iloc[:chunk_start]
                ])

            if len(train) < 100 or len(chunk) < 5:
                continue

            medians = train[avail].median()
            X_tr = train[avail].fillna(medians)
            X_te = chunk[avail].fillna(medians)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            lr = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
            lr.fit(X_tr_s, train["target"])
            lr_probs = lr.predict_proba(X_te_s)[:, 1]

            xgb = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                eval_metric="logloss",
            )
            xgb.fit(X_tr_s, train["target"])
            xgb_probs = xgb.predict_proba(X_te_s)[:, 1]

            ens_probs = (lr_probs + xgb_probs) / 2

            for i, (_, row) in enumerate(chunk.iterrows()):
                market_p = implied_from_odds(row["over_odds"], row["under_odds"])
                edge = ens_probs[i] - market_p
                ensemble_results.append({
                    "game_id": int(row["game_id"]), "season": int(row["season"]),
                    "edge": edge, "abs_edge": abs(edge),
                    "bet_over": edge > 0, "actual_over": row["target"] == 1,
                    "correct": (edge > 0) == (row["target"] == 1),
                    "over_odds": row["over_odds"], "under_odds": row["under_odds"],
                })

    summarize("Weekly ensemble (LR+XGB)", ensemble_results)

    # ==========================================
    # FINAL COMPARISON
    # ==========================================
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON AT ≥1% AND ≥3% EDGE")
    print(f"{'='*70}")

    all_approaches = {
        "Static (pre-season)": static_results,
        "Monthly retrain": monthly_results,
        "Weekly retrain LR": weekly_results,
        "Weekly retrain XGB": weekly_xgb_results,
        "Weekly ensemble": ensemble_results,
    }

    print(f"\n  {'Model':<25s} {'≥1% bets':>8s} {'≥1% win':>8s} {'≥1% ROI':>8s} {'≥3% bets':>8s} {'≥3% win':>8s} {'≥3% ROI':>8s}")
    print(f"  {'-'*75}")

    for name, results in all_approaches.items():
        rdf = pd.DataFrame(results)
        s1 = rdf[rdf["abs_edge"] >= 0.01]
        s3 = rdf[rdf["abs_edge"] >= 0.03]
        w1 = s1["correct"].mean() if len(s1) > 0 else 0
        w3 = s3["correct"].mean() if len(s3) > 0 else 0
        r1 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s1.iterrows()) / max(len(s1)*100,1) * 100
        r3 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s3.iterrows()) / max(len(s3)*100,1) * 100
        print(f"  {name:<25s} {len(s1):>8d} {w1:>7.1%} {r1:>+7.1f}% {len(s3):>8d} {w3:>7.1%} {r3:>+7.1f}%")


if __name__ == "__main__":
    main()
