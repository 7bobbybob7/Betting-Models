"""
models/mlb/investigate_totals_deep.py - Deep investigation of totals edge.

Covers:
1. Why 2022 was the only losing year
2. Over/under asymmetry
3. September collapse
4. Opening vs closing lines
5. Park factor interactions
6. Combined totals + moneyline signal

Usage:
    python -m models.mlb.investigate_totals_deep
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from db.db import query


TOTALS_FEATURES = [
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


def get_total_lines():
    odds = query("""
        SELECT o.game_id, o.total_line, o.over_odds, o.under_odds, o.sportsbook
        FROM odds o
        WHERE o.market = 'total' AND o.total_line IS NOT NULL AND o.is_closing = true
        ORDER BY o.game_id
    """)
    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel",
                     "betmgm", "caesars", "espn_bet"]
    best = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        book = r["sportsbook"]
        if gid not in best:
            best[gid] = {"total_line": float(r["total_line"]), "sportsbook": book}
        else:
            cur_p = book_priority.index(best[gid]["sportsbook"]) if best[gid]["sportsbook"] in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {"total_line": float(r["total_line"]), "sportsbook": book}
    return {k: v["total_line"] for k, v in best.items()}


def build_predictions(df, total_lines, available):
    """Build predictions for each season using expanding window."""
    all_results = []

    for test_year in range(2019, 2025):
        train = df[df["season"].between(2016, test_year - 1)].copy()
        test = df[df["season"] == test_year].copy()

        medians = train[available].median()
        X_train = train[available].fillna(medians)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_s, train["total_runs"])

        X_test = test[available].fillna(medians)
        preds = model.predict(scaler.transform(X_test))

        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue

            market_total = total_lines[gid]
            pred_total = preds[idx]
            actual_total = row["total_runs"]
            edge = pred_total - market_total
            side = "over" if edge > 0 else "under"

            if side == "over":
                correct = actual_total > market_total
                push = actual_total == market_total
            else:
                correct = actual_total < market_total
                push = actual_total == market_total

            all_results.append({
                "game_id": gid,
                "season": test_year,
                "game_date": row["game_date"],
                "month": row["game_date"].month if hasattr(row["game_date"], "month") else pd.to_datetime(row["game_date"]).month,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "pred_total": pred_total,
                "market_total": market_total,
                "actual_total": actual_total,
                "edge": edge,
                "abs_edge": abs(edge),
                "side": side,
                "correct": correct,
                "push": push,
                "park_factor": row.get("park_factor"),
                "weather_temp": row.get("weather_temp"),
                "weather_wind": row.get("weather_wind"),
                "venue": row.get("home_team"),  # proxy
                "home_b_rpg_15": row.get("home_b_rpg_15"),
                "away_b_rpg_15": row.get("away_b_rpg_15"),
            })

    return pd.DataFrame(all_results)


def main():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in TOTALS_FEATURES if c in df.columns]
    total_lines = get_total_lines()

    print(f"\n{'='*60}")
    print("  TOTALS DEEP INVESTIGATION")
    print(f"{'='*60}")

    rdf = build_predictions(df, total_lines, available)
    rdf_edge = rdf[rdf["abs_edge"] >= 1.5].copy()  # focus on actionable edge

    print(f"\n  Total predictions: {len(rdf)}")
    print(f"  ≥1.5 run edge: {len(rdf_edge)}")

    # ==========================================
    # 1. WHY 2022 WAS THE ONLY LOSING YEAR
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. WHY 2022 LOST")
    print(f"{'='*60}")

    # League-wide run environment by year
    runs_by_year = df.groupby("season")["total_runs"].agg(["mean", "std", "count"])
    print(f"\n  League-wide runs per game:")
    for yr, row in runs_by_year.iterrows():
        if yr >= 2019:
            print(f"    {int(yr)}: {row['mean']:.2f} RPG (std={row['std']:.2f})")

    # Model prediction error by year
    print(f"\n  Model prediction error by year (≥1.5 edge games):")
    for yr in range(2019, 2025):
        yr_data = rdf_edge[rdf_edge["season"] == yr]
        if len(yr_data) > 0:
            bias = (yr_data["pred_total"] - yr_data["actual_total"]).mean()
            correct = yr_data["correct"].mean()
            market_bias = (yr_data["market_total"] - yr_data["actual_total"]).mean()
            print(f"    {yr}: correct={correct:.1%} ({len(yr_data):3d} games), "
                  f"model bias={bias:+.2f}, market bias={market_bias:+.2f}")

    # 2022 specifically: was the market also wrong, or just the model?
    yr2022 = rdf[rdf["season"] == 2022]
    print(f"\n  2022 detailed:")
    print(f"    All games: model bias={(yr2022['pred_total'] - yr2022['actual_total']).mean():+.2f}, "
          f"market bias={(yr2022['market_total'] - yr2022['actual_total']).mean():+.2f}")
    print(f"    2022 actual RPG: {yr2022['actual_total'].mean():.2f}, "
          f"model pred: {yr2022['pred_total'].mean():.2f}, "
          f"market line: {yr2022['market_total'].mean():.2f}")

    # Compare with other years
    for yr in [2021, 2023, 2024]:
        yrx = rdf[rdf["season"] == yr]
        if len(yrx) > 0:
            print(f"    {yr} actual RPG: {yrx['actual_total'].mean():.2f}, "
                  f"model pred: {yrx['pred_total'].mean():.2f}, "
                  f"market line: {yrx['market_total'].mean():.2f}")

    # ==========================================
    # 2. OVER/UNDER ASYMMETRY
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. OVER/UNDER ASYMMETRY")
    print(f"{'='*60}")

    for yr in range(2019, 2025):
        yr_edge = rdf_edge[rdf_edge["season"] == yr]
        if len(yr_edge) > 0:
            overs = yr_edge[yr_edge["side"] == "over"]
            unders = yr_edge[yr_edge["side"] == "under"]
            print(f"    {yr}: {len(overs):3d} overs ({overs['correct'].mean():.1%} correct), "
                  f"{len(unders):3d} unders ({unders['correct'].mean():.1%} correct)" if len(unders) > 0
                  else f"    {yr}: {len(overs):3d} overs ({overs['correct'].mean():.1%} correct), "
                  f"  0 unders")

    # Overall
    all_overs = rdf_edge[rdf_edge["side"] == "over"]
    all_unders = rdf_edge[rdf_edge["side"] == "under"]
    print(f"\n    Overall overs:  {len(all_overs):4d} bets, {all_overs['correct'].mean():.1%} correct")
    print(f"    Overall unders: {len(all_unders):4d} bets, {all_unders['correct'].mean():.1%} correct")

    # Is the model biased or is the market biased?
    print(f"\n    Avg model pred:  {rdf['pred_total'].mean():.2f}")
    print(f"    Avg market line: {rdf['market_total'].mean():.2f}")
    print(f"    Avg actual:      {rdf['actual_total'].mean():.2f}")

    # ==========================================
    # 3. SEPTEMBER COLLAPSE
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. MONTHLY PERFORMANCE (≥1.5 run edge)")
    print(f"{'='*60}")

    monthly = rdf_edge.groupby("month").agg(
        games=("correct", "count"),
        correct_pct=("correct", "mean"),
        avg_edge=("abs_edge", "mean"),
    )
    for month, row in monthly.iterrows():
        flag = " ←" if row["correct_pct"] < 0.50 else ""
        print(f"    Month {int(month):2d}: {int(row['games']):4d} games, "
              f"correct={row['correct_pct']:.1%}, avg |edge|={row['avg_edge']:.2f}{flag}")

    # September across years
    print(f"\n  September by year (≥1.5 edge):")
    sept = rdf_edge[rdf_edge["month"] == 9]
    for yr in range(2019, 2025):
        s = sept[sept["season"] == yr]
        if len(s) > 0:
            print(f"    {yr}: {len(s):3d} games, correct={s['correct'].mean():.1%}")

    # ==========================================
    # 4. PARK FACTOR INTERACTIONS
    # ==========================================
    print(f"\n{'='*60}")
    print("  4. PARK FACTOR INTERACTIONS (≥1.5 run edge)")
    print(f"{'='*60}")

    rdf_edge["pf_bucket"] = pd.cut(
        rdf_edge["park_factor"].fillna(1.0),
        bins=[0.8, 0.93, 0.97, 1.03, 1.07, 1.3],
        labels=["very_low", "low", "neutral", "high", "very_high"]
    )

    for bucket in ["very_low", "low", "neutral", "high", "very_high"]:
        b = rdf_edge[rdf_edge["pf_bucket"] == bucket]
        if len(b) > 10:
            overs = b[b["side"] == "over"]
            unders = b[b["side"] == "under"]
            print(f"    {bucket:10s}: {len(b):4d} games, correct={b['correct'].mean():.1%} "
                  f"(overs={len(overs)}, unders={len(unders)})")

    # ==========================================
    # 5. COMBINED TOTALS + MONEYLINE SIGNAL
    # ==========================================
    print(f"\n{'='*60}")
    print("  5. COMBINED TOTALS + MONEYLINE SIGNAL")
    print(f"{'='*60}")

    # Load moneyline predictions
    ml_preds = query("""
        SELECT p.game_id, p.predicted_prob, p.edge as ml_edge
        FROM predictions p
        WHERE p.model_name = 'mlb_logreg_v1' AND p.market = 'moneyline'
    """)

    # Merge with totals predictions
    combined = rdf.merge(ml_preds, on="game_id", how="inner")
    print(f"\n  Games with both totals and ML predictions: {len(combined)}")

    if len(combined) > 0:
        # When ML model has strong opinion AND totals has edge
        combined["ml_confident"] = combined["ml_edge"].abs() >= 0.05
        combined["totals_edge"] = combined["abs_edge"] >= 1.5

        # Does ML confidence improve totals accuracy?
        for ml_conf, label in [(True, "ML confident (5%+ edge)"), (False, "ML not confident")]:
            subset = combined[(combined["ml_confident"] == ml_conf) & combined["totals_edge"]]
            if len(subset) > 10:
                print(f"    {label}: {len(subset):4d} totals bets, correct={subset['correct'].mean():.1%}")

        # When both agree on direction (high total + home favored = likely high scoring)
        combined["ml_home_favored"] = combined["predicted_prob"] > 0.55
        combined["model_says_over"] = combined["edge"] > 1.5

        both_agree = combined[combined["ml_home_favored"] & combined["model_says_over"]]
        if len(both_agree) > 10:
            print(f"\n    ML favors home + totals says over: {len(both_agree)} games, "
                  f"over correct={both_agree['correct'].mean():.1%}")

        neither = combined[~combined["ml_home_favored"] & combined["model_says_over"]]
        if len(neither) > 10:
            print(f"    ML favors away + totals says over: {len(neither)} games, "
                  f"over correct={neither['correct'].mean():.1%}")

    # ==========================================
    # 6. EDGE SIZE DISTRIBUTION — IS THE MARKET GETTING SHARPER?
    # ==========================================
    print(f"\n{'='*60}")
    print("  6. IS THE MARKET GETTING SHARPER?")
    print(f"{'='*60}")

    print(f"\n  Avg |edge| by season (all games with total lines):")
    for yr in range(2019, 2025):
        yr_data = rdf[rdf["season"] == yr]
        if len(yr_data) > 0:
            avg_edge = yr_data["abs_edge"].mean()
            big_edge = (yr_data["abs_edge"] >= 1.5).sum()
            print(f"    {yr}: avg |edge|={avg_edge:.2f} runs, "
                  f"≥1.5 run edge: {big_edge} games ({big_edge/len(yr_data)*100:.1f}%)")


if __name__ == "__main__":
    main()
