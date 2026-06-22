"""
models/mlb/totals_model.py - Run totals prediction model.

Predicts total runs scored in a game (home + away).
Uses the same features as the game model but targets total runs instead of winner.

Usage:
    python -m models.mlb.totals_model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import PoissonRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson

from db.db import query


TOTALS_FEATURES = [
    # ELO (team strength affects run scoring)
    "elo_diff",
    # Pitcher quality (lower = more runs)
    "home_p_fip_5", "away_p_fip_5",
    "home_p_era_szn", "away_p_era_szn",
    "home_p_kpct_5", "away_p_kpct_5",
    # Bullpen
    "home_bp_era_7d", "away_bp_era_7d",
    # Team/lineup batting (run production)
    "home_b_rpg_15", "away_b_rpg_15",
    "home_b_ops_15", "away_b_ops_15",
    "home_b_woba_15", "away_b_woba_15",
    "home_b_iso_15", "away_b_iso_15",
    # Park and weather (major for totals)
    "park_factor",
    "weather_temp", "weather_wind",
]


def main():
    print(f"\n{'='*60}")
    print("  TOTALS MODEL (Run Scoring)")
    print(f"{'='*60}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in TOTALS_FEATURES if c in df.columns]

    X_train = train[available].copy()
    y_train = train["total_runs"]
    X_val = val[available].copy()
    y_val = val["total_runs"]
    X_test = test[available].copy()
    y_test = test["total_runs"]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"  Features: {len(available)}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Mean total runs: {y_train.mean():.2f}")

    # Linear regression for run totals
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Feature importance
    coefs = pd.Series(model.coef_, index=available)
    top = coefs.abs().sort_values(ascending=False)
    print(f"\n  Top features:")
    for feat in top.head(10).index:
        print(f"    {feat:30s} {coefs[feat]:+.4f}")

    # Evaluate
    for name, X_s, y in [("val (2023)", X_val_s, y_val), ("test (2024)", X_test_s, y_test)]:
        preds = model.predict(X_s)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print(f"\n  --- {name} ---")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Mean pred: {preds.mean():.2f}, actual: {y.mean():.2f}")

        # Over/under calibration for common lines
        print(f"  Over/under calibration:")
        for line in [7.5, 8.5, 9.5, 10.5]:
            pred_over = (preds > line).mean()
            actual_over = (y.values > line).mean()
            diff = abs(pred_over - actual_over)
            flag = " *" if diff > 0.03 else ""
            print(f"    Total > {line}: pred={pred_over:.3f} actual={actual_over:.3f} diff={diff:.3f}{flag}")

    # Compare with CLV against total lines
    print(f"\n  --- CLV on Totals ---")
    odds = query("""
        SELECT o.game_id, o.total_line, o.sportsbook
        FROM odds o
        WHERE o.market = 'total' AND o.total_line IS NOT NULL
        ORDER BY o.game_id
    """)

    # Best total line per game
    best_totals = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        if gid not in best_totals:
            best_totals[gid] = float(r["total_line"])

    test_preds = model.predict(X_test_s)
    matched = 0
    clv_vals = []

    for idx, (_, row) in enumerate(test.iterrows()):
        gid = int(row["game_id"])
        if gid in best_totals:
            market_total = best_totals[gid]
            pred_total = test_preds[idx]
            actual_total = row["total_runs"]

            # Edge: model says over if pred > market, under if pred < market
            edge = pred_total - market_total
            # CLV: did the side we'd bet on win?
            if edge > 0.5:  # bet over
                correct = actual_total > market_total
            elif edge < -0.5:  # bet under
                correct = actual_total < market_total
            else:
                continue  # no edge

            clv_vals.append({
                "pred_total": pred_total,
                "market_total": market_total,
                "actual_total": actual_total,
                "edge": edge,
                "correct": correct,
            })
            matched += 1

    if clv_vals:
        clv_df = pd.DataFrame(clv_vals)
        print(f"  Games with total lines: {matched}")
        print(f"  Correct side: {clv_df['correct'].mean():.1%}")
        print(f"  Mean |edge|: {clv_df['edge'].abs().mean():.2f} runs")

        # By edge size
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            big = clv_df[clv_df["edge"].abs() >= threshold]
            if len(big) > 10:
                print(f"  |Edge| >= {threshold}: {len(big)} games, correct={big['correct'].mean():.1%}")

    # Baseline
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"\n  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Model improvement: {(baseline_mae - test_mae) / baseline_mae:.1%}")

    # Save
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": available, "medians": medians}
    with open("models/mlb/saved/totals_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved to models/mlb/saved/totals_model.pkl")


if __name__ == "__main__":
    main()
