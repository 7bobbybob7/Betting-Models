"""
models/mlb/hitter_model.py - Hitter prop models (hits, total bases, HR).

Uses Negative Binomial regression (hitting stats have higher variance than mean).
Features from both box score rolling stats and Statcast batted ball data.

Usage:
    python -m models.mlb.hitter_model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import poisson, nbinom
import statsmodels.api as sm

from db.db import query


def build_hitter_dataset():
    """Build per-batter-per-game dataset with rolling features."""
    print("\nBuilding hitter prop dataset...")

    # Load batting game logs with Statcast aggregates
    print("  Loading batting stats...")
    batters = query("""
        SELECT
            bg.game_id, bg.player_id, bg.team_id, bg.batting_order,
            bg.pa, bg.ab, bg.hits, bg.doubles, bg.triples, bg.hr,
            bg.bb, bg.so, bg.hbp,
            g.game_date, g.home_team_id, g.away_team_id,
            s.year as season,
            p.name as player_name
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        JOIN players p ON bg.player_id = p.player_id
        WHERE bg.pa > 0 AND bg.batting_order IS NOT NULL
          AND bg.batting_order BETWEEN 1 AND 9
          AND g.sport_id = 2 AND g.status = 'final'
        ORDER BY g.game_date, bg.game_id, bg.batting_order
    """)
    print(f"    {len(batters)} batter-game rows")

    # Load Statcast batted ball data aggregated per batter per game
    print("  Loading Statcast batted ball stats...")
    statcast = query("""
        SELECT
            p.batter_id as player_id, p.game_id,
            COUNT(*) as pitches_seen,
            AVG(CASE WHEN p.is_in_play AND p.launch_speed IS NOT NULL THEN p.launch_speed END) as avg_exit_velo,
            AVG(CASE WHEN p.is_in_play AND p.launch_angle IS NOT NULL THEN p.launch_angle END) as avg_launch_angle,
            AVG(CASE WHEN p.is_in_play THEN p.xba END) as avg_xba,
            AVG(CASE WHEN p.is_in_play THEN p.xwoba END) as avg_xwoba,
            SUM(CASE WHEN p.is_in_play AND p.launch_speed >= 95 THEN 1 ELSE 0 END) as hard_hits,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END) as balls_in_play,
            SUM(CASE WHEN p.is_whiff THEN 1 ELSE 0 END) as whiffs,
            SUM(CASE WHEN p.is_swing THEN 1 ELSE 0 END) as swings
        FROM mlb_pitches p
        GROUP BY p.batter_id, p.game_id
    """)
    print(f"    {len(statcast)} batter-game Statcast rows")

    # Merge
    merged = batters.merge(statcast, on=["player_id", "game_id"], how="left")

    # Compute per-game metrics
    merged["tb"] = merged["hits"] + merged["doubles"] + 2 * merged["triples"] + 3 * merged["hr"]
    merged["hard_hit_rate"] = merged["hard_hits"] / merged["balls_in_play"].replace(0, np.nan)
    merged["whiff_rate"] = merged["whiffs"] / merged["swings"].replace(0, np.nan)

    # Build rolling features per batter
    print("  Building rolling features...")
    merged = merged.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    rolling_data = []
    for pid, grp in merged.groupby("player_id"):
        if len(grp) < 10:
            continue

        grp = grp.sort_values("game_date").reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.iloc[i]
            prior = grp.iloc[:i]

            if len(prior) < 10:
                continue

            last15 = prior.tail(15)
            last30 = prior.tail(30)
            season_prior = prior[prior["season"] == row["season"]]

            feat = {
                "game_id": int(row["game_id"]),
                "player_id": int(pid),
                "game_date": row["game_date"],
                "season": int(row["season"]),
                "batting_order": row["batting_order"],
                "actual_hits": int(row["hits"]) if pd.notna(row["hits"]) else 0,
                "actual_tb": int(row["tb"]) if pd.notna(row["tb"]) else 0,
                "actual_hr": int(row["hr"]) if pd.notna(row["hr"]) else 0,
                "actual_pa": int(row["pa"]) if pd.notna(row["pa"]) else 0,
                # Rolling box score
                "avg_15": _sdiv(last15["hits"].sum(), last15["ab"].sum()),
                "avg_30": _sdiv(last30["hits"].sum(), last30["ab"].sum()),
                "slg_15": _sdiv(last15["tb"].sum(), last15["ab"].sum()),
                "iso_15": _sdiv((last15["tb"] - last15["hits"]).sum(), last15["ab"].sum()),
                "hr_rate_30": _sdiv(last30["hr"].sum(), last30["ab"].sum()),
                "pa_per_game_15": round(last15["pa"].mean(), 1),
                "kpct_15": _sdiv(last15["so"].sum(), last15["pa"].sum()),
                "bbpct_15": _sdiv(last15["bb"].sum(), last15["pa"].sum()),
                "hits_per_game_15": round(last15["hits"].mean(), 2),
                "hits_per_game_30": round(last30["hits"].mean(), 2),
                "tb_per_game_15": round(last15["tb"].mean(), 2),
                "hr_per_game_30": round(last30["hr"].mean(), 2),
                # Statcast rolling
                "sc_exit_velo_15": last15["avg_exit_velo"].dropna().mean() if last15["avg_exit_velo"].notna().any() else None,
                "sc_xba_15": last15["avg_xba"].dropna().mean() if last15["avg_xba"].notna().any() else None,
                "sc_xwoba_15": last15["avg_xwoba"].dropna().mean() if last15["avg_xwoba"].notna().any() else None,
                "sc_hard_hit_15": last15["hard_hit_rate"].dropna().mean() if last15["hard_hit_rate"].notna().any() else None,
                "sc_whiff_rate_15": last15["whiff_rate"].dropna().mean() if last15["whiff_rate"].notna().any() else None,
            }

            # Season stats
            if len(season_prior) >= 5:
                feat["avg_szn"] = _sdiv(season_prior["hits"].sum(), season_prior["ab"].sum())
                feat["hr_rate_szn"] = _sdiv(season_prior["hr"].sum(), season_prior["ab"].sum())
                feat["hits_per_game_szn"] = round(season_prior["hits"].mean(), 2)
            else:
                feat["avg_szn"] = None
                feat["hr_rate_szn"] = None
                feat["hits_per_game_szn"] = None

            rolling_data.append(feat)

    df = pd.DataFrame(rolling_data)
    print(f"  Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def _sdiv(num, denom):
    if denom is None or denom == 0 or pd.isna(denom):
        return None
    return round(num / denom, 4)


HITS_FEATURES = [
    "avg_15", "avg_30", "avg_szn",
    "hits_per_game_15", "hits_per_game_30", "hits_per_game_szn",
    "kpct_15", "bbpct_15", "pa_per_game_15",
    "batting_order",
    "sc_exit_velo_15", "sc_xba_15", "sc_hard_hit_15", "sc_whiff_rate_15",
]

TB_FEATURES = [
    "slg_15", "iso_15", "tb_per_game_15",
    "avg_15", "hits_per_game_15",
    "hr_rate_30", "pa_per_game_15",
    "batting_order",
    "sc_exit_velo_15", "sc_xba_15", "sc_xwoba_15", "sc_hard_hit_15",
]

HR_FEATURES = [
    "hr_per_game_30", "hr_rate_30", "hr_rate_szn",
    "iso_15", "slg_15",
    "pa_per_game_15", "batting_order",
    "sc_exit_velo_15", "sc_hard_hit_15", "sc_xwoba_15",
]


def train_prop_model(df, target, features, name):
    """Train a Poisson model for a hitter prop."""
    print(f"\n{'='*60}")
    print(f"  HITTER {name.upper()} MODEL (Poisson)")
    print(f"{'='*60}")

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in features if c in df.columns]

    X_train = train[available].copy()
    y_train = train[target]
    X_val = val[available].copy()
    y_val = val[target]
    X_test = test[available].copy()
    y_test = test[target]

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
    print(f"  Mean {target}: {y_train.mean():.3f}")

    # Train
    best_alpha = 0
    best_mae = 999
    for alpha in [0, 0.01, 0.1, 1.0]:
        model = PoissonRegressor(alpha=alpha, max_iter=1000)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_val_s)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    model = PoissonRegressor(alpha=best_alpha, max_iter=1000)
    model.fit(X_train_s, y_train)

    # Feature importance
    coefs = pd.Series(model.coef_, index=available)
    top = coefs.abs().sort_values(ascending=False).head(8)
    print(f"\n  Best alpha={best_alpha}")
    print(f"  Top features:")
    for feat in top.index:
        print(f"    {feat:30s} {coefs[feat]:+.4f}")

    # Evaluate
    for split_name, X_s, y in [("val (2023)", X_val_s, y_val), ("test (2024)", X_test_s, y_test)]:
        preds = model.predict(X_s)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        print(f"\n  --- {split_name} ---")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Mean pred: {preds.mean():.3f}, actual: {y.mean():.3f}")

        # Calibration for common lines
        if target == "actual_hits":
            lines = [0.5, 1.5, 2.5]
        elif target == "actual_tb":
            lines = [0.5, 1.5, 2.5, 3.5]
        else:  # HR
            lines = [0.5]

        print(f"  Prop line calibration:")
        for line in lines:
            p_over = np.array([1 - poisson.cdf(line, max(mu, 0.01)) for mu in preds])
            actual_over = (y.values > line).astype(float)
            model_mean = p_over.mean()
            actual_mean = actual_over.mean()
            diff = abs(model_mean - actual_mean)
            flag = " *" if diff > 0.03 else ""
            print(f"    {target.replace('actual_', '')} > {line}: pred={model_mean:.3f} actual={actual_mean:.3f} diff={diff:.3f}{flag}")

    # Baseline
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    test_preds = model.predict(X_test_s)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"\n  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Model improvement: {(baseline_mae - test_mae) / baseline_mae:.1%}")

    # Save
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": available, "medians": medians}
    path = f"models/mlb/saved/hitter_{target.replace('actual_', '')}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved to {path}")

    return model, test_mae


def main():
    df = build_hitter_dataset()

    results = {}

    # Hits model
    _, hits_mae = train_prop_model(df, "actual_hits", HITS_FEATURES, "Hits")
    results["hits"] = hits_mae

    # Total bases model
    _, tb_mae = train_prop_model(df, "actual_tb", TB_FEATURES, "Total Bases")
    results["total_bases"] = tb_mae

    # HR model
    _, hr_mae = train_prop_model(df, "actual_hr", HR_FEATURES, "Home Runs")
    results["hr"] = hr_mae

    # Summary
    print(f"\n{'='*60}")
    print("  HITTER PROP MODEL COMPARISON")
    print(f"{'='*60}")
    for prop, mae in results.items():
        print(f"  {prop:15s} MAE: {mae:.3f}")


if __name__ == "__main__":
    main()
