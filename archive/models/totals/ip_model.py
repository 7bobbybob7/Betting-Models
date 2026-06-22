"""
models/mlb/ip_model.py - Pitcher innings pitched prediction model.

Predicts how deep a starter goes. Coupled with K model:
projected K = predicted_K/9 * predicted_IP / 9

Usage:
    python -m models.mlb.ip_model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from db.db import query
from models.mlb.statcast_features import build_statcast_features


IP_FEATURES = [
    "ip_per_start_5", "ip_per_start_szn",
    "pitches_per_start_5",
    "sc_zone_rate_5", "sc_fb_velo_5",
    "sc_chase_rate_5", "sc_swstr_rate_5",
    "opp_k_rate_15",
    "weather_temp",
    "k_per_start_5",  # better pitchers go deeper
]


def build_ip_dataset():
    """Build dataset for IP prediction."""
    print("\nBuilding IP prediction dataset...")
    starters = query("""
        SELECT pg.game_id, pg.player_id as pitcher_id, pg.team_id,
               pg.ip as actual_ip, pg.so as actual_k, pg.pitches, pg.earned_runs,
               g.game_date, g.home_team_id, g.away_team_id,
               s.year as season,
               gi.weather_temp
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE pg.is_starter = true AND pg.ip IS NOT NULL
          AND g.sport_id = 2 AND g.status = 'final'
        ORDER BY g.game_date
    """)

    pitcher_feats, team_feats = build_statcast_features()

    # Build rolling features
    starters_sorted = starters.sort_values("game_date")
    rolling_feats = {}

    for pid, grp in starters_sorted.groupby("pitcher_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 3:
                rolling_feats[(gid, int(pid))] = {
                    "ip_per_start_5": None, "ip_per_start_szn": None,
                    "pitches_per_start_5": None, "k_per_start_5": None,
                }
                continue

            last5 = prior.tail(5)
            season_prior = prior[prior["season"] == row["season"]] if "season" in prior.columns else prior

            rolling_feats[(gid, int(pid))] = {
                "ip_per_start_5": round(last5["actual_ip"].mean(), 2),
                "ip_per_start_szn": round(season_prior["actual_ip"].mean(), 2) if len(season_prior) > 0 else None,
                "pitches_per_start_5": round(last5["pitches"].mean(), 1) if last5["pitches"].notna().all() else None,
                "k_per_start_5": round(last5["actual_k"].mean(), 2) if last5["actual_k"].notna().all() else None,
            }

    # Assemble
    rows = []
    for _, row in starters_sorted.iterrows():
        gid = int(row["game_id"])
        pid = int(row["pitcher_id"])
        tid = int(row["team_id"])
        opp_tid = int(row["away_team_id"]) if tid == row["home_team_id"] else int(row["home_team_id"])

        r = {
            "game_id": gid, "pitcher_id": pid,
            "game_date": row["game_date"], "season": int(row["season"]),
            "actual_ip": float(row["actual_ip"]),
            "weather_temp": row.get("weather_temp"),
        }
        r.update(pitcher_feats.get((gid, pid), {}))
        r.update(rolling_feats.get((gid, pid), {}))
        r.update(team_feats.get((gid, opp_tid), {}))
        rows.append(r)

    df = pd.DataFrame(rows)
    print(f"  Dataset: {df.shape}")
    return df


def train_ip_model(df):
    """Train IP prediction model."""
    print(f"\n{'='*60}")
    print("  PITCHER IP MODEL")
    print(f"{'='*60}")

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in IP_FEATURES if c in df.columns]
    print(f"  Features: {len(available)}")

    X_train = train[available].copy()
    y_train = train["actual_ip"]
    X_val = val[available].copy()
    y_val = val["actual_ip"]
    X_test = test[available].copy()
    y_test = test["actual_ip"]

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    # Feature importance
    coefs = pd.Series(model.coef_, index=available).abs().sort_values(ascending=False)
    print(f"\n  Top features:")
    for feat, c in coefs.head(8).items():
        print(f"    {feat:30s} {model.coef_[available.index(feat)]:+.4f}")

    # Evaluate
    for name, X_s, y in [("val (2023)", X_val_s, y_val), ("test (2024)", X_test_s, y_test)]:
        preds = np.clip(model.predict(X_s), 0.1, 9.0)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        print(f"\n  --- {name} ---")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Mean pred: {preds.mean():.2f}, actual: {y.mean():.2f}")

        # By IP bucket
        for lo, hi in [(0, 4), (4, 5.1), (5.1, 6.1), (6.1, 9)]:
            mask = (y >= lo) & (y < hi)
            if mask.sum() > 20:
                print(f"    IP {lo}-{hi}: {mask.sum():4d} starts, pred={preds[mask].mean():.2f} actual={y.values[mask].mean():.2f}")

    # Baseline
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    test_preds = np.clip(model.predict(X_test_s), 0.1, 9.0)
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"\n  Baseline MAE (predict mean={y_train.mean():.2f}): {baseline_mae:.3f}")
    print(f"  Model improvement: {(baseline_mae - test_mae) / baseline_mae:.1%}")

    # Save
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {"model": model, "scaler": scaler, "features": available, "medians": medians}
    with open("models/mlb/saved/ip_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved to models/mlb/saved/ip_model.pkl")

    return model, scaler


def main():
    df = build_ip_dataset()
    train_ip_model(df)


if __name__ == "__main__":
    main()
