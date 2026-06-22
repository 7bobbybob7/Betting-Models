"""
models/mlb/player_model.py - Player-level game outcome model.

Replaces team-level batting features with lineup-specific features.
Blends lineup-level + pitcher + ELO + bullpen + contextual features.

Compares head-to-head with Phase 1 team-level model on CLV.

Usage:
    python -m models.mlb.player_model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
from xgboost import XGBClassifier

from db.db import query
from models.mlb.train import compute_clv, evaluate_model


def build_player_feature_matrix():
    """Build feature matrix with lineup-specific features."""
    print("\n=== BUILDING PLAYER-LEVEL FEATURE MATRIX ===")

    # Load the existing team-level features (has ELO, pitcher, bullpen, contextual)
    print("  Loading team-level features...")
    team_df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    print(f"    {len(team_df)} games")

    # Build lineup features
    from models.mlb.lineup_features import build_lineup_features
    lineup_feats = build_lineup_features()

    # Merge lineup features into the game matrix
    print("\n  Merging lineup features...")

    # We need game_id -> home_team_id, away_team_id mapping
    game_teams = query("""
        SELECT game_id, home_team_id, away_team_id
        FROM games WHERE sport_id = 2
    """)
    team_map = {}
    for _, r in game_teams.iterrows():
        team_map[int(r["game_id"])] = (int(r["home_team_id"]), int(r["away_team_id"]))

    # Add lineup features to each game
    home_lu_cols = []
    away_lu_cols = []

    for _, row in team_df.iterrows():
        gid = int(row["game_id"])
        if gid not in team_map:
            continue
        home_tid, away_tid = team_map[gid]

        h_lu = lineup_feats.get((gid, home_tid), {})
        a_lu = lineup_feats.get((gid, away_tid), {})

        for k, v in h_lu.items():
            team_df.loc[team_df["game_id"] == gid, f"home_{k}"] = v
        for k, v in a_lu.items():
            team_df.loc[team_df["game_id"] == gid, f"away_{k}"] = v

    # Faster approach: build lookup then vectorize
    print("  Vectorizing lineup merge...")
    for prefix, tid_col in [("home", "home_team_id"), ("away", "away_team_id")]:
        for lu_key in ["lu_woba", "lu_kpct", "lu_iso", "lu_exit_velo", "lu_xba",
                        "lu_hard_hit", "lu_top_bot_gap", "lu_players_with_stats"]:
            col_name = f"{prefix}_{lu_key}"
            vals = []
            for _, row in team_df.iterrows():
                gid = int(row["game_id"])
                if gid in team_map:
                    tid = team_map[gid][0] if prefix == "home" else team_map[gid][1]
                    lu = lineup_feats.get((gid, tid), {})
                    vals.append(lu.get(lu_key))
                else:
                    vals.append(None)
            team_df[col_name] = vals

    # Add lineup differentials
    for col in ["lu_woba", "lu_kpct", "lu_iso", "lu_exit_velo", "lu_xba", "lu_hard_hit"]:
        h = f"home_{col}"
        a = f"away_{col}"
        if h in team_df.columns and a in team_df.columns:
            invert = col in ["lu_kpct"]  # lower K% is better
            if invert:
                team_df[f"diff_{col}"] = team_df[a] - team_df[h]
            else:
                team_df[f"diff_{col}"] = team_df[h] - team_df[a]

    print(f"  Final matrix: {team_df.shape}")
    return team_df


# Player-level features (lineup + pitcher + ELO + contextual)
PLAYER_FEATURES = [
    # ELO
    "elo_diff", "elo_win_prob",
    # Pitcher
    "home_p_fip_5", "away_p_fip_5",
    "home_p_kpct_5", "away_p_kpct_5",
    "home_p_era_szn", "away_p_era_szn",
    "home_p_rest_days", "away_p_rest_days",
    # Bullpen
    "home_bp_era_7d", "away_bp_era_7d",
    "home_bp_ip_3d", "away_bp_ip_3d",
    # Lineup (replaces team batting)
    "home_lu_woba", "away_lu_woba",
    "home_lu_kpct", "away_lu_kpct",
    "home_lu_iso", "away_lu_iso",
    "home_lu_exit_velo", "away_lu_exit_velo",
    "home_lu_xba", "away_lu_xba",
    "home_lu_hard_hit", "away_lu_hard_hit",
    "home_lu_top_bot_gap", "away_lu_top_bot_gap",
    # Differentials
    "diff_lu_woba", "diff_lu_kpct", "diff_lu_iso",
    "diff_lu_exit_velo", "diff_lu_xba", "diff_lu_hard_hit",
    # Contextual
    "park_factor", "weather_temp",
]

# Blended: team-level + lineup features
BLENDED_FEATURES = PLAYER_FEATURES + [
    "home_b_woba_15", "away_b_woba_15",
    "home_b_ops_15", "away_b_ops_15",
    "home_b_rpg_15", "away_b_rpg_15",
    "diff_b_woba_15", "diff_b_ops_15", "diff_b_rpg_15",
]


def train_and_evaluate(df, features, model_name):
    """Train and evaluate a model variant."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    available = [c for c in features if c in df.columns]

    X_train = train[available].copy()
    y_train = train["home_win"].astype(int)
    X_val = val[available].copy()
    y_val = val["home_win"].astype(int)
    X_test = test[available].copy()
    y_test = test["home_win"].astype(int)

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    print(f"  Features: {len(available)}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # LogReg
    best_c, best_brier = None, 1.0
    for c in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(penalty="l1", C=c, solver="saga", max_iter=5000, random_state=42)
        m.fit(X_train_s, y_train)
        p = m.predict_proba(X_val_s)[:, 1]
        b = brier_score_loss(y_val, p)
        if b < best_brier:
            best_brier = b
            best_c = c

    lr = LogisticRegression(penalty="l1", C=best_c, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_train_s, y_train)

    # Non-zero features
    coefs = pd.Series(lr.coef_[0], index=available)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"\n  LogReg: C={best_c}, {len(nonzero)}/{len(available)} non-zero features")
    for feat, c in nonzero.head(10).items():
        print(f"    {feat:35s} {coefs[feat]:+.4f}")

    # Evaluate
    test_probs = lr.predict_proba(X_test_s)[:, 1]
    metrics = evaluate_model(model_name, y_test, test_probs, "test (2024)")

    # CLV
    pred_df = pd.DataFrame({
        "game_id": test["game_id"].values,
        "model_prob": test_probs,
        "home_win": test["home_win"].values,
    })
    clv_result = compute_clv(pred_df)

    # Save
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {"model": lr, "scaler": scaler, "features": available, "medians": medians}
    path = f"models/mlb/saved/{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    return metrics, clv_result


def main():
    df = build_player_feature_matrix()

    # Save the player-level feature matrix
    df.to_csv("data/mlb_features_player.csv", index=False)
    print(f"\nSaved player-level features to data/mlb_features_player.csv")

    # Model 1: Player-level only (lineup replaces team batting)
    m1_metrics, m1_clv = train_and_evaluate(df, PLAYER_FEATURES, "Player-Level")

    # Model 2: Blended (lineup + team batting)
    m2_metrics, m2_clv = train_and_evaluate(df, BLENDED_FEATURES, "Blended")

    # Summary comparison with Phase 1
    print(f"\n{'='*60}")
    print("  HEAD-TO-HEAD: PHASE 1 vs PHASE 3")
    print(f"{'='*60}")
    print(f"\n  {'Model':<25s} {'Accuracy':>10s} {'Brier':>10s} {'Mean CLV':>10s}")
    print(f"  {'-'*60}")
    print(f"  {'Phase 1 LogReg (9 feat)':<25s} {'0.5583':>10s} {'0.2429':>10s} {'+0.0037':>10s}")
    print(f"  {'Player-Level':<25s} {m1_metrics['accuracy']:>10.4f} {m1_metrics['brier']:>10.4f} {m1_clv.get('mean_clv', 0):>+10.4f}")
    print(f"  {'Blended':<25s} {m2_metrics['accuracy']:>10.4f} {m2_metrics['brier']:>10.4f} {m2_clv.get('mean_clv', 0):>+10.4f}")


if __name__ == "__main__":
    main()
