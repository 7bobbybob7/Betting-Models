"""
models/mlb/train.py - Train and evaluate MLB game outcome models.

Trains logistic regression (baseline) and XGBoost on the feature matrix.
Time-series split: train 2016-2023, validate 2024, test 2025.
Outputs calibration plots, CLV analysis, and model comparison.

Usage:
    python -m models.mlb.train
    python -m models.mlb.train --features data/mlb_features.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from xgboost import XGBClassifier

from db.db import query


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
METADATA_COLS = [
    "game_id", "game_date", "season", "home_team", "away_team",
    "home_score", "away_score", "home_win", "is_postseason",
]

# Core features (~30) — simpler model, less overfitting risk
CORE_FEATURES = [
    "elo_diff", "elo_win_prob",
    "home_p_fip_5", "away_p_fip_5",
    "home_p_kpct_5", "away_p_kpct_5",
    "home_p_bbpct_5", "away_p_bbpct_5",
    "home_p_era_szn", "away_p_era_szn",
    "home_p_rest_days", "away_p_rest_days",
    "home_b_woba_15", "away_b_woba_15",
    "home_b_ops_15", "away_b_ops_15",
    "home_b_rpg_15", "away_b_rpg_15",
    "home_b_kpct_15", "away_b_kpct_15",
    "home_bp_era_7d", "away_bp_era_7d",
    "home_bp_ip_3d", "away_bp_ip_3d",
    "park_factor", "weather_temp",
]

# Extended features — all differentials + raw features
def get_extended_features(df):
    """Get all numeric feature columns (exclude metadata)."""
    exclude = set(METADATA_COLS)
    return [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(df, feature_cols):
    """Prepare train/val/test splits with proper temporal ordering."""
    # Drop rows with missing target
    df = df[df["home_win"].notna()].copy()

    # Time-series split (2024 test has full odds coverage)
    train = df[df["season"].between(2016, 2022)].copy()
    val = df[df["season"] == 2023].copy()
    test = df[df["season"] == 2024].copy()

    # Filter to available features
    available = [c for c in feature_cols if c in df.columns]

    X_train = train[available].copy()
    y_train = train["home_win"].astype(int)
    X_val = val[available].copy()
    y_val = val["home_win"].astype(int)
    X_test = test[available].copy()
    y_test = test["home_win"].astype(int)

    # Fill NaN with column medians (from training set only — no leakage)
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    print(f"  Train: {len(X_train)} games (2016-2022)")
    print(f"  Val:   {len(X_val)} games (2023)")
    print(f"  Test:  {len(X_test)} games (2024)")
    print(f"  Features: {len(available)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, medians, available, train, val, test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_logistic(X_train, y_train, X_val, y_val):
    """Train logistic regression with L1 regularization."""
    print("\n--- Logistic Regression (L1) ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Try multiple C values
    best_c = None
    best_brier = 1.0

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(
            penalty="l1", C=c, solver="saga", max_iter=5000, random_state=42
        )
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        brier = brier_score_loss(y_val, probs)
        acc = accuracy_score(y_val, (probs > 0.5).astype(int))
        print(f"  C={c:6.3f}  Brier={brier:.4f}  Acc={acc:.3f}")

        if brier < best_brier:
            best_brier = brier
            best_c = c

    # Retrain with best C
    model = LogisticRegression(
        penalty="l1", C=best_c, solver="saga", max_iter=5000, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Feature importance (non-zero coefficients)
    coefs = pd.Series(model.coef_[0], index=X_train.columns)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"\n  Best C={best_c}, {len(nonzero)}/{len(coefs)} non-zero features")
    print(f"  Top 10 features:")
    for feat, coef in nonzero.head(10).items():
        print(f"    {feat:35s} {coefs[feat]:+.4f}")

    return model, scaler, best_c


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    print("\n--- XGBoost ---")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")

    # Feature importance (permutation-based would be better, but gain is quick)
    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print(f"  Top 10 features (gain):")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:35s} {imp:.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(name, y_true, probs, split_name="val"):
    """Compute and print evaluation metrics."""
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    brier = brier_score_loss(y_true, probs)
    logloss = log_loss(y_true, probs)

    print(f"\n  {name} — {split_name}:")
    print(f"    Accuracy:   {acc:.4f}")
    print(f"    Brier:      {brier:.4f}")
    print(f"    Log loss:   {logloss:.4f}")

    # Calibration by decile
    print(f"    Calibration:")
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            pred_mean = probs[mask].mean()
            actual_mean = y_true.values[mask].mean()
            diff = abs(pred_mean - actual_mean)
            n = mask.sum()
            flag = " *" if diff > 0.03 else ""
            print(f"      {bins[i]:.1f}-{bins[i+1]:.1f}: pred={pred_mean:.3f} actual={actual_mean:.3f} n={n:5d} diff={diff:.3f}{flag}")

    return {"accuracy": acc, "brier": brier, "logloss": logloss}


def compute_clv(predictions_df, sport_id=2):
    """
    Compute CLV by comparing model probabilities against closing implied probabilities.
    Returns CLV metrics.
    """
    print("\n--- CLV Analysis ---")

    # Get closing moneyline odds for games in our predictions
    game_ids = predictions_df["game_id"].tolist()
    if not game_ids:
        print("  No game IDs for CLV analysis")
        return {}

    # Get de-vigged closing implied probabilities
    # Use the sharpest available book (Pinnacle > Bet365 > DraftKings > any)
    odds = query("""
        SELECT game_id, sportsbook, home_implied, away_implied
        FROM odds
        WHERE market = 'moneyline'
          AND home_implied IS NOT NULL
          AND is_closing = true
        ORDER BY game_id, sportsbook
    """)

    if len(odds) == 0:
        print("  No closing odds found in database")
        return {}

    # For each game, pick the best available book
    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel",
                     "betmgm", "caesars", "espn_bet"]

    best_odds = {}
    for _, row in odds.iterrows():
        gid = row["game_id"]
        book = row["sportsbook"]
        if gid not in best_odds:
            best_odds[gid] = row
        else:
            current_book = best_odds[gid]["sportsbook"]
            current_priority = book_priority.index(current_book) if current_book in book_priority else 999
            new_priority = book_priority.index(book) if book in book_priority else 999
            if new_priority < current_priority:
                best_odds[gid] = row

    # Match predictions with odds
    matched = 0
    clv_values = []

    for _, pred in predictions_df.iterrows():
        gid = pred["game_id"]
        if gid in best_odds:
            market_implied = best_odds[gid]["home_implied"]
            model_prob = pred["model_prob"]
            if pd.notna(market_implied) and pd.notna(model_prob):
                clv = model_prob - float(market_implied)
                clv_values.append({
                    "game_id": gid,
                    "model_prob": model_prob,
                    "market_implied": float(market_implied),
                    "clv": clv,
                    "home_win": pred.get("home_win", None),
                })
                matched += 1

    if not clv_values:
        print("  No games matched between predictions and odds")
        return {}

    clv_df = pd.DataFrame(clv_values)
    mean_clv = clv_df["clv"].mean()
    median_clv = clv_df["clv"].median()
    positive_pct = (clv_df["clv"] > 0).mean()

    print(f"  Games with closing odds: {matched}")
    print(f"  Mean CLV:     {mean_clv:+.4f} ({'positive' if mean_clv > 0 else 'negative'})")
    print(f"  Median CLV:   {median_clv:+.4f}")
    print(f"  CLV > 0:      {positive_pct:.1%}")

    # CLV by predicted edge bucket
    clv_df["edge"] = clv_df["clv"].abs()
    for threshold in [0.02, 0.05, 0.10]:
        big_edge = clv_df[clv_df["edge"] >= threshold]
        if len(big_edge) > 0:
            print(f"  Games with |edge| >= {threshold:.0%}: {len(big_edge)}, mean CLV: {big_edge['clv'].mean():+.4f}")

    return {
        "matched_games": matched,
        "mean_clv": mean_clv,
        "median_clv": median_clv,
        "positive_pct": positive_pct,
        "clv_df": clv_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MLB game outcome models")
    parser.add_argument("--features", type=str, default="data/mlb_features.csv")
    parser.add_argument("--save-models", action="store_true", help="Save trained models to disk")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  MLB MODEL TRAINING")
    print(f"{'='*60}")

    # Load feature matrix
    print(f"\nLoading features from {args.features}...")
    df = pd.read_csv(args.features, parse_dates=["game_date"])
    print(f"  Shape: {df.shape}")

    # ==========================================
    # CORE FEATURES — Logistic Regression
    # ==========================================
    print(f"\n{'='*60}")
    print("  MODEL 1: Logistic Regression (Core Features)")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test, medians, features_used, train_df, val_df, test_df = \
        prepare_data(df, CORE_FEATURES)

    lr_model, lr_scaler, lr_best_c = train_logistic(X_train, y_train, X_val, y_val)

    # Evaluate on validation
    lr_val_probs = lr_model.predict_proba(lr_scaler.transform(X_val))[:, 1]
    lr_val_metrics = evaluate_model("LogReg (core)", y_val, lr_val_probs, "val (2023)")

    # Evaluate on test
    lr_test_probs = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]
    lr_test_metrics = evaluate_model("LogReg (core)", y_test, lr_test_probs, "test (2024)")

    # ==========================================
    # EXTENDED FEATURES — XGBoost
    # ==========================================
    print(f"\n{'='*60}")
    print("  MODEL 2: XGBoost (Extended Features)")
    print(f"{'='*60}")

    ext_features = get_extended_features(df)
    X_train_ext, y_train_ext, X_val_ext, y_val_ext, X_test_ext, y_test_ext, medians_ext, features_ext, _, _, _ = \
        prepare_data(df, ext_features)

    xgb_model = train_xgboost(X_train_ext, y_train_ext, X_val_ext, y_val_ext)

    # Evaluate on validation
    xgb_val_probs = xgb_model.predict_proba(X_val_ext)[:, 1]
    xgb_val_metrics = evaluate_model("XGBoost (extended)", y_val_ext, xgb_val_probs, "val (2023)")

    # Evaluate on test
    xgb_test_probs = xgb_model.predict_proba(X_test_ext)[:, 1]
    xgb_test_metrics = evaluate_model("XGBoost (extended)", y_test_ext, xgb_test_probs, "test (2024)")

    # ==========================================
    # CLV ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("  CLV ANALYSIS")
    print(f"{'='*60}")

    # Build predictions DataFrames for CLV
    for name, probs, split_df, split_name in [
        ("LogReg", lr_test_probs, test_df, "test"),
        ("XGBoost", xgb_test_probs, test_df, "test"),
    ]:
        pred_df = pd.DataFrame({
            "game_id": split_df["game_id"].values,
            "model_prob": probs,
            "home_win": split_df["home_win"].values,
        })
        print(f"\n  {name} ({split_name} set):")
        compute_clv(pred_df)

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<20s} {'LogReg (val)':<15s} {'LogReg (test)':<15s} {'XGB (val)':<15s} {'XGB (test)':<15s}")
    print(f"  {'-'*80}")
    for metric in ["accuracy", "brier", "logloss"]:
        print(f"  {metric:<20s} {lr_val_metrics[metric]:<15.4f} {lr_test_metrics[metric]:<15.4f} {xgb_val_metrics[metric]:<15.4f} {xgb_test_metrics[metric]:<15.4f}")

    # ==========================================
    # SAVE MODELS
    # ==========================================
    if args.save_models:
        os.makedirs("models/mlb/saved", exist_ok=True)
        with open("models/mlb/saved/lr_core.pkl", "wb") as f:
            pickle.dump({"model": lr_model, "scaler": lr_scaler, "features": features_used, "medians": medians}, f)
        with open("models/mlb/saved/xgb_extended.pkl", "wb") as f:
            pickle.dump({"model": xgb_model, "features": features_ext, "medians": medians_ext}, f)
        print(f"\n  Models saved to models/mlb/saved/")


if __name__ == "__main__":
    main()
