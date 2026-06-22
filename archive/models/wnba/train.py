"""
models/wnba/train.py - Train and evaluate WNBA models.

Moneyline (LogReg) + Totals (Linear regression).
Train on 2015-2023 (excluding 2020 bubble), test on 2024.

Usage:
    python -m models.wnba.train
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score, mean_absolute_error
from scipy.stats import binomtest

from db.db import query


BUBBLE_YEAR = 2020

ML_FEATURES = [
    "elo_diff", "elo_win_prob",
    "home_off_eff_5", "away_off_eff_5",
    "home_def_eff_5", "away_def_eff_5",
    "home_efg_5", "away_efg_5",
    "home_tov_pct_5", "away_tov_pct_5",
    "home_ppg_5", "away_ppg_5",
    "home_pace_5", "away_pace_5",
]

TOTALS_FEATURES = [
    "elo_diff",
    "home_off_eff_5", "away_off_eff_5",
    "home_def_eff_5", "away_def_eff_5",
    "home_pace_5", "away_pace_5",
    "home_ppg_5", "away_ppg_5",
    "home_efg_5", "away_efg_5",
    "home_ft_rate_5", "away_ft_rate_5",
]


def get_wnba_odds():
    """Get closing odds for WNBA games."""
    # Check what WNBA odds we have
    odds = query("""
        SELECT o.game_id, o.sportsbook, o.market,
               o.home_line, o.away_line, o.total_line,
               o.home_implied, o.away_implied
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        WHERE g.sport_id = 3 AND o.is_closing = true
    """)
    return odds


def main():
    print(f"\n{'='*60}")
    print("  WNBA MODEL TRAINING")
    print(f"{'='*60}")

    # Load features
    df = pd.read_csv("data/wnba_features.csv", parse_dates=["game_date"])
    print(f"\n  Total games: {len(df)}")

    # Split: train 2015-2023 (excl 2020), test 2024
    train = df[(df["season"].between(2015, 2023)) & (df["season"] != BUBBLE_YEAR)].copy()
    test = df[df["season"] == 2024].copy()

    print(f"  Train: {len(train)} games (2015-2023, excl 2020 bubble)")
    print(f"  Test:  {len(test)} games (2024)")

    # ==========================================
    # MONEYLINE MODEL
    # ==========================================
    print(f"\n{'='*60}")
    print("  MONEYLINE MODEL (LogReg)")
    print(f"{'='*60}")

    avail_ml = [c for c in ML_FEATURES if c in df.columns]
    X_train_ml = train[avail_ml].copy()
    y_train_ml = train["home_win"].astype(int)
    X_test_ml = test[avail_ml].copy()
    y_test_ml = test["home_win"].astype(int)

    medians_ml = X_train_ml.median()
    X_train_ml = X_train_ml.fillna(medians_ml)
    X_test_ml = X_test_ml.fillna(medians_ml)

    scaler_ml = StandardScaler()
    X_train_mls = scaler_ml.fit_transform(X_train_ml)
    X_test_mls = scaler_ml.transform(X_test_ml)

    # Tune C
    best_c = 0.01
    best_score = 0
    for c in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(penalty="l1", C=c, solver="saga", max_iter=5000, random_state=42)
        m.fit(X_train_mls, y_train_ml)
        score = m.score(X_train_mls, y_train_ml)
        print(f"  C={c}: train acc={score:.3f}")
        if score > best_score:
            best_score = score
            best_c = c

    lr = LogisticRegression(penalty="l1", C=best_c, solver="saga", max_iter=5000, random_state=42)
    lr.fit(X_train_mls, y_train_ml)

    # Non-zero features
    coefs = pd.Series(lr.coef_[0], index=avail_ml)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"\n  Best C={best_c}, {len(nonzero)}/{len(avail_ml)} non-zero features")
    for feat, c in nonzero.head(8).items():
        print(f"    {feat:30s} {coefs[feat]:+.4f}")

    # Test evaluation
    ml_probs = lr.predict_proba(X_test_mls)[:, 1]
    ml_acc = accuracy_score(y_test_ml, (ml_probs > 0.5).astype(int))
    ml_brier = brier_score_loss(y_test_ml, ml_probs)

    print(f"\n  Test (2024):")
    print(f"    Accuracy: {ml_acc:.3f}")
    print(f"    Brier:    {ml_brier:.4f}")
    print(f"    Home win rate: {y_test_ml.mean():.3f}")

    # Calibration
    print(f"    Calibration:")
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (ml_probs >= bins[i]) & (ml_probs < bins[i + 1])
        if mask.sum() >= 5:
            pred_mean = ml_probs[mask].mean()
            actual_mean = y_test_ml.values[mask].mean()
            diff = abs(pred_mean - actual_mean)
            flag = " *" if diff > 0.05 else ""
            print(f"      {bins[i]:.1f}-{bins[i+1]:.1f}: pred={pred_mean:.3f} actual={actual_mean:.3f} n={mask.sum():4d} diff={diff:.3f}{flag}")

    # ==========================================
    # TOTALS MODEL
    # ==========================================
    print(f"\n{'='*60}")
    print("  TOTALS MODEL (Linear Regression)")
    print(f"{'='*60}")

    avail_tot = [c for c in TOTALS_FEATURES if c in df.columns]
    X_train_tot = train[avail_tot].copy()
    y_train_tot = train["total_points"]
    X_test_tot = test[avail_tot].copy()
    y_test_tot = test["total_points"]

    medians_tot = X_train_tot.median()
    X_train_tot = X_train_tot.fillna(medians_tot)
    X_test_tot = X_test_tot.fillna(medians_tot)

    scaler_tot = StandardScaler()
    X_train_tots = scaler_tot.fit_transform(X_train_tot)
    X_test_tots = scaler_tot.transform(X_test_tot)

    tot_model = LinearRegression()
    tot_model.fit(X_train_tots, y_train_tot)

    tot_preds = tot_model.predict(X_test_tots)
    tot_mae = mean_absolute_error(y_test_tot, tot_preds)
    baseline_mae = mean_absolute_error(y_test_tot, np.full(len(y_test_tot), y_train_tot.mean()))

    print(f"\n  Test (2024):")
    print(f"    MAE:      {tot_mae:.2f}")
    print(f"    Baseline: {baseline_mae:.2f} (predict mean={y_train_tot.mean():.1f})")
    print(f"    Improvement: {(baseline_mae - tot_mae) / baseline_mae:.1%}")
    print(f"    Mean pred: {tot_preds.mean():.1f}, actual: {y_test_tot.mean():.1f}")

    # Feature importance
    coefs_tot = pd.Series(tot_model.coef_, index=avail_tot).abs().sort_values(ascending=False)
    print(f"\n  Top features:")
    for feat in coefs_tot.head(8).index:
        print(f"    {feat:30s} {tot_model.coef_[avail_tot.index(feat)]:+.4f}")

    # ==========================================
    # CLV ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("  CLV ANALYSIS")
    print(f"{'='*60}")

    odds = get_wnba_odds()
    print(f"\n  WNBA odds in DB: {len(odds)}")

    if len(odds) > 0:
        # Moneyline CLV
        ml_odds = odds[odds["market"] == "moneyline"]
        best_ml = {}
        for _, r in ml_odds.iterrows():
            gid = r["game_id"]
            if gid not in best_ml and pd.notna(r["home_implied"]):
                best_ml[gid] = float(r["home_implied"])

        clv_vals = []
        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid in best_ml:
                clv = float(ml_probs[idx]) - best_ml[gid]
                clv_vals.append(clv)

        if clv_vals:
            mean_clv = np.mean(clv_vals)
            print(f"\n  Moneyline CLV:")
            print(f"    Games with odds: {len(clv_vals)}")
            print(f"    Mean CLV: {mean_clv:+.4f}")
            print(f"    CLV > 0: {np.mean([c > 0 for c in clv_vals]):.1%}")
        else:
            print("\n  No moneyline odds matched for CLV")

        # Totals CLV
        tot_odds = odds[odds["market"] == "total"]
        best_tot = {}
        for _, r in tot_odds.iterrows():
            gid = r["game_id"]
            if gid not in best_tot and pd.notna(r["total_line"]):
                best_tot[gid] = float(r["total_line"])

        if best_tot:
            tot_correct = 0
            tot_matched = 0
            for idx, (_, row) in enumerate(test.iterrows()):
                gid = int(row["game_id"])
                if gid in best_tot:
                    market_total = best_tot[gid]
                    pred_total = tot_preds[idx]
                    actual_total = row["total_points"]
                    edge = pred_total - market_total

                    if abs(edge) < 1.5:
                        continue

                    tot_matched += 1
                    if (edge > 0 and actual_total > market_total) or \
                       (edge < 0 and actual_total < market_total):
                        tot_correct += 1

            if tot_matched > 0:
                print(f"\n  Totals (≥1.5 edge):")
                print(f"    Games: {tot_matched}")
                print(f"    Correct side: {tot_correct/tot_matched:.1%}")
        else:
            print("\n  No totals odds matched for CLV")
    else:
        print("  No WNBA odds in database — cannot compute CLV")
        print("  Need to scrape WNBA odds from ESPN or SBR")

    # ==========================================
    # SAVE MODELS
    # ==========================================
    os.makedirs("models/wnba/saved", exist_ok=True)

    with open("models/wnba/saved/lr_ml.pkl", "wb") as f:
        pickle.dump({"model": lr, "scaler": scaler_ml, "features": avail_ml, "medians": medians_ml}, f)

    with open("models/wnba/saved/totals.pkl", "wb") as f:
        pickle.dump({"model": tot_model, "scaler": scaler_tot, "features": avail_tot, "medians": medians_tot}, f)

    print(f"\n  Models saved to models/wnba/saved/")

    # ==========================================
    # COMPARISON WITH MLB
    # ==========================================
    print(f"\n{'='*60}")
    print("  WNBA vs MLB COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<25s} {'WNBA':>10s} {'MLB':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'ML Accuracy':<25s} {ml_acc:>10.3f} {'0.558':>10s}")
    print(f"  {'ML Brier':<25s} {ml_brier:>10.4f} {'0.2429':>10s}")
    print(f"  {'Totals MAE':<25s} {tot_mae:>10.2f} {'3.38':>10s}")
    print(f"  {'Totals Improvement':<25s} {(baseline_mae-tot_mae)/baseline_mae:>10.1%} {'1.7%':>10s}")


if __name__ == "__main__":
    main()
