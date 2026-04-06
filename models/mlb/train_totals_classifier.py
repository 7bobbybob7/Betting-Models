"""
models/mlb/train_totals_classifier.py - Train and save the totals classifier.

Direct over/under classifier (LogReg L1) using game features + market line.
Trained on all available data (2016-2024) for live deployment.

The classifier needs the market line as an input feature at prediction time,
so it requires odds to be available before generating predictions.

Usage:
    python -m models.mlb.train_totals_classifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
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

# Additional features that require the market line
LINE_FEATURES = ["market_line", "rpg_vs_line"]


def main():
    print(f"\n{'='*60}")
    print("  TRAIN TOTALS CLASSIFIER")
    print(f"{'='*60}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in FEATURES if c in df.columns]

    # Get closing total lines
    odds = query("""
        SELECT game_id, total_line
        FROM odds
        WHERE market = 'total' AND total_line IS NOT NULL AND is_closing = true
    """)
    best_lines = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        if gid not in best_lines:
            best_lines[gid] = float(r["total_line"])

    # Build training data: all games 2016-2024 with total lines
    rows = []
    for _, row in df[df["season"].between(2016, 2024)].iterrows():
        gid = int(row["game_id"])
        if gid not in best_lines:
            continue
        market_total = best_lines[gid]
        actual = row["total_runs"]
        if actual == market_total:
            continue  # push

        feat = {c: row[c] for c in available}
        home_rpg = row.get("home_b_rpg_15", 0) or 0
        away_rpg = row.get("away_b_rpg_15", 0) or 0
        feat["market_line"] = market_total
        feat["rpg_vs_line"] = home_rpg + away_rpg - market_total
        feat["target"] = 1 if actual > market_total else 0
        feat["season"] = int(row["season"])
        rows.append(feat)

    train_df = pd.DataFrame(rows)
    print(f"\n  Training samples: {len(train_df)}")
    print(f"  Over rate: {train_df['target'].mean():.1%}")

    clf_features = available + LINE_FEATURES
    clf_avail = [c for c in clf_features if c in train_df.columns]
    print(f"  Features: {len(clf_avail)}")

    X = train_df[clf_avail].fillna(train_df[clf_avail].median())
    y = train_df["target"]
    medians = train_df[clf_avail].median()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
    clf.fit(X_scaled, y)

    # In-sample check
    probs = clf.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    print(f"\n  In-sample accuracy: {accuracy_score(y, preds):.1%}")
    print(f"  In-sample Brier: {brier_score_loss(y, probs):.4f}")

    # Feature importance
    coefs = pd.Series(clf.coef_[0], index=clf_avail)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"\n  Non-zero features: {len(nonzero)}/{len(clf_avail)}")
    for feat in nonzero.head(8).index:
        print(f"    {feat:30s} {coefs[feat]:+.4f}")

    # Quick expanding-window validation
    print(f"\n  Expanding-window validation:")
    for test_year in [2023, 2024]:
        tr = train_df[train_df["season"] < test_year]
        te = train_df[train_df["season"] == test_year]
        X_tr = tr[clf_avail].fillna(medians)
        X_te = te[clf_avail].fillna(medians)
        sc_v = StandardScaler()
        clf_v = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf_v.fit(sc_v.fit_transform(X_tr), tr["target"])
        p = clf_v.predict_proba(sc_v.transform(X_te))[:, 1]
        acc = accuracy_score(te["target"], (p > 0.5).astype(int))
        print(f"    {test_year}: {len(te)} games, accuracy={acc:.1%}")

    # Save
    os.makedirs("models/mlb/saved", exist_ok=True)
    bundle = {
        "model": clf,
        "scaler": scaler,
        "features": clf_avail,
        "medians": medians,
    }
    with open("models/mlb/saved/totals_classifier.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved to models/mlb/saved/totals_classifier.pkl")


if __name__ == "__main__":
    main()
