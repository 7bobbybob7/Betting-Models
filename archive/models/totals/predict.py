"""
models/mlb/predict.py - Generate predictions and log to database.

Loads trained models, generates predictions for specified games,
computes edge against closing odds, and writes to the predictions table.

Usage:
    python -m models.mlb.predict --season 2024           # backfill 2024 predictions
    python -m models.mlb.predict --season 2024 --dry-run  # preview without writing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
import pandas as pd
import numpy as np

from db.db import query, execute, bulk_insert


def load_model(path):
    """Load a saved model bundle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_closing_implied(game_ids):
    """Get best available closing implied probability per game."""
    if not game_ids:
        return {}

    odds = query("""
        SELECT game_id, sportsbook, home_implied, away_implied
        FROM odds
        WHERE market = 'moneyline'
          AND home_implied IS NOT NULL
          AND is_closing = true
    """)

    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel",
                     "betmgm", "caesars", "espn_bet"]

    best = {}
    for _, row in odds.iterrows():
        gid = row["game_id"]
        if gid not in game_ids:
            continue
        book = row["sportsbook"]
        if gid not in best:
            best[gid] = float(row["home_implied"])
        else:
            cur_book = None  # we don't track which book was picked, just override by priority
            new_p = book_priority.index(book) if book in book_priority else 999
            # Simple: keep first sharp book found
            if new_p < 5:  # top-5 priority book
                best[gid] = float(row["home_implied"])

    return best


def get_existing_predictions(model_name):
    """Get game_ids that already have predictions for this model."""
    df = query(
        "SELECT game_id FROM predictions WHERE model_name = %s AND market = 'moneyline'",
        [model_name]
    )
    return set(df["game_id"])


def generate_predictions(model_bundle, features_df, model_name):
    """Generate predictions from a model bundle and feature DataFrame."""
    model = model_bundle["model"]
    features = model_bundle["features"]
    medians = model_bundle["medians"]

    available = [c for c in features if c in features_df.columns]
    X = features_df[available].fillna(medians)

    if "scaler" in model_bundle:
        X = model_bundle["scaler"].transform(X)

    probs = model.predict_proba(X)[:, 1]

    return probs


def log_predictions(features_df, probs, model_name, closing_implied, dry_run=False):
    """Write predictions to the predictions table."""
    existing = get_existing_predictions(model_name)

    rows = []
    for i, (_, game) in enumerate(features_df.iterrows()):
        gid = int(game["game_id"])

        if gid in existing:
            continue

        prob = float(probs[i])
        market_imp = closing_implied.get(gid)
        edge = round(prob - market_imp, 4) if market_imp else None

        # Determine outcome
        home_win = game.get("home_win")
        if pd.notna(home_win):
            outcome = "win" if home_win == 1 else "loss"
        else:
            outcome = None

        rows.append((
            gid,
            model_name,
            "moneyline",
            round(prob, 4),
            None,  # predicted_value (not used for moneyline)
            edge,
            False,  # bet_placed
            None,   # bet_amount
            None,   # bet_odds
            outcome,
            None,   # pnl
        ))

    if dry_run:
        print(f"  Would insert {len(rows)} predictions (dry run)")
        if rows:
            print(f"  Sample: game_id={rows[0][0]}, prob={rows[0][3]}, edge={rows[0][5]}, outcome={rows[0][9]}")
        return len(rows)

    if rows:
        cols = [
            "game_id", "model_name", "market",
            "predicted_prob", "predicted_value", "edge",
            "bet_placed", "bet_amount", "bet_odds",
            "outcome", "pnl"
        ]
        # Insert in chunks
        chunk_size = 5000
        for i in range(0, len(rows), chunk_size):
            bulk_insert("predictions", cols, rows[i:i + chunk_size])

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate and log MLB predictions")
    parser.add_argument("--season", type=int, required=True, help="Season to predict")
    parser.add_argument("--features", type=str, default="data/mlb_features.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MLB PREDICTION LOGGING — {args.season}")
    print(f"{'='*60}")

    # Load features
    print(f"\nLoading features...")
    df = pd.read_csv(args.features, parse_dates=["game_date"])
    season_df = df[df["season"] == args.season].copy()
    print(f"  {len(season_df)} games in {args.season}")

    # Get closing odds
    game_ids = set(season_df["game_id"].astype(int))
    closing = get_closing_implied(game_ids)
    print(f"  {len(closing)} games with closing odds")

    # Load and run each model
    models = [
        ("mlb_logreg_v1", "models/mlb/saved/lr_core.pkl"),
        ("mlb_xgb_v1", "models/mlb/saved/xgb_extended.pkl"),
    ]

    for model_name, model_path in models:
        if not os.path.exists(model_path):
            print(f"\n  {model_name}: model file not found at {model_path}, skipping")
            continue

        print(f"\n  --- {model_name} ---")
        bundle = load_model(model_path)
        probs = generate_predictions(bundle, season_df, model_name)

        print(f"  Predictions generated: {len(probs)}")
        print(f"  Prob range: {probs.min():.3f} - {probs.max():.3f}")
        print(f"  Mean prob: {probs.mean():.3f}")

        n = log_predictions(season_df, probs, model_name, closing, dry_run=args.dry_run)
        print(f"  Logged: {n} new predictions")

    # Summary
    if not args.dry_run:
        total = query("SELECT model_name, COUNT(*) as cnt FROM predictions WHERE market = 'moneyline' GROUP BY model_name ORDER BY model_name")
        print(f"\n  Predictions in DB:")
        for _, r in total.iterrows():
            print(f"    {r['model_name']:25s} {int(r['cnt']):,}")


if __name__ == "__main__":
    main()
