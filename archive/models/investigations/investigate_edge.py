"""
models/mlb/investigate_edge.py - Deep investigation of high-edge CLV signal.

Questions to answer:
1. Is the signal stable across multiple seasons?
2. What types of games are these? (teams, pitchers, park, month)
3. Does CLV scale proportionally with edge size?
4. What does simulated ROI look like at different Kelly fractions?
5. What's the max drawdown?

Usage:
    python -m models.mlb.investigate_edge
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from db.db import query
from models.mlb.train import CORE_FEATURES, get_extended_features, compute_clv


def load_features():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    return df


def get_closing_odds():
    """Get best available closing moneyline implied prob per game."""
    odds = query("""
        SELECT game_id, sportsbook, home_implied, away_implied, home_line, away_line
        FROM odds
        WHERE market = 'moneyline'
          AND home_implied IS NOT NULL
          AND is_closing = true
        ORDER BY game_id, sportsbook
    """)

    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel",
                     "betmgm", "caesars", "espn_bet"]

    best = {}
    for _, row in odds.iterrows():
        gid = row["game_id"]
        book = row["sportsbook"]
        if gid not in best:
            best[gid] = row
        else:
            cur = best[gid]["sportsbook"]
            cur_p = book_priority.index(cur) if cur in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = row

    return best


def american_to_decimal(american):
    """Convert American odds to decimal odds."""
    if american is None or pd.isna(american):
        return None
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def train_on_window(df, train_end_year, features):
    """Train logistic regression on data up to train_end_year, return model + scaler."""
    train = df[df["season"] <= train_end_year].copy()
    available = [c for c in features if c in df.columns]

    X = train[available].copy()
    y = train["home_win"].astype(int)
    medians = X.median()
    X = X.fillna(medians)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty="l1", C=0.01, solver="saga", max_iter=5000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, medians, available


def main():
    print(f"\n{'='*60}")
    print("  HIGH-EDGE CLV INVESTIGATION")
    print(f"{'='*60}")

    df = load_features()
    closing_odds = get_closing_odds()

    # ==========================================
    # 1. CROSS-SEASON STABILITY
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. CROSS-SEASON STABILITY")
    print(f"{'='*60}")
    print("  Training expanding window, testing each season separately.\n")

    # For each test season, train on all prior years
    season_results = []
    for test_year in range(2019, 2025):
        train_end = test_year - 1

        model, scaler, medians, feats = train_on_window(df, train_end, CORE_FEATURES)

        test = df[df["season"] == test_year].copy()
        X_test = test[feats].fillna(medians)
        X_test_scaled = scaler.transform(X_test)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        # Match with odds
        clv_rows = []
        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid in closing_odds:
                market_imp = closing_odds[gid]["home_implied"]
                if pd.notna(market_imp):
                    clv = probs[idx] - float(market_imp)
                    clv_rows.append({
                        "game_id": gid,
                        "model_prob": probs[idx],
                        "market_implied": float(market_imp),
                        "clv": clv,
                        "home_win": int(row["home_win"]),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "game_date": row["game_date"],
                        "home_line": closing_odds[gid].get("home_line"),
                        "away_line": closing_odds[gid].get("away_line"),
                        "season": test_year,
                        "is_postseason": row["is_postseason"],
                    })

        if not clv_rows:
            print(f"  {test_year}: no odds data")
            continue

        clv_df = pd.DataFrame(clv_rows)
        clv_df["edge"] = (clv_df["model_prob"] - clv_df["market_implied"]).abs()

        # Overall
        n_total = len(clv_df)
        mean_clv = clv_df["clv"].mean()
        pos_pct = (clv_df["clv"] > 0).mean()

        # High edge subsets
        edge_5 = clv_df[clv_df["edge"] >= 0.05]
        edge_10 = clv_df[clv_df["edge"] >= 0.10]

        result = {
            "season": test_year,
            "games": n_total,
            "mean_clv": mean_clv,
            "clv_pos_pct": pos_pct,
            "edge5_n": len(edge_5),
            "edge5_clv": edge_5["clv"].mean() if len(edge_5) > 0 else None,
            "edge10_n": len(edge_10),
            "edge10_clv": edge_10["clv"].mean() if len(edge_10) > 0 else None,
        }
        season_results.append(result)

        print(f"  {test_year}: {n_total:4d} games | CLV={mean_clv:+.4f} | CLV>0={pos_pct:.1%} | "
              f"5%+ edge: {len(edge_5):3d} games CLV={edge_5['clv'].mean():+.4f} | "
              f"10%+ edge: {len(edge_10):3d} games CLV={edge_10['clv'].mean() if len(edge_10) > 0 else 0:+.4f}")

    # Summary
    results_df = pd.DataFrame(season_results)
    print(f"\n  --- Summary ---")
    print(f"  Seasons with positive overall CLV: {(results_df['mean_clv'] > 0).sum()} / {len(results_df)}")
    print(f"  Seasons with positive 5%+ edge CLV: {(results_df['edge5_clv'] > 0).sum()} / {results_df['edge5_clv'].notna().sum()}")
    print(f"  Mean CLV across seasons: {results_df['mean_clv'].mean():+.4f}")
    print(f"  Mean 5%+ edge CLV across seasons: {results_df['edge5_clv'].mean():+.4f}")

    # ==========================================
    # 2. GAME COMPOSITION ANALYSIS (2024 test set)
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. GAME COMPOSITION (2024 test set, 5%+ edge games)")
    print(f"{'='*60}")

    # Retrain for 2024 test
    model, scaler, medians, feats = train_on_window(df, 2022, CORE_FEATURES)
    test_2024 = df[df["season"] == 2024].copy()
    X_2024 = test_2024[feats].fillna(medians)
    probs_2024 = model.predict_proba(scaler.transform(X_2024))[:, 1]

    edge_games = []
    for idx, (_, row) in enumerate(test_2024.iterrows()):
        gid = int(row["game_id"])
        if gid in closing_odds:
            market_imp = closing_odds[gid]["home_implied"]
            if pd.notna(market_imp):
                edge = probs_2024[idx] - float(market_imp)
                if abs(edge) >= 0.05:
                    edge_games.append({
                        "game_id": gid,
                        "model_prob": probs_2024[idx],
                        "market_implied": float(market_imp),
                        "edge": edge,
                        "clv": edge,
                        "home_win": int(row["home_win"]),
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "game_date": row["game_date"],
                        "month": row["game_date"].month if hasattr(row["game_date"], "month") else pd.to_datetime(row["game_date"]).month,
                        "is_postseason": row["is_postseason"],
                        "park_factor": row.get("park_factor"),
                        "home_line": closing_odds[gid].get("home_line"),
                        "away_line": closing_odds[gid].get("away_line"),
                    })

    edge_df = pd.DataFrame(edge_games)
    print(f"\n  Total 5%+ edge games: {len(edge_df)}")

    # By month
    print(f"\n  By month:")
    monthly = edge_df.groupby("month").agg(
        games=("clv", "count"),
        mean_clv=("clv", "mean"),
        win_rate=("home_win", "mean"),
    )
    for month, row in monthly.iterrows():
        print(f"    Month {int(month):2d}: {int(row['games']):3d} games, CLV={row['mean_clv']:+.4f}")

    # By side (model favors home vs away)
    print(f"\n  By model side:")
    home_fav = edge_df[edge_df["edge"] > 0]
    away_fav = edge_df[edge_df["edge"] < 0]
    if len(home_fav) > 0:
        print(f"    Model favors HOME: {len(home_fav):3d} games, CLV={home_fav['clv'].mean():+.4f}")
    if len(away_fav) > 0:
        print(f"    Model favors AWAY: {len(away_fav):3d} games, CLV={away_fav['clv'].mean():+.4f}")

    # By favorite vs underdog (does model find edge on dogs?)
    print(f"\n  By market favorite/underdog:")
    edge_df["model_side"] = np.where(edge_df["edge"] > 0, "home", "away")
    edge_df["market_fav"] = np.where(edge_df["market_implied"] > 0.5, "home", "away")
    edge_df["betting_dog"] = edge_df["model_side"] != edge_df["market_fav"]

    dogs = edge_df[edge_df["betting_dog"]]
    favs = edge_df[~edge_df["betting_dog"]]
    if len(dogs) > 0:
        print(f"    Betting UNDERDOGS: {len(dogs):3d} games, CLV={dogs['clv'].mean():+.4f}")
    if len(favs) > 0:
        print(f"    Betting FAVORITES: {len(favs):3d} games, CLV={favs['clv'].mean():+.4f}")

    # Most frequent teams
    print(f"\n  Top 10 teams appearing in 5%+ edge games:")
    all_teams = pd.concat([edge_df["home_team"], edge_df["away_team"]])
    team_counts = all_teams.value_counts().head(10)
    for team, count in team_counts.items():
        print(f"    {team:30s} {count:3d} games")

    # Postseason vs regular
    ps = edge_df[edge_df["is_postseason"] == True]
    rs = edge_df[edge_df["is_postseason"] == False]
    print(f"\n  Postseason: {len(ps)} games" + (f", CLV={ps['clv'].mean():+.4f}" if len(ps) > 0 else ""))
    print(f"  Regular:    {len(rs)} games, CLV={rs['clv'].mean():+.4f}")

    # ==========================================
    # 3. CLV BY EDGE BUCKET
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. CLV BY EDGE SIZE (2024)")
    print(f"{'='*60}")

    # Rebuild full CLV for 2024
    all_clv = []
    for idx, (_, row) in enumerate(test_2024.iterrows()):
        gid = int(row["game_id"])
        if gid in closing_odds:
            market_imp = closing_odds[gid]["home_implied"]
            if pd.notna(market_imp):
                edge = probs_2024[idx] - float(market_imp)
                all_clv.append({
                    "model_prob": probs_2024[idx],
                    "market_implied": float(market_imp),
                    "edge": edge,
                    "abs_edge": abs(edge),
                    "home_win": int(row["home_win"]),
                    "home_line": closing_odds[gid].get("home_line"),
                    "away_line": closing_odds[gid].get("away_line"),
                })

    all_clv_df = pd.DataFrame(all_clv)

    buckets = [(0, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 1.0)]
    print(f"\n  {'Edge bucket':<15s} {'Games':>6s} {'Mean CLV':>10s} {'CLV>0':>8s} {'Model acc':>10s}")
    print(f"  {'-'*55}")
    for lo, hi in buckets:
        mask = (all_clv_df["abs_edge"] >= lo) & (all_clv_df["abs_edge"] < hi)
        subset = all_clv_df[mask]
        if len(subset) > 0:
            # For accuracy: did model's favored side win?
            subset_copy = subset.copy()
            subset_copy["model_correct"] = np.where(
                subset_copy["edge"] > 0,
                subset_copy["home_win"] == 1,
                subset_copy["home_win"] == 0
            )
            acc = subset_copy["model_correct"].mean()
            print(f"  {lo:.0%}-{hi:.0%}          {len(subset):6d} {subset['edge'].mean():+10.4f} {(subset['edge']>0).mean():8.1%} {acc:10.1%}")

    # ==========================================
    # 4. SIMULATED ROI (Kelly betting)
    # ==========================================
    print(f"\n{'='*60}")
    print("  4. SIMULATED ROI (2024, LogReg)")
    print(f"{'='*60}")

    # Simulate betting on games with edge >= threshold
    for threshold in [0.02, 0.05, 0.10]:
        for kelly_frac in [0.25]:
            bankroll = 10000
            initial_bankroll = bankroll
            max_bankroll = bankroll
            max_drawdown = 0
            bets = 0
            wins = 0
            total_wagered = 0

            for _, game in all_clv_df.iterrows():
                edge = game["edge"]
                if abs(edge) < threshold:
                    continue

                # Determine which side to bet
                if edge > 0:  # bet home
                    model_prob = game["model_prob"]
                    decimal_odds = american_to_decimal(game["home_line"])
                    won = game["home_win"] == 1
                else:  # bet away
                    model_prob = 1 - game["model_prob"]
                    decimal_odds = american_to_decimal(game["away_line"])
                    won = game["home_win"] == 0

                if decimal_odds is None or decimal_odds <= 1:
                    continue

                # Kelly sizing
                b = decimal_odds - 1
                q = 1 - model_prob
                kelly = (b * model_prob - q) / b
                if kelly <= 0:
                    continue

                bet_size = bankroll * kelly * kelly_frac
                bet_size = min(bet_size, bankroll * 0.03)  # 3% max cap
                total_wagered += bet_size
                bets += 1

                if won:
                    bankroll += bet_size * b
                    wins += 1
                else:
                    bankroll -= bet_size

                max_bankroll = max(max_bankroll, bankroll)
                drawdown = (max_bankroll - bankroll) / max_bankroll
                max_drawdown = max(max_drawdown, drawdown)

            roi = (bankroll - initial_bankroll) / total_wagered * 100 if total_wagered > 0 else 0
            profit = bankroll - initial_bankroll

            print(f"\n  Edge >= {threshold:.0%}, {kelly_frac}x Kelly:")
            print(f"    Bets:          {bets}")
            print(f"    Wins:          {wins} ({wins/bets*100:.1f}%)" if bets > 0 else "")
            print(f"    Wagered:       ${total_wagered:,.0f}")
            print(f"    Final bankroll: ${bankroll:,.0f}")
            print(f"    Profit:        ${profit:+,.0f}")
            print(f"    ROI:           {roi:+.2f}%")
            print(f"    Max drawdown:  {max_drawdown:.1%}")


if __name__ == "__main__":
    main()
