"""
models/mlb/investigate_totals_deep2.py - Second round of totals investigation.

1. Backtest refined strategy (filtered)
2. Run environment adjustment
3. Individual park analysis
4. Weather deep dive
5. Pitcher matchup interactions
6. Line movement analysis

Usage:
    python -m models.mlb.investigate_totals_deep2
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
            best[gid] = {
                "total_line": float(r["total_line"]),
                "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                "sportsbook": book,
            }
        else:
            cur_p = book_priority.index(best[gid]["sportsbook"]) if best[gid]["sportsbook"] in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {
                    "total_line": float(r["total_line"]),
                    "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                    "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                    "sportsbook": book,
                }
    return best


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def build_all_predictions(df, total_lines, available, adjust_environment=False):
    """Build predictions for all seasons with expanding window."""
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

        # Run environment adjustment
        if adjust_environment:
            # Use first 30 days of the test season to calibrate
            test_sorted = test.sort_values("game_date").reset_index(drop=True)
            if len(test_sorted) > 100:
                early = test_sorted.head(100)
                early_actual_mean = early["total_runs"].mean()
                early_pred_indices = test.index.isin(early.index)
                early_pred_mean = preds[test.index.get_indexer(early.index)].mean() if sum(early_pred_indices) > 0 else preds[:100].mean()
                adjustment = early_actual_mean - early_pred_mean
                preds = preds + adjustment

        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue

            market = total_lines[gid]
            pred_total = preds[idx]
            actual_total = row["total_runs"]
            market_total = market["total_line"]
            edge = pred_total - market_total
            side = "over" if edge > 0 else "under"

            if side == "over":
                correct = actual_total > market_total
                push = actual_total == market_total
                decimal_odds = american_to_decimal(market["over_odds"])
            else:
                correct = actual_total < market_total
                push = actual_total == market_total
                decimal_odds = american_to_decimal(market["under_odds"])

            all_results.append({
                "game_id": gid,
                "season": test_year,
                "game_date": row["game_date"],
                "month": pd.to_datetime(row["game_date"]).month if not hasattr(row["game_date"], "month") else row["game_date"].month,
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
                "decimal_odds": decimal_odds,
                "park_factor": row.get("park_factor"),
                "weather_temp": row.get("weather_temp"),
                "weather_wind": row.get("weather_wind"),
                "home_p_fip_5": row.get("home_p_fip_5"),
                "away_p_fip_5": row.get("away_p_fip_5"),
                "is_postseason": row.get("is_postseason", False),
            })

    return pd.DataFrame(all_results)


def simulate_roi(rdf, label, threshold=1.5):
    """Simulate flat $100 bets on a subset."""
    subset = rdf[(rdf["abs_edge"] >= threshold) & (~rdf["push"])].copy()
    if len(subset) == 0:
        print(f"    {label}: 0 bets")
        return

    bets = len(subset)
    wins = subset["correct"].sum()
    # Flat $100
    profit = 0
    for _, g in subset.iterrows():
        if g["correct"]:
            profit += 100 * (g["decimal_odds"] - 1)
        else:
            profit -= 100

    roi = profit / (bets * 100) * 100
    print(f"    {label}: {bets:4d} bets, {wins/bets:.1%} win, ROI={roi:+.1f}%, P&L=${profit:+,.0f}")
    return roi


def main():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in TOTALS_FEATURES if c in df.columns]
    total_lines = get_total_lines()

    print(f"\n{'='*60}")
    print("  TOTALS DEEP INVESTIGATION — ROUND 2")
    print(f"{'='*60}")

    rdf = build_all_predictions(df, total_lines, available, adjust_environment=False)
    rdf_adj = build_all_predictions(df, total_lines, available, adjust_environment=True)

    # ==========================================
    # 1. BACKTEST REFINED STRATEGY
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. REFINED STRATEGY BACKTEST")
    print(f"{'='*60}")

    print(f"\n  Baseline (all games, ≥1.5 run edge):")
    simulate_roi(rdf, "All games")

    # Apply filters
    filtered = rdf[
        (rdf["month"] >= 5) & (rdf["month"] <= 9) &  # May-Sept only
        (rdf["is_postseason"] == False)  # No postseason
    ].copy()
    print(f"\n  Filter: May-Sept, no postseason:")
    simulate_roi(filtered, "May-Sept regular")

    # Add park filter
    hitter_parks = filtered[filtered["park_factor"] >= 1.0]
    print(f"\n  Filter: May-Sept, no postseason, park factor ≥ 1.0:")
    simulate_roi(hitter_parks, "Hitter-neutral parks")

    very_hitter = filtered[filtered["park_factor"] >= 1.05]
    print(f"\n  Filter: May-Sept, no postseason, park factor ≥ 1.05:")
    simulate_roi(very_hitter, "Hitter parks only")

    # By season for best filter
    print(f"\n  Best filter (May-Sept, ≥1.0 PF) by season:")
    for yr in range(2019, 2025):
        yr_data = hitter_parks[hitter_parks["season"] == yr]
        simulate_roi(yr_data, f"  {yr}")

    # ==========================================
    # 2. RUN ENVIRONMENT ADJUSTMENT
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. RUN ENVIRONMENT ADJUSTMENT")
    print(f"{'='*60}")

    print(f"\n  Without adjustment:")
    simulate_roi(rdf, "Original")

    print(f"\n  With environment adjustment (first 100 games calibration):")
    simulate_roi(rdf_adj, "Adjusted")

    print(f"\n  Adjusted by season:")
    for yr in range(2019, 2025):
        yr_orig = rdf[rdf["season"] == yr]
        yr_adj = rdf_adj[rdf_adj["season"] == yr]
        orig_roi = simulate_roi(yr_orig, f"  {yr} orig")
        adj_roi = simulate_roi(yr_adj, f"  {yr} adj ")

    # ==========================================
    # 3. INDIVIDUAL PARK ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. INDIVIDUAL PARK ANALYSIS (≥1.5 run edge)")
    print(f"{'='*60}")

    edge_games = rdf[rdf["abs_edge"] >= 1.5].copy()

    # Group by home team (proxy for park)
    park_stats = edge_games.groupby("home_team").agg(
        games=("correct", "count"),
        correct_pct=("correct", "mean"),
        avg_pf=("park_factor", "mean"),
        avg_edge=("abs_edge", "mean"),
    ).sort_values("correct_pct", ascending=False)

    print(f"\n  {'Park (home team)':<30s} {'Games':>6s} {'Correct':>8s} {'PF':>6s}")
    print(f"  {'-'*55}")
    for team, r in park_stats.iterrows():
        if r["games"] >= 15:
            print(f"  {team:<30s} {int(r['games']):>6d} {r['correct_pct']:>7.1%} {r['avg_pf']:>6.3f}")

    # ==========================================
    # 4. WEATHER DEEP DIVE
    # ==========================================
    print(f"\n{'='*60}")
    print("  4. WEATHER DEEP DIVE (≥1.5 run edge)")
    print(f"{'='*60}")

    # Temperature
    temp_games = edge_games[edge_games["weather_temp"].notna()].copy()
    temp_games["temp_bucket"] = pd.cut(
        temp_games["weather_temp"],
        bins=[30, 55, 65, 75, 85, 110],
        labels=["<55", "55-65", "65-75", "75-85", "85+"]
    )

    print(f"\n  By temperature:")
    for bucket in ["<55", "55-65", "65-75", "75-85", "85+"]:
        b = temp_games[temp_games["temp_bucket"] == bucket]
        if len(b) > 10:
            overs = b[b["side"] == "over"]
            print(f"    {bucket:8s}: {len(b):4d} games, correct={b['correct'].mean():.1%}, "
                  f"overs={len(overs)} ({overs['correct'].mean():.1%} correct)" if len(overs) > 0 else "")

    # Wind
    wind_games = edge_games[edge_games["weather_wind"].notna()].copy()
    wind_games["wind_bucket"] = pd.cut(
        wind_games["weather_wind"],
        bins=[-1, 5, 10, 15, 50],
        labels=["calm", "light", "moderate", "strong"]
    )

    print(f"\n  By wind speed:")
    for bucket in ["calm", "light", "moderate", "strong"]:
        b = wind_games[wind_games["wind_bucket"] == bucket]
        if len(b) > 10:
            print(f"    {bucket:10s}: {len(b):4d} games, correct={b['correct'].mean():.1%}")

    # ==========================================
    # 5. PITCHER MATCHUP INTERACTIONS
    # ==========================================
    print(f"\n{'='*60}")
    print("  5. PITCHER MATCHUP INTERACTIONS (≥1.5 run edge)")
    print(f"{'='*60}")

    pitcher_games = edge_games[
        edge_games["home_p_fip_5"].notna() & edge_games["away_p_fip_5"].notna()
    ].copy()

    pitcher_games["avg_fip"] = (pitcher_games["home_p_fip_5"] + pitcher_games["away_p_fip_5"]) / 2
    pitcher_games["fip_bucket"] = pd.cut(
        pitcher_games["avg_fip"],
        bins=[0, 3.5, 4.0, 4.5, 5.0, 20],
        labels=["elite (<3.5)", "good (3.5-4)", "avg (4-4.5)", "below (4.5-5)", "bad (5+)"]
    )

    print(f"\n  By average starter FIP:")
    for bucket in ["elite (<3.5)", "good (3.5-4)", "avg (4-4.5)", "below (4.5-5)", "bad (5+)"]:
        b = pitcher_games[pitcher_games["fip_bucket"] == bucket]
        if len(b) > 10:
            pct_over = (b["side"] == "over").mean()
            print(f"    {bucket:18s}: {len(b):4d} games, correct={b['correct'].mean():.1%}, "
                  f"{pct_over:.0%} overs")

    # Both starters bad (high total expected)
    both_bad = pitcher_games[
        (pitcher_games["home_p_fip_5"] > 4.5) & (pitcher_games["away_p_fip_5"] > 4.5)
    ]
    both_good = pitcher_games[
        (pitcher_games["home_p_fip_5"] < 3.5) & (pitcher_games["away_p_fip_5"] < 3.5)
    ]
    print(f"\n    Both starters FIP > 4.5: {len(both_bad):3d} games, "
          f"correct={both_bad['correct'].mean():.1%}" if len(both_bad) > 5 else "")
    print(f"    Both starters FIP < 3.5: {len(both_good):3d} games, "
          f"correct={both_good['correct'].mean():.1%}" if len(both_good) > 5 else "")

    # ==========================================
    # 6. LINE MOVEMENT ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("  6. LINE MOVEMENT ANALYSIS")
    print(f"{'='*60}")

    # Check if we have multiple total lines per game (different sportsbooks = proxy for line variation)
    multi_lines = query("""
        SELECT game_id,
               MIN(total_line) as min_line,
               MAX(total_line) as max_line,
               AVG(total_line) as avg_line,
               COUNT(DISTINCT total_line) as n_lines
        FROM odds
        WHERE market = 'total' AND total_line IS NOT NULL AND is_closing = true
        GROUP BY game_id
        HAVING COUNT(DISTINCT total_line) > 1
    """)

    if len(multi_lines) > 0:
        print(f"\n  Games with line variation across books: {len(multi_lines)}")
        print(f"  Avg spread between min/max line: {(multi_lines['max_line'] - multi_lines['min_line']).mean():.2f} runs")

        # Merge with our predictions
        multi_lines["game_id"] = multi_lines["game_id"].astype(int)
        edge_with_spread = edge_games.merge(multi_lines, on="game_id", how="inner")

        if len(edge_with_spread) > 0:
            edge_with_spread["line_spread"] = edge_with_spread["max_line"] - edge_with_spread["min_line"]
            edge_with_spread["tight_market"] = edge_with_spread["line_spread"] <= 0.5
            edge_with_spread["wide_market"] = edge_with_spread["line_spread"] > 0.5

            tight = edge_with_spread[edge_with_spread["tight_market"]]
            wide = edge_with_spread[edge_with_spread["wide_market"]]

            print(f"\n  Tight market (books agree within 0.5): {len(tight)} games, correct={tight['correct'].mean():.1%}")
            print(f"  Wide market (books disagree > 0.5):    {len(wide)} games, correct={wide['correct'].mean():.1%}")
    else:
        print(f"\n  No multi-line data available for movement analysis")

    # ==========================================
    # FINAL: OPTIMAL STRATEGY
    # ==========================================
    print(f"\n{'='*60}")
    print("  OPTIMAL STRATEGY BACKTEST")
    print(f"{'='*60}")

    # Apply all best filters
    optimal = rdf[
        (rdf["abs_edge"] >= 1.5) &
        (rdf["month"] >= 5) & (rdf["month"] <= 9) &
        (rdf["is_postseason"] == False) &
        (rdf["park_factor"] >= 1.0) &
        (~rdf["push"])
    ].copy()

    print(f"\n  Filters: ≥1.5 edge, May-Sept, no postseason, PF ≥ 1.0")
    print(f"  Total qualifying bets: {len(optimal)}")

    if len(optimal) > 0:
        total_bets = len(optimal)
        wins = optimal["correct"].sum()
        profit = 0
        for _, g in optimal.iterrows():
            if g["correct"]:
                profit += 100 * (g["decimal_odds"] - 1)
            else:
                profit -= 100

        roi = profit / (total_bets * 100) * 100
        print(f"  Win rate: {wins/total_bets:.1%}")
        print(f"  ROI: {roi:+.2f}%")
        print(f"  P&L: ${profit:+,.0f}")
        print(f"  Avg bets per season: {total_bets / 6:.0f}")

        # By season
        print(f"\n  By season:")
        for yr in range(2019, 2025):
            yr_data = optimal[optimal["season"] == yr]
            if len(yr_data) > 0:
                yr_wins = yr_data["correct"].sum()
                yr_profit = 0
                for _, g in yr_data.iterrows():
                    if g["correct"]:
                        yr_profit += 100 * (g["decimal_odds"] - 1)
                    else:
                        yr_profit -= 100
                yr_roi = yr_profit / (len(yr_data) * 100) * 100
                print(f"    {yr}: {len(yr_data):3d} bets, {yr_wins/len(yr_data):.1%} win, "
                      f"ROI={yr_roi:+.1f}%, P&L=${yr_profit:+,.0f}")


if __name__ == "__main__":
    main()
