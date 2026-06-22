"""
models/mlb/investigate_totals.py - Deep investigation of totals edge signal.

Questions:
1. Is 88.5% correct at 2+ run edge stable across seasons?
2. What types of games trigger high edge? (parks, weather, pitchers)
3. Simulated ROI at various sizing strategies
4. Does the edge survive vig?

Usage:
    python -m models.mlb.investigate_totals
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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
    """Get best available total line per game."""
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
        return 1.909  # default -110
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def main():
    print(f"\n{'='*60}")
    print("  TOTALS EDGE INVESTIGATION")
    print(f"{'='*60}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    total_lines = get_total_lines()
    available = [c for c in TOTALS_FEATURES if c in df.columns]

    # ==========================================
    # 1. CROSS-SEASON STABILITY
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. CROSS-SEASON STABILITY")
    print(f"{'='*60}")

    season_results = []

    for test_year in range(2019, 2025):
        train_end = test_year - 1
        train = df[df["season"].between(2016, train_end)].copy()
        test = df[df["season"] == test_year].copy()

        X_train = train[available].fillna(train[available].median())
        y_train = train["total_runs"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_s, y_train)

        X_test = test[available].fillna(train[available].median())
        X_test_s = scaler.transform(X_test)
        preds = model.predict(X_test_s)

        # Match with total lines
        results = []
        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue

            market = total_lines[gid]
            pred_total = preds[idx]
            actual_total = row["total_runs"]
            market_total = market["total_line"]
            edge = pred_total - market_total

            results.append({
                "game_id": gid,
                "pred_total": pred_total,
                "market_total": market_total,
                "actual_total": actual_total,
                "edge": edge,
                "abs_edge": abs(edge),
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "park_factor": row.get("park_factor"),
                "weather_temp": row.get("weather_temp"),
                "weather_wind": row.get("weather_wind"),
                "over_odds": market["over_odds"],
                "under_odds": market["under_odds"],
            })

        if not results:
            continue

        rdf = pd.DataFrame(results)

        # Correct side analysis
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            subset = rdf[rdf["abs_edge"] >= threshold]
            if len(subset) > 5:
                correct = 0
                for _, g in subset.iterrows():
                    if g["edge"] > 0 and g["actual_total"] > g["market_total"]:
                        correct += 1
                    elif g["edge"] < 0 and g["actual_total"] < g["market_total"]:
                        correct += 1
                pct = correct / len(subset)
                season_results.append({
                    "season": test_year,
                    "threshold": threshold,
                    "games": len(subset),
                    "correct": correct,
                    "pct": pct,
                })

    sr = pd.DataFrame(season_results)

    print(f"\n  Correct side % by season and edge threshold:\n")
    print(f"  {'Season':<8s}", end="")
    for t in [0.5, 1.0, 1.5, 2.0]:
        print(f"  {'≥'+str(t)+'r':>12s}", end="")
    print()
    print(f"  {'-'*60}")

    for year in range(2019, 2025):
        print(f"  {year:<8d}", end="")
        for t in [0.5, 1.0, 1.5, 2.0]:
            row = sr[(sr["season"] == year) & (sr["threshold"] == t)]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {r['pct']:.1%} ({int(r['games']):>3d})", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()

    # Cross-season average
    print(f"  {'Avg':<8s}", end="")
    for t in [0.5, 1.0, 1.5, 2.0]:
        subset = sr[sr["threshold"] == t]
        if len(subset) > 0:
            avg_pct = subset["pct"].mean()
            avg_games = subset["games"].mean()
            print(f"  {avg_pct:.1%} ({avg_games:>3.0f})", end="")
    print()

    # ==========================================
    # 2. COMPOSITION ANALYSIS (2024)
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. COMPOSITION ANALYSIS (high-edge games, 2024)")
    print(f"{'='*60}")

    # Retrain for 2024
    train = df[df["season"].between(2016, 2022)].copy()
    test_2024 = df[df["season"] == 2024].copy()
    X_train = train[available].fillna(train[available].median())
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_s, train["total_runs"])

    X_test = test_2024[available].fillna(train[available].median())
    preds_2024 = model.predict(scaler.transform(X_test))

    edge_games = []
    for idx, (_, row) in enumerate(test_2024.iterrows()):
        gid = int(row["game_id"])
        if gid not in total_lines:
            continue
        market = total_lines[gid]
        edge = preds_2024[idx] - market["total_line"]
        if abs(edge) >= 1.0:
            correct = (edge > 0 and row["total_runs"] > market["total_line"]) or \
                      (edge < 0 and row["total_runs"] < market["total_line"])
            edge_games.append({
                "edge": edge,
                "abs_edge": abs(edge),
                "correct": correct,
                "pred": preds_2024[idx],
                "market": market["total_line"],
                "actual": row["total_runs"],
                "park_factor": row.get("park_factor"),
                "weather_temp": row.get("weather_temp"),
                "weather_wind": row.get("weather_wind"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "month": row["game_date"].month if hasattr(row["game_date"], "month") else pd.to_datetime(row["game_date"]).month,
                "side": "over" if edge > 0 else "under",
            })

    edf = pd.DataFrame(edge_games)
    print(f"\n  Total 1+ run edge games: {len(edf)}")

    # By side (over vs under)
    print(f"\n  By side:")
    for side in ["over", "under"]:
        s = edf[edf["side"] == side]
        print(f"    {side:6s}: {len(s):3d} games, correct={s['correct'].mean():.1%}")

    # By month
    print(f"\n  By month:")
    for month in sorted(edf["month"].unique()):
        m = edf[edf["month"] == month]
        print(f"    Month {int(month):2d}: {len(m):3d} games, correct={m['correct'].mean():.1%}")

    # By park factor
    print(f"\n  By park factor:")
    edf["pf_bucket"] = pd.cut(edf["park_factor"], bins=[0.8, 0.95, 1.05, 1.3], labels=["pitcher", "neutral", "hitter"])
    for bucket in ["pitcher", "neutral", "hitter"]:
        b = edf[edf["pf_bucket"] == bucket]
        if len(b) > 5:
            print(f"    {bucket:10s}: {len(b):3d} games, correct={b['correct'].mean():.1%}")

    # By weather
    temps = edf["weather_temp"].dropna()
    if len(temps) > 10:
        print(f"\n  By temperature:")
        edf["temp_bucket"] = pd.cut(edf["weather_temp"], bins=[30, 60, 75, 110], labels=["cold", "mild", "hot"])
        for bucket in ["cold", "mild", "hot"]:
            b = edf[edf["temp_bucket"] == bucket]
            if len(b) > 5:
                print(f"    {bucket:10s}: {len(b):3d} games, correct={b['correct'].mean():.1%}")

    # Top teams appearing
    print(f"\n  Top teams in 1+ run edge games:")
    all_teams = pd.concat([edf["home_team"], edf["away_team"]])
    for team, count in all_teams.value_counts().head(8).items():
        print(f"    {team:30s} {count:3d}")

    # ==========================================
    # 3. SIMULATED ROI
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. SIMULATED ROI (all seasons, expanding window)")
    print(f"{'='*60}")

    for strategy_name, threshold, sizing in [
        ("Flat $100, 1+ run edge", 1.0, "flat"),
        ("Flat $100, 1.5+ run edge", 1.5, "flat"),
        ("Flat $100, 2+ run edge", 2.0, "flat"),
        ("Quarter-Kelly, 1+ run edge", 1.0, "kelly"),
        ("Quarter-Kelly, 1.5+ run edge", 1.5, "kelly"),
        ("Quarter-Kelly, 2+ run edge", 2.0, "kelly"),
    ]:
        bankroll = 10000
        initial = bankroll
        max_bankroll = bankroll
        max_drawdown = 0
        total_bets = 0
        total_wins = 0
        total_wagered = 0
        season_pnl = {}

        for test_year in range(2019, 2025):
            train_end = test_year - 1
            tr = df[df["season"].between(2016, train_end)].copy()
            te = df[df["season"] == test_year].copy()

            X_tr = tr[available].fillna(tr[available].median())
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            mdl = LinearRegression()
            mdl.fit(X_tr_s, tr["total_runs"])

            X_te = te[available].fillna(tr[available].median())
            pr = mdl.predict(sc.transform(X_te))

            year_pnl = 0
            year_bets = 0

            for idx, (_, row) in enumerate(te.iterrows()):
                gid = int(row["game_id"])
                if gid not in total_lines:
                    continue

                market = total_lines[gid]
                edge = pr[idx] - market["total_line"]

                if abs(edge) < threshold:
                    continue

                # Determine side and odds
                if edge > 0:  # bet over
                    decimal_odds = american_to_decimal(market["over_odds"])
                    won = row["total_runs"] > market["total_line"]
                    push = row["total_runs"] == market["total_line"]
                else:  # bet under
                    decimal_odds = american_to_decimal(market["under_odds"])
                    won = row["total_runs"] < market["total_line"]
                    push = row["total_runs"] == market["total_line"]

                if push:
                    continue

                # Sizing
                if sizing == "flat":
                    bet_size = 100
                else:  # kelly
                    # Estimate win prob from historical correct rate at this edge level
                    est_prob = min(0.5 + abs(edge) * 0.1, 0.85)  # rough heuristic
                    b = decimal_odds - 1
                    kelly = (b * est_prob - (1 - est_prob)) / b
                    if kelly <= 0:
                        continue
                    bet_size = bankroll * kelly * 0.25
                    bet_size = min(bet_size, bankroll * 0.03)

                total_wagered += bet_size
                total_bets += 1
                year_bets += 1

                if won:
                    profit = bet_size * (decimal_odds - 1)
                    bankroll += profit
                    year_pnl += profit
                    total_wins += 1
                else:
                    bankroll -= bet_size
                    year_pnl -= bet_size

                max_bankroll = max(max_bankroll, bankroll)
                dd = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
                max_drawdown = max(max_drawdown, dd)

            season_pnl[test_year] = year_pnl

        roi = (bankroll - initial) / total_wagered * 100 if total_wagered > 0 else 0
        profit = bankroll - initial

        print(f"\n  {strategy_name}:")
        print(f"    Total bets:    {total_bets}")
        print(f"    Win rate:      {total_wins/total_bets:.1%}" if total_bets > 0 else "")
        print(f"    Total wagered: ${total_wagered:,.0f}")
        print(f"    Final bankroll: ${bankroll:,.0f}")
        print(f"    Profit:        ${profit:+,.0f}")
        print(f"    ROI:           {roi:+.2f}%")
        print(f"    Max drawdown:  {max_drawdown:.1%}")
        print(f"    By season: ", end="")
        for yr, pnl in sorted(season_pnl.items()):
            print(f"{yr}=${pnl:+.0f}  ", end="")
        print()


if __name__ == "__main__":
    main()
