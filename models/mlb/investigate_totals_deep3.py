"""
models/mlb/investigate_totals_deep3.py - Third round of totals investigation.

1. Exclude losing parks
2. Day vs night games
3. Series position / bullpen fatigue
4. Edge decay through season
5. Drawdown path analysis
6. Opening vs closing line edge
7. Compound filters
8. Edge by posted total line level

Usage:
    python -m models.mlb.investigate_totals_deep3
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

LOSING_PARKS = ["New York Mets", "St. Louis Cardinals", "Kansas City Royals",
                "Atlanta Braves", "Houston Astros", "Milwaukee Brewers"]


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


def build_all_predictions(df, total_lines, available):
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

        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue
            market = total_lines[gid]
            pred_total = preds[idx]
            market_total = market["total_line"]
            actual_total = row["total_runs"]
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
                "game_id": gid, "season": test_year,
                "game_date": row["game_date"],
                "month": pd.to_datetime(row["game_date"]).month if not hasattr(row["game_date"], "month") else row["game_date"].month,
                "home_team": row["home_team"], "away_team": row["away_team"],
                "pred_total": pred_total, "market_total": market_total,
                "actual_total": actual_total, "edge": edge, "abs_edge": abs(edge),
                "side": side, "correct": correct, "push": push,
                "decimal_odds": decimal_odds,
                "park_factor": row.get("park_factor"),
                "weather_temp": row.get("weather_temp"),
                "home_p_fip_5": row.get("home_p_fip_5"),
                "away_p_fip_5": row.get("away_p_fip_5"),
                "is_postseason": row.get("is_postseason", False),
            })

    return pd.DataFrame(all_results)


def roi_summary(subset, label):
    s = subset[~subset["push"]].copy()
    if len(s) == 0:
        print(f"    {label}: 0 bets")
        return None
    wins = s["correct"].sum()
    profit = sum(100 * (g["decimal_odds"] - 1) if g["correct"] else -100 for _, g in s.iterrows())
    roi = profit / (len(s) * 100) * 100
    print(f"    {label}: {len(s):4d} bets, {wins/len(s):.1%} win, ROI={roi:+.1f}%, P&L=${profit:+,.0f}")
    return roi


def main():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in TOTALS_FEATURES if c in df.columns]
    total_lines = get_total_lines()

    rdf = build_all_predictions(df, total_lines, available)

    # Base filter for all analyses
    base = rdf[
        (rdf["abs_edge"] >= 1.5) &
        (rdf["month"] >= 5) & (rdf["month"] <= 9) &
        (rdf["is_postseason"] == False) &
        (rdf["park_factor"] >= 1.0)
    ].copy()

    print(f"\n{'='*60}")
    print("  TOTALS DEEP INVESTIGATION — ROUND 3")
    print(f"{'='*60}")
    print(f"\n  Base strategy (May-Sept, PF≥1.0, ≥1.5 edge): {len(base)} bets")
    roi_summary(base, "Base")

    # ==========================================
    # 1. EXCLUDE LOSING PARKS
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. EXCLUDE LOSING PARKS")
    print(f"{'='*60}")

    no_losers = base[~base["home_team"].isin(LOSING_PARKS)]
    print(f"\n  Excluding: {', '.join(LOSING_PARKS)}")
    roi_summary(no_losers, "Excl. losers")

    print(f"\n  By season:")
    for yr in range(2019, 2025):
        roi_summary(no_losers[no_losers["season"] == yr], f"  {yr}")

    # ==========================================
    # 2. DAY VS NIGHT (proxy: game month + temp)
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. GAME TIME PROXY (temperature as day/night proxy)")
    print(f"{'='*60}")

    # We don't have game time, but high temp in summer = likely day game
    # Low temp in summer = likely night game
    summer = base[base["month"].isin([6, 7, 8])].copy()
    summer_temp = summer[summer["weather_temp"].notna()]

    if len(summer_temp) > 20:
        hot_games = summer_temp[summer_temp["weather_temp"] >= 85]
        cool_games = summer_temp[summer_temp["weather_temp"] < 75]
        mid_games = summer_temp[(summer_temp["weather_temp"] >= 75) & (summer_temp["weather_temp"] < 85)]

        print(f"\n  Summer games (June-Aug):")
        roi_summary(hot_games, "Hot (≥85°F, likely day)")
        roi_summary(mid_games, "Mild (75-85°F)")
        roi_summary(cool_games, "Cool (<75°F, likely night)")

    # ==========================================
    # 3. SERIES POSITION / BULLPEN FATIGUE
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. BULLPEN FATIGUE PROXY")
    print(f"{'='*60}")

    # Use home_bp_ip_3d from features as bullpen fatigue proxy
    bp_data = df[["game_id", "home_bp_ip_3d", "away_bp_ip_3d"]].copy()
    base_bp = base.merge(bp_data, on="game_id", how="left")
    base_bp["max_bp_fatigue"] = base_bp[["home_bp_ip_3d", "away_bp_ip_3d"]].max(axis=1)

    fatigued = base_bp[base_bp["max_bp_fatigue"] >= 10]  # heavy usage
    rested = base_bp[base_bp["max_bp_fatigue"] < 5]  # light usage
    mid_bp = base_bp[(base_bp["max_bp_fatigue"] >= 5) & (base_bp["max_bp_fatigue"] < 10)]

    print(f"\n  By max bullpen IP in last 3 days:")
    roi_summary(rested, "Rested (<5 IP)")
    roi_summary(mid_bp, "Moderate (5-10 IP)")
    roi_summary(fatigued, "Fatigued (≥10 IP)")

    # ==========================================
    # 4. EDGE DECAY THROUGH SEASON
    # ==========================================
    print(f"\n{'='*60}")
    print("  4. EDGE DECAY THROUGH SEASON")
    print(f"{'='*60}")

    print(f"\n  By month (base strategy):")
    for month in range(5, 10):
        m = base[base["month"] == month]
        roi_summary(m, f"Month {month}")

    # First half vs second half of each season
    print(f"\n  First half (May-Jun) vs Second half (Jul-Sept):")
    first_half = base[base["month"].isin([5, 6])]
    second_half = base[base["month"].isin([7, 8, 9])]
    roi_summary(first_half, "May-Jun")
    roi_summary(second_half, "Jul-Sept")

    # ==========================================
    # 5. DRAWDOWN PATH
    # ==========================================
    print(f"\n{'='*60}")
    print("  5. DRAWDOWN PATH (base strategy, chronological)")
    print(f"{'='*60}")

    chrono = base.sort_values("game_date").reset_index(drop=True)
    chrono = chrono[~chrono["push"]].reset_index(drop=True)

    bankroll = 10000
    peak = bankroll
    max_dd = 0
    streak = 0
    max_lose_streak = 0
    max_win_streak = 0
    cur_streak_type = None

    pnl_path = []
    for _, g in chrono.iterrows():
        if g["correct"]:
            bankroll += 100 * (g["decimal_odds"] - 1)
            if cur_streak_type == "W":
                streak += 1
            else:
                cur_streak_type = "W"
                streak = 1
            max_win_streak = max(max_win_streak, streak)
        else:
            bankroll -= 100
            if cur_streak_type == "L":
                streak += 1
            else:
                cur_streak_type = "L"
                streak = 1
            max_lose_streak = max(max_lose_streak, streak)

        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)
        pnl_path.append({"date": g["game_date"], "bankroll": bankroll, "drawdown": dd})

    pnl_df = pd.DataFrame(pnl_path)
    print(f"\n  Starting bankroll: $10,000")
    print(f"  Final bankroll:    ${bankroll:,.0f}")
    print(f"  Max drawdown:      {max_dd:.1%}")
    print(f"  Max losing streak: {max_lose_streak}")
    print(f"  Max winning streak: {max_win_streak}")

    # Quarterly drawdown
    print(f"\n  Bankroll checkpoints:")
    for i in range(0, len(pnl_df), max(1, len(pnl_df) // 8)):
        r = pnl_df.iloc[i]
        print(f"    Bet #{i+1:4d} ({str(r['date'])[:10]}): ${r['bankroll']:,.0f} (DD={r['drawdown']:.1%})")
    print(f"    Bet #{len(pnl_df):4d} ({str(pnl_df.iloc[-1]['date'])[:10]}): ${pnl_df.iloc[-1]['bankroll']:,.0f}")

    # ==========================================
    # 6. OPENING VS CLOSING LINE
    # ==========================================
    print(f"\n{'='*60}")
    print("  6. LINE LEVEL ANALYSIS")
    print(f"{'='*60}")

    # We don't have opening lines separately, but we can check
    # if edge correlates with how far the line is from round numbers
    base_lines = base.copy()
    base_lines["line_is_half"] = (base_lines["market_total"] % 1 == 0.5)
    base_lines["line_is_whole"] = (base_lines["market_total"] % 1 == 0)

    half = base_lines[base_lines["line_is_half"]]
    whole = base_lines[base_lines["line_is_whole"]]
    print(f"\n  Half-point lines (e.g., 8.5): {len(half)} bets")
    roi_summary(half, "Half lines")
    print(f"  Whole lines (e.g., 9.0): {len(whole)} bets")
    roi_summary(whole, "Whole lines")

    # ==========================================
    # 7. COMPOUND FILTERS
    # ==========================================
    print(f"\n{'='*60}")
    print("  7. COMPOUND FILTERS")
    print(f"{'='*60}")

    # Base + exclude losers
    f1 = base[~base["home_team"].isin(LOSING_PARKS)]
    print(f"\n  A) Base + exclude losing parks:")
    roi_summary(f1, "Excl losers")

    # Base + bad pitchers
    pitcher_data = df[["game_id", "home_p_fip_5", "away_p_fip_5"]].copy()
    base_p = base.merge(pitcher_data.drop_duplicates("game_id"), on="game_id", how="left", suffixes=("", "_feat"))

    bad_pitchers = base_p[
        (base_p["home_p_fip_5_feat"].fillna(4) > 4.5) | (base_p["away_p_fip_5_feat"].fillna(4) > 4.5)
    ]
    print(f"\n  B) Base + at least 1 bad starter (FIP > 4.5):")
    roi_summary(bad_pitchers, "Bad pitcher")

    # Base + exclude losers + bad pitchers
    f3 = bad_pitchers[~bad_pitchers["home_team"].isin(LOSING_PARKS)]
    print(f"\n  C) Base + excl losers + bad pitcher:")
    roi_summary(f3, "Combined")

    # ==========================================
    # 8. EDGE BY POSTED TOTAL LINE LEVEL
    # ==========================================
    print(f"\n{'='*60}")
    print("  8. EDGE BY POSTED TOTAL LINE")
    print(f"{'='*60}")

    base["line_bucket"] = pd.cut(
        base["market_total"],
        bins=[5, 7.5, 8.5, 9.5, 10.5, 15],
        labels=["≤7.5", "7.5-8.5", "8.5-9.5", "9.5-10.5", "10.5+"]
    )

    print(f"\n  By market total line:")
    for bucket in ["≤7.5", "7.5-8.5", "8.5-9.5", "9.5-10.5", "10.5+"]:
        b = base[base["line_bucket"] == bucket]
        if len(b) > 10:
            pct_over = (b["side"] == "over").mean()
            roi_summary(b, f"{bucket} ({pct_over:.0%} overs)")

    # ==========================================
    # ULTIMATE STRATEGY
    # ==========================================
    print(f"\n{'='*60}")
    print("  ULTIMATE STRATEGY COMPARISON")
    print(f"{'='*60}")

    strategies = {
        "Base (May-Sept, PF≥1.0, ≥1.5)": base,
        "Excl losing parks": base[~base["home_team"].isin(LOSING_PARKS)],
        "Hitter parks only (PF≥1.05)": rdf[
            (rdf["abs_edge"] >= 1.5) & (rdf["month"] >= 5) & (rdf["month"] <= 9) &
            (rdf["is_postseason"] == False) & (rdf["park_factor"] >= 1.05)
        ],
        "≥2.0 run edge (base)": base[base["abs_edge"] >= 2.0],
    }

    print()
    for name, s in strategies.items():
        roi_summary(s, name)


if __name__ == "__main__":
    main()
