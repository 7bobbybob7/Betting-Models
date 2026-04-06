"""
models/mlb/totals_line_shopping.py - Rerun all analysis with line shopping.

Instead of betting at one book's odds, find the best available odds
across all sportsbooks for each bet. This is what real bettors do.

Usage:
    python -m models.mlb.totals_line_shopping
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
LINE_FEATURES = ["market_line", "rpg_vs_line", "home_rpg_vs_half_line", "away_rpg_vs_half_line"]

REAL_BOOKS = ['bet365', 'draftkings', 'fanduel', 'betmgm', 'caesars', 'espn_bet',
              'bovada.lv', 'betonline.ag', 'pinnaclesports.com', 'westgate',
              'pointsbet', 'betrivers', 'unibet', 'sugarhouse', 'mgm']


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    return 1 + american / 100 if american >= 0 else 1 + 100 / abs(american)


def implied_from_odds(over_odds, under_odds):
    def to_imp(odds):
        return 100 / (odds + 100) if odds >= 0 else abs(odds) / (abs(odds) + 100)
    o, u = to_imp(over_odds), to_imp(under_odds)
    return o / (o + u)


def get_line_shopping_odds():
    """Get best available odds per game per side from all real sportsbooks."""
    book_list = ",".join([f"'{b}'" for b in REAL_BOOKS])
    odds = query(f"""
        SELECT o.game_id, o.sportsbook, o.total_line, o.over_odds, o.under_odds
        FROM odds o
        WHERE o.market = 'total' AND o.total_line IS NOT NULL AND o.is_closing = true
          AND o.sportsbook IN ({book_list})
          AND o.over_odds IS NOT NULL AND o.under_odds IS NOT NULL
    """)

    # For each game: best over odds, best under odds, consensus line
    result = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        if gid not in result:
            result[gid] = {
                "total_line": float(r["total_line"]),
                "best_over": float(r["over_odds"]),
                "best_under": float(r["under_odds"]),
                "worst_over": float(r["over_odds"]),
                "worst_under": float(r["under_odds"]),
                "all_over": [float(r["over_odds"])],
                "all_under": [float(r["under_odds"])],
                "n_books": 1,
            }
        else:
            result[gid]["all_over"].append(float(r["over_odds"]))
            result[gid]["all_under"].append(float(r["under_odds"]))
            # Best = highest (least negative or most positive)
            if float(r["over_odds"]) > result[gid]["best_over"]:
                result[gid]["best_over"] = float(r["over_odds"])
            if float(r["under_odds"]) > result[gid]["best_under"]:
                result[gid]["best_under"] = float(r["under_odds"])
            if float(r["over_odds"]) < result[gid]["worst_over"]:
                result[gid]["worst_over"] = float(r["over_odds"])
            if float(r["under_odds"]) < result[gid]["worst_under"]:
                result[gid]["worst_under"] = float(r["under_odds"])
            result[gid]["n_books"] += 1

    # Compute median odds (what a single-book bettor gets)
    for gid in result:
        result[gid]["median_over"] = np.median(result[gid]["all_over"])
        result[gid]["median_under"] = np.median(result[gid]["all_under"])

    return result


def build_dataset(df, line_data):
    rows = []
    for _, row in df.iterrows():
        gid = int(row["game_id"])
        if gid not in line_data:
            continue
        market = line_data[gid]
        actual = row["total_runs"]
        if actual == market["total_line"]:
            continue

        feat = {c: row[c] for c in FEATURES if c in row.index}
        home_rpg = row.get("home_b_rpg_15", 0) or 0
        away_rpg = row.get("away_b_rpg_15", 0) or 0
        feat["market_line"] = market["total_line"]
        feat["rpg_vs_line"] = home_rpg + away_rpg - market["total_line"]
        feat["home_rpg_vs_half_line"] = home_rpg - market["total_line"] / 2
        feat["away_rpg_vs_half_line"] = away_rpg - market["total_line"] / 2
        feat["target"] = 1 if actual > market["total_line"] else 0
        feat["game_id"] = gid
        feat["season"] = int(row["season"])
        feat["best_over"] = market["best_over"]
        feat["best_under"] = market["best_under"]
        feat["median_over"] = market["median_over"]
        feat["median_under"] = market["median_under"]
        feat["worst_over"] = market["worst_over"]
        feat["worst_under"] = market["worst_under"]
        feat["n_books"] = market["n_books"]
        rows.append(feat)

    return pd.DataFrame(rows).sort_values("game_id").reset_index(drop=True)


def evaluate_with_shopping(results, label, use_best=True):
    """Evaluate using best available odds (line shopping) vs median odds (single book)."""
    rdf = pd.DataFrame(results)

    print(f"\n  {label}:")
    print(f"  {'Threshold':<10s} {'Bets':>6s} {'Win%':>7s} {'Best ROI':>9s} {'Med ROI':>9s} {'Worst ROI':>10s} {'p-val':>7s}")
    print(f"  {'-'*65}")

    for t in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
        s = rdf[rdf["abs_edge"] >= t]
        if len(s) < 20:
            continue

        wins = int(s["correct"].sum())
        bets = len(s)
        win_rate = wins / bets

        # ROI with best available odds (line shopping)
        best_profit = 0
        med_profit = 0
        worst_profit = 0
        for _, g in s.iterrows():
            if g["bet_over"]:
                best_dec = american_to_decimal(g["best_over"])
                med_dec = american_to_decimal(g["median_over"])
                worst_dec = american_to_decimal(g["worst_over"])
            else:
                best_dec = american_to_decimal(g["best_under"])
                med_dec = american_to_decimal(g["median_under"])
                worst_dec = american_to_decimal(g["worst_under"])

            if g["correct"]:
                best_profit += 100 * (best_dec - 1)
                med_profit += 100 * (med_dec - 1)
                worst_profit += 100 * (worst_dec - 1)
            else:
                best_profit -= 100
                med_profit -= 100
                worst_profit -= 100

        best_roi = best_profit / (bets * 100) * 100
        med_roi = med_profit / (bets * 100) * 100
        worst_roi = worst_profit / (bets * 100) * 100
        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue
        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""

        print(f"  ≥{t:<8.1%} {bets:>6d} {win_rate:>6.1%} {best_roi:>+8.1f}% {med_roi:>+8.1f}% {worst_roi:>+9.1f}% {pval:>6.3f}{marker}")

    return rdf


def main():
    print(f"\n{'='*70}")
    print("  LINE SHOPPING ANALYSIS — COMPLETE RERUN")
    print(f"{'='*70}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    line_data = get_line_shopping_odds()
    full = build_dataset(df, line_data)
    avail = [c for c in FEATURES + LINE_FEATURES if c in full.columns]

    # Filter to multi-book games only (line shopping requires 2+ books)
    multi = full[full["n_books"] >= 2].copy()

    print(f"\n  Total games: {len(full)}")
    print(f"  Games with 2+ books: {len(multi)}")
    print(f"  Avg books per game: {multi['n_books'].mean():.1f}")

    # Show what line shopping buys us
    print(f"\n  ODDS IMPROVEMENT FROM LINE SHOPPING:")
    print(f"  {'':>20s} {'Over':>10s} {'Under':>10s}")
    print(f"  {'Best available':>20s} {multi['best_over'].mean():>+10.1f} {multi['best_under'].mean():>+10.1f}")
    print(f"  {'Median (single book)':>20s} {multi['median_over'].mean():>+10.1f} {multi['median_under'].mean():>+10.1f}")
    print(f"  {'Worst available':>20s} {multi['worst_over'].mean():>+10.1f} {multi['worst_under'].mean():>+10.1f}")

    # Best over breakeven
    avg_best = multi["best_over"].mean()
    avg_med = multi["median_over"].mean()
    def be(odds):
        if odds >= 0: return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)
    print(f"\n  Breakeven at best avg odds ({avg_best:+.0f}): {be(avg_best):.1%}")
    print(f"  Breakeven at median odds ({avg_med:+.0f}):  {be(avg_med):.1%}")
    print(f"  Breakeven at -110 (no shopping):      52.4%")

    # ==========================================
    # CLASSIFIER WITH LINE SHOPPING
    # ==========================================
    print(f"\n{'='*70}")
    print("  CLASSIFIER RESULTS: BEST vs MEDIAN vs WORST ODDS")
    print(f"{'='*70}")

    all_results = []
    for test_year in range(2019, 2025):
        train = multi[multi["season"].between(2016, test_year - 1)]
        test = multi[multi["season"] == test_year]
        if len(train) < 100 or len(test) < 20:
            continue

        X_tr = train[avail].fillna(train[avail].median())
        X_te = test[avail].fillna(train[avail].median())
        sc = StandardScaler()
        clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf.fit(sc.fit_transform(X_tr), train["target"])
        probs = clf.predict_proba(sc.transform(X_te))[:, 1]

        for i, (_, row) in enumerate(test.iterrows()):
            market_p = implied_from_odds(row["median_over"], row["median_under"])
            edge = probs[i] - market_p
            all_results.append({
                "game_id": int(row["game_id"]), "season": int(row["season"]),
                "edge": edge, "abs_edge": abs(edge),
                "bet_over": edge > 0,
                "actual_over": row["target"] == 1,
                "correct": (edge > 0) == (row["target"] == 1),
                "best_over": row["best_over"], "best_under": row["best_under"],
                "median_over": row["median_over"], "median_under": row["median_under"],
                "worst_over": row["worst_over"], "worst_under": row["worst_under"],
            })

    rdf = evaluate_with_shopping(all_results, "Direct Classifier (expanding window)")

    # ==========================================
    # BY SEASON WITH LINE SHOPPING
    # ==========================================
    print(f"\n{'='*70}")
    print("  BY SEASON AT ≥1% EDGE (line shopping vs no shopping)")
    print(f"{'='*70}")

    print(f"\n  {'Season':<8s} {'Bets':>6s} {'Win%':>7s} {'Best ROI':>9s} {'Med ROI':>9s}")
    print(f"  {'-'*45}")

    rdf_full = pd.DataFrame(all_results)
    for yr in range(2019, 2025):
        s = rdf_full[(rdf_full["season"] == yr) & (rdf_full["abs_edge"] >= 0.01)]
        if len(s) < 10:
            continue
        wins = s["correct"].sum()
        best_p = sum(100 * (american_to_decimal(g["best_over"] if g["bet_over"] else g["best_under"]) - 1)
                     if g["correct"] else -100 for _, g in s.iterrows())
        med_p = sum(100 * (american_to_decimal(g["median_over"] if g["bet_over"] else g["median_under"]) - 1)
                    if g["correct"] else -100 for _, g in s.iterrows())
        best_roi = best_p / (len(s) * 100) * 100
        med_roi = med_p / (len(s) * 100) * 100
        print(f"  {yr:<8d} {len(s):>6d} {wins/len(s):>6.1%} {best_roi:>+8.1f}% {med_roi:>+8.1f}%")

    # ==========================================
    # CUMULATIVE P&L COMPARISON
    # ==========================================
    print(f"\n{'='*70}")
    print("  CUMULATIVE P&L: $100 FLAT BETS, ≥1% EDGE, ALL SEASONS")
    print(f"{'='*70}")

    s = rdf_full[rdf_full["abs_edge"] >= 0.01].copy()
    best_total = sum(100 * (american_to_decimal(g["best_over"] if g["bet_over"] else g["best_under"]) - 1)
                     if g["correct"] else -100 for _, g in s.iterrows())
    med_total = sum(100 * (american_to_decimal(g["median_over"] if g["bet_over"] else g["median_under"]) - 1)
                    if g["correct"] else -100 for _, g in s.iterrows())
    worst_total = sum(100 * (american_to_decimal(g["worst_over"] if g["bet_over"] else g["worst_under"]) - 1)
                      if g["correct"] else -100 for _, g in s.iterrows())

    bets = len(s)
    print(f"\n  Bets: {bets} ({bets/6:.0f}/year)")
    print(f"  Win rate: {s['correct'].mean():.1%}")
    print(f"\n  P&L with best odds (line shopping):  ${best_total:+,.0f} (ROI: {best_total/(bets*100)*100:+.1f}%)")
    print(f"  P&L with median odds (single book):  ${med_total:+,.0f} (ROI: {med_total/(bets*100)*100:+.1f}%)")
    print(f"  P&L with worst odds:                 ${worst_total:+,.0f} (ROI: {worst_total/(bets*100)*100:+.1f}%)")
    print(f"\n  Line shopping advantage: ${best_total - med_total:+,.0f} over {bets} bets")
    print(f"  Per bet: ${(best_total - med_total)/bets:+.2f}")


if __name__ == "__main__":
    main()
