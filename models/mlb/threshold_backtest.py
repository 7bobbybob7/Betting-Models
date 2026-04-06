"""
models/mlb/threshold_backtest.py - Backtest edge thresholds using the exact
methodology from the live pipeline (2-step de-vig framework).

Tests BOTH approaches:
1. Regression → P(over) via norm.cdf (what the live pipeline uses)
2. Direct classifier (LogReg on over/under target)

Each with:
- Multiplicative de-vig on MEDIAN book odds
- Line shopping (best available odds across books)
- 2-step: info_edge >= threshold AND model_p > breakeven at best odds

Usage:
    python -m models.mlb.threshold_backtest
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, binomtest
from collections import defaultdict

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


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    if american >= 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def implied_from_american(american):
    if american >= 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def get_odds_by_game():
    """Get ALL closing odds per game per book.
    Filter to reasonable odds only (between -300 and +300).
    """
    odds = query("""
        SELECT game_id, total_line, over_odds, under_odds, sportsbook
        FROM odds
        WHERE market = 'total' AND total_line IS NOT NULL AND is_closing = true
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND over_odds BETWEEN -300 AND 300
          AND under_odds BETWEEN -300 AND 300
    """)

    by_game = defaultdict(list)
    for _, r in odds.iterrows():
        by_game[r["game_id"]].append({
            "sportsbook": r["sportsbook"],
            "total_line": float(r["total_line"]),
            "over_odds": float(r["over_odds"]),
            "under_odds": float(r["under_odds"]),
        })
    return by_game


def run_threshold_table(rdf, label, thresholds):
    """Print threshold comparison table."""
    print(f"\n  {label}")
    print(f"  {'Thresh':<8s} {'Bets':>6s} {'B/yr':>6s} {'Win%':>6s} {'ROI@Best':>9s} {'ROI@-110':>9s} {'P&L@Best':>10s} {'p-val':>7s}")
    print(f"  {'-'*70}")

    best_roi = -999
    best_thresh = None

    for threshold in thresholds:
        subset = rdf[(rdf["info_edge"] >= threshold) & (rdf["bet_ev"] > 0)].copy()
        if len(subset) < 20:
            continue

        wins = int(subset["correct"].sum())
        bets = len(subset)
        win_rate = wins / bets
        bets_per_year = bets / 6

        profit_best = sum(100 * (g["best_odds_decimal"] - 1) if g["correct"] else -100 for _, g in subset.iterrows())
        roi_best = profit_best / (bets * 100) * 100

        profit_110 = sum(100 * (american_to_decimal(-110) - 1) if g["correct"] else -100 for _, g in subset.iterrows())
        roi_110 = profit_110 / (bets * 100) * 100

        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue
        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""
        print(f"  ≥{threshold:<5.1%} {bets:>6d} {bets_per_year:>5.0f} {win_rate:>5.1%} {roi_best:>+8.1f}% {roi_110:>+8.1f}% ${profit_best:>+9,.0f} {pval:>6.3f}{marker}")

        if roi_best > best_roi:
            best_roi = roi_best
            best_thresh = threshold

    return best_thresh, best_roi


def run_season_breakdown(rdf, threshold):
    """Print per-season breakdown."""
    subset = rdf[(rdf["info_edge"] >= threshold) & (rdf["bet_ev"] > 0)]
    for yr in range(2019, 2025):
        s = subset[subset["season"] == yr]
        if len(s) == 0:
            print(f"    {yr}: 0 bets")
            continue
        wins = int(s["correct"].sum())
        profit = sum(100 * (g["best_odds_decimal"] - 1) if g["correct"] else -100 for _, g in s.iterrows())
        roi = profit / (len(s) * 100) * 100
        avg_books = s["n_books"].mean()
        print(f"    {yr}: {len(s):4d} bets, {wins/len(s):.1%} win, ROI={roi:+.1f}%, books={avg_books:.1f}")


def main():
    print(f"\n{'='*70}")
    print("  THRESHOLD BACKTEST — REGRESSION vs CLASSIFIER")
    print(f"  Median-book de-vig | Line shopping | 2-step framework")
    print(f"{'='*70}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in FEATURES if c in df.columns]
    odds_by_game = get_odds_by_game()

    print(f"\n  Games with multi-book odds: {len(odds_by_game)}")
    print(f"  Feature data: {len(df)} games")

    # Check odds sanity
    all_over = []
    all_under = []
    for gid, books in odds_by_game.items():
        for b in books:
            all_over.append(b["over_odds"])
            all_under.append(b["under_odds"])
    print(f"  Odds range: over [{min(all_over):.0f}, {max(all_over):.0f}], "
          f"under [{min(all_under):.0f}, {max(all_under):.0f}]")
    print(f"  Median odds: over {np.median(all_over):.0f}, under {np.median(all_under):.0f}")

    thresholds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

    # ==========================================================
    # APPROACH 1: REGRESSION → P(over) via norm.cdf
    # ==========================================================
    print(f"\n{'='*70}")
    print("  APPROACH 1: REGRESSION → P(over) [matches live pipeline]")
    print(f"{'='*70}")

    for scale in [3.4, 4.5]:
        reg_results = []

        for test_year in range(2019, 2025):
            train = df[df["season"].between(2016, test_year - 1)].copy()
            test = df[df["season"] == test_year].copy()

            medians = train[available].median()
            X_train = train[available].fillna(medians)
            X_test = test[available].fillna(medians)

            sc = StandardScaler()
            reg = LinearRegression()
            reg.fit(sc.fit_transform(X_train), train["total_runs"])
            preds = reg.predict(sc.transform(X_test))

            for idx, (_, row) in enumerate(test.iterrows()):
                gid = int(row["game_id"])
                if gid not in odds_by_game:
                    continue

                books = odds_by_game[gid]
                line_counts = defaultdict(int)
                for b in books:
                    line_counts[b["total_line"]] += 1
                market_total = max(line_counts, key=line_counts.get)
                books_on_line = [b for b in books if b["total_line"] == market_total]

                actual_total = row["total_runs"]
                if actual_total == market_total:
                    continue

                pred_total = preds[idx]

                over_odds_list = [b["over_odds"] for b in books_on_line]
                under_odds_list = [b["under_odds"] for b in books_on_line]

                med_over = np.median(over_odds_list)
                med_under = np.median(under_odds_list)

                raw_over_imp = implied_from_american(med_over)
                raw_under_imp = implied_from_american(med_under)
                total_imp = raw_over_imp + raw_under_imp
                devig_over = raw_over_imp / total_imp
                devig_under = raw_under_imp / total_imp

                model_p_over = 1 - norm.cdf(market_total, loc=pred_total, scale=scale)
                model_p_under = 1 - model_p_over

                over_info_edge = model_p_over - devig_over
                under_info_edge = model_p_under - devig_under

                if over_info_edge > under_info_edge:
                    bet_side = "over"
                    info_edge = over_info_edge
                    model_win_prob = model_p_over
                else:
                    bet_side = "under"
                    info_edge = under_info_edge
                    model_win_prob = model_p_under

                if bet_side == "over":
                    best_american = max(over_odds_list)
                else:
                    best_american = max(under_odds_list)

                best_decimal = american_to_decimal(best_american)
                breakeven = 1 / best_decimal
                bet_ev = model_win_prob - breakeven

                actual_over = actual_total > market_total
                bet_over = bet_side == "over"
                correct = bet_over == actual_over

                reg_results.append({
                    "game_id": gid, "season": int(row["season"]),
                    "info_edge": info_edge, "model_win_prob": model_win_prob,
                    "breakeven": breakeven, "bet_ev": bet_ev,
                    "bet_over": bet_over, "correct": correct,
                    "best_odds_decimal": best_decimal, "n_books": len(books_on_line),
                })

        rdf_reg = pd.DataFrame(reg_results)
        run_threshold_table(rdf_reg, f"Regression → norm.cdf(scale={scale})", thresholds)

    # ==========================================================
    # APPROACH 2: DIRECT CLASSIFIER
    # ==========================================================
    print(f"\n{'='*70}")
    print("  APPROACH 2: DIRECT CLASSIFIER (LogReg on over/under)")
    print(f"{'='*70}")

    clf_results = []

    for test_year in range(2019, 2025):
        train_rows = []
        test_rows = []

        for split_df, target_list in [(df[df["season"].between(2016, test_year - 1)], train_rows),
                                       (df[df["season"] == test_year], test_rows)]:
            for _, row in split_df.iterrows():
                gid = int(row["game_id"])
                if gid not in odds_by_game:
                    continue

                books = odds_by_game[gid]
                line_counts = defaultdict(int)
                for b in books:
                    line_counts[b["total_line"]] += 1
                market_total = max(line_counts, key=line_counts.get)
                books_on_line = [b for b in books if b["total_line"] == market_total]

                actual_total = row["total_runs"]
                if actual_total == market_total:
                    continue

                feat = {c: row[c] for c in available}
                home_rpg = row.get("home_b_rpg_15", 0) or 0
                away_rpg = row.get("away_b_rpg_15", 0) or 0
                feat["market_line"] = market_total
                feat["rpg_vs_line"] = home_rpg + away_rpg - market_total
                feat["target"] = 1 if actual_total > market_total else 0
                feat["game_id"] = gid
                feat["season"] = int(row["season"])
                feat["over_odds_list"] = [b["over_odds"] for b in books_on_line]
                feat["under_odds_list"] = [b["under_odds"] for b in books_on_line]
                feat["n_books"] = len(books_on_line)
                target_list.append(feat)

        if not train_rows or not test_rows:
            continue

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)

        clf_features = available + ["market_line", "rpg_vs_line"]
        clf_avail = [c for c in clf_features if c in train_df.columns]

        X_tr = train_df[clf_avail].fillna(train_df[clf_avail].median())
        X_te = test_df[clf_avail].fillna(train_df[clf_avail].median())

        sc = StandardScaler()
        clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf.fit(sc.fit_transform(X_tr), train_df["target"])
        probs = clf.predict_proba(sc.transform(X_te))[:, 1]

        for i, (_, row) in enumerate(test_df.iterrows()):
            over_odds_list = row["over_odds_list"]
            under_odds_list = row["under_odds_list"]

            med_over = np.median(over_odds_list)
            med_under = np.median(under_odds_list)

            raw_over_imp = implied_from_american(med_over)
            raw_under_imp = implied_from_american(med_under)
            total_imp = raw_over_imp + raw_under_imp
            devig_over = raw_over_imp / total_imp
            devig_under = raw_under_imp / total_imp

            model_p_over = probs[i]
            model_p_under = 1 - model_p_over

            over_info_edge = model_p_over - devig_over
            under_info_edge = model_p_under - devig_under

            if over_info_edge > under_info_edge:
                bet_side = "over"
                info_edge = over_info_edge
                model_win_prob = model_p_over
            else:
                bet_side = "under"
                info_edge = under_info_edge
                model_win_prob = model_p_under

            if bet_side == "over":
                best_american = max(over_odds_list)
            else:
                best_american = max(under_odds_list)

            best_decimal = american_to_decimal(best_american)
            breakeven = 1 / best_decimal
            bet_ev = model_win_prob - breakeven

            actual_over = row["target"] == 1
            bet_over = bet_side == "over"
            correct = bet_over == actual_over

            clf_results.append({
                "game_id": int(row["game_id"]), "season": int(row["season"]),
                "info_edge": info_edge, "model_win_prob": model_win_prob,
                "breakeven": breakeven, "bet_ev": bet_ev,
                "bet_over": bet_over, "correct": correct,
                "best_odds_decimal": best_decimal, "n_books": int(row["n_books"]),
            })

    rdf_clf = pd.DataFrame(clf_results)
    run_threshold_table(rdf_clf, "Direct Classifier (LogReg L1)", thresholds)

    # ==========================================================
    # PER-SEASON for best approach
    # ==========================================================
    print(f"\n{'='*70}")
    print("  PER-SEASON BREAKDOWN: CLASSIFIER ≥1.0% + EV gate")
    print(f"{'='*70}")
    run_season_breakdown(rdf_clf, 0.01)

    print(f"\n  PER-SEASON BREAKDOWN: CLASSIFIER ≥0.5% + EV gate")
    run_season_breakdown(rdf_clf, 0.005)

    # ==========================================================
    # BANKROLL SIMULATION
    # ==========================================================
    print(f"\n{'='*70}")
    print(f"  BANKROLL SIM: CLASSIFIER ($10K start, $100 flat)")
    print(f"{'='*70}")

    for threshold in [0.005, 0.01, 0.02]:
        subset = rdf_clf[(rdf_clf["info_edge"] >= threshold) & (rdf_clf["bet_ev"] > 0)].sort_values(["season", "game_id"])
        if len(subset) == 0:
            continue

        bankroll = 10000
        peak = bankroll
        max_dd = 0
        max_lose = 0
        cur_lose = 0

        for _, g in subset.iterrows():
            if g["correct"]:
                bankroll += 100 * (g["best_odds_decimal"] - 1)
                cur_lose = 0
            else:
                bankroll -= 100
                cur_lose += 1
                max_lose = max(max_lose, cur_lose)
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak
            max_dd = max(max_dd, dd)

        total_profit = bankroll - 10000
        print(f"\n  ≥{threshold:.1%}: {len(subset)} bets over 6 years")
        print(f"    Final: ${bankroll:,.0f} (${total_profit:+,.0f})")
        print(f"    Max drawdown: {max_dd:.1%}")
        print(f"    Max losing streak: {max_lose}")


if __name__ == "__main__":
    main()
