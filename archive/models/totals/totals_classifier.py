"""
models/mlb/totals_classifier.py - High-volume totals betting approach.

Instead of predicting exact total and comparing to line,
directly classify: will this game go over or under the market line?

Two approaches:
1. Probability-based: regression prediction → probability via distribution
2. Direct classifier: train on "did it go over the line?" as target

Tests at high volume with thin edges to beat variance through volume.

Usage:
    python -m models.mlb.totals_classifier
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
from scipy.stats import norm, binomtest

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


def get_total_lines():
    odds = query("""
        SELECT o.game_id, o.total_line, o.over_odds, o.under_odds, o.sportsbook
        FROM odds o WHERE o.market = 'total' AND o.total_line IS NOT NULL AND o.is_closing = true
    """)
    book_priority = ["pinnaclesports.com", "bet365", "draftkings", "fanduel", "betmgm", "caesars"]
    best = {}
    for _, r in odds.iterrows():
        gid = r["game_id"]
        book = r["sportsbook"]
        if gid not in best:
            best[gid] = {
                "total_line": float(r["total_line"]),
                "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
            }
        else:
            cur_p = book_priority.index(best[gid].get("_sb", "x")) if best[gid].get("_sb", "x") in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {
                    "total_line": float(r["total_line"]),
                    "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                    "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                    "_sb": book,
                }
    return best


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def implied_from_odds(over_odds, under_odds):
    """De-vig to get true implied P(over)."""
    def to_imp(odds):
        if odds >= 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    over_imp = to_imp(over_odds)
    under_imp = to_imp(under_odds)
    total = over_imp + under_imp
    return over_imp / total  # de-vigged P(over)


def main():
    print(f"\n{'='*60}")
    print("  HIGH-VOLUME TOTALS ANALYSIS")
    print(f"{'='*60}")

    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    total_lines = get_total_lines()
    available = [c for c in FEATURES if c in df.columns]

    # ==========================================
    # APPROACH 1: Probability-based (regression → distribution)
    # ==========================================
    print(f"\n{'='*60}")
    print("  APPROACH 1: REGRESSION → PROBABILITY")
    print(f"{'='*60}")

    # Expanding window, test each year
    all_results = []

    for test_year in range(2019, 2025):
        train = df[df["season"].between(2016, test_year - 1)].copy()
        test = df[df["season"] == test_year].copy()

        medians = train[available].median()
        X_train = train[available].fillna(medians)
        X_test = test[available].fillna(medians)

        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)

        # Train regression
        reg = LinearRegression()
        reg.fit(X_train_s, train["total_runs"])
        preds = reg.predict(X_test_s)

        # Compute residual std from training data
        train_preds = reg.predict(X_train_s)
        residual_std = np.std(train["total_runs"].values - train_preds)

        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue

            market = total_lines[gid]
            market_line = market["total_line"]
            pred_total = preds[idx]
            actual_total = row["total_runs"]

            # Model's P(over) using normal distribution
            model_p_over = 1 - norm.cdf(market_line, loc=pred_total, scale=residual_std)

            # Market's implied P(over)
            market_p_over = implied_from_odds(market["over_odds"], market["under_odds"])

            # Edge
            edge = model_p_over - market_p_over

            # Actual result
            if actual_total > market_line:
                actual_over = True
            elif actual_total < market_line:
                actual_over = False
            else:
                continue  # push

            # Decide: bet over if edge > 0, under if edge < 0
            bet_over = edge > 0

            all_results.append({
                "game_id": gid,
                "season": test_year,
                "pred_total": pred_total,
                "market_line": market_line,
                "actual_total": actual_total,
                "model_p_over": model_p_over,
                "market_p_over": market_p_over,
                "edge": edge,
                "abs_edge": abs(edge),
                "bet_over": bet_over,
                "actual_over": actual_over,
                "correct": bet_over == actual_over,
                "over_odds": market["over_odds"],
                "under_odds": market["under_odds"],
            })

    rdf = pd.DataFrame(all_results)
    print(f"\n  Total games with odds: {len(rdf)}")

    # Test at various edge thresholds
    print(f"\n  {'Threshold':<12s} {'Bets':>6s} {'Bets/yr':>8s} {'Win%':>7s} {'ROI':>8s} {'P&L':>10s} {'p-value':>8s}")
    print(f"  {'-'*65}")

    for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]:
        subset = rdf[rdf["abs_edge"] >= threshold].copy()
        if len(subset) < 20:
            continue

        wins = subset["correct"].sum()
        bets = len(subset)
        win_rate = wins / bets

        # Compute ROI
        profit = 0
        for _, g in subset.iterrows():
            if g["bet_over"]:
                dec_odds = american_to_decimal(g["over_odds"])
            else:
                dec_odds = american_to_decimal(g["under_odds"])

            if g["correct"]:
                profit += 100 * (dec_odds - 1)
            else:
                profit -= 100

        roi = profit / (bets * 100) * 100
        bets_per_year = bets / 6

        # P-value vs breakeven
        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue

        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""
        print(f"  ≥{threshold:<9.1%} {bets:>6d} {bets_per_year:>7.0f} {win_rate:>6.1%} {roi:>+7.1f}% ${profit:>+9,.0f} {pval:>7.3f}{marker}")

    # ==========================================
    # APPROACH 2: Direct classifier
    # ==========================================
    print(f"\n{'='*60}")
    print("  APPROACH 2: DIRECT OVER/UNDER CLASSIFIER")
    print(f"{'='*60}")

    # For each game, create target: did it go over the market line?
    # Features include the line itself as context

    classifier_results = []

    for test_year in range(2019, 2025):
        train = df[df["season"].between(2016, test_year - 1)].copy()
        test = df[df["season"] == test_year].copy()

        # Add market line as a feature + line-relative features
        train_rows = []
        test_rows = []

        for split_df, target_list in [(train, train_rows), (test, test_rows)]:
            for _, row in split_df.iterrows():
                gid = int(row["game_id"])
                if gid not in total_lines:
                    continue

                market = total_lines[gid]
                market_line = market["total_line"]
                actual_total = row["total_runs"]

                if actual_total == market_line:
                    continue  # push

                feat = {c: row[c] for c in available if c in row.index}
                feat["market_line"] = market_line
                # Line-relative features
                feat["rpg_vs_line"] = (row.get("home_b_rpg_15", 0) or 0) + (row.get("away_b_rpg_15", 0) or 0) - market_line
                feat["target"] = 1 if actual_total > market_line else 0
                feat["game_id"] = gid
                feat["season"] = int(row["season"])
                feat["actual_total"] = actual_total
                feat["market_line"] = market_line
                feat["over_odds"] = market["over_odds"]
                feat["under_odds"] = market["under_odds"]

                target_list.append(feat)

        if not train_rows or not test_rows:
            continue

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)

        clf_features = available + ["market_line", "rpg_vs_line"]
        clf_avail = [c for c in clf_features if c in train_df.columns]

        X_train_c = train_df[clf_avail].fillna(train_df[clf_avail].median())
        y_train_c = train_df["target"]
        X_test_c = test_df[clf_avail].fillna(train_df[clf_avail].median())
        y_test_c = test_df["target"]

        sc_c = StandardScaler()
        X_train_cs = sc_c.fit_transform(X_train_c)
        X_test_cs = sc_c.transform(X_test_c)

        clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf.fit(X_train_cs, y_train_c)

        probs = clf.predict_proba(X_test_cs)[:, 1]  # P(over)

        for i, (_, row) in enumerate(test_df.iterrows()):
            market_p_over = implied_from_odds(row["over_odds"], row["under_odds"])
            model_p_over = probs[i]
            edge = model_p_over - market_p_over

            bet_over = edge > 0
            actual_over = row["target"] == 1

            classifier_results.append({
                "game_id": int(row["game_id"]),
                "season": int(row["season"]),
                "model_p_over": model_p_over,
                "market_p_over": market_p_over,
                "edge": edge,
                "abs_edge": abs(edge),
                "bet_over": bet_over,
                "actual_over": actual_over,
                "correct": bet_over == actual_over,
                "over_odds": row["over_odds"],
                "under_odds": row["under_odds"],
            })

    cdf = pd.DataFrame(classifier_results)
    print(f"\n  Total games: {len(cdf)}")

    print(f"\n  {'Threshold':<12s} {'Bets':>6s} {'Bets/yr':>8s} {'Win%':>7s} {'ROI':>8s} {'P&L':>10s} {'p-value':>8s}")
    print(f"  {'-'*65}")

    for threshold in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]:
        subset = cdf[cdf["abs_edge"] >= threshold].copy()
        if len(subset) < 20:
            continue

        wins = subset["correct"].sum()
        bets = len(subset)
        win_rate = wins / bets

        profit = 0
        for _, g in subset.iterrows():
            if g["bet_over"]:
                dec_odds = american_to_decimal(g["over_odds"])
            else:
                dec_odds = american_to_decimal(g["under_odds"])
            if g["correct"]:
                profit += 100 * (dec_odds - 1)
            else:
                profit -= 100

        roi = profit / (bets * 100) * 100
        bets_per_year = bets / 6

        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue

        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""
        print(f"  ≥{threshold:<9.1%} {bets:>6d} {bets_per_year:>7.0f} {win_rate:>6.1%} {roi:>+7.1f}% ${profit:>+9,.0f} {pval:>7.3f}{marker}")

    # ==========================================
    # COMPARISON
    # ==========================================
    print(f"\n{'='*60}")
    print("  HEAD-TO-HEAD: APPROACH 1 vs APPROACH 2 (at ≥1% edge)")
    print(f"{'='*60}")

    for label, results_df in [("Regression→Probability", rdf), ("Direct Classifier", cdf)]:
        s = results_df[results_df["abs_edge"] >= 0.01]
        if len(s) > 0:
            wins = s["correct"].sum()
            bets = len(s)
            profit = 0
            for _, g in s.iterrows():
                dec = american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])
                profit += 100 * (dec - 1) if g["correct"] else -100
            roi = profit / (bets * 100) * 100
            pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue
            print(f"\n  {label}:")
            print(f"    Bets: {bets} ({bets/6:.0f}/yr), Win: {wins/bets:.1%}, ROI: {roi:+.1f}%, p={pval:.3f}")

    # By season for best approach
    print(f"\n  By season (≥1% edge):")
    for label, results_df in [("Reg→Prob", rdf), ("Classifier", cdf)]:
        print(f"\n  {label}:")
        for yr in range(2019, 2025):
            s = results_df[(results_df["season"] == yr) & (results_df["abs_edge"] >= 0.01)]
            if len(s) > 0:
                wins = s["correct"].sum()
                profit = sum(100 * (american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"]) - 1) if g["correct"] else -100 for _, g in s.iterrows())
                roi = profit / (len(s) * 100) * 100
                print(f"    {yr}: {len(s):4d} bets, {wins/len(s):.1%} win, ROI={roi:+.1f}%")


if __name__ == "__main__":
    main()
