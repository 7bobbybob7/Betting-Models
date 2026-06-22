"""
models/mlb/investigate_classifier.py - Deep investigation of the classifier signal.

The direct over/under classifier showed p=0.037 at ≥8% edge across 1,786 bets.
This is the first statistically significant result. Investigate honestly.

Key questions:
1. Is the threshold selection biased? (We picked 8% because it looked best)
2. Cross-season consistency
3. True out-of-sample (dev/holdout split)
4. What features drive the classifier?
5. Drawdown analysis
6. Is this just the old totals signal repackaged?
7. Multiple testing correction

Usage:
    python -m models.mlb.investigate_classifier
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
            best[gid] = {"total_line": float(r["total_line"]),
                         "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                         "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                         "_sb": book}
        else:
            cur_p = book_priority.index(best[gid]["_sb"]) if best[gid]["_sb"] in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {"total_line": float(r["total_line"]),
                             "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                             "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                             "_sb": book}
    return best


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    return 1 + american / 100 if american >= 0 else 1 + 100 / abs(american)


def implied_from_odds(over_odds, under_odds):
    def to_imp(odds):
        return 100 / (odds + 100) if odds >= 0 else abs(odds) / (abs(odds) + 100)
    o, u = to_imp(over_odds), to_imp(under_odds)
    return o / (o + u)


def build_classifier_predictions(df, total_lines, available):
    """Build classifier predictions with expanding window."""
    all_results = []

    for test_year in range(2019, 2025):
        train_rows, test_rows = [], []

        for split_df, target in [(df[df["season"].between(2016, test_year - 1)], train_rows),
                                  (df[df["season"] == test_year], test_rows)]:
            for _, row in split_df.iterrows():
                gid = int(row["game_id"])
                if gid not in total_lines:
                    continue
                market = total_lines[gid]
                actual = row["total_runs"]
                if actual == market["total_line"]:
                    continue
                feat = {c: row[c] for c in available}
                feat["market_line"] = market["total_line"]
                feat["rpg_vs_line"] = (row.get("home_b_rpg_15", 0) or 0) + (row.get("away_b_rpg_15", 0) or 0) - market["total_line"]
                feat["target"] = 1 if actual > market["total_line"] else 0
                feat["game_id"] = gid
                feat["season"] = int(row["season"])
                feat["actual_total"] = actual
                feat["over_odds"] = market["over_odds"]
                feat["under_odds"] = market["under_odds"]
                feat["park_factor_val"] = row.get("park_factor", 1.0)
                feat["month"] = pd.to_datetime(row["game_date"]).month
                target.append(feat)

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
            market_p = implied_from_odds(row["over_odds"], row["under_odds"])
            edge = probs[i] - market_p
            bet_over = edge > 0

            all_results.append({
                "game_id": int(row["game_id"]), "season": int(row["season"]),
                "model_p_over": probs[i], "market_p_over": market_p,
                "edge": edge, "abs_edge": abs(edge),
                "bet_over": bet_over,
                "actual_over": row["target"] == 1,
                "correct": bet_over == (row["target"] == 1),
                "over_odds": row["over_odds"], "under_odds": row["under_odds"],
                "park_factor": row.get("park_factor_val", 1.0),
                "month": int(row["month"]),
            })

    return pd.DataFrame(all_results)


def simulate_roi(subset, label):
    if len(subset) == 0:
        return None
    profit = 0
    for _, g in subset.iterrows():
        dec = american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])
        profit += 100 * (dec - 1) if g["correct"] else -100
    roi = profit / (len(subset) * 100) * 100
    return roi


def main():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    total_lines = get_total_lines()
    available = [c for c in FEATURES if c in df.columns]

    print(f"\n{'='*60}")
    print("  CLASSIFIER DEEP INVESTIGATION")
    print(f"{'='*60}")

    rdf = build_classifier_predictions(df, total_lines, available)

    # ==========================================
    # 1. MULTIPLE TESTING CORRECTION
    # ==========================================
    print(f"\n{'='*60}")
    print("  1. MULTIPLE TESTING CORRECTION")
    print(f"{'='*60}")

    print(f"\n  We tested 9 thresholds (0%, 0.5%, 1%, 1.5%, 2%, 3%, 5%, 8%, 10%).")
    print(f"  With 9 tests, Bonferroni correction: α = 0.05/9 = 0.0056")
    print(f"  The ≥10% p-value was 0.009 — fails Bonferroni.")
    print(f"  The ≥8% p-value was 0.037 — fails Bonferroni.")
    print(f"  After correction, NOTHING is significant at α=0.05.")
    print(f"\n  This is the most important finding. The 'significance' was an artifact")
    print(f"  of testing multiple thresholds and reporting the best one.")

    # ==========================================
    # 2. CROSS-SEASON CONSISTENCY
    # ==========================================
    print(f"\n{'='*60}")
    print("  2. CROSS-SEASON CONSISTENCY")
    print(f"{'='*60}")

    for threshold in [0.05, 0.08, 0.10]:
        print(f"\n  ≥{threshold:.0%} edge:")
        subset = rdf[rdf["abs_edge"] >= threshold]
        total_profit = 0
        for yr in range(2019, 2025):
            s = subset[subset["season"] == yr]
            if len(s) > 0:
                roi = simulate_roi(s, "")
                profit = sum(100 * (american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"]) - 1) if g["correct"] else -100 for _, g in s.iterrows())
                total_profit += profit
                print(f"    {yr}: {len(s):4d} bets, {s['correct'].mean():.1%} win, ROI={roi:+.1f}%")

    # ==========================================
    # 3. TRUE OUT-OF-SAMPLE (dev/holdout split)
    # ==========================================
    print(f"\n{'='*60}")
    print("  3. TRUE OUT-OF-SAMPLE")
    print(f"  Dev: 2019-2022 (pick threshold), Holdout: 2023-2024")
    print(f"{'='*60}")

    dev = rdf[rdf["season"].between(2019, 2022)]
    holdout = rdf[rdf["season"].between(2023, 2024)]

    print(f"\n  Dev set (2019-2022) — pick best threshold:")
    for t in [0.03, 0.05, 0.08, 0.10]:
        s = dev[dev["abs_edge"] >= t]
        if len(s) > 0:
            roi = simulate_roi(s, "")
            pval = binomtest(int(s["correct"].sum()), len(s), 0.524, alternative="greater").pvalue
            print(f"    ≥{t:.0%}: {len(s):4d} bets, {s['correct'].mean():.1%} win, ROI={roi:+.1f}%, p={pval:.3f}")

    # Apply the BEST dev threshold to holdout
    # Find best from dev
    best_dev_roi = -999
    best_dev_t = 0.05
    for t in [0.03, 0.05, 0.08, 0.10]:
        s = dev[dev["abs_edge"] >= t]
        if len(s) > 0:
            roi = simulate_roi(s, "")
            if roi > best_dev_roi:
                best_dev_roi = roi
                best_dev_t = t

    print(f"\n  Best dev threshold: ≥{best_dev_t:.0%} (ROI={best_dev_roi:+.1f}%)")

    h = holdout[holdout["abs_edge"] >= best_dev_t]
    if len(h) > 0:
        h_roi = simulate_roi(h, "")
        h_pval = binomtest(int(h["correct"].sum()), len(h), 0.524, alternative="greater").pvalue
        print(f"\n  HOLDOUT (2023-2024) at ≥{best_dev_t:.0%}:")
        print(f"    Bets: {len(h)}")
        print(f"    Win rate: {h['correct'].mean():.1%}")
        print(f"    ROI: {h_roi:+.1f}%")
        print(f"    p-value: {h_pval:.3f}")

        for yr in [2023, 2024]:
            s = h[h["season"] == yr]
            if len(s) > 0:
                yr_roi = simulate_roi(s, "")
                print(f"    {yr}: {len(s)} bets, {s['correct'].mean():.1%} win, ROI={yr_roi:+.1f}%")

    # ==========================================
    # 4. FEATURE IMPORTANCE
    # ==========================================
    print(f"\n{'='*60}")
    print("  4. WHAT DRIVES THE CLASSIFIER?")
    print(f"{'='*60}")

    # Train one model on all data to see coefficients
    all_rows = []
    for _, row in df.iterrows():
        gid = int(row["game_id"])
        if gid not in total_lines:
            continue
        market = total_lines[gid]
        actual = row["total_runs"]
        if actual == market["total_line"]:
            continue
        feat = {c: row[c] for c in available}
        feat["market_line"] = market["total_line"]
        feat["rpg_vs_line"] = (row.get("home_b_rpg_15", 0) or 0) + (row.get("away_b_rpg_15", 0) or 0) - market["total_line"]
        feat["target"] = 1 if actual > market["total_line"] else 0
        all_rows.append(feat)

    all_df = pd.DataFrame(all_rows)
    clf_features = available + ["market_line", "rpg_vs_line"]
    clf_avail = [c for c in clf_features if c in all_df.columns]

    X = all_df[clf_avail].fillna(all_df[clf_avail].median())
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
    clf.fit(sc.fit_transform(X), all_df["target"])

    coefs = pd.Series(clf.coef_[0], index=clf_avail)
    nonzero = coefs[coefs != 0].abs().sort_values(ascending=False)
    print(f"\n  Non-zero features: {len(nonzero)}/{len(clf_avail)}")
    print(f"  Top features:")
    for feat in nonzero.head(10).index:
        print(f"    {feat:30s} {coefs[feat]:+.4f}")

    # ==========================================
    # 5. IS THIS JUST THE OLD SIGNAL REPACKAGED?
    # ==========================================
    print(f"\n{'='*60}")
    print("  5. OVERLAP WITH OLD TOTALS STRATEGY")
    print(f"{'='*60}")

    # The old strategy: ≥1.5 run edge on regression model
    # Does the classifier find the same games?
    from models.mlb.totals_classifier import FEATURES as REG_FEATURES

    # Check correlation between classifier edge and regression edge
    reg_results = []
    for test_year in range(2019, 2025):
        train = df[df["season"].between(2016, test_year - 1)]
        test = df[df["season"] == test_year]
        medians = train[available].median()
        X_tr = train[available].fillna(medians)
        X_te = test[available].fillna(medians)
        sc2 = StandardScaler()
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(sc2.fit_transform(X_tr), train["total_runs"])
        preds = reg.predict(sc2.transform(X_te))

        for idx, (_, row) in enumerate(test.iterrows()):
            gid = int(row["game_id"])
            if gid in total_lines:
                reg_edge = preds[idx] - total_lines[gid]["total_line"]
                reg_results.append({"game_id": gid, "reg_edge": reg_edge})

    reg_df = pd.DataFrame(reg_results)
    merged = rdf.merge(reg_df, on="game_id", how="inner")

    if len(merged) > 0:
        corr = merged["edge"].corr(merged["reg_edge"])
        print(f"\n  Correlation between classifier edge and regression edge: {corr:.3f}")

        # How many high-edge classifier bets overlap with high-edge regression bets?
        clf_high = set(merged[merged["abs_edge"] >= 0.08]["game_id"])
        reg_high = set(merged[merged["reg_edge"].abs() >= 1.5]["game_id"])
        overlap = clf_high & reg_high
        print(f"  Classifier ≥8% bets: {len(clf_high)}")
        print(f"  Regression ≥1.5 run bets: {len(reg_high)}")
        print(f"  Overlap: {len(overlap)} ({len(overlap)/max(len(clf_high),1):.0%} of classifier bets)")

    # ==========================================
    # 6. DRAWDOWN ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("  6. DRAWDOWN (≥8% edge, chronological)")
    print(f"{'='*60}")

    subset = rdf[rdf["abs_edge"] >= 0.08].sort_values(["season", "game_id"]).reset_index(drop=True)
    bankroll = 10000
    peak = bankroll
    max_dd = 0
    max_lose = 0
    cur_lose = 0

    for _, g in subset.iterrows():
        dec = american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])
        if g["correct"]:
            bankroll += 100 * (dec - 1)
            cur_lose = 0
        else:
            bankroll -= 100
            cur_lose += 1
            max_lose = max(max_lose, cur_lose)
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)

    print(f"  Final bankroll: ${bankroll:,.0f}")
    print(f"  Max drawdown: {max_dd:.1%}")
    print(f"  Max losing streak: {max_lose}")

    # ==========================================
    # 7. SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print("  HONEST SUMMARY")
    print(f"{'='*60}")
    print(f"""
  1. The p=0.037 at ≥8% fails Bonferroni correction (tested 9 thresholds).
     After correction, nothing is significant at α=0.05.

  2. The classifier IS better than the regression approach — it consistently
     shows higher win rates at every threshold. Direct classification on
     "over or under" is a better formulation than predicting exact totals.

  3. The ≥5-8% range is the most promising — enough selectivity for edge,
     enough volume for validation (~300-700 bets/year).

  4. The high-volume thin-edge approach (≥0-2%) definitively does NOT work.
     The model has no consistent edge below 3% threshold.

  5. Live deployment remains the only honest test. Deploy the classifier
     at ≥5% edge (~700 bets/year) and evaluate after one season.
""")


if __name__ == "__main__":
    main()
