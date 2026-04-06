"""
Quick check: are classifier high-edge bets getting systematically better odds?
If so, it's selecting for line shopping opportunity, not information.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
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


def main():
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]
    available = [c for c in FEATURES if c in df.columns]
    odds_by_game = get_odds_by_game()

    # Build both regression and classifier results with full odds info
    reg_results = []
    clf_train_data = []
    clf_test_data = []

    for test_year in range(2019, 2025):
        train = df[df["season"].between(2016, test_year - 1)].copy()
        test = df[df["season"] == test_year].copy()

        medians = train[available].median()
        X_train = train[available].fillna(medians)
        X_test = test[available].fillna(medians)

        sc_r = StandardScaler()
        reg = LinearRegression()
        reg.fit(sc_r.fit_transform(X_train), train["total_runs"])
        preds = reg.predict(sc_r.transform(X_test))

        # Classifier data
        train_rows = []
        test_rows = []

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

            over_odds_list = [b["over_odds"] for b in books_on_line]
            under_odds_list = [b["under_odds"] for b in books_on_line]

            med_over = np.median(over_odds_list)
            med_under = np.median(under_odds_list)

            raw_over_imp = implied_from_american(med_over)
            raw_under_imp = implied_from_american(med_under)
            total_imp = raw_over_imp + raw_under_imp
            devig_over = raw_over_imp / total_imp
            devig_under = raw_under_imp / total_imp

            # Regression approach
            pred_total = preds[idx]
            model_p_over = 1 - norm.cdf(market_total, loc=pred_total, scale=4.5)
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
                median_american = med_over
                worst_american = min(over_odds_list)
            else:
                best_american = max(under_odds_list)
                median_american = med_under
                worst_american = min(under_odds_list)

            best_decimal = american_to_decimal(best_american)
            median_decimal = american_to_decimal(median_american)
            worst_decimal = american_to_decimal(worst_american)
            breakeven = 1 / best_decimal
            bet_ev = model_win_prob - breakeven

            actual_over = actual_total > market_total
            bet_over = bet_side == "over"
            correct = bet_over == actual_over

            odds_spread = best_decimal - worst_decimal

            reg_results.append({
                "game_id": gid, "season": int(row["season"]),
                "info_edge": info_edge, "bet_ev": bet_ev,
                "correct": correct,
                "best_american": best_american,
                "best_decimal": best_decimal,
                "median_decimal": median_decimal,
                "worst_decimal": worst_decimal,
                "odds_spread": odds_spread,
                "n_books": len(books_on_line),
                "breakeven": breakeven,
            })

        # Build classifier training/test data
        for split_df, target_list in [(train, train_rows), (test, test_rows)]:
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

        train_df_c = pd.DataFrame(train_rows)
        test_df_c = pd.DataFrame(test_rows)

        clf_features = available + ["market_line", "rpg_vs_line"]
        clf_avail = [c for c in clf_features if c in train_df_c.columns]

        X_tr = train_df_c[clf_avail].fillna(train_df_c[clf_avail].median())
        X_te = test_df_c[clf_avail].fillna(train_df_c[clf_avail].median())

        sc_c = StandardScaler()
        clf = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        clf.fit(sc_c.fit_transform(X_tr), train_df_c["target"])
        probs = clf.predict_proba(sc_c.transform(X_te))[:, 1]

        for i, (_, row) in enumerate(test_df_c.iterrows()):
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
                median_american = med_over
                worst_american = min(over_odds_list)
            else:
                best_american = max(under_odds_list)
                median_american = med_under
                worst_american = min(under_odds_list)

            best_decimal = american_to_decimal(best_american)
            median_decimal = american_to_decimal(median_american)
            worst_decimal = american_to_decimal(worst_american)
            breakeven = 1 / best_decimal
            bet_ev = model_win_prob - breakeven

            actual_over = row["target"] == 1
            bet_over = bet_side == "over"
            correct = bet_over == actual_over

            odds_spread = best_decimal - worst_decimal

            clf_test_data.append({
                "game_id": int(row["game_id"]), "season": int(row["season"]),
                "info_edge": info_edge, "bet_ev": bet_ev,
                "correct": correct,
                "best_american": best_american,
                "best_decimal": best_decimal,
                "median_decimal": median_decimal,
                "worst_decimal": worst_decimal,
                "odds_spread": odds_spread,
                "n_books": int(row["n_books"]),
                "breakeven": breakeven,
            })

    rdf_reg = pd.DataFrame(reg_results)
    rdf_clf = pd.DataFrame(clf_test_data)

    # ==========================================
    # COMPARE ODDS CHARACTERISTICS
    # ==========================================
    print(f"\n{'='*70}")
    print("  ODDS CHARACTERISTICS BY APPROACH & THRESHOLD")
    print(f"  (All bets passing info_edge >= T AND +EV gate)")
    print(f"{'='*70}")

    print(f"\n  {'Approach':<20s} {'Thresh':<8s} {'Bets':>6s} {'Avg Best':>10s} {'Avg Med':>10s} {'Avg Spread':>11s} {'Avg Books':>10s} {'Breakeven':>10s}")
    print(f"  {'-'*90}")

    for label, rdf in [("Regression (4.5)", rdf_reg), ("Classifier", rdf_clf)]:
        for t in [0.0, 0.01, 0.02, 0.03, 0.05]:
            s = rdf[(rdf["info_edge"] >= t) & (rdf["bet_ev"] > 0)]
            if len(s) < 20:
                continue
            print(f"  {label:<20s} ≥{t:<5.1%} {len(s):>6d} "
                  f"{s['best_american'].mean():>+9.1f} "
                  f"{s['median_decimal'].mean():>9.3f} "
                  f"{s['odds_spread'].mean():>10.3f} "
                  f"{s['n_books'].mean():>9.1f} "
                  f"{s['breakeven'].mean():>9.1%}")

    # ==========================================
    # SAME GAMES COMPARISON
    # ==========================================
    print(f"\n{'='*70}")
    print("  SAME-GAME COMPARISON: When both bet on the same game,")
    print("  does the classifier get better odds?")
    print(f"{'='*70}")

    for t in [0.0, 0.01, 0.03]:
        reg_games = rdf_reg[(rdf_reg["info_edge"] >= t) & (rdf_reg["bet_ev"] > 0)]
        clf_games = rdf_clf[(rdf_clf["info_edge"] >= t) & (rdf_clf["bet_ev"] > 0)]

        reg_ids = set(reg_games["game_id"])
        clf_ids = set(clf_games["game_id"])
        shared = reg_ids & clf_ids
        clf_only = clf_ids - reg_ids
        reg_only = reg_ids - clf_ids

        print(f"\n  At ≥{t:.1%}:")
        print(f"    Regression bets: {len(reg_ids)}, Classifier bets: {len(clf_ids)}")
        print(f"    Shared: {len(shared)}, Classifier-only: {len(clf_only)}, Regression-only: {len(reg_only)}")

        if len(shared) > 0:
            shared_reg = reg_games[reg_games["game_id"].isin(shared)]
            shared_clf = clf_games[clf_games["game_id"].isin(shared)]
            print(f"    Shared games — Reg win%: {shared_reg['correct'].mean():.1%}, Clf win%: {shared_clf['correct'].mean():.1%}")
            print(f"    Shared games — Reg avg best: {shared_reg['best_american'].mean():+.1f}, Clf avg best: {shared_clf['best_american'].mean():+.1f}")

        if len(clf_only) > 0:
            clf_excl = clf_games[clf_games["game_id"].isin(clf_only)]
            print(f"    Classifier-only — win%: {clf_excl['correct'].mean():.1%}, avg best: {clf_excl['best_american'].mean():+.1f}, avg spread: {clf_excl['odds_spread'].mean():.3f}")

        if len(reg_only) > 0:
            reg_excl = reg_games[reg_games["game_id"].isin(reg_only)]
            print(f"    Regression-only — win%: {reg_excl['correct'].mean():.1%}, avg best: {reg_excl['best_american'].mean():+.1f}, avg spread: {reg_excl['odds_spread'].mean():.3f}")

    # ==========================================
    # ROI DECOMPOSITION
    # ==========================================
    print(f"\n{'='*70}")
    print("  ROI DECOMPOSITION: How much comes from win rate vs odds quality?")
    print(f"{'='*70}")

    for label, rdf in [("Regression (4.5)", rdf_reg), ("Classifier", rdf_clf)]:
        for t in [0.0, 0.03, 0.05]:
            s = rdf[(rdf["info_edge"] >= t) & (rdf["bet_ev"] > 0)]
            if len(s) < 20:
                continue

            # ROI at best odds
            profit_best = sum(100 * (g["best_decimal"] - 1) if g["correct"] else -100 for _, g in s.iterrows())
            roi_best = profit_best / (len(s) * 100) * 100

            # ROI at median odds (removes line shopping)
            profit_med = sum(100 * (g["median_decimal"] - 1) if g["correct"] else -100 for _, g in s.iterrows())
            roi_med = profit_med / (len(s) * 100) * 100

            # ROI at flat -110 (pure win rate signal)
            profit_110 = sum(100 * (1.909 - 1) if g["correct"] else -100 for _, g in s.iterrows())
            roi_110 = profit_110 / (len(s) * 100) * 100

            print(f"\n  {label} ≥{t:.1%} ({len(s)} bets):")
            print(f"    ROI @-110 (pure model):    {roi_110:+.2f}%")
            print(f"    ROI @median (single book):  {roi_med:+.2f}%")
            print(f"    ROI @best (line shopping):  {roi_best:+.2f}%")
            print(f"    Line shopping boost:        {roi_best - roi_110:+.2f}%")


if __name__ == "__main__":
    main()
