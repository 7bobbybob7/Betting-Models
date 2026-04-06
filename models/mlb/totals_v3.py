"""
models/mlb/totals_v3.py - Comprehensive model improvement for high-volume totals.

Goal: improve from 51.7% at thin edges to 53%+ for profitable high-volume betting.

Tests:
1. New features (dome, rest days, weather condition, schedule density)
2. XGBoost classifier
3. Time-weighted training
4. Line-relative features
5. Ensemble
6. Pitcher matchup features from Statcast

Usage:
    python -m models.mlb.totals_v3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.stats import binomtest

from db.db import query


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_enhanced_features():
    """Load base features + build new features."""
    print("Loading base features...")
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    print("Loading game info for new features...")
    game_info = query("""
        SELECT g.game_id, g.game_date, g.venue,
               g.home_team_id, g.away_team_id,
               gi.weather_cond
        FROM games g
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = 2
    """)

    # Merge weather condition
    gi_map = {}
    for _, r in game_info.iterrows():
        gi_map[int(r["game_id"])] = {
            "weather_cond": r.get("weather_cond", ""),
            "venue": r.get("venue", ""),
            "home_team_id": int(r["home_team_id"]),
            "away_team_id": int(r["away_team_id"]),
        }

    # === NEW FEATURE 1: Dome/roof ===
    dome_keywords = ["dome", "roof closed"]
    df["is_dome"] = df["game_id"].apply(
        lambda gid: 1 if any(k in str(gi_map.get(gid, {}).get("weather_cond", "")).lower() for k in dome_keywords) else 0
    )

    # === NEW FEATURE 2: Rest days per team ===
    print("Computing rest days...")
    all_games = query("""
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM games WHERE sport_id = 2 ORDER BY game_date
    """)

    # For each team, compute days since last game
    team_last_game = {}
    rest_days = {}  # (game_id, team_id) -> rest_days

    for _, g in all_games.iterrows():
        gid = int(g["game_id"])
        gd = g["game_date"]
        for tid in [int(g["home_team_id"]), int(g["away_team_id"])]:
            if tid in team_last_game:
                delta = (gd - team_last_game[tid]).days
                rest_days[(gid, tid)] = delta
            else:
                rest_days[(gid, tid)] = 5  # default for first game
            team_last_game[tid] = gd

    df["home_rest_days"] = df.apply(
        lambda r: rest_days.get((int(r["game_id"]), gi_map.get(int(r["game_id"]), {}).get("home_team_id", 0)), 1), axis=1
    )
    df["away_rest_days"] = df.apply(
        lambda r: rest_days.get((int(r["game_id"]), gi_map.get(int(r["game_id"]), {}).get("away_team_id", 0)), 1), axis=1
    )

    # === NEW FEATURE 3: Schedule density (games in last 7 days) ===
    # Already captured by rest_days — a team with rest=1 played yesterday

    # === NEW FEATURE 4: Weather categories ===
    df["is_clear"] = df["game_id"].apply(
        lambda gid: 1 if "clear" in str(gi_map.get(gid, {}).get("weather_cond", "")).lower() or
                        "sunny" in str(gi_map.get(gid, {}).get("weather_cond", "")).lower() else 0
    )
    df["is_precip"] = df["game_id"].apply(
        lambda gid: 1 if any(w in str(gi_map.get(gid, {}).get("weather_cond", "")).lower()
                             for w in ["rain", "drizzle", "snow"]) else 0
    )

    # === NEW FEATURE 5: Season phase ===
    df["month"] = pd.to_datetime(df["game_date"]).dt.month
    df["is_early_season"] = (df["month"] <= 4).astype(int)
    df["is_summer"] = (df["month"].between(6, 8)).astype(int)

    # === NEW FEATURE 6: Combined team RPG vs line ===
    # This is the most important — how does combined team scoring compare to the market line?
    # Need to add market line to the dataframe
    # This happens at prediction time, not here

    print(f"Enhanced features: {df.shape}")
    return df


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


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
BASE_FEATURES = [
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

NEW_FEATURES = [
    "is_dome", "home_rest_days", "away_rest_days",
    "is_clear", "is_precip", "is_early_season", "is_summer",
]

LINE_FEATURES = [
    "market_line", "rpg_vs_line",
    "home_rpg_vs_half_line", "away_rpg_vs_half_line",
]


def build_training_data(df, total_lines, features, add_line_features=True):
    """Build classifier training data with market line context."""
    rows = []
    for _, row in df.iterrows():
        gid = int(row["game_id"])
        if gid not in total_lines:
            continue
        market = total_lines[gid]
        actual = row["total_runs"]
        if actual == market["total_line"]:
            continue

        feat = {c: row[c] for c in features if c in row.index}

        if add_line_features:
            feat["market_line"] = market["total_line"]
            home_rpg = row.get("home_b_rpg_15", 0) or 0
            away_rpg = row.get("away_b_rpg_15", 0) or 0
            feat["rpg_vs_line"] = home_rpg + away_rpg - market["total_line"]
            feat["home_rpg_vs_half_line"] = home_rpg - market["total_line"] / 2
            feat["away_rpg_vs_half_line"] = away_rpg - market["total_line"] / 2

        feat["target"] = 1 if actual > market["total_line"] else 0
        feat["game_id"] = gid
        feat["season"] = int(row["season"])
        feat["over_odds"] = market["over_odds"]
        feat["under_odds"] = market["under_odds"]
        rows.append(feat)

    return pd.DataFrame(rows)


def evaluate_model(train_df, test_df, clf_features, model_type="logreg", time_weighted=False):
    """Train and evaluate a model. Returns (predictions_df, win_rate, roi)."""
    avail = [c for c in clf_features if c in train_df.columns]

    X_train = train_df[avail].fillna(train_df[avail].median())
    y_train = train_df["target"]
    X_test = test_df[avail].fillna(train_df[avail].median())

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)

    # Sample weights for time-weighting
    weights = None
    if time_weighted:
        max_season = train_df["season"].max()
        weights = train_df["season"].apply(lambda s: 1 + (s - 2016) * 0.2).values

    if model_type == "logreg":
        model = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        model.fit(X_train_s, y_train, sample_weight=weights)
        probs = model.predict_proba(X_test_s)[:, 1]
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=1.0, random_state=42,
            eval_metric="logloss",
        )
        if time_weighted:
            model.fit(X_train_s, y_train, sample_weight=weights)
        else:
            model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

    # Compute edges and results
    results = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        market_p = implied_from_odds(row["over_odds"], row["under_odds"])
        edge = probs[i] - market_p
        bet_over = edge > 0
        actual_over = row["target"] == 1

        results.append({
            "game_id": int(row["game_id"]),
            "season": int(row["season"]),
            "edge": edge,
            "abs_edge": abs(edge),
            "bet_over": bet_over,
            "actual_over": actual_over,
            "correct": bet_over == actual_over,
            "over_odds": row["over_odds"],
            "under_odds": row["under_odds"],
        })

    return pd.DataFrame(results)


def summarize(label, rdf, thresholds=[0.0, 0.01, 0.03, 0.05, 0.08]):
    """Print summary at various thresholds."""
    print(f"\n  {label}:")
    for t in thresholds:
        s = rdf[rdf["abs_edge"] >= t]
        if len(s) < 20:
            continue
        wins = s["correct"].sum()
        bets = len(s)
        profit = sum(100 * (american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"]) - 1)
                     if g["correct"] else -100 for _, g in s.iterrows())
        roi = profit / (bets * 100) * 100
        pval = binomtest(wins, bets, 0.524, alternative="greater").pvalue
        marker = " ***" if pval < 0.05 else " *" if pval < 0.10 else ""
        print(f"    ≥{t:5.1%}: {bets:5d} bets ({bets/6:4.0f}/yr), {wins/bets:5.1%} win, ROI={roi:+5.1f}%, p={pval:.3f}{marker}")


def main():
    df = load_enhanced_features()
    total_lines = get_total_lines()

    print(f"\n{'='*70}")
    print("  COMPREHENSIVE MODEL IMPROVEMENT — TARGET: 53%+ AT HIGH VOLUME")
    print(f"{'='*70}")

    # Build full dataset with all features
    all_features = BASE_FEATURES + NEW_FEATURES
    full_data = build_training_data(df, total_lines, all_features, add_line_features=True)
    all_clf_features = all_features + LINE_FEATURES

    print(f"\n  Total games with odds: {len(full_data)}")

    # ==========================================
    # BASELINE: LogReg, base features + line
    # ==========================================
    print(f"\n{'='*70}")
    print("  TEST 1: BASELINE vs NEW FEATURES")
    print(f"{'='*70}")

    base_clf = BASE_FEATURES + LINE_FEATURES
    enhanced_clf = all_features + LINE_FEATURES

    for label, features in [
        ("A) Baseline (20 feat + line)", base_clf),
        ("B) + Dome/rest/weather/season", enhanced_clf),
    ]:
        all_results = []
        for test_year in range(2019, 2025):
            train = full_data[full_data["season"].between(2016, test_year - 1)]
            test = full_data[full_data["season"] == test_year]
            if len(train) < 100 or len(test) < 20:
                continue
            r = evaluate_model(train, test, features, "logreg")
            all_results.append(r)
        if all_results:
            rdf = pd.concat(all_results)
            summarize(label, rdf)

    # ==========================================
    # TEST 2: XGBoost vs LogReg
    # ==========================================
    print(f"\n{'='*70}")
    print("  TEST 2: LOGREG vs XGBOOST")
    print(f"{'='*70}")

    for model_type, label in [("logreg", "C) LogReg (enhanced)"), ("xgboost", "D) XGBoost (enhanced)")]:
        all_results = []
        for test_year in range(2019, 2025):
            train = full_data[full_data["season"].between(2016, test_year - 1)]
            test = full_data[full_data["season"] == test_year]
            if len(train) < 100 or len(test) < 20:
                continue
            r = evaluate_model(train, test, enhanced_clf, model_type)
            all_results.append(r)
        if all_results:
            rdf = pd.concat(all_results)
            summarize(label, rdf)

    # ==========================================
    # TEST 3: TIME-WEIGHTED TRAINING
    # ==========================================
    print(f"\n{'='*70}")
    print("  TEST 3: TIME-WEIGHTED vs EQUAL-WEIGHTED")
    print(f"{'='*70}")

    for tw, label in [(False, "E) Equal-weighted"), (True, "F) Time-weighted (recent=higher)")]:
        all_results = []
        for test_year in range(2019, 2025):
            train = full_data[full_data["season"].between(2016, test_year - 1)]
            test = full_data[full_data["season"] == test_year]
            if len(train) < 100 or len(test) < 20:
                continue
            r = evaluate_model(train, test, enhanced_clf, "logreg", time_weighted=tw)
            all_results.append(r)
        if all_results:
            rdf = pd.concat(all_results)
            summarize(label, rdf)

    # ==========================================
    # TEST 4: XGBoost + TIME-WEIGHTED
    # ==========================================
    print(f"\n{'='*70}")
    print("  TEST 4: XGBOOST + TIME-WEIGHTED")
    print(f"{'='*70}")

    for tw, label in [(False, "G) XGBoost equal-weight"), (True, "H) XGBoost time-weighted")]:
        all_results = []
        for test_year in range(2019, 2025):
            train = full_data[full_data["season"].between(2016, test_year - 1)]
            test = full_data[full_data["season"] == test_year]
            if len(train) < 100 or len(test) < 20:
                continue
            r = evaluate_model(train, test, enhanced_clf, "xgboost", time_weighted=tw)
            all_results.append(r)
        if all_results:
            rdf = pd.concat(all_results)
            summarize(label, rdf)

    # ==========================================
    # TEST 5: ENSEMBLE (LogReg + XGBoost average)
    # ==========================================
    print(f"\n{'='*70}")
    print("  TEST 5: ENSEMBLE (LogReg + XGBoost)")
    print(f"{'='*70}")

    ensemble_results = []
    for test_year in range(2019, 2025):
        train = full_data[full_data["season"].between(2016, test_year - 1)]
        test = full_data[full_data["season"] == test_year]
        if len(train) < 100 or len(test) < 20:
            continue

        avail = [c for c in enhanced_clf if c in train.columns]
        X_tr = train[avail].fillna(train[avail].median())
        X_te = test[avail].fillna(train[avail].median())
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # LogReg
        lr = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=42)
        lr.fit(X_tr_s, train["target"])
        lr_probs = lr.predict_proba(X_te_s)[:, 1]

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=1.0, random_state=42,
            eval_metric="logloss",
        )
        xgb.fit(X_tr_s, train["target"])
        xgb_probs = xgb.predict_proba(X_te_s)[:, 1]

        # Ensemble
        ens_probs = (lr_probs + xgb_probs) / 2

        for i, (_, row) in enumerate(test.iterrows()):
            market_p = implied_from_odds(row["over_odds"], row["under_odds"])
            edge = ens_probs[i] - market_p
            bet_over = edge > 0
            actual_over = row["target"] == 1

            ensemble_results.append({
                "game_id": int(row["game_id"]),
                "season": int(row["season"]),
                "edge": edge, "abs_edge": abs(edge),
                "bet_over": bet_over, "actual_over": actual_over,
                "correct": bet_over == actual_over,
                "over_odds": row["over_odds"], "under_odds": row["under_odds"],
            })

    edf = pd.DataFrame(ensemble_results)
    summarize("I) Ensemble (LR + XGB avg)", edf)

    # ==========================================
    # FINAL COMPARISON AT KEY THRESHOLDS
    # ==========================================
    print(f"\n{'='*70}")
    print("  FINAL: ALL MODELS AT ≥1% AND ≥3% EDGE (high volume)")
    print(f"{'='*70}")

    # Rerun all and compare at the two key thresholds
    models_to_compare = {
        "Baseline LogReg": (base_clf, "logreg", False),
        "Enhanced LogReg": (enhanced_clf, "logreg", False),
        "Enhanced XGBoost": (enhanced_clf, "xgboost", False),
        "TimeWeight LogReg": (enhanced_clf, "logreg", True),
        "TimeWeight XGBoost": (enhanced_clf, "xgboost", True),
    }

    print(f"\n  {'Model':<25s} {'≥1% bets':>8s} {'≥1% win':>8s} {'≥1% ROI':>8s} {'≥3% bets':>8s} {'≥3% win':>8s} {'≥3% ROI':>8s}")
    print(f"  {'-'*75}")

    for name, (features, mtype, tw) in models_to_compare.items():
        all_r = []
        for test_year in range(2019, 2025):
            train = full_data[full_data["season"].between(2016, test_year - 1)]
            test = full_data[full_data["season"] == test_year]
            if len(train) < 100 or len(test) < 20:
                continue
            r = evaluate_model(train, test, features, mtype, tw)
            all_r.append(r)
        if not all_r:
            continue
        rdf = pd.concat(all_r)

        for t_label, t in [("≥1%", 0.01), ("≥3%", 0.03)]:
            s = rdf[rdf["abs_edge"] >= t]
            if len(s) > 0:
                wins = s["correct"].sum()
                profit = sum(100 * (american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"]) - 1)
                             if g["correct"] else -100 for _, g in s.iterrows())
                roi = profit / (len(s) * 100) * 100

        s1 = rdf[rdf["abs_edge"] >= 0.01]
        s3 = rdf[rdf["abs_edge"] >= 0.03]
        w1 = s1["correct"].mean() if len(s1) > 0 else 0
        w3 = s3["correct"].mean() if len(s3) > 0 else 0
        r1 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s1.iterrows()) / max(len(s1)*100,1) * 100
        r3 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s3.iterrows()) / max(len(s3)*100,1) * 100

        print(f"  {name:<25s} {len(s1):>8d} {w1:>7.1%} {r1:>+7.1f}% {len(s3):>8d} {w3:>7.1%} {r3:>+7.1f}%")

    # Also add ensemble
    s1 = edf[edf["abs_edge"] >= 0.01]
    s3 = edf[edf["abs_edge"] >= 0.03]
    w1 = s1["correct"].mean() if len(s1) > 0 else 0
    w3 = s3["correct"].mean() if len(s3) > 0 else 0
    r1 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s1.iterrows()) / max(len(s1)*100,1) * 100
    r3 = sum(100*(american_to_decimal(g["over_odds"] if g["bet_over"] else g["under_odds"])-1) if g["correct"] else -100 for _,g in s3.iterrows()) / max(len(s3)*100,1) * 100
    print(f"  {'Ensemble (LR+XGB)':<25s} {len(s1):>8d} {w1:>7.1%} {r1:>+7.1f}% {len(s3):>8d} {w3:>7.1%} {r3:>+7.1f}%")


if __name__ == "__main__":
    main()
