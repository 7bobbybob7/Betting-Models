"""
models/mlb/totals_v2.py - Enhanced totals model with Statcast + lineup features.

Improvements over v1:
    1. Statcast pitcher features (whiff rate, chase rate, velo, pitch mix)
    2. Lineup-specific features (wOBA, K%, exit velo, xBA)

Evaluated on dev set (2019-2022) ONLY — holdout (2023-2024) reserved for live comparison.

Usage:
    python -m models.mlb.totals_v2
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
from models.mlb.statcast_features import build_statcast_features
from models.mlb.lineup_features import build_lineup_features


# V1 features (baseline)
V1_FEATURES = [
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

# V2: V1 + Statcast pitcher features
STATCAST_FEATURES = [
    "home_sc_whiff_rate_5", "away_sc_whiff_rate_5",
    "home_sc_chase_rate_5", "away_sc_chase_rate_5",
    "home_sc_swstr_rate_5", "away_sc_swstr_rate_5",
    "home_sc_fb_velo_5", "away_sc_fb_velo_5",
    "home_sc_velo_trend", "away_sc_velo_trend",
    "home_sc_k_per_start_5", "away_sc_k_per_start_5",
    "home_sc_zone_rate_5", "away_sc_zone_rate_5",
]

# V3: V2 + Lineup features
LINEUP_FEATURES = [
    "home_lu_woba", "away_lu_woba",
    "home_lu_kpct", "away_lu_kpct",
    "home_lu_iso", "away_lu_iso",
    "home_lu_exit_velo", "away_lu_exit_velo",
    "home_lu_xba", "away_lu_xba",
    "home_lu_hard_hit", "away_lu_hard_hit",
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
                         "sb": book}
        else:
            cur_p = book_priority.index(best[gid].get("sb", "x")) if best[gid].get("sb", "x") in book_priority else 999
            new_p = book_priority.index(book) if book in book_priority else 999
            if new_p < cur_p:
                best[gid] = {"total_line": float(r["total_line"]),
                             "over_odds": float(r["over_odds"]) if pd.notna(r["over_odds"]) else -110,
                             "under_odds": float(r["under_odds"]) if pd.notna(r["under_odds"]) else -110,
                             "sb": book}
    return best


def american_to_decimal(american):
    if american is None or pd.isna(american):
        return 1.909
    if american >= 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def build_enhanced_features():
    """Build feature matrix with V1 + Statcast + Lineup features."""
    print("\n=== BUILDING ENHANCED TOTALS FEATURES ===")

    # Load base features
    print("  Loading base features...")
    df = pd.read_csv("data/mlb_features.csv", parse_dates=["game_date"])
    df["total_runs"] = df["home_score"] + df["away_score"]

    # Build Statcast features
    pitcher_sc, team_k = build_statcast_features()

    # Build lineup features
    lineup_feats = build_lineup_features()

    # Get game -> team mapping
    game_teams = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id = 2")
    team_map = {}
    for _, r in game_teams.iterrows():
        team_map[int(r["game_id"])] = (int(r["home_team_id"]), int(r["away_team_id"]))

    # Get game -> starter mapping
    starters = query("""
        SELECT g.game_id,
               hp.player_id as home_starter, ap.player_id as away_starter
        FROM games g
        LEFT JOIN mlb_pitching_game hp ON g.game_id = hp.game_id
            AND hp.team_id = g.home_team_id AND hp.is_starter = true
        LEFT JOIN mlb_pitching_game ap ON g.game_id = ap.game_id
            AND ap.team_id = g.away_team_id AND ap.is_starter = true
        WHERE g.sport_id = 2
    """)
    starter_map = {}
    for _, r in starters.iterrows():
        starter_map[int(r["game_id"])] = (
            int(r["home_starter"]) if pd.notna(r["home_starter"]) else None,
            int(r["away_starter"]) if pd.notna(r["away_starter"]) else None,
        )

    # Merge Statcast pitcher features
    print("  Merging Statcast pitcher features...")
    for col_prefix, starter_idx in [("home_sc_", 0), ("away_sc_", 1)]:
        sc_cols = {
            "whiff_rate_5": [], "chase_rate_5": [], "swstr_rate_5": [],
            "fb_velo_5": [], "velo_trend": [], "k_per_start_5": [], "zone_rate_5": [],
        }

        for _, row in df.iterrows():
            gid = int(row["game_id"])
            starter_pair = starter_map.get(gid, (None, None))
            pid = starter_pair[starter_idx]

            sc = pitcher_sc.get((gid, pid), {}) if pid else {}
            for key in sc_cols:
                sc_cols[key].append(sc.get(f"sc_{key}"))

        for key, vals in sc_cols.items():
            df[f"{col_prefix}{key}"] = vals

    # Merge lineup features
    print("  Merging lineup features...")
    for prefix, tid_idx in [("home_lu_", 0), ("away_lu_", 1)]:
        lu_cols = {
            "woba": [], "kpct": [], "iso": [], "exit_velo": [],
            "xba": [], "hard_hit": [],
        }

        for _, row in df.iterrows():
            gid = int(row["game_id"])
            team_pair = team_map.get(gid, (None, None))
            tid = team_pair[tid_idx]

            lu = lineup_feats.get((gid, tid), {}) if tid else {}
            for key in lu_cols:
                lu_cols[key].append(lu.get(f"lu_{key}"))

        for key, vals in lu_cols.items():
            df[f"{prefix}{key}"] = vals

    print(f"  Enhanced matrix: {df.shape}")
    return df


def evaluate_model(df, features, label, total_lines, train_years, dev_years):
    """Train on train_years, evaluate on dev_years."""
    available = [c for c in features if c in df.columns]

    train = df[df["season"].between(*train_years)].copy()
    dev = df[df["season"].between(*dev_years)].copy()

    medians = train[available].median()
    X_train = train[available].fillna(medians)
    X_dev = dev[available].fillna(medians)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_dev_s = sc.transform(X_dev)

    mdl = LinearRegression()
    mdl.fit(X_train_s, train["total_runs"])
    dev_preds = mdl.predict(X_dev_s)

    # Evaluate on dev set
    bets = 0
    wins = 0
    profit = 0

    for idx, (_, row) in enumerate(dev.iterrows()):
        gid = int(row["game_id"])
        if gid not in total_lines:
            continue
        market = total_lines[gid]
        edge = dev_preds[idx] - market["total_line"]
        if abs(edge) < 1.5:
            continue

        side_over = edge > 0
        if side_over:
            correct = row["total_runs"] > market["total_line"]
            dec_odds = american_to_decimal(market["over_odds"])
        else:
            correct = row["total_runs"] < market["total_line"]
            dec_odds = american_to_decimal(market["under_odds"])

        if row["total_runs"] == market["total_line"]:
            continue

        bets += 1
        if correct:
            wins += 1
            profit += 100 * (dec_odds - 1)
        else:
            profit -= 100

    roi = profit / (bets * 100) * 100 if bets > 0 else 0
    mae = mean_absolute_error(dev["total_runs"], dev_preds)
    win_rate = wins / bets if bets > 0 else 0

    print(f"  {label}:")
    print(f"    Features: {len(available)}, Bets: {bets}, Win: {win_rate:.1%}, ROI: {roi:+.1f}%, MAE: {mae:.3f}")

    # Per-year breakdown
    for yr in range(dev_years[0], dev_years[1] + 1):
        yr_dev = dev[dev["season"] == yr]
        yr_preds = dev_preds[dev["season"].values == yr]
        yr_bets = 0
        yr_wins = 0
        yr_profit = 0
        for i, (_, row) in enumerate(yr_dev.iterrows()):
            gid = int(row["game_id"])
            if gid not in total_lines:
                continue
            market = total_lines[gid]
            edge = yr_preds[i] - market["total_line"]
            if abs(edge) < 1.5:
                continue
            side_over = edge > 0
            if side_over:
                correct = row["total_runs"] > market["total_line"]
                dec_odds = american_to_decimal(market["over_odds"])
            else:
                correct = row["total_runs"] < market["total_line"]
                dec_odds = american_to_decimal(market["under_odds"])
            if row["total_runs"] == market["total_line"]:
                continue
            yr_bets += 1
            if correct:
                yr_wins += 1
                yr_profit += 100 * (dec_odds - 1)
            else:
                yr_profit -= 100
        yr_roi = yr_profit / (yr_bets * 100) * 100 if yr_bets > 0 else 0
        if yr_bets > 0:
            print(f"      {yr}: {yr_bets} bets, {yr_wins/yr_bets:.1%} win, ROI={yr_roi:+.1f}%")

    return bets, win_rate, roi, mae


def main():
    df = build_enhanced_features()
    total_lines = get_total_lines()

    print(f"\n{'='*60}")
    print("  TOTALS V2: DEV SET EVALUATION (2019-2022)")
    print("  Train: 2016-2018, Dev: 2019-2022")
    print("  Holdout (2023-2024) NOT TOUCHED")
    print(f"{'='*60}\n")

    # V1: Baseline (box score features only)
    v1_feats = V1_FEATURES
    evaluate_model(df, v1_feats, "V1 (baseline)", total_lines, (2016, 2018), (2019, 2022))

    # V2: + Statcast pitcher features
    v2_feats = V1_FEATURES + STATCAST_FEATURES
    evaluate_model(df, v2_feats, "V2 (+Statcast)", total_lines, (2016, 2018), (2019, 2022))

    # V3: + Lineup features
    v3_feats = V1_FEATURES + STATCAST_FEATURES + LINEUP_FEATURES
    evaluate_model(df, v3_feats, "V3 (+Statcast +Lineup)", total_lines, (2016, 2018), (2019, 2022))

    # V4: Lineup replaces team batting (test if lineup subsumes team)
    v4_feats = [f for f in V1_FEATURES if not f.startswith(("home_b_", "away_b_"))] + STATCAST_FEATURES + LINEUP_FEATURES
    evaluate_model(df, v4_feats, "V4 (Lineup replaces team)", total_lines, (2016, 2018), (2019, 2022))

    # Also test with expanding window (more training data)
    print(f"\n{'='*60}")
    print("  EXPANDING WINDOW (more training data)")
    print(f"{'='*60}\n")

    for label, feats in [
        ("V1 baseline", V1_FEATURES),
        ("V2 +Statcast", V1_FEATURES + STATCAST_FEATURES),
        ("V3 +Statcast +Lineup", V1_FEATURES + STATCAST_FEATURES + LINEUP_FEATURES),
    ]:
        # Expanding window: for each dev year, train on all prior
        total_bets = 0
        total_wins = 0
        total_profit = 0

        for test_yr in range(2019, 2023):
            available = [c for c in feats if c in df.columns]
            train = df[df["season"].between(2016, test_yr - 1)].copy()
            test = df[df["season"] == test_yr].copy()

            medians = train[available].median()
            X_tr = train[available].fillna(medians)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            mdl = LinearRegression()
            mdl.fit(X_tr_s, train["total_runs"])

            X_te = test[available].fillna(medians)
            preds = mdl.predict(sc.transform(X_te))

            for idx, (_, row) in enumerate(test.iterrows()):
                gid = int(row["game_id"])
                if gid not in total_lines:
                    continue
                market = total_lines[gid]
                edge = preds[idx] - market["total_line"]
                if abs(edge) < 1.5:
                    continue
                side_over = edge > 0
                if side_over:
                    correct = row["total_runs"] > market["total_line"]
                    dec_odds = american_to_decimal(market["over_odds"])
                else:
                    correct = row["total_runs"] < market["total_line"]
                    dec_odds = american_to_decimal(market["under_odds"])
                if row["total_runs"] == market["total_line"]:
                    continue
                total_bets += 1
                if correct:
                    total_wins += 1
                    total_profit += 100 * (dec_odds - 1)
                else:
                    total_profit -= 100

        roi = total_profit / (total_bets * 100) * 100 if total_bets > 0 else 0
        wr = total_wins / total_bets if total_bets > 0 else 0
        print(f"  {label}: {total_bets} bets, {wr:.1%} win, ROI={roi:+.1f}%")


if __name__ == "__main__":
    main()
