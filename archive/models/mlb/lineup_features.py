"""
models/mlb/lineup_features.py - Lineup-specific features for player-level game model.

Replaces generic team rolling stats with features specific to the 9 batters
actually in the lineup for each game. Weighted by batting order position.

Features per lineup:
    - Lineup aggregate wOBA, K%, BB%, ISO (rolling per-player, then aggregated)
    - Lineup Statcast contact quality (exit velo, hard hit rate, xBA)
    - Batting order quality (top-of-order vs bottom-of-order strength)
    - Lineup vs opposing starter history (if sufficient sample)

Usage:
    from models.mlb.lineup_features import build_lineup_features
    features = build_lineup_features()  # dict: (game_id, team_id) -> feat dict
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from tqdm import tqdm

from db.db import query


# Batting order weights — top of order gets more PAs and has more leverage
ORDER_WEIGHTS = {1: 1.15, 2: 1.12, 3: 1.10, 4: 1.08, 5: 1.05, 6: 1.0, 7: 0.95, 8: 0.90, 9: 0.85}


def load_player_game_stats():
    """Load per-player-per-game batting stats with Statcast."""
    print("  Loading player game stats...")
    df = query("""
        SELECT
            bg.game_id, bg.player_id, bg.team_id, bg.batting_order,
            bg.pa, bg.ab, bg.hits, bg.doubles, bg.triples, bg.hr,
            bg.bb, bg.so, bg.hbp,
            g.game_date, g.home_team_id, g.away_team_id,
            g.season_id, s.year as season
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE bg.batting_order BETWEEN 1 AND 9
          AND bg.pa > 0
          AND g.sport_id = 2 AND g.status = 'final'
        ORDER BY g.game_date, bg.game_id, bg.batting_order
    """)
    print(f"    {len(df)} rows")
    return df


def load_player_statcast():
    """Load per-player Statcast aggregates."""
    print("  Loading player Statcast aggregates...")
    df = query("""
        SELECT
            p.batter_id as player_id, p.game_id,
            AVG(CASE WHEN p.is_in_play AND p.launch_speed IS NOT NULL THEN p.launch_speed END) as exit_velo,
            AVG(CASE WHEN p.is_in_play THEN p.xba END) as xba,
            SUM(CASE WHEN p.is_in_play AND p.launch_speed >= 95 THEN 1 ELSE 0 END) as hard_hits,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END) as bip,
            SUM(CASE WHEN p.is_whiff THEN 1 ELSE 0 END) as whiffs,
            SUM(CASE WHEN p.is_swing THEN 1 ELSE 0 END) as swings
        FROM mlb_pitches p
        GROUP BY p.batter_id, p.game_id
    """)
    print(f"    {len(df)} rows")
    return df


def build_player_rolling_stats(batting_df, statcast_df):
    """Build rolling per-player stats. Returns dict: (game_id, player_id) -> stats."""
    print("  Building per-player rolling stats...")

    # Merge batting with Statcast
    merged = batting_df.merge(statcast_df, on=["player_id", "game_id"], how="left")

    # Compute per-game metrics
    merged["tb"] = merged["hits"] + merged["doubles"] + 2 * merged["triples"] + 3 * merged["hr"]
    merged["singles"] = merged["hits"] - merged["doubles"] - merged["triples"] - merged["hr"]

    # wOBA weights
    W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.69, 0.72, 0.88, 1.27, 1.62, 2.10
    merged["woba_num"] = (W_BB * merged["bb"] + W_HBP * merged["hbp"].fillna(0) +
                          W_1B * merged["singles"].clip(lower=0) + W_2B * merged["doubles"] +
                          W_3B * merged["triples"] + W_HR * merged["hr"])
    merged["woba_den"] = merged["ab"] + merged["bb"] + merged["hbp"].fillna(0)
    merged["hard_hit_rate"] = merged["hard_hits"] / merged["bip"].replace(0, np.nan)
    merged["whiff_rate"] = merged["whiffs"] / merged["swings"].replace(0, np.nan)

    merged = merged.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    player_stats = {}

    for pid, grp in tqdm(merged.groupby("player_id"), desc="  Players", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 10:
                player_stats[(gid, int(pid))] = None
                continue

            last20 = prior.tail(20)
            ab = last20["ab"].sum()
            pa = last20["pa"].sum()

            if ab == 0 or pa == 0:
                player_stats[(gid, int(pid))] = None
                continue

            stats = {
                "p_woba": round(last20["woba_num"].sum() / last20["woba_den"].sum(), 4) if last20["woba_den"].sum() > 0 else None,
                "p_kpct": round(last20["so"].sum() / pa, 4),
                "p_bbpct": round(last20["bb"].sum() / pa, 4),
                "p_iso": round((last20["tb"].sum() - last20["hits"].sum()) / ab, 4),
                "p_avg": round(last20["hits"].sum() / ab, 4),
                "p_exit_velo": round(last20["exit_velo"].dropna().mean(), 1) if last20["exit_velo"].notna().any() else None,
                "p_xba": round(last20["xba"].dropna().mean(), 4) if last20["xba"].notna().any() else None,
                "p_hard_hit": round(last20["hard_hit_rate"].dropna().mean(), 4) if last20["hard_hit_rate"].notna().any() else None,
            }
            player_stats[(gid, int(pid))] = stats

    return player_stats


def build_lineup_features(player_stats=None, batting_df=None, statcast_df=None):
    """
    Build lineup-level features by aggregating per-player stats for each game's lineup.
    Returns dict: (game_id, team_id) -> feature dict
    """
    print("\n=== LINEUP FEATURE ENGINEERING ===")

    if batting_df is None:
        batting_df = load_player_game_stats()
    if statcast_df is None:
        statcast_df = load_player_statcast()
    if player_stats is None:
        player_stats = build_player_rolling_stats(batting_df, statcast_df)

    print("  Aggregating lineup features...")

    # Get the starting lineup (batting_order 1-9) for each game/team
    # Take the first player at each position (starter, not substitute)
    starters = batting_df.sort_values(["game_id", "team_id", "batting_order", "pa"], ascending=[True, True, True, False])
    starters = starters.drop_duplicates(subset=["game_id", "team_id", "batting_order"], keep="first")

    lineup_features = {}
    games = starters.groupby(["game_id", "team_id"])

    for (gid, tid), lineup in tqdm(games, desc="  Lineups", leave=False):
        lineup = lineup.sort_values("batting_order")

        if len(lineup) < 7:  # need at least 7 of 9 spots filled
            lineup_features[(int(gid), int(tid))] = _empty_lineup_features()
            continue

        # Collect per-player stats with batting order weights
        woba_vals, kpct_vals, iso_vals, ev_vals, xba_vals, hh_vals = [], [], [], [], [], []
        weights = []
        players_with_stats = 0

        for _, batter in lineup.iterrows():
            pid = int(batter["player_id"])
            order = int(batter["batting_order"])
            weight = ORDER_WEIGHTS.get(order, 1.0)

            ps = player_stats.get((int(gid), pid))
            if ps is None:
                continue

            players_with_stats += 1
            weights.append(weight)

            if ps["p_woba"] is not None:
                woba_vals.append(ps["p_woba"] * weight)
            if ps["p_kpct"] is not None:
                kpct_vals.append(ps["p_kpct"] * weight)
            if ps["p_iso"] is not None:
                iso_vals.append(ps["p_iso"] * weight)
            if ps["p_exit_velo"] is not None:
                ev_vals.append(ps["p_exit_velo"] * weight)
            if ps["p_xba"] is not None:
                xba_vals.append(ps["p_xba"] * weight)
            if ps["p_hard_hit"] is not None:
                hh_vals.append(ps["p_hard_hit"] * weight)

        if players_with_stats < 5:
            lineup_features[(int(gid), int(tid))] = _empty_lineup_features()
            continue

        total_weight = sum(weights)
        feat = {
            "lu_woba": round(sum(woba_vals) / total_weight, 4) if woba_vals else None,
            "lu_kpct": round(sum(kpct_vals) / total_weight, 4) if kpct_vals else None,
            "lu_iso": round(sum(iso_vals) / total_weight, 4) if iso_vals else None,
            "lu_exit_velo": round(sum(ev_vals) / total_weight, 1) if ev_vals else None,
            "lu_xba": round(sum(xba_vals) / total_weight, 4) if xba_vals else None,
            "lu_hard_hit": round(sum(hh_vals) / total_weight, 4) if hh_vals else None,
            "lu_players_with_stats": players_with_stats,
        }

        # Top-of-order vs bottom-of-order split
        top = [player_stats.get((int(gid), int(b["player_id"]))) for _, b in lineup.iterrows() if b["batting_order"] <= 4]
        bot = [player_stats.get((int(gid), int(b["player_id"]))) for _, b in lineup.iterrows() if b["batting_order"] >= 6]
        top = [s for s in top if s is not None and s.get("p_woba") is not None]
        bot = [s for s in bot if s is not None and s.get("p_woba") is not None]

        if top and bot:
            top_woba = np.mean([s["p_woba"] for s in top])
            bot_woba = np.mean([s["p_woba"] for s in bot])
            feat["lu_top_bot_gap"] = round(top_woba - bot_woba, 4)
        else:
            feat["lu_top_bot_gap"] = None

        lineup_features[(int(gid), int(tid))] = feat

    print(f"  Built {len(lineup_features)} lineup feature sets")
    return lineup_features


def _empty_lineup_features():
    return {k: None for k in [
        "lu_woba", "lu_kpct", "lu_iso", "lu_exit_velo", "lu_xba",
        "lu_hard_hit", "lu_players_with_stats", "lu_top_bot_gap",
    ]}


if __name__ == "__main__":
    features = build_lineup_features()
    # Sample
    sample_key = [k for k in features.keys() if features[k].get("lu_woba") is not None][100]
    print(f"\nSample lineup features (game={sample_key[0]}, team={sample_key[1]}):")
    for k, v in features[sample_key].items():
        print(f"  {k:25s} {v}")
