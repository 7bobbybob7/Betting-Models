"""
models/mlb/statcast_features.py - Aggregate Statcast pitch-level data into
per-pitcher and per-team features for the K prop model.

Features built:
    Pitcher (from mlb_pitches):
        - Overall whiff rate, swing rate, chase rate
        - Whiff rate by pitch type (FF, SL, CH, CU, etc.)
        - Pitch mix composition (% of each pitch type)
        - Average fastball velocity (+ recent trend)
        - K rate by zone (in-zone vs out-of-zone)
        - Swinging strike rate
        - Called strike + swinging strike rate

    Team batting (opposing lineup aggregate):
        - Team K rate (from mlb_batting_game)
        - Team whiff rate, chase rate, swing rate (from mlb_pitches as batters)

Usage:
    from models.mlb.statcast_features import build_statcast_features
    pitcher_feats, team_feats = build_statcast_features()
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from tqdm import tqdm

from db.db import query


# Major pitch type groups
PITCH_GROUPS = {
    "fastball": ["FF", "SI", "FC", "FA"],
    "breaking": ["SL", "CU", "KC", "ST", "SV", "KN"],
    "offspeed": ["CH", "FS", "FO", "EP"],
}

TRACKED_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "ST"]


def load_pitch_data():
    """Load aggregated pitch stats per pitcher per game."""
    print("  Loading pitch-level aggregates...")
    df = query("""
        SELECT
            p.game_id,
            p.pitcher_id,
            g.game_date,
            g.season_id,
            g.home_team_id,
            g.away_team_id,
            pg.team_id as pitcher_team_id,
            pg.so as game_k,
            pg.ip,
            COUNT(*) as total_pitches,
            SUM(CASE WHEN p.is_swing THEN 1 ELSE 0 END) as swings,
            SUM(CASE WHEN p.is_whiff THEN 1 ELSE 0 END) as whiffs,
            SUM(CASE WHEN p.is_strike THEN 1 ELSE 0 END) as strikes,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END) as in_play,
            SUM(CASE WHEN p.zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) as in_zone,
            SUM(CASE WHEN p.zone >= 11 THEN 1 ELSE 0 END) as out_zone,
            SUM(CASE WHEN p.zone >= 11 AND p.is_swing THEN 1 ELSE 0 END) as chase,
            SUM(CASE WHEN p.zone BETWEEN 1 AND 9 AND p.is_whiff THEN 1 ELSE 0 END) as zone_whiff,
            SUM(CASE WHEN p.zone >= 11 AND p.is_whiff THEN 1 ELSE 0 END) as chase_whiff,
            AVG(CASE WHEN p.pitch_type IN ('FF', 'SI', 'FC') THEN p.release_speed END) as avg_fb_velo,
            -- Pitch type counts
            SUM(CASE WHEN p.pitch_type = 'FF' THEN 1 ELSE 0 END) as n_ff,
            SUM(CASE WHEN p.pitch_type = 'SL' THEN 1 ELSE 0 END) as n_sl,
            SUM(CASE WHEN p.pitch_type = 'CH' THEN 1 ELSE 0 END) as n_ch,
            SUM(CASE WHEN p.pitch_type = 'CU' THEN 1 ELSE 0 END) as n_cu,
            SUM(CASE WHEN p.pitch_type = 'SI' THEN 1 ELSE 0 END) as n_si,
            SUM(CASE WHEN p.pitch_type = 'FC' THEN 1 ELSE 0 END) as n_fc,
            SUM(CASE WHEN p.pitch_type = 'ST' THEN 1 ELSE 0 END) as n_st,
            -- Whiffs by pitch type
            SUM(CASE WHEN p.pitch_type = 'FF' AND p.is_whiff THEN 1 ELSE 0 END) as whiff_ff,
            SUM(CASE WHEN p.pitch_type = 'SL' AND p.is_whiff THEN 1 ELSE 0 END) as whiff_sl,
            SUM(CASE WHEN p.pitch_type = 'CH' AND p.is_whiff THEN 1 ELSE 0 END) as whiff_ch,
            SUM(CASE WHEN p.pitch_type = 'CU' AND p.is_whiff THEN 1 ELSE 0 END) as whiff_cu,
            -- Swings by pitch type
            SUM(CASE WHEN p.pitch_type = 'FF' AND p.is_swing THEN 1 ELSE 0 END) as swing_ff,
            SUM(CASE WHEN p.pitch_type = 'SL' AND p.is_swing THEN 1 ELSE 0 END) as swing_sl,
            SUM(CASE WHEN p.pitch_type = 'CH' AND p.is_swing THEN 1 ELSE 0 END) as swing_ch,
            SUM(CASE WHEN p.pitch_type = 'CU' AND p.is_swing THEN 1 ELSE 0 END) as swing_cu
        FROM mlb_pitches p
        JOIN games g ON p.game_id = g.game_id
        JOIN mlb_pitching_game pg ON p.game_id = pg.game_id
            AND p.pitcher_id = pg.player_id AND pg.is_starter = true
        GROUP BY p.game_id, p.pitcher_id, g.game_date, g.season_id,
                 g.home_team_id, g.away_team_id, pg.team_id, pg.so, pg.ip
        ORDER BY g.game_date, p.game_id
    """)
    print(f"    {len(df)} pitcher-game rows loaded")
    return df


def load_team_batting_k_rates():
    """Load team K rates per game for opposing lineup features."""
    print("  Loading team batting K rates...")
    df = query("""
        SELECT
            bg.game_id, bg.team_id, g.game_date,
            SUM(bg.so) as team_so,
            SUM(bg.pa) as team_pa,
            SUM(bg.ab) as team_ab
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        GROUP BY bg.game_id, bg.team_id, g.game_date
        ORDER BY g.game_date
    """)
    print(f"    {len(df)} team-game rows loaded")
    return df


def build_pitcher_statcast_features(pitch_df):
    """
    Build rolling Statcast features per pitcher.
    Returns dict: (game_id, pitcher_id) -> feature dict
    """
    print("  Building pitcher Statcast features...")
    pitch_df = pitch_df.sort_values("game_date").copy()

    features = {}

    for pid, grp in tqdm(pitch_df.groupby("pitcher_id"), desc="  Pitchers", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 3:
                features[(gid, int(pid))] = _empty_statcast_pitcher()
                continue

            feat = {}

            # Rolling windows
            for window, suffix in [(5, "5"), (10, "10")]:
                w = prior.tail(window)
                tp = w["total_pitches"].sum()
                sw = w["swings"].sum()
                wh = w["whiffs"].sum()
                oz = w["out_zone"].sum()
                ch = w["chase"].sum()

                feat[f"sc_whiff_rate_{suffix}"] = _sdiv(wh, sw)
                feat[f"sc_swstr_rate_{suffix}"] = _sdiv(wh, tp)  # swinging strike rate
                feat[f"sc_chase_rate_{suffix}"] = _sdiv(ch, oz)
                feat[f"sc_zone_rate_{suffix}"] = _sdiv(w["in_zone"].sum(), tp)

                # K rate per start
                feat[f"sc_k_per_start_{suffix}"] = round(w["game_k"].mean(), 2) if w["game_k"].notna().any() else None

                # Fastball velo
                velos = w["avg_fb_velo"].dropna()
                feat[f"sc_fb_velo_{suffix}"] = round(velos.mean(), 1) if len(velos) > 0 else None

            # Pitch mix (last 5 starts)
            w5 = prior.tail(5)
            tp5 = w5["total_pitches"].sum()
            if tp5 > 0:
                feat["sc_pct_ff"] = _sdiv(w5["n_ff"].sum(), tp5)
                feat["sc_pct_sl"] = _sdiv(w5["n_sl"].sum(), tp5)
                feat["sc_pct_ch"] = _sdiv(w5["n_ch"].sum(), tp5)
                feat["sc_pct_cu"] = _sdiv(w5["n_cu"].sum(), tp5)
                feat["sc_pct_si"] = _sdiv(w5["n_si"].sum(), tp5)
                feat["sc_pct_fc"] = _sdiv(w5["n_fc"].sum(), tp5)
            else:
                for pt in ["ff", "sl", "ch", "cu", "si", "fc"]:
                    feat[f"sc_pct_{pt}"] = None

            # Whiff rate by pitch type (last 5)
            for pt in ["ff", "sl", "ch", "cu"]:
                swings_pt = w5[f"swing_{pt}"].sum()
                whiffs_pt = w5[f"whiff_{pt}"].sum()
                feat[f"sc_whiff_{pt}"] = _sdiv(whiffs_pt, swings_pt)

            # Velo trend (last 3 vs last 10)
            v3 = prior.tail(3)["avg_fb_velo"].dropna()
            v10 = prior.tail(10)["avg_fb_velo"].dropna()
            if len(v3) > 0 and len(v10) > 0:
                feat["sc_velo_trend"] = round(v3.mean() - v10.mean(), 1)
            else:
                feat["sc_velo_trend"] = None

            features[(gid, int(pid))] = feat

    return features


def build_team_k_features(team_batting_df):
    """
    Build rolling team K-rate features (for opposing lineup).
    Returns dict: (game_id, team_id) -> feature dict
    """
    print("  Building team K-rate features...")
    team_batting_df = team_batting_df.sort_values("game_date").copy()

    features = {}

    for tid, grp in tqdm(team_batting_df.groupby("team_id"), desc="  Teams", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 5:
                features[(gid, int(tid))] = {"opp_k_rate_15": None, "opp_k_rate_30": None}
                continue

            feat = {}
            for window, suffix in [(15, "15"), (30, "30")]:
                w = prior.tail(window)
                feat[f"opp_k_rate_{suffix}"] = _sdiv(w["team_so"].sum(), w["team_pa"].sum())

            features[(gid, int(tid))] = feat

    return features


def _empty_statcast_pitcher():
    feat = {}
    for suffix in ["5", "10"]:
        for k in ["sc_whiff_rate", "sc_swstr_rate", "sc_chase_rate", "sc_zone_rate",
                   "sc_k_per_start", "sc_fb_velo"]:
            feat[f"{k}_{suffix}"] = None
    for pt in ["ff", "sl", "ch", "cu", "si", "fc"]:
        feat[f"sc_pct_{pt}"] = None
    for pt in ["ff", "sl", "ch", "cu"]:
        feat[f"sc_whiff_{pt}"] = None
    feat["sc_velo_trend"] = None
    return feat


def _sdiv(num, denom):
    if denom is None or denom == 0:
        return None
    return round(num / denom, 4)


def build_statcast_features():
    """Build all Statcast features. Returns (pitcher_feats, team_feats) dicts."""
    print("\n=== STATCAST FEATURE ENGINEERING ===")
    pitch_df = load_pitch_data()
    team_bat_df = load_team_batting_k_rates()

    pitcher_feats = build_pitcher_statcast_features(pitch_df)
    team_feats = build_team_k_features(team_bat_df)

    print(f"\n  Pitcher features: {len(pitcher_feats)} entries")
    print(f"  Team K-rate features: {len(team_feats)} entries")

    return pitcher_feats, team_feats


if __name__ == "__main__":
    p, t = build_statcast_features()
    # Sample output
    sample_key = list(p.keys())[1000]
    print(f"\nSample pitcher features (game_id={sample_key[0]}, pitcher={sample_key[1]}):")
    for k, v in p[sample_key].items():
        print(f"  {k:25s} {v}")
