"""
models/mlb/batter_arsenal_features.py — Per-batter Statcast + box-score features for hitter prop modeling.

Per-pitch-type 90-day rolling features (FF/SI/FC/SL/ST/SV/CU/KC/CH/FS/FO) with bucket fallbacks
(FB/BR/OS), plate discipline, vs-handedness splits, and recent box-score form.

STRICT NO-LEAKAGE CONTRACT
    Every rolling window is HALF-OPEN: [as_of_date - 90D, as_of_date). The prediction
    date's own stats are EXCLUDED. Enforced via pandas rolling closed='left' AND the
    `g.game_date < as_of_date` SQL bound on the as-of grid.

API
    build_training_set(start_date, end_date)   -> DataFrame   per-batter per-game over [start, end]
    compute_batch(as_of_date, player_ids=None) -> DataFrame   one as-of, all active batters
    compute_for(player_id, as_of_date)         -> dict        single-row predict-time helper

Notes
    HRR label (hits + runs + rbi) requires a `runs` column on mlb_batting_game which is
    currently NOT captured. This module exposes `bat_h_plus_rbi_per_pa_30d` as a proxy
    until a runs backfill lands. Pull rate is replaced with GB/LD/FB rate (we don't store
    hit location in mlb_pitches).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import argparse
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from db.db import query


# ----------------------------------------------------------------------------
# Pitch grouping
# ----------------------------------------------------------------------------

INDIVIDUAL_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CH', 'FS', 'FO']

# Bucket assignments — coarse fallback when per-type sample is thin.
# v1 decision: cutter (FC) groups with breaking since most modern cutters behave slider-like.
BUCKETS = {
    'FB': ['FF', 'SI'],
    'BR': ['SL', 'ST', 'SV', 'CU', 'KC', 'FC'],
    'OS': ['CH', 'FS', 'FO'],
}
PITCH_TO_BUCKET = {pt: bk for bk, pts in BUCKETS.items() for pt in pts}


# ----------------------------------------------------------------------------
# Minimum sample requirements — below threshold returns NaN (model imputes)
# ----------------------------------------------------------------------------

MIN_BIP_INDIVIDUAL    = 20   # xwOBA per individual pitch type
MIN_BIP_BUCKET        = 60   # xwOBA per bucket
MIN_SWINGS_INDIVIDUAL = 30   # whiff rate per individual pitch type
MIN_SWINGS_BUCKET     = 80   # whiff rate per bucket
MIN_OZ_PITCHES        = 100  # chase rate
MIN_Z_PITCHES         = 100  # zone-swing rate
MIN_BIP_BATTED_BALL   = 30   # hard-hit/barrel/launch-angle distribution
MIN_BIP_HAND          = 50   # vs RHP/LHP xwOBA
MIN_SWINGS_HAND       = 80   # vs RHP/LHP whiff rate
MIN_SWINGS_OVERALL    = 100  # overall contact rate
MIN_PA_30D            = 30   # recent-form per-PA rates
MIN_AB_30D            = 30   # ISO
MIN_GP_15D            = 5    # PA/game

ROLLING_PITCH_WINDOW   = '90D'
ROLLING_FORM_30D       = '30D'
ROLLING_FORM_15D       = '15D'


# ----------------------------------------------------------------------------
# Raw aggregate pulls — one row per (batter, game_date, [pitch_type|hand])
# Always pulls (start - 90 days) so rolling windows have history at start.
# ----------------------------------------------------------------------------

def _pull_pitch_aggregates(start: date, end: date) -> pd.DataFrame:
    """Per (batter, game_date, pitch_type): pitches/swings/whiffs/bip/xwoba_sum."""
    sql = """
        SELECT
            p.batter_id        AS player_id,
            g.game_date,
            p.pitch_type,
            COUNT(*)                                              AS pitches,
            SUM(CASE WHEN p.is_swing   THEN 1 ELSE 0 END)         AS swings,
            SUM(CASE WHEN p.is_whiff   THEN 1 ELSE 0 END)         AS whiffs,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END)         AS bip,
            SUM(CASE WHEN p.is_in_play THEN p.xwoba ELSE 0 END)   AS xwoba_sum
        FROM mlb_pitches p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND p.pitch_type IS NOT NULL
        GROUP BY p.batter_id, g.game_date, p.pitch_type
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_plate_discipline(start: date, end: date) -> pd.DataFrame:
    """Per (batter, game_date): pitch-level discipline + batted-ball quality."""
    sql = """
        SELECT
            p.batter_id AS player_id,
            g.game_date,
            COUNT(*)                                                  AS pitches,
            SUM(CASE WHEN p.is_swing   THEN 1 ELSE 0 END)             AS swings,
            SUM(CASE WHEN p.is_whiff   THEN 1 ELSE 0 END)             AS whiffs,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END)             AS bip,
            SUM(CASE WHEN p.zone BETWEEN 1  AND 9  THEN 1 ELSE 0 END) AS z_pitches,
            SUM(CASE WHEN p.zone BETWEEN 11 AND 14 THEN 1 ELSE 0 END) AS oz_pitches,
            SUM(CASE WHEN p.is_swing AND p.zone BETWEEN 1  AND 9  THEN 1 ELSE 0 END) AS z_swings,
            SUM(CASE WHEN p.is_swing AND p.zone BETWEEN 11 AND 14 THEN 1 ELSE 0 END) AS oz_swings,
            SUM(CASE WHEN p.is_in_play AND p.launch_speed >= 95                          THEN 1 ELSE 0 END) AS hard_hits,
            SUM(CASE WHEN p.is_in_play AND p.launch_speed >= 98
                                       AND p.launch_angle BETWEEN 26 AND 30             THEN 1 ELSE 0 END) AS barrels,
            SUM(CASE WHEN p.is_in_play AND p.launch_angle < 10                            THEN 1 ELSE 0 END) AS gb,
            SUM(CASE WHEN p.is_in_play AND p.launch_angle BETWEEN 10 AND 25               THEN 1 ELSE 0 END) AS ld,
            SUM(CASE WHEN p.is_in_play AND p.launch_angle > 25                            THEN 1 ELSE 0 END) AS fb_hits
        FROM mlb_pitches p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
        GROUP BY p.batter_id, g.game_date
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_handedness_aggregates(start: date, end: date) -> pd.DataFrame:
    """Per (batter, game_date, pitcher_throws): bip/xwoba_sum/swings/whiffs."""
    sql = """
        SELECT
            p.batter_id AS player_id,
            g.game_date,
            pl.throws   AS p_throws,
            SUM(CASE WHEN p.is_swing   THEN 1 ELSE 0 END)         AS swings,
            SUM(CASE WHEN p.is_whiff   THEN 1 ELSE 0 END)         AS whiffs,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END)         AS bip,
            SUM(CASE WHEN p.is_in_play THEN p.xwoba ELSE 0 END)   AS xwoba_sum
        FROM mlb_pitches p
        JOIN games   g  ON p.game_id    = g.game_id
        JOIN players pl ON p.pitcher_id = pl.player_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND pl.throws IN ('L', 'R')
        GROUP BY p.batter_id, g.game_date, pl.throws
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_box_score_form(start: date, end: date) -> pd.DataFrame:
    """Per (batter, game_date): box-score line. NB: runs scored is NOT in our schema."""
    sql = """
        SELECT
            bg.player_id,
            g.game_date,
            bg.pa, bg.ab, bg.hits, bg.doubles, bg.triples, bg.hr,
            bg.rbi, bg.bb, bg.so
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND bg.pa > 0
    """
    return query(sql, params={'start': start, 'end': end})


# ----------------------------------------------------------------------------
# Rolling — left-closed window per batter so as-of date is EXCLUDED
# ----------------------------------------------------------------------------

def _roll_sum(df: pd.DataFrame, value_cols: list, window: str,
              group_col: str = 'player_id') -> pd.DataFrame:
    """For each group, sum value_cols over a half-open [date - window, date) window.
    `closed='left'` is the leakage barrier: the current date row never contributes."""
    if df.empty:
        return df.copy()
    df = df.sort_values([group_col, 'game_date'])
    out_chunks = []
    for grp_id, grp in df.groupby(group_col, sort=False):
        rolled = (grp.set_index('game_date')[value_cols]
                     .rolling(window, closed='left')
                     .sum()
                     .reset_index())
        rolled[group_col] = grp_id
        out_chunks.append(rolled)
    return pd.concat(out_chunks, ignore_index=True)


def _safe_div(num: pd.Series, den: pd.Series, min_den: float) -> np.ndarray:
    """Compute num/den only where den >= min_den, else NaN."""
    den_filled = den.fillna(0)
    mask = den_filled >= min_den
    result = np.full(len(num), np.nan)
    safe_den = den.where(mask, np.nan)
    result = np.where(mask, num / safe_den, np.nan)
    return result


# ----------------------------------------------------------------------------
# Feature assemblers — each produces (player_id, game_date, bat_* features)
# ----------------------------------------------------------------------------

def _assemble_arsenal(pitch_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch-type xwOBA + whiff rate over 90d, plus bucket-level fallbacks."""
    out = as_of.copy()

    # Individual pitch types
    for pt in INDIVIDUAL_PITCH_TYPES:
        sub = pitch_agg[pitch_agg['pitch_type'] == pt]
        if sub.empty:
            out[f'bat_xwoba_vs_{pt}_90d']      = np.nan
            out[f'bat_whiff_rate_vs_{pt}_90d'] = np.nan
            continue
        rolled = _roll_sum(sub, ['swings', 'whiffs', 'bip', 'xwoba_sum'], ROLLING_PITCH_WINDOW)
        rolled = rolled.rename(columns={
            'swings': f'sw_{pt}', 'whiffs': f'wh_{pt}',
            'bip': f'bip_{pt}', 'xwoba_sum': f'xs_{pt}',
        })
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out[f'bat_xwoba_vs_{pt}_90d']      = _safe_div(out[f'xs_{pt}'], out[f'bip_{pt}'], MIN_BIP_INDIVIDUAL)
        out[f'bat_whiff_rate_vs_{pt}_90d'] = _safe_div(out[f'wh_{pt}'], out[f'sw_{pt}'], MIN_SWINGS_INDIVIDUAL)

    # Bucket fallbacks
    pitch_agg_with_bucket = pitch_agg.copy()
    pitch_agg_with_bucket['bucket'] = pitch_agg_with_bucket['pitch_type'].map(PITCH_TO_BUCKET)
    pitch_agg_with_bucket = pitch_agg_with_bucket.dropna(subset=['bucket'])

    for bk in BUCKETS:
        sub = (pitch_agg_with_bucket[pitch_agg_with_bucket['bucket'] == bk]
               .groupby(['player_id', 'game_date'], as_index=False)
               [['swings', 'whiffs', 'bip', 'xwoba_sum']].sum())
        if sub.empty:
            out[f'bat_xwoba_vs_{bk}_90d']      = np.nan
            out[f'bat_whiff_rate_vs_{bk}_90d'] = np.nan
            continue
        rolled = _roll_sum(sub, ['swings', 'whiffs', 'bip', 'xwoba_sum'], ROLLING_PITCH_WINDOW)
        rolled = rolled.rename(columns={
            'swings': f'sw_{bk}', 'whiffs': f'wh_{bk}',
            'bip': f'bip_{bk}', 'xwoba_sum': f'xs_{bk}',
        })
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out[f'bat_xwoba_vs_{bk}_90d']      = _safe_div(out[f'xs_{bk}'], out[f'bip_{bk}'], MIN_BIP_BUCKET)
        out[f'bat_whiff_rate_vs_{bk}_90d'] = _safe_div(out[f'wh_{bk}'], out[f'sw_{bk}'], MIN_SWINGS_BUCKET)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('bat_')]
    return out[keep]


def _assemble_discipline(disc_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """Chase / zswing / contact / hard-hit / barrel / launch-angle distribution."""
    rolled = _roll_sum(
        disc_agg,
        ['pitches', 'swings', 'whiffs', 'bip',
         'z_pitches', 'oz_pitches', 'z_swings', 'oz_swings',
         'hard_hits', 'barrels', 'gb', 'ld', 'fb_hits'],
        ROLLING_PITCH_WINDOW,
    )
    out = as_of.merge(rolled, on=['player_id', 'game_date'], how='left')

    out['bat_chase_rate_90d']     = _safe_div(out['oz_swings'], out['oz_pitches'], MIN_OZ_PITCHES)
    out['bat_zswing_rate_90d']    = _safe_div(out['z_swings'],  out['z_pitches'],  MIN_Z_PITCHES)
    out['bat_contact_rate_90d']   = _safe_div(out['swings'] - out['whiffs'].fillna(0),
                                              out['swings'], MIN_SWINGS_OVERALL)
    out['bat_hard_hit_rate_90d']  = _safe_div(out['hard_hits'], out['bip'], MIN_BIP_BATTED_BALL)
    out['bat_barrel_rate_90d']    = _safe_div(out['barrels'],   out['bip'], MIN_BIP_BATTED_BALL)
    out['bat_gb_rate_90d']        = _safe_div(out['gb'],        out['bip'], MIN_BIP_BATTED_BALL)
    out['bat_ld_rate_90d']        = _safe_div(out['ld'],        out['bip'], MIN_BIP_BATTED_BALL)
    out['bat_fb_rate_90d']        = _safe_div(out['fb_hits'],   out['bip'], MIN_BIP_BATTED_BALL)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('bat_')]
    return out[keep]


def _assemble_handedness(hand_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """xwOBA and whiff rate vs RHP / vs LHP."""
    out = as_of.copy()
    for hand in ('R', 'L'):
        sub = hand_agg[hand_agg['p_throws'] == hand]
        if sub.empty:
            out[f'bat_xwoba_vs_{hand}HP_90d']      = np.nan
            out[f'bat_whiff_rate_vs_{hand}HP_90d'] = np.nan
            continue
        rolled = _roll_sum(sub, ['bip', 'xwoba_sum', 'swings', 'whiffs'], ROLLING_PITCH_WINDOW)
        rolled = rolled.rename(columns={
            'bip': f'bip_{hand}', 'xwoba_sum': f'xs_{hand}',
            'swings': f'sw_{hand}', 'whiffs': f'wh_{hand}',
        })
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out[f'bat_xwoba_vs_{hand}HP_90d']      = _safe_div(out[f'xs_{hand}'], out[f'bip_{hand}'], MIN_BIP_HAND)
        out[f'bat_whiff_rate_vs_{hand}HP_90d'] = _safe_div(out[f'wh_{hand}'], out[f'sw_{hand}'], MIN_SWINGS_HAND)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('bat_')]
    return out[keep]


def _assemble_form(box_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """Recent box-score form: PA/game (15d), per-PA rates over 30d."""
    box_agg = box_agg.copy()
    box_agg['gp'] = 1

    rolled30 = _roll_sum(
        box_agg,
        ['pa', 'ab', 'hits', 'doubles', 'triples', 'hr', 'rbi', 'bb', 'so'],
        ROLLING_FORM_30D,
    )
    rolled30 = rolled30.rename(columns={c: f'{c}_30d' for c in
                                        ['pa', 'ab', 'hits', 'doubles', 'triples', 'hr', 'rbi', 'bb', 'so']})

    rolled15 = _roll_sum(box_agg, ['pa', 'gp'], ROLLING_FORM_15D)
    rolled15 = rolled15.rename(columns={'pa': 'pa_15d', 'gp': 'gp_15d'})

    out = as_of.merge(rolled30, on=['player_id', 'game_date'], how='left')
    out = out.merge(rolled15, on=['player_id', 'game_date'], how='left')

    out['bat_pa_per_game_15d']      = _safe_div(out['pa_15d'], out['gp_15d'], MIN_GP_15D)
    out['bat_hits_per_pa_30d']      = _safe_div(out['hits_30d'], out['pa_30d'], MIN_PA_30D)
    out['bat_iso_30d']              = _safe_div(
        out['doubles_30d'] + 2 * out['triples_30d'] + 3 * out['hr_30d'],
        out['ab_30d'], MIN_AB_30D)
    out['bat_k_rate_30d']           = _safe_div(out['so_30d'], out['pa_30d'], MIN_PA_30D)
    out['bat_bb_rate_30d']          = _safe_div(out['bb_30d'], out['pa_30d'], MIN_PA_30D)
    # Proxy for HRR until runs-scored backfill lands
    out['bat_h_plus_rbi_per_pa_30d'] = _safe_div(out['hits_30d'] + out['rbi_30d'],
                                                  out['pa_30d'], MIN_PA_30D)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('bat_')]
    return out[keep]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_training_set(start_date: date, end_date: date) -> pd.DataFrame:
    """Build per-batter per-game feature rows over [start_date, end_date] inclusive.

    Each row's as-of-date is the row's game_date. Features are computed strictly
    from data with game_date < as_of_date.
    """
    pull_start = start_date - timedelta(days=92)  # 90-day window + small buffer
    print(f"[batter_arsenal] pulling raw aggregates {pull_start} -> {end_date}...")

    pitch_agg = _pull_pitch_aggregates(pull_start, end_date + timedelta(days=1))
    disc_agg  = _pull_plate_discipline(pull_start, end_date + timedelta(days=1))
    hand_agg  = _pull_handedness_aggregates(pull_start, end_date + timedelta(days=1))
    box_agg   = _pull_box_score_form(pull_start, end_date + timedelta(days=1))
    print(f"  pitches:    {len(pitch_agg):>8,} (batter, date, pitch_type)")
    print(f"  discipline: {len(disc_agg):>8,} (batter, date)")
    print(f"  handedness: {len(hand_agg):>8,} (batter, date, hand)")
    print(f"  box-form:   {len(box_agg):>8,} (batter, date)")

    # Normalize dtypes — all date columns to pandas datetime
    for df in (pitch_agg, disc_agg, hand_agg, box_agg):
        df['game_date'] = pd.to_datetime(df['game_date'])

    # The as-of grid: every (batter, game_date) where they actually played, within window
    as_of = (box_agg[['player_id', 'game_date']]
             .drop_duplicates()
             .sort_values(['player_id', 'game_date'])
             .reset_index(drop=True))
    as_of = as_of[as_of['game_date'] >= pd.Timestamp(start_date)]
    as_of = as_of[as_of['game_date'] <= pd.Timestamp(end_date)].reset_index(drop=True)
    print(f"  as-of grid: {len(as_of):>8,} (batter, game_date) rows in window")

    print("[batter_arsenal] assembling features...")
    arsenal     = _assemble_arsenal(pitch_agg, as_of)
    discipline  = _assemble_discipline(disc_agg, as_of)
    handedness  = _assemble_handedness(hand_agg, as_of)
    form        = _assemble_form(box_agg, as_of)

    out = as_of
    for block in (arsenal, discipline, handedness, form):
        out = out.merge(block, on=['player_id', 'game_date'], how='left')

    print(f"[batter_arsenal] done: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out


def compute_batch(as_of_date: date, player_ids: list | None = None) -> pd.DataFrame:
    """Features for a single as-of date (all batters who played that date)."""
    df = build_training_set(as_of_date, as_of_date)
    if player_ids:
        df = df[df['player_id'].isin(player_ids)].reset_index(drop=True)
    return df


def compute_for(player_id: int, as_of_date: date) -> dict:
    """Single-row feature dict for one (player, as-of). Predict-time helper."""
    df = compute_batch(as_of_date, player_ids=[player_id])
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-06-15")
    parser.add_argument("--end",   type=str, default="2025-06-21")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    df = build_training_set(start, end)

    feature_cols = [c for c in df.columns if c.startswith('bat_')]
    print(f"\n=== Feature coverage over {start} -> {end} ===")
    for c in feature_cols:
        nn = df[c].notna().sum()
        print(f"  {c:42s}  {nn:>5,} / {len(df):,}  ({nn / len(df) * 100:5.1f}%)")
