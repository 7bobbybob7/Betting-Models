"""
models/mlb/pitcher_arsenal_features.py — Per-pitcher Statcast + box-score features for hitter prop modeling.

Mirrors batter_arsenal_features.py architecture. The key extra dimension is BATTER-HAND
conditioning on pitch mix: a RHP throws very different pitches to RHB (lots of sliders)
vs LHB (more changeups). Without this conditioning the matchup math is wrong.

Switch hitters always bat opposite the pitcher's throwing hand — that's the canonical
strategy — so they're mapped to the opposite hand in the aggregation.

STRICT NO-LEAKAGE CONTRACT
    Every rolling window is HALF-OPEN: [as_of - window, as_of). The prediction date
    itself is EXCLUDED. Enforced via pandas rolling closed='left' AND the SQL
    `game_date < as_of_date` bound on the as-of grid.

API
    build_training_set(start_date, end_date)   -> DataFrame   per-pitcher per-appearance
    compute_batch(as_of_date, pitcher_ids=None)-> DataFrame
    compute_for(pitcher_id, as_of_date)        -> dict

Notes
    Dataset assembler should use pandas merge_asof to attach these features to game rows —
    pitcher features at "most recent appearance" before the predict date. Pitchers don't
    pitch every day, so we only emit rows at appearance dates.
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
# Pitch grouping — must match batter_arsenal_features.py
# ----------------------------------------------------------------------------

INDIVIDUAL_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CH', 'FS', 'FO']

BUCKETS = {
    'FB': ['FF', 'SI'],
    'BR': ['SL', 'ST', 'SV', 'CU', 'KC', 'FC'],
    'OS': ['CH', 'FS', 'FO'],
}
PITCH_TO_BUCKET = {pt: bk for bk, pts in BUCKETS.items() for pt in pts}


# ----------------------------------------------------------------------------
# Minimum sample requirements
# ----------------------------------------------------------------------------

MIN_PITCHES_MIX_OVERALL     = 200  # for pit_pct_*_30d baseline (overall)
MIN_PITCHES_MIX_HAND        = 150  # for pit_pct_*_vs_HAND_30d
MIN_BIP_ALLOWED_INDIV       = 20   # per-pitch-type xwOBA allowed
MIN_BIP_ALLOWED_BUCKET      = 60
MIN_BIP_ALLOWED_HAND        = 50   # vs RHB / vs LHB aggregate xwOBA allowed
MIN_SWINGS_HAND             = 80   # vs RHB / vs LHB whiff rate
MIN_SWINGS_OVERALL          = 100  # overall whiff rate
MIN_FB_PITCHES_VELO         = 30   # avg FF velo
MIN_FB_PITCHES_VELO_BASE    = 100  # baseline for velo trend
MIN_PA_SZN                  = 100  # season-level rates from box scores
MIN_STARTS_FOR_IP_PER_START = 3    # workload metric

ROLLING_MIX_WINDOW   = '30D'   # mix changes fast; recent
ROLLING_RESULTS_WIN  = '90D'   # results need more sample
ROLLING_SEASON_WIN   = '180D'  # approximate season-to-date (pitchers don't pitch off-season)
ROLLING_WORKLOAD_WIN = '30D'   # last ~5 starts for typical starter


# ----------------------------------------------------------------------------
# Raw aggregate pulls
# Switch hitters always bat opposite the pitcher's throwing hand — encoded in SQL.
# ----------------------------------------------------------------------------

def _pull_pitch_aggregates(start: date, end: date) -> pd.DataFrame:
    """Per (pitcher, game_date, pitch_type, batter_hand) — base aggregate for mix + results."""
    sql = """
        SELECT
            p.pitcher_id AS player_id,
            g.game_date,
            p.pitch_type,
            CASE
                WHEN batter.bats = 'S' AND pitcher.throws = 'R' THEN 'L'
                WHEN batter.bats = 'S' AND pitcher.throws = 'L' THEN 'R'
                ELSE batter.bats
            END AS batter_hand,
            COUNT(*)                                                AS pitches,
            SUM(CASE WHEN p.is_swing   THEN 1 ELSE 0 END)           AS swings,
            SUM(CASE WHEN p.is_whiff   THEN 1 ELSE 0 END)           AS whiffs,
            SUM(CASE WHEN p.is_in_play THEN 1 ELSE 0 END)           AS bip,
            SUM(CASE WHEN p.is_in_play THEN p.xwoba ELSE 0 END)     AS xwoba_sum
        FROM mlb_pitches p
        JOIN games   g       ON p.game_id    = g.game_id
        JOIN players batter  ON p.batter_id  = batter.player_id
        JOIN players pitcher ON p.pitcher_id = pitcher.player_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND p.pitch_type IS NOT NULL
          AND batter.bats  IN ('L', 'R', 'S')
          AND pitcher.throws IN ('L', 'R')
        GROUP BY p.pitcher_id, g.game_date, p.pitch_type,
                 CASE WHEN batter.bats = 'S' AND pitcher.throws = 'R' THEN 'L'
                      WHEN batter.bats = 'S' AND pitcher.throws = 'L' THEN 'R'
                      ELSE batter.bats END
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_velocity(start: date, end: date) -> pd.DataFrame:
    """Per (pitcher, game_date): avg fastball (FB bucket = FF + SI + FC) velocity.
    Sinker-heavy pitchers like Framber barely throw FF — narrowing to FF would lose them entirely."""
    sql = """
        SELECT
            p.pitcher_id   AS player_id,
            g.game_date,
            SUM(p.release_speed)                                                AS velo_sum,
            SUM(CASE WHEN p.release_speed IS NOT NULL THEN 1 ELSE 0 END)        AS velo_n
        FROM mlb_pitches p
        JOIN games g ON p.game_id = g.game_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND p.pitch_type IN ('FF', 'SI', 'FC')
          AND p.release_speed IS NOT NULL
        GROUP BY p.pitcher_id, g.game_date
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_pitching_box(start: date, end: date) -> pd.DataFrame:
    """Per (pitcher, game_date): box-score line for season-level rates + workload."""
    sql = """
        SELECT
            pg.player_id,
            g.game_date,
            pg.is_starter::int AS is_starter,
            pg.ip,
            COALESCE(pg.pitches, 0) AS pitches,
            pg.so,
            pg.bb,
            pg.hr_allowed,
            -- PA estimate: outs (ip*3) + hits + bb + hbp (no hbp in our schema, skip)
            pg.ip * 3 + COALESCE(pg.hits_allowed, 0) + COALESCE(pg.bb, 0) AS pa_est
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND pg.ip IS NOT NULL
          AND pg.ip > 0
    """
    return query(sql, params={'start': start, 'end': end})


# ----------------------------------------------------------------------------
# Rolling helpers (same shape as batter file)
# ----------------------------------------------------------------------------

def _roll_sum(df: pd.DataFrame, value_cols: list, window: str,
              group_col: str = 'player_id') -> pd.DataFrame:
    """Half-open rolling-window sum — current row EXCLUDED via closed='left'."""
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
    """num/den only where den >= min_den, else NaN."""
    mask = den.fillna(0) >= min_den
    return np.where(mask, num / den.where(mask, np.nan), np.nan)


# ----------------------------------------------------------------------------
# Feature assemblers
# ----------------------------------------------------------------------------

def _assemble_mix(pitch_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """Pitch mix percentages per (pitch type | batter hand), 30d window.
    Plus overall (handedness-agnostic) bucket-level mix baseline."""
    out = as_of.copy()

    # ---- Overall pitch counts per (pitcher, date) — denominator for overall mix ----
    overall = pitch_agg.groupby(['player_id', 'game_date'], as_index=False)['pitches'].sum()
    overall = overall.rename(columns={'pitches': 'pitches_total'})
    overall_rolled = _roll_sum(overall, ['pitches_total'], ROLLING_MIX_WINDOW)
    out = out.merge(overall_rolled, on=['player_id', 'game_date'], how='left')

    # ---- Per-hand pitch counts per (pitcher, date) — denominator for vs-hand mix ----
    by_hand = pitch_agg.groupby(['player_id', 'game_date', 'batter_hand'], as_index=False)['pitches'].sum()
    for hand in ('R', 'L'):
        sub = by_hand[by_hand['batter_hand'] == hand][['player_id', 'game_date', 'pitches']]
        if sub.empty:
            out[f'pitches_vs_{hand}HB_30d'] = np.nan
            continue
        sub_rolled = _roll_sum(sub, ['pitches'], ROLLING_MIX_WINDOW)
        sub_rolled = sub_rolled.rename(columns={'pitches': f'pitches_vs_{hand}HB_30d'})
        out = out.merge(sub_rolled, on=['player_id', 'game_date'], how='left')

    # ---- Per-pitch-type vs each batter hand ----
    # A pitcher who throws ZERO of a pitch type still has a meaningful 0% — we explicitly
    # fillna(0) on the numerator so the merge's NaN-on-no-rows becomes a real 0% reading.
    for pt in INDIVIDUAL_PITCH_TYPES:
        for hand in ('R', 'L'):
            sub = pitch_agg[
                (pitch_agg['pitch_type'] == pt) & (pitch_agg['batter_hand'] == hand)
            ][['player_id', 'game_date', 'pitches']]
            if sub.empty:
                out[f'pt_count_{pt}_vs_{hand}HB'] = 0.0
            else:
                sub = sub.groupby(['player_id', 'game_date'], as_index=False)['pitches'].sum()
                sub_rolled = _roll_sum(sub, ['pitches'], ROLLING_MIX_WINDOW)
                sub_rolled = sub_rolled.rename(columns={'pitches': f'pt_count_{pt}_vs_{hand}HB'})
                out = out.merge(sub_rolled, on=['player_id', 'game_date'], how='left')
                out[f'pt_count_{pt}_vs_{hand}HB'] = out[f'pt_count_{pt}_vs_{hand}HB'].fillna(0)
            out[f'pit_pct_{pt}_vs_{hand}HB_30d'] = _safe_div(
                out[f'pt_count_{pt}_vs_{hand}HB'],
                out[f'pitches_vs_{hand}HB_30d'],
                MIN_PITCHES_MIX_HAND,
            )

    # ---- Per-bucket overall (handedness-agnostic) baseline ----
    pitch_agg = pitch_agg.copy()
    pitch_agg['bucket'] = pitch_agg['pitch_type'].map(PITCH_TO_BUCKET)
    pitch_agg = pitch_agg.dropna(subset=['bucket'])
    for bk in BUCKETS:
        sub = pitch_agg[pitch_agg['bucket'] == bk].groupby(
            ['player_id', 'game_date'], as_index=False
        )['pitches'].sum()
        if sub.empty:
            out[f'bk_count_{bk}'] = 0.0
        else:
            sub_rolled = _roll_sum(sub, ['pitches'], ROLLING_MIX_WINDOW)
            sub_rolled = sub_rolled.rename(columns={'pitches': f'bk_count_{bk}'})
            out = out.merge(sub_rolled, on=['player_id', 'game_date'], how='left')
            out[f'bk_count_{bk}'] = out[f'bk_count_{bk}'].fillna(0)
        out[f'pit_pct_{bk}_30d'] = _safe_div(
            out[f'bk_count_{bk}'],
            out['pitches_total'],
            MIN_PITCHES_MIX_OVERALL,
        )

    # ---- Per-bucket × hand fallback ----
    for bk in BUCKETS:
        for hand in ('R', 'L'):
            sub = pitch_agg[
                (pitch_agg['bucket'] == bk) & (pitch_agg['batter_hand'] == hand)
            ].groupby(['player_id', 'game_date'], as_index=False)['pitches'].sum()
            if sub.empty:
                out[f'bk_h_count_{bk}_{hand}'] = 0.0
            else:
                sub_rolled = _roll_sum(sub, ['pitches'], ROLLING_MIX_WINDOW)
                sub_rolled = sub_rolled.rename(columns={'pitches': f'bk_h_count_{bk}_{hand}'})
                out = out.merge(sub_rolled, on=['player_id', 'game_date'], how='left')
                out[f'bk_h_count_{bk}_{hand}'] = out[f'bk_h_count_{bk}_{hand}'].fillna(0)
            out[f'pit_pct_{bk}_vs_{hand}HB_30d'] = _safe_div(
                out[f'bk_h_count_{bk}_{hand}'],
                out[f'pitches_vs_{hand}HB_30d'],
                MIN_PITCHES_MIX_HAND,
            )

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('pit_pct_')]
    return out[keep]


def _assemble_results_allowed(pitch_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """xwOBA + whiff rate allowed: vs RHB / vs LHB aggregate, plus per-bucket allowed."""
    out = as_of.copy()

    # vs hand aggregates
    by_hand = pitch_agg.groupby(['player_id', 'game_date', 'batter_hand'], as_index=False)[
        ['swings', 'whiffs', 'bip', 'xwoba_sum']
    ].sum()
    for hand in ('R', 'L'):
        sub = by_hand[by_hand['batter_hand'] == hand][
            ['player_id', 'game_date', 'swings', 'whiffs', 'bip', 'xwoba_sum']
        ]
        if sub.empty:
            out[f'pit_xwoba_allowed_vs_{hand}HB_90d'] = np.nan
            out[f'pit_whiff_rate_vs_{hand}HB_90d']    = np.nan
            continue
        rolled = _roll_sum(sub, ['swings', 'whiffs', 'bip', 'xwoba_sum'], ROLLING_RESULTS_WIN)
        rolled = rolled.rename(columns={
            'swings': f'sw_{hand}', 'whiffs': f'wh_{hand}',
            'bip': f'bip_{hand}', 'xwoba_sum': f'xs_{hand}',
        })
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out[f'pit_xwoba_allowed_vs_{hand}HB_90d'] = _safe_div(
            out[f'xs_{hand}'], out[f'bip_{hand}'], MIN_BIP_ALLOWED_HAND)
        out[f'pit_whiff_rate_vs_{hand}HB_90d'] = _safe_div(
            out[f'wh_{hand}'], out[f'sw_{hand}'], MIN_SWINGS_HAND)

    # Overall whiff rate (regardless of batter hand)
    overall = pitch_agg.groupby(['player_id', 'game_date'], as_index=False)[
        ['swings', 'whiffs']
    ].sum()
    if not overall.empty:
        rolled = _roll_sum(overall, ['swings', 'whiffs'], ROLLING_RESULTS_WIN)
        rolled = rolled.rename(columns={'swings': 'sw_all', 'whiffs': 'wh_all'})
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out['pit_whiff_rate_90d'] = _safe_div(out['wh_all'], out['sw_all'], MIN_SWINGS_OVERALL)
    else:
        out['pit_whiff_rate_90d'] = np.nan

    # Per-bucket xwOBA allowed (handedness-agnostic, just lets the model see "this guy
    # gets crushed when his FB is put in play even if it doesn't happen often")
    pitch_agg2 = pitch_agg.copy()
    pitch_agg2['bucket'] = pitch_agg2['pitch_type'].map(PITCH_TO_BUCKET)
    pitch_agg2 = pitch_agg2.dropna(subset=['bucket'])
    for bk in BUCKETS:
        sub = pitch_agg2[pitch_agg2['bucket'] == bk].groupby(
            ['player_id', 'game_date'], as_index=False
        )[['bip', 'xwoba_sum']].sum()
        if sub.empty:
            out[f'pit_xwoba_allowed_vs_{bk}_90d'] = np.nan
            continue
        rolled = _roll_sum(sub, ['bip', 'xwoba_sum'], ROLLING_RESULTS_WIN)
        rolled = rolled.rename(columns={'bip': f'bk_bip_{bk}', 'xwoba_sum': f'bk_xs_{bk}'})
        out = out.merge(rolled, on=['player_id', 'game_date'], how='left')
        out[f'pit_xwoba_allowed_vs_{bk}_90d'] = _safe_div(
            out[f'bk_xs_{bk}'], out[f'bk_bip_{bk}'], MIN_BIP_ALLOWED_BUCKET)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('pit_')]
    return out[keep]


def _assemble_velocity(velo_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """Avg FF velocity (30d) + velo trend (30d minus 180d baseline)."""
    if velo_agg.empty:
        out = as_of.copy()
        out['pit_fb_velo_30d']   = np.nan
        out['pit_fb_velo_trend'] = np.nan
        return out

    rolled30  = _roll_sum(velo_agg, ['velo_sum', 'velo_n'], ROLLING_MIX_WINDOW)
    rolled180 = _roll_sum(velo_agg, ['velo_sum', 'velo_n'], ROLLING_SEASON_WIN)
    rolled30  = rolled30.rename(columns={'velo_sum': 'velo_sum_30',  'velo_n': 'velo_n_30'})
    rolled180 = rolled180.rename(columns={'velo_sum': 'velo_sum_180', 'velo_n': 'velo_n_180'})

    out = as_of.merge(rolled30, on=['player_id', 'game_date'], how='left')
    out = out.merge(rolled180, on=['player_id', 'game_date'], how='left')

    velo_30  = _safe_div(out['velo_sum_30'],  out['velo_n_30'],  MIN_FB_PITCHES_VELO)
    velo_180 = _safe_div(out['velo_sum_180'], out['velo_n_180'], MIN_FB_PITCHES_VELO_BASE)
    out['pit_fb_velo_30d']   = velo_30
    out['pit_fb_velo_trend'] = velo_30 - velo_180

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('pit_')]
    return out[keep]


def _assemble_box_rates(box_agg: pd.DataFrame, as_of: pd.DataFrame) -> pd.DataFrame:
    """K/BB rates (season-window), HR/9 (season-window), IP per start (30d)."""
    if box_agg.empty:
        out = as_of.copy()
        for c in ['pit_k_rate_szn', 'pit_bb_rate_szn', 'pit_hr_per_9_szn', 'pit_ip_per_start_30d']:
            out[c] = np.nan
        return out

    rolled_szn = _roll_sum(
        box_agg, ['ip', 'so', 'bb', 'hr_allowed', 'pa_est'], ROLLING_SEASON_WIN
    )
    rolled_szn = rolled_szn.rename(columns={
        'ip': 'ip_szn', 'so': 'so_szn', 'bb': 'bb_szn',
        'hr_allowed': 'hr_szn', 'pa_est': 'pa_szn',
    })

    rolled_wl = _roll_sum(box_agg, ['ip', 'is_starter'], ROLLING_WORKLOAD_WIN)
    rolled_wl = rolled_wl.rename(columns={'ip': 'ip_wl', 'is_starter': 'starts_wl'})

    out = as_of.merge(rolled_szn, on=['player_id', 'game_date'], how='left')
    out = out.merge(rolled_wl,  on=['player_id', 'game_date'], how='left')

    out['pit_k_rate_szn']      = _safe_div(out['so_szn'], out['pa_szn'], MIN_PA_SZN)
    out['pit_bb_rate_szn']     = _safe_div(out['bb_szn'], out['pa_szn'], MIN_PA_SZN)
    out['pit_hr_per_9_szn']    = _safe_div(out['hr_szn'] * 9.0, out['ip_szn'], 30.0)  # min 30 IP
    out['pit_ip_per_start_30d'] = _safe_div(out['ip_wl'], out['starts_wl'], MIN_STARTS_FOR_IP_PER_START)

    keep = ['player_id', 'game_date'] + [c for c in out.columns if c.startswith('pit_')]
    return out[keep]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_training_set(start_date: date, end_date: date) -> pd.DataFrame:
    """One row per (pitcher_id, game_date) where pitcher appeared in window.

    Each row's features are computed from data strictly before game_date.
    Dataset assembler will use merge_asof to attach features to game prediction rows.
    """
    pull_start = start_date - timedelta(days=182)  # need 180d window + buffer
    print(f"[pitcher_arsenal] pulling raw aggregates {pull_start} -> {end_date}...")

    pitch_agg = _pull_pitch_aggregates(pull_start, end_date + timedelta(days=1))
    velo_agg  = _pull_velocity(pull_start, end_date + timedelta(days=1))
    box_agg   = _pull_pitching_box(pull_start, end_date + timedelta(days=1))
    print(f"  pitches: {len(pitch_agg):>8,} (pitcher, date, pitch_type, hand) rows")
    print(f"  velo:    {len(velo_agg):>8,} (pitcher, date) rows")
    print(f"  box:     {len(box_agg):>8,} (pitcher, date) rows")

    for df in (pitch_agg, velo_agg, box_agg):
        df['game_date'] = pd.to_datetime(df['game_date'])

    # As-of grid: every (pitcher, game_date) where they actually pitched in window
    as_of = (box_agg[['player_id', 'game_date']]
             .drop_duplicates()
             .sort_values(['player_id', 'game_date'])
             .reset_index(drop=True))
    as_of = as_of[as_of['game_date'] >= pd.Timestamp(start_date)]
    as_of = as_of[as_of['game_date'] <= pd.Timestamp(end_date)].reset_index(drop=True)
    print(f"  as-of:   {len(as_of):>8,} (pitcher, game_date) rows in window")

    print("[pitcher_arsenal] assembling features...")
    mix     = _assemble_mix(pitch_agg, as_of)
    results = _assemble_results_allowed(pitch_agg, as_of)
    velo    = _assemble_velocity(velo_agg, as_of)
    box     = _assemble_box_rates(box_agg, as_of)

    out = as_of
    for block in (mix, results, velo, box):
        out = out.merge(block, on=['player_id', 'game_date'], how='left')

    print(f"[pitcher_arsenal] done: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out


def compute_batch(as_of_date: date, pitcher_ids: list | None = None) -> pd.DataFrame:
    df = build_training_set(as_of_date, as_of_date)
    if pitcher_ids:
        df = df[df['player_id'].isin(pitcher_ids)].reset_index(drop=True)
    return df


def compute_for(pitcher_id: int, as_of_date: date) -> dict:
    df = compute_batch(as_of_date, pitcher_ids=[pitcher_id])
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-06-15")
    parser.add_argument("--end",   type=str, default="2025-06-17")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    df = build_training_set(start, end)

    feature_cols = [c for c in df.columns if c.startswith('pit_')]
    print(f"\n=== Feature coverage over {start} -> {end} ===")
    print(f"Total features: {len(feature_cols)}")
    for c in feature_cols:
        nn = df[c].notna().sum()
        print(f"  {c:42s}  {nn:>5,} / {len(df):,}  ({nn / len(df) * 100:5.1f}%)")
