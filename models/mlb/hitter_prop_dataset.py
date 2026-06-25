"""
models/mlb/hitter_prop_dataset.py — Assemble the full training dataset.

Joins all four feature modules + the v1 target labels into one DataFrame, one row
per (batter, game) where the batter started.

DOES NOT attach Underdog odds — that's the backtest layer's job (so name/team matching
complexity stays out of the modeling code).

INVARIANTS
    No feature is computed from the game being predicted.
    Pitcher features are merge_asof'd by (opposing_starter_id, game_date) using
    direction='backward', allow_exact_matches=True — and the pitcher arsenal module
    already uses closed='left' rolling, so same-day match is safe.

API
    build_dataset(start_date, end_date) -> DataFrame
        Columns:
          - identity:     player_id, batter_name, game_id, game_date, opposing_starter_id
          - 98 features:  bat_* (36) + pit_* (45) + ctx_* (11) + mu_* (6)
          - 3 labels:     lbl_hrr_over_15, lbl_tb_over_15, lbl_rbi_over_05
          - raw counts:   lbl_hits, lbl_runs, lbl_rbi, lbl_tb, lbl_hrr (debugging)
          - validity:     lbl_hrr_valid (False if runs column NULL — backfill pending)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from datetime import date, datetime

import numpy as np
import pandas as pd

from db.db import query
from models.mlb import batter_arsenal_features as bat_mod
from models.mlb import pitcher_arsenal_features as pit_mod
from models.mlb import context_features as ctx_mod
from models.mlb import matchup_features as mf


# ----------------------------------------------------------------------------
# Pull starter info + raw box-score outcomes (for labels)
# ----------------------------------------------------------------------------

def _pull_starter_rows(start_date: date, end_date: date) -> pd.DataFrame:
    """Per (batter, game): identity + box score outcomes + opposing starter + handedness."""
    sql = """
        SELECT
            bg.player_id,
            bg.game_id,
            g.game_date,
            bg.team_id            AS batter_team_id,
            bg.batting_order,
            -- Box-score outcomes (label inputs)
            bg.pa, bg.ab,
            bg.hits, bg.doubles, bg.triples, bg.hr,
            bg.rbi, bg.runs,
            -- Game / opposing-starter context
            g.home_team_id, g.away_team_id,
            opp_pg.player_id      AS opposing_starter_id,
            -- Handedness for matchup features (need switch-hitter resolution)
            batter_pl.bats        AS bat_hand,
            batter_pl.name        AS batter_name,
            pitcher_pl.throws     AS pit_throws
        FROM mlb_batting_game bg
        JOIN games g           ON bg.game_id = g.game_id
        JOIN players batter_pl ON bg.player_id = batter_pl.player_id
        LEFT JOIN mlb_pitching_game opp_pg
               ON opp_pg.game_id = bg.game_id
              AND opp_pg.is_starter
              AND opp_pg.team_id != bg.team_id
        LEFT JOIN players pitcher_pl ON opp_pg.player_id = pitcher_pl.player_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND g.game_date >= %(start)s AND g.game_date <= %(end)s
          AND bg.batting_order BETWEEN 1 AND 9
          AND bg.pa > 0
    """
    return query(sql, params={'start': start_date, 'end': end_date})


# ----------------------------------------------------------------------------
# Label computation
# ----------------------------------------------------------------------------

def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute label columns. lbl_hrr_valid=False where runs is NULL (backfill pending)."""
    df = df.copy()
    # Raw counts (kept for debugging + downstream analysis)
    df['lbl_tb']  = df['hits'] + df['doubles'] + 2 * df['triples'] + 3 * df['hr']
    df['lbl_hrr'] = df['hits'].astype('Int64') + df['runs'].astype('Int64') + df['rbi'].astype('Int64')

    # Binary targets — > line means must be at least line+1 since counts are integers
    df['lbl_hrr_over_15'] = (df['lbl_hrr'] > 1.5).astype('Int64')
    df['lbl_tb_over_15']  = (df['lbl_tb']  > 1.5).astype('Int64')
    df['lbl_rbi_over_05'] = (df['rbi']     > 0.5).astype('Int64')

    # Mark HRR label validity (runs column was added recently; backfill in progress)
    df['lbl_hrr_valid'] = df['runs'].notna()
    df.loc[~df['lbl_hrr_valid'], 'lbl_hrr_over_15'] = pd.NA
    df.loc[~df['lbl_hrr_valid'], 'lbl_hrr']         = pd.NA

    return df


# ----------------------------------------------------------------------------
# Main assembler
# ----------------------------------------------------------------------------

def build_dataset(start_date: date, end_date: date) -> pd.DataFrame:
    print(f"[dataset] window: {start_date} -> {end_date}")

    # === 1. Pull row spine + raw outcomes ===
    print("[dataset] pulling starter rows + outcomes...")
    spine = _pull_starter_rows(start_date, end_date)
    spine['game_date'] = pd.to_datetime(spine['game_date'])
    print(f"  {len(spine):,} (batter, game) rows; "
          f"opposing starter present: {spine['opposing_starter_id'].notna().sum():,}; "
          f"runs label valid: {spine['runs'].notna().sum():,}")

    # === 2. Build feature blocks ===
    print("[dataset] building batter features...")
    bat_df = bat_mod.build_training_set(start_date, end_date)
    bat_df['game_date'] = pd.to_datetime(bat_df['game_date'])

    print("[dataset] building pitcher features...")
    pit_df = pit_mod.build_training_set(start_date, end_date)
    pit_df['game_date'] = pd.to_datetime(pit_df['game_date'])

    print("[dataset] building context features...")
    ctx_df = ctx_mod.build_training_set(start_date, end_date)
    ctx_df['game_date'] = pd.to_datetime(ctx_df['game_date'])

    # === 3. Join batter features (direct by player_id + game_date) ===
    print("[dataset] joining batter features...")
    out = spine.merge(bat_df, on=['player_id', 'game_date'], how='left')

    # === 4. Join context features (by player_id + game_id + game_date) ===
    print("[dataset] joining context features...")
    out = out.merge(ctx_df, on=['player_id', 'game_id', 'game_date'], how='left')

    # === 5. Join opposing-starter pitcher features via merge_asof ===
    print("[dataset] joining opposing starter features (merge_asof)...")
    # Rename pit_df's player_id → opposing_starter_id for the merge key
    pit_df_renamed = pit_df.rename(columns={'player_id': 'opposing_starter_id'})
    # merge_asof requires both sides sorted by the 'on' key globally
    out = out.sort_values('game_date').reset_index(drop=True)
    pit_df_renamed = pit_df_renamed.sort_values('game_date').reset_index(drop=True)
    out = pd.merge_asof(
        out, pit_df_renamed, on='game_date', by='opposing_starter_id',
        direction='backward', allow_exact_matches=True,
    )

    # === 6. Compute matchup features ===
    print("[dataset] computing matchup features...")
    out = mf.compute_all(out)

    # === 7. Compute labels ===
    print("[dataset] computing labels...")
    out = _add_labels(out)

    # === 8. Trim to final column set ===
    identity_cols = ['player_id', 'batter_name', 'game_id', 'game_date',
                     'opposing_starter_id', 'bat_hand', 'pit_throws',
                     'batter_team_id', 'batting_order']
    # bat_hand / pit_throws match the bat_/pit_ prefix but are identity metadata, not
    # model features — exclude them here so they don't duplicate identity_cols (parquet
    # rejects duplicate column names).
    feature_cols  = [c for c in out.columns
                     if c.startswith(('bat_', 'pit_', 'ctx_', 'mu_'))
                     and c not in identity_cols]
    label_cols    = ['lbl_hits', 'lbl_runs', 'lbl_rbi', 'lbl_tb', 'lbl_hrr',
                     'lbl_hrr_over_15', 'lbl_tb_over_15', 'lbl_rbi_over_05',
                     'lbl_hrr_valid']
    # Add raw counts that we use for label sanity
    out['lbl_hits'] = out['hits']
    out['lbl_runs'] = out['runs']
    out['lbl_rbi']  = out['rbi']
    keep = identity_cols + feature_cols + label_cols
    # De-dup while preserving order (defensive — any accidental overlap collapses to one)
    seen = set()
    keep = [c for c in keep if c in out.columns and not (c in seen or seen.add(c))]

    out = out[keep]
    print(f"[dataset] done: {out.shape[0]:,} rows x {out.shape[1]} cols "
          f"({len(feature_cols)} features, {len([c for c in label_cols if c.startswith('lbl_') and 'over' in c])} binary labels)")
    return out


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-06-15")
    parser.add_argument("--end",   type=str, default="2025-06-17")
    parser.add_argument("--save",  type=str, default=None,
                        help="Optional CSV/parquet path to save the dataset")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    df = build_dataset(start, end)

    print(f"\n=== Dataset summary ===")
    print(f"  rows:        {len(df):,}")
    print(f"  columns:     {df.shape[1]}")
    feature_cols = [c for c in df.columns if c.startswith(('bat_', 'pit_', 'ctx_', 'mu_'))]
    print(f"  features:    {len(feature_cols)}")
    print(f"  HRR label valid: {df['lbl_hrr_valid'].sum():,} / {len(df):,}")
    print(f"  HRR > 1.5 base rate: {df['lbl_hrr_over_15'].mean() if df['lbl_hrr_valid'].any() else 'N/A':.3f}")
    print(f"  TB  > 1.5 base rate: {df['lbl_tb_over_15'].mean():.3f}")
    print(f"  RBI > 0.5 base rate: {df['lbl_rbi_over_05'].mean():.3f}")

    print(f"\n=== Feature coverage (% non-null) ===")
    for prefix in ('bat_', 'pit_', 'ctx_', 'mu_'):
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            cov = df[cols].notna().mean().mean() * 100
            print(f"  {prefix:8s} ({len(cols):>2d} cols): {cov:5.1f}% avg coverage")

    if args.save:
        if args.save.endswith('.parquet'):
            df.to_parquet(args.save, index=False)
        else:
            df.to_csv(args.save, index=False)
        print(f"\nSaved to {args.save}")
