"""
models/mlb/context_features.py — Game-level + lineup-position features for hitter prop modeling.

Captures everything that's neither batter-specific nor pitcher-specific:
    - Weather (temp, wind speed, wind direction relative to offense)
    - Dome flag
    - Park factors (3-year rolling per venue, computed from prior years only)
    - Home/away
    - Batting order position
    - Lineup support — OBP of batters in front (RBI opportunity)
                    — SLG of batters behind (run-scoring opportunity)
    - Team runs-per-game baseline

STRICT NO-LEAKAGE CONTRACT
    Park factors use only prior 3 SEASONS of data (excludes current season entirely).
    Per-player OBP/SLG used in lineup support uses closed='left' rolling on season window —
    same pattern as batter_arsenal_features.

API (mirrors the arsenal modules)
    build_training_set(start_date, end_date)   -> per (player_id, game_id) row
    compute_batch(as_of_date)                  -> all starters on that date
    compute_for(player_id, game_id, as_of_date)-> single row
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import re
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from db.db import query


# ----------------------------------------------------------------------------
# Min sample thresholds
# ----------------------------------------------------------------------------

MIN_PA_FOR_OBP_SLG     = 50    # season-to-date PA needed for player's OBP/SLG to be meaningful
MIN_LINEUP_TEAMMATES   = 1     # need at least 1 teammate to compute in-front/behind aggregates
MIN_GAMES_FOR_PARK     = 50    # min games at venue (over 365d) for park factor to be reliable
PARK_FACTOR_WINDOW     = '365D'


# ----------------------------------------------------------------------------
# Raw pulls
# ----------------------------------------------------------------------------

def _pull_lineup_rows(start: date, end: date) -> pd.DataFrame:
    """Per (game, player, batting_order) for every starter (order 1-9) in window."""
    sql = """
        SELECT
            bg.player_id, bg.team_id, bg.game_id, bg.batting_order,
            bg.pa, bg.ab, bg.hits, bg.doubles, bg.triples, bg.hr, bg.bb,
            g.game_date, g.home_team_id, g.away_team_id, g.venue
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND bg.batting_order BETWEEN 1 AND 9
          AND bg.pa > 0
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_game_meta(start: date, end: date) -> pd.DataFrame:
    """Per game: weather + venue meta."""
    sql = """
        SELECT
            g.game_id,
            g.game_date,
            g.venue,
            g.home_team_id,
            gi.weather_temp,
            gi.weather_wind,
            LOWER(COALESCE(gi.weather_dir, ''))  AS weather_dir,
            LOWER(COALESCE(gi.weather_cond, '')) AS weather_cond
        FROM games g
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND g.game_date >= %(start)s
          AND g.game_date <  %(end)s
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_park_history(start: date, end: date) -> pd.DataFrame:
    """Per game: venue + runs scored + HRs hit. Used to compute rolling 365D park factors.

    Why per-game instead of per-year: ballparks modify dimensions between seasons
    (Camden Yards 2022 LF wall, 2025 LF wall walk-back, etc.). A 3-yr window can't
    react to recent changes. Rolling 365D per game updates within one calendar year.
    """
    sql = """
        SELECT
            g.game_id,
            g.game_date,
            g.venue,
            (g.home_score + g.away_score)         AS runs,
            COALESCE(bg_hr.hr, 0)                 AS hr,
            1                                     AS games
        FROM games g
        LEFT JOIN (
            SELECT game_id, SUM(hr) AS hr FROM mlb_batting_game GROUP BY game_id
        ) bg_hr ON g.game_id = bg_hr.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND g.venue IS NOT NULL
          AND g.game_date >= %(start)s
          AND g.game_date <  %(end)s
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_player_box_history(start: date, end: date) -> pd.DataFrame:
    """Per (player, game_date): box-score line for rolling season OBP/SLG."""
    sql = """
        SELECT
            bg.player_id,
            g.game_date,
            EXTRACT(YEAR FROM g.game_date)::INT AS season,
            bg.pa, bg.ab, bg.hits, bg.doubles, bg.triples, bg.hr, bg.bb
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND g.game_date >= %(start)s
          AND g.game_date <  %(end)s
          AND bg.pa > 0
    """
    return query(sql, params={'start': start, 'end': end})


def _pull_team_history(start: date, end: date) -> pd.DataFrame:
    """Per (team, game_date): runs scored (from games table)."""
    # Each game contributes runs scored to BOTH teams (home + away)
    sql = """
        SELECT home_team_id AS team_id, game_date, home_score AS runs
        FROM games WHERE sport_id = 2 AND status = 'final'
          AND game_date >= %(start)s AND game_date < %(end)s
        UNION ALL
        SELECT away_team_id AS team_id, game_date, away_score AS runs
        FROM games WHERE sport_id = 2 AND status = 'final'
          AND game_date >= %(start)s AND game_date < %(end)s
    """
    return query(sql, params={'start': start, 'end': end})


# ----------------------------------------------------------------------------
# Park factors — computed per (venue, year) using prior 3 years ONLY
# Strict no-leakage: factor for 2025 games uses 2022-2024 data only.
# ----------------------------------------------------------------------------

def _compute_park_factors(park_hist: pd.DataFrame) -> pd.DataFrame:
    """Returns per (venue, game_date): ctx_park_runs_factor, ctx_park_hr_factor.
    Rolling 365-day window, closed='left' so the game's own day is excluded.

    Reacts to ballpark changes within 1 calendar year (Camden 2022/2025 wall, Mets
    CF dimensions, new stadium openings). 3-year smoothing was masking these.
    """
    park_hist = park_hist.copy()
    park_hist['game_date'] = pd.to_datetime(park_hist['game_date'])

    # Per-venue rolling 365D sums
    venue_rolled = _roll_sum(park_hist, ['runs', 'hr', 'games'], PARK_FACTOR_WINDOW, group_col='venue')
    venue_rolled = venue_rolled.rename(columns={
        'runs': 'venue_runs', 'hr': 'venue_hr', 'games': 'venue_games',
    })

    # League rolling 365D (sum across all venues for same window)
    league_daily = (park_hist
                    .groupby('game_date', as_index=False)[['runs', 'hr', 'games']]
                    .sum()
                    .sort_values('game_date')
                    .set_index('game_date'))
    league_rolled = league_daily.rolling(PARK_FACTOR_WINDOW, closed='left').sum().reset_index()
    league_rolled = league_rolled.rename(columns={
        'runs': 'league_runs', 'hr': 'league_hr', 'games': 'league_games',
    })

    out = venue_rolled.merge(league_rolled, on='game_date', how='left')

    # Per-game rates
    venue_rpg  = out['venue_runs'] / out['venue_games'].replace(0, np.nan)
    venue_hrpg = out['venue_hr']   / out['venue_games'].replace(0, np.nan)
    league_rpg  = out['league_runs'] / out['league_games'].replace(0, np.nan)
    league_hrpg = out['league_hr']   / out['league_games'].replace(0, np.nan)

    mask = out['venue_games'].fillna(0) >= MIN_GAMES_FOR_PARK
    out['ctx_park_runs_factor'] = np.where(mask, venue_rpg / league_rpg, np.nan)
    out['ctx_park_hr_factor']   = np.where(mask, venue_hrpg / league_hrpg, np.nan)

    return out[['venue', 'game_date', 'ctx_park_runs_factor', 'ctx_park_hr_factor']]


# ----------------------------------------------------------------------------
# Rolling per-player OBP/SLG over season window (closed='left' → no leakage)
# ----------------------------------------------------------------------------

def _roll_sum(df: pd.DataFrame, value_cols: list, window: str,
              group_col: str = 'player_id') -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.sort_values([group_col, 'game_date'])
    chunks = []
    for grp_id, grp in df.groupby(group_col, sort=False):
        rolled = (grp.set_index('game_date')[value_cols]
                     .rolling(window, closed='left').sum()
                     .reset_index())
        rolled[group_col] = grp_id
        chunks.append(rolled)
    return pd.concat(chunks, ignore_index=True)


def _compute_player_obp_slg(box_hist: pd.DataFrame) -> pd.DataFrame:
    """Per (player_id, game_date): season-to-date OBP and SLG.
    Uses 180-day rolling window (approximates current season, since pitchers/batters
    don't accumulate stats during the off-season anyway).
    """
    rolled = _roll_sum(box_hist,
                       ['pa', 'ab', 'hits', 'doubles', 'triples', 'hr', 'bb'],
                       '180D')
    # Compute TB and OBP-proxy + SLG, with min sample threshold
    rolled['_tb'] = rolled['hits'] + rolled['doubles'] + 2*rolled['triples'] + 3*rolled['hr']
    mask = rolled['pa'].fillna(0) >= MIN_PA_FOR_OBP_SLG
    rolled['szn_obp_proxy'] = np.where(mask,
                                        (rolled['hits'] + rolled['bb']) / rolled['pa'],
                                        np.nan)
    mask_slg = rolled['ab'].fillna(0) >= MIN_PA_FOR_OBP_SLG
    rolled['szn_slg'] = np.where(mask_slg,
                                  rolled['_tb'] / rolled['ab'],
                                  np.nan)
    return rolled[['player_id', 'game_date', 'szn_obp_proxy', 'szn_slg']]


def _compute_team_rpg(team_hist: pd.DataFrame) -> pd.DataFrame:
    """Per (team_id, game_date): rolling 180-day team runs-per-game."""
    team_hist = team_hist.copy()
    team_hist['gp'] = 1
    rolled = _roll_sum(team_hist, ['runs', 'gp'], '180D', group_col='team_id')
    rolled['ctx_team_runs_per_game_szn'] = np.where(
        rolled['gp'].fillna(0) >= 20,
        rolled['runs'] / rolled['gp'], np.nan)
    return rolled[['team_id', 'game_date', 'ctx_team_runs_per_game_szn']]


# ----------------------------------------------------------------------------
# Weather / venue parsing
# ----------------------------------------------------------------------------

DOME_VENUES = {
    "rogers centre", "tropicana field", "minute maid park", "globe life field",
    "chase field", "t-mobile park", "american family field", "loandepot park",
    "miller park",  # old name for AmFam
}

def _is_dome(venue: str, weather_dir: str, weather_cond: str) -> bool:
    """True if game is indoors (closed roof / dome / weather marked 'none' or 'dome')."""
    v = (venue or "").lower()
    if any(d in v for d in DOME_VENUES):
        return True
    if 'dome' in (weather_cond or ''):
        return True
    if weather_dir == 'none.':
        return True
    return False


def _wind_offense_score(weather_dir: str) -> float:
    """+1 if wind helps offense (blowing out), -1 if wind hurts (blowing in), 0 if neutral/dome.
    NaN if missing/varies/calm — let the model handle as missing."""
    if not weather_dir:
        return np.nan
    if weather_dir.startswith('out to'):
        return 1.0
    if weather_dir.startswith('in from'):
        return -1.0
    if weather_dir in ('none.', 'calm.'):
        return 0.0
    return np.nan  # 'l to r.' / 'r to l.' / 'varies.' — neutral but ambiguous


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_training_set(start_date: date, end_date: date) -> pd.DataFrame:
    """Per-batter per-game context features for [start_date, end_date]."""
    pull_start = start_date - timedelta(days=182)

    print(f"[context] pulling raw data...")
    lineup    = _pull_lineup_rows(start_date, end_date + timedelta(days=1))
    meta      = _pull_game_meta(start_date, end_date + timedelta(days=1))
    box_hist  = _pull_player_box_history(pull_start, end_date + timedelta(days=1))
    team_hist = _pull_team_history(pull_start, end_date + timedelta(days=1))
    # Park factors need 365 days of prior data for the earliest game in our window
    park_hist = _pull_park_history(pull_start - timedelta(days=365), end_date + timedelta(days=1))
    print(f"  lineup:    {len(lineup):>8,} (game, player, order) rows in window")
    print(f"  meta:      {len(meta):>8,} games")
    print(f"  box hist:  {len(box_hist):>8,} (player, game_date) for OBP/SLG rolling")
    print(f"  team hist: {len(team_hist):>8,} (team, game_date) for team RPG")
    print(f"  park hist: {len(park_hist):>8,} (venue, game) rows for rolling park factors")

    for df in (lineup, meta, box_hist, team_hist):
        df['game_date'] = pd.to_datetime(df['game_date'])

    print(f"[context] computing rolling 365D park factors...")
    park_factors = _compute_park_factors(park_hist)
    print(f"  {len(park_factors):,} (venue, game_date) park factor rows")

    print(f"[context] computing rolling player OBP/SLG and team RPG...")
    obp_slg = _compute_player_obp_slg(box_hist)
    team_rpg = _compute_team_rpg(team_hist)

    print(f"[context] assembling lineup support features (this is the big one)...")
    out = _assemble_lineup_support(lineup, obp_slg)

    print(f"[context] merging weather / park / team / simple fields...")
    out = _merge_weather_park(out, meta, park_factors)
    out = _merge_team_rpg(out, lineup, team_rpg)
    out = _add_simple_fields(out, lineup)

    keep = ['player_id', 'game_id', 'game_date'] + [c for c in out.columns if c.startswith('ctx_')]
    out = out[keep]
    print(f"[context] done: {out.shape[0]:,} rows x {out.shape[1]} cols")
    return out


def _assemble_lineup_support(lineup: pd.DataFrame, obp_slg: pd.DataFrame) -> pd.DataFrame:
    """For each (target_batter, game), aggregate teammates' OBP (in front) and SLG (behind)."""
    # Attach each player's season-to-date OBP/SLG AS OF game_date
    lineup_with_stats = lineup.merge(
        obp_slg, on=['player_id', 'game_date'], how='left'
    )

    # For each game/team, list of (order, obp, slg) tuples for all 9 batters
    # Then for each target row, aggregate teammates' OBP in front + SLG behind
    rows_out = []
    grp_cols = ['game_id', 'team_id']
    for (gid, tid), grp in lineup_with_stats.groupby(grp_cols):
        # Sort by order for clarity (not strictly needed)
        grp = grp.sort_values('batting_order')
        for _, row in grp.iterrows():
            order = row['batting_order']
            in_front = grp[grp['batting_order'] < order]
            behind   = grp[grp['batting_order'] > order]

            obp_in_front = in_front['szn_obp_proxy'].dropna().mean() if len(in_front) > 0 else np.nan
            slg_behind   = behind['szn_slg'].dropna().mean()         if len(behind) > 0 else np.nan

            rows_out.append({
                'player_id': row['player_id'],
                'game_id':   gid,
                'game_date': row['game_date'],
                'ctx_lineup_obp_in_front': obp_in_front if not np.isnan(obp_in_front) else None,
                'ctx_lineup_slg_behind':   slg_behind   if not np.isnan(slg_behind)   else None,
                'ctx_batter_order_position': int(order),
            })
    return pd.DataFrame(rows_out)


def _merge_weather_park(out: pd.DataFrame, meta: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    """Attach weather + park factor to each row."""
    meta = meta.copy()
    meta['ctx_temp_f']             = meta['weather_temp']
    meta['ctx_wind_speed']         = meta['weather_wind']
    meta['ctx_is_dome']            = meta.apply(
        lambda r: int(_is_dome(r['venue'], r['weather_dir'], r['weather_cond'])), axis=1
    )
    meta['ctx_wind_offense_score'] = meta['weather_dir'].apply(_wind_offense_score)

    out = out.merge(
        meta[['game_id', 'venue', 'game_date',
              'ctx_temp_f', 'ctx_wind_speed', 'ctx_is_dome', 'ctx_wind_offense_score']],
        on=['game_id', 'game_date'], how='left'
    )
    out = out.merge(park_factors, on=['venue', 'game_date'], how='left')
    return out.drop(columns=['venue'])


def _merge_team_rpg(out: pd.DataFrame, lineup: pd.DataFrame, team_rpg: pd.DataFrame) -> pd.DataFrame:
    """Attach team's season-to-date RPG (the batter's own team).
    Uses merge_asof with allow_exact_matches=False — strict-less-than for no leakage."""
    out = out.merge(
        lineup[['player_id', 'game_id', 'team_id']].drop_duplicates(),
        on=['player_id', 'game_id'], how='left',
    )
    # merge_asof requires BOTH frames sorted by the 'on' key globally
    out = out.sort_values('game_date').reset_index(drop=True)
    team_rpg_sorted = team_rpg.sort_values('game_date').reset_index(drop=True)
    out = pd.merge_asof(
        out, team_rpg_sorted, on='game_date', by='team_id',
        direction='backward', allow_exact_matches=False,
    )
    return out.drop(columns=['team_id'])


def _add_simple_fields(out: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
    """is_home derived from batter's team vs home_team_id."""
    join = lineup[['player_id', 'game_id', 'team_id', 'home_team_id']].drop_duplicates()
    out = out.merge(join, on=['player_id', 'game_id'], how='left')
    out['ctx_is_home'] = (out['team_id'] == out['home_team_id']).astype(int)
    return out.drop(columns=['team_id', 'home_team_id'])


def compute_batch(as_of_date: date, player_ids: list | None = None) -> pd.DataFrame:
    df = build_training_set(as_of_date, as_of_date)
    if player_ids:
        df = df[df['player_id'].isin(player_ids)].reset_index(drop=True)
    return df


def compute_for(player_id: int, game_id: int, as_of_date: date) -> dict:
    df = compute_batch(as_of_date, player_ids=[player_id])
    df = df[df['game_id'] == game_id]
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

    feature_cols = [c for c in df.columns if c.startswith('ctx_')]
    print(f"\n=== Feature coverage over {start} -> {end} ===")
    print(f"Total features: {len(feature_cols)}")
    for c in feature_cols:
        nn = df[c].notna().sum()
        print(f"  {c:38s}  {nn:>5,} / {len(df):,}  ({nn / len(df) * 100:5.1f}%)")
