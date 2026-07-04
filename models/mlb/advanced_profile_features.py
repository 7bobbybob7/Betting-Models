"""
models/mlb/advanced_profile_features.py — LEG 1 v2 / Attack 3: features from the newly
backfilled Statcast columns (mlb_pitch_extras). These are the datasets the market has had
least time to absorb, and the ones that let us measure the user's archetype for real.

Batter profiles (rolling 120d, closed='left', per batter):
    bat_pull_rate_120d        pulled BIP / BIP   (TRUE pull%, from hc_x/hc_y + stand)
    bat_oppo_rate_120d        opposite-field BIP / BIP
    bat_bat_speed_120d        avg swing speed (mph)
    bat_swing_len_120d        avg swing length (ft)
    bat_attack_angle_120d     avg attack angle (deg)
    bat_fast_swing_rate_120d  share of swings >= 75 mph ("A-swings")

Opposing-catcher framing (rolling 200d, closed='left', per catcher, joined via game start):
    ctx_catcher_framing_120d  called-strike rate on EDGE takes, minus league avg
                              (positive = steals strikes -> more Ks against)

NO-LEAKAGE: every rolling window closed='left'; catcher framing keyed to the opposing
starter's battery is approximated by the catcher who caught the most pitches for the
opposing team that game (see _game_catcher). All strictly < as_of_date.

Pull/oppo geometry: hc_x,hc_y are Statcast hit coords (home plate ~ (125, 199), y increases
toward the field pointing "up"). Spray angle = atan2(hc_x-125, 199-hc_y). For a RHB, pulled
= hit to the left side (negative angle); mirror for LHB. Switch hitters use `stand`.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from db.db import query

BAT_WINDOW = '120D'
FRAME_WINDOW = '200D'
MIN_BIP = 30
MIN_SWINGS = 40
MIN_FRAME_TAKES = 200
FAST_SWING = 75.0    # mph threshold for an "A-swing"


# ----------------------------------------------------------------------------
# Raw pulls (join mlb_pitches for stand/is_swing/in_play + zone; extras for coords/tracking)
# ----------------------------------------------------------------------------

def _pull_batted_balls(start: date, end: date) -> pd.DataFrame:
    """Per BIP: batter, date, stand, hc_x, hc_y (for pull/oppo)."""
    sql = """
        SELECT p.batter_id AS player_id, g.game_date, pl.bats AS stand,
               e.hc_x, e.hc_y
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e
          ON e.game_id = p.game_id AND e.at_bat_number = p.at_bat_number
         AND e.pitch_number = p.pitch_number
        JOIN games g ON p.game_id = g.game_id
        JOIN players pl ON p.batter_id = pl.player_id
        WHERE p.is_in_play = true AND e.hc_x IS NOT NULL AND e.hc_y IS NOT NULL
          AND pl.bats IN ('L','R','S')
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


def _pull_swings(start: date, end: date) -> pd.DataFrame:
    """Per swing with bat-tracking: batter, date, bat_speed, swing_length, attack_angle."""
    sql = """
        SELECT p.batter_id AS player_id, g.game_date,
               e.bat_speed, e.swing_length, e.attack_angle
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e
          ON e.game_id = p.game_id AND e.at_bat_number = p.at_bat_number
         AND e.pitch_number = p.pitch_number
        JOIN games g ON p.game_id = g.game_id
        WHERE e.bat_speed IS NOT NULL
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


def _pull_edge_takes(start: date, end: date) -> pd.DataFrame:
    """Per edge take (no swing, near zone): game, date, catcher, called_strike flag."""
    sql = """
        SELECT e.catcher_mlbam, g.game_date, p.game_id,
               CASE WHEN p.description = 'called_strike' THEN 1 ELSE 0 END AS called_strike
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e
          ON e.game_id = p.game_id AND e.at_bat_number = p.at_bat_number
         AND e.pitch_number = p.pitch_number
        JOIN games g ON p.game_id = g.game_id
        WHERE p.is_swing = false AND e.catcher_mlbam IS NOT NULL
          AND p.plate_x IS NOT NULL AND ABS(p.plate_x) BETWEEN 0.7 AND 1.1
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


def _pull_spine(start: date, end: date) -> pd.DataFrame:
    """(game_id, player_id, game_date, opp_team_id) for batters who started."""
    sql = """
        SELECT bg.game_id, bg.player_id, g.game_date,
               CASE WHEN bg.team_id = g.home_team_id THEN g.away_team_id
                    ELSE g.home_team_id END AS opp_team_id
        FROM mlb_batting_game bg JOIN games g ON bg.game_id = g.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND bg.batting_order BETWEEN 1 AND 9 AND bg.pa > 0
          AND g.game_date >= %(s)s AND g.game_date <= %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


def _game_catcher(start: date, end: date) -> pd.DataFrame:
    """Primary catcher per (game_id, fielding team) = who caught the most pitches."""
    sql = """
        SELECT p.game_id,
               CASE WHEN p.top_bottom = 'top' THEN g.home_team_id ELSE g.away_team_id END AS field_team,
               e.catcher_mlbam, COUNT(*) AS pitches
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e
          ON e.game_id = p.game_id AND e.at_bat_number = p.at_bat_number
         AND e.pitch_number = p.pitch_number
        JOIN games g ON p.game_id = g.game_id
        WHERE e.catcher_mlbam IS NOT NULL
          AND g.game_date >= %(s)s AND g.game_date <= %(e)s
        GROUP BY 1, 2, 3
    """
    df = query(sql, params={'s': start, 'e': end})
    if df.empty:
        return df
    return (df.sort_values('pitches').groupby(['game_id', 'field_team'], as_index=False)
              .last()[['game_id', 'field_team', 'catcher_mlbam']])


# ----------------------------------------------------------------------------
# Feature computation
# ----------------------------------------------------------------------------

def _spray_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add pulled/oppo booleans from hc_x/hc_y + handedness."""
    df = df.copy()
    ang = np.degrees(np.arctan2(df['hc_x'] - 125.0, 199.0 - df['hc_y']))  # neg=left field
    stand = df['stand'].where(df['stand'] != 'S', 'R')  # switch: approx as RHB (faced LHP mostly)
    # RHB pulls to left (negative angle); LHB pulls to right (positive)
    df['pulled'] = np.where(stand == 'R', ang < -15, ang > 15).astype(float)
    df['oppo']   = np.where(stand == 'R', ang > 15, ang < -15).astype(float)
    return df


def _roll_rate(df, num_cols, window, group='player_id'):
    df = df.sort_values([group, 'game_date'])
    out = []
    for gid, grp in df.groupby(group, sort=False):
        r = grp.set_index('game_date')[num_cols].rolling(window, closed='left').sum().reset_index(drop=True)
        r[group] = gid
        r['game_date'] = grp['game_date'].values
        out.append(r)
    return pd.concat(out, ignore_index=True)


def build_training_set(start_date: date, end_date: date) -> pd.DataFrame:
    pull_start = start_date - timedelta(days=210)
    print(f"[adv_profile] pulling raw {pull_start} -> {end_date}...")
    bb = _pull_batted_balls(pull_start, end_date + timedelta(days=1))
    sw = _pull_swings(pull_start, end_date + timedelta(days=1))
    et = _pull_edge_takes(pull_start, end_date + timedelta(days=1))
    spine = _pull_spine(start_date, end_date)
    gc = _game_catcher(start_date - timedelta(days=1), end_date)
    for d in (bb, sw, et, spine):
        d['game_date'] = pd.to_datetime(d['game_date'])
    print(f"  bip={len(bb):,} swings={len(sw):,} edge_takes={len(et):,} spine={len(spine):,}")

    # out is the spine, one row per (game_id, player_id); every feature left-merges onto it.
    out = spine[['game_id', 'player_id', 'game_date', 'opp_team_id']].drop_duplicates(
        ['game_id', 'player_id']).copy()
    spine_dates = out[['player_id', 'game_date']].drop_duplicates()

    # ---- batter spray (pull/oppo) ----
    bb = _spray_labels(bb); bb['bip'] = 1.0
    r = _roll_rate(bb, ['pulled', 'oppo', 'bip'], BAT_WINDOW)
    sp = spine_dates.merge(r, on=['player_id', 'game_date'], how='left')
    ok = sp['bip'] >= MIN_BIP
    sp['bat_pull_rate_120d'] = np.where(ok, sp['pulled'] / sp['bip'], np.nan)
    sp['bat_oppo_rate_120d'] = np.where(ok, sp['oppo'] / sp['bip'], np.nan)
    out = out.merge(sp[['player_id', 'game_date', 'bat_pull_rate_120d', 'bat_oppo_rate_120d']],
                    on=['player_id', 'game_date'], how='left')

    # ---- bat tracking ----
    sw['n'] = 1.0
    sw['fast'] = (sw['bat_speed'].astype(float) >= FAST_SWING).astype(float)
    sw['bs_sum'] = sw['bat_speed'].astype(float)
    sw['sl_sum'] = sw['swing_length'].astype(float)
    sw['aa_sum'] = sw['attack_angle'].astype(float)
    r = _roll_rate(sw, ['bs_sum', 'sl_sum', 'aa_sum', 'fast', 'n'], BAT_WINDOW)
    t = spine_dates.merge(r, on=['player_id', 'game_date'], how='left')
    ok = t['n'] >= MIN_SWINGS
    t['bat_bat_speed_120d']       = np.where(ok, t['bs_sum'] / t['n'], np.nan)
    t['bat_swing_len_120d']       = np.where(ok, t['sl_sum'] / t['n'], np.nan)
    t['bat_attack_angle_120d']    = np.where(ok, t['aa_sum'] / t['n'], np.nan)
    t['bat_fast_swing_rate_120d'] = np.where(ok, t['fast'] / t['n'], np.nan)
    out = out.merge(t[['player_id', 'game_date', 'bat_bat_speed_120d', 'bat_swing_len_120d',
                       'bat_attack_angle_120d', 'bat_fast_swing_rate_120d']],
                    on=['player_id', 'game_date'], how='left')

    # ---- opposing-catcher framing: per-catcher rolling rate, joined via the game's
    #      primary catcher for the batter's OPPONENT ----
    lg = et['called_strike'].mean()
    et['cs'] = et['called_strike'].astype(float); et['takes'] = 1.0
    fr = _roll_rate(et.rename(columns={'catcher_mlbam': 'player_id'}),
                    ['cs', 'takes'], FRAME_WINDOW).rename(columns={'player_id': 'catcher_mlbam'})
    okf = fr['takes'] >= MIN_FRAME_TAKES
    fr['ctx_catcher_framing_120d'] = np.where(okf, fr['cs'] / fr['takes'] - lg, np.nan)
    gc2 = gc.rename(columns={'field_team': 'opp_team_id'})     # (game_id, opp_team_id) -> catcher
    out = out.merge(gc2, on=['game_id', 'opp_team_id'], how='left')
    out = out.merge(fr[['catcher_mlbam', 'game_date', 'ctx_catcher_framing_120d']],
                    on=['catcher_mlbam', 'game_date'], how='left')

    feats = [c for c in out.columns if c.startswith(('bat_', 'ctx_'))]
    out = out[['game_id', 'player_id'] + feats].drop_duplicates(['game_id', 'player_id'])
    print(f"[adv_profile] done: {len(out):,} rows x {len(feats)} features")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end",   default="2026-05-15")
    args = ap.parse_args()
    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end,   "%Y-%m-%d").date()
    df = build_training_set(s, e)
    feats = [c for c in df.columns if c.startswith(('bat_', 'ctx_'))]
    print("\n=== coverage / distribution ===")
    for c in feats:
        col = df[c].astype(float)
        print(f"  {c:30s} nn={col.notna().mean()*100:5.1f}%  mean={col.mean():7.3f}  "
              f"p10={col.quantile(.1):7.3f}  p90={col.quantile(.9):7.3f}")
