"""
models/mlb/game_context_features.py — LEG 1 v2 / Attack 1: fresh game-context features.

The features the prop market plausibly prices lazily (see docs/LEG1_MODEL_V2_PRD.md):

    ctx_ump_k_rate_365d          HP umpire's K/PA in games officiated, rolling 365d
    ctx_ump_bb_rate_365d         same for BB/PA (small zone -> more walks)
    ctx_ump_runs_pg_365d         runs/game in his games (overall run environment)
    ctx_opp_bullpen_relievers_1d # opposing relievers who pitched the PRIOR day
    ctx_opp_bullpen_ip_2d        opposing bullpen IP over the prior 2 days
    ctx_opp_bullpen_ip_3d        opposing bullpen IP over the prior 3 days
    ctx_batter_rest_days         days since the batter's previous game (capped at 7)
    ctx_batter_games_7d          batter's games played in the prior 7 days

STRICT NO-LEAKAGE CONTRACT
    Umpire tendencies: rolling 365d with closed='left' — the game being predicted never
    contributes to its own ump features. Bullpen: strictly PRIOR calendar days. Rest:
    strictly prior appearances. The ump's IDENTITY is game metadata known before first
    pitch (announced day-of), so using it is not a leak — but live use is day-of only.

API
    build_training_set(start_date, end_date) -> one row per (game_id, player_id) for every
    batter-game in the window, ready to left-merge onto the cached dataset parquets on
    (game_id, player_id).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import argparse
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from db.db import query

UMP_WINDOW = '365D'
MIN_UMP_GAMES = 20        # below this the ump features are NaN (model imputes)
REST_CAP = 7


# ----------------------------------------------------------------------------
# Raw pulls
# ----------------------------------------------------------------------------

def _pull_ump_games(start: date, end: date) -> pd.DataFrame:
    """One row per game with HP ump + that game's K/BB/PA/runs totals (both teams)."""
    sql = """
        SELECT g.game_id, g.game_date, gi.umpire_hp,
               g.home_score + g.away_score AS runs,
               bg.so_total, bg.bb_total, bg.pa_total
        FROM games g
        JOIN mlb_game_info gi ON g.game_id = gi.game_id
        JOIN (SELECT game_id, SUM(so) AS so_total, SUM(bb) AS bb_total, SUM(pa) AS pa_total
              FROM mlb_batting_game GROUP BY game_id) bg ON g.game_id = bg.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND gi.umpire_hp IS NOT NULL
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


def _pull_bullpen_daily(start: date, end: date) -> pd.DataFrame:
    """Per (team_id, game_date): relievers used + bullpen IP that day."""
    sql = """
        SELECT pg.team_id, g.game_date,
               COUNT(*) AS relievers_used,
               SUM(pg.ip) AS bullpen_ip
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE g.sport_id = 2 AND g.status = 'final' AND pg.is_starter = false
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
        GROUP BY pg.team_id, g.game_date
    """
    return query(sql, params={'s': start, 'e': end})


def _pull_spine(start: date, end: date) -> pd.DataFrame:
    """(game_id, player_id, game_date, batter team, opponent team, ump)."""
    sql = """
        SELECT bg.game_id, bg.player_id, g.game_date, bg.team_id AS batter_team_id,
               CASE WHEN bg.team_id = g.home_team_id THEN g.away_team_id
                    ELSE g.home_team_id END AS opp_team_id,
               gi.umpire_hp
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = 2 AND g.status = 'final'
          AND bg.batting_order BETWEEN 1 AND 9 AND bg.pa > 0
          AND g.game_date >= %(s)s AND g.game_date <= %(e)s
    """
    return query(sql, params={'s': start, 'e': end})


# ----------------------------------------------------------------------------
# Feature builders
# ----------------------------------------------------------------------------

def _ump_features(ump_games: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id): rolling 365d closed-left K/PA, BB/PA, runs/game for that game's ump."""
    df = ump_games.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['gp'] = 1
    df = df.sort_values(['umpire_hp', 'game_date'])
    chunks = []
    for ump, grp in df.groupby('umpire_hp', sort=False):
        g = grp.set_index('game_date')[['so_total', 'bb_total', 'pa_total', 'runs', 'gp']]
        rolled = g.rolling(UMP_WINDOW, closed='left').sum()
        rolled['game_id'] = grp['game_id'].values
        chunks.append(rolled.reset_index(drop=True))
    r = pd.concat(chunks, ignore_index=True)
    enough = r['gp'] >= MIN_UMP_GAMES
    out = pd.DataFrame({'game_id': r['game_id']})
    out['ctx_ump_k_rate_365d']  = np.where(enough, r['so_total'] / r['pa_total'], np.nan)
    out['ctx_ump_bb_rate_365d'] = np.where(enough, r['bb_total'] / r['pa_total'], np.nan)
    out['ctx_ump_runs_pg_365d'] = np.where(enough, r['runs'] / r['gp'], np.nan)
    return out


def _bullpen_lookup(bp_daily: pd.DataFrame) -> dict:
    """(team_id, date) -> (relievers_used, bullpen_ip)."""
    bp_daily = bp_daily.copy()
    bp_daily['game_date'] = pd.to_datetime(bp_daily['game_date']).dt.date
    return {(int(r.team_id), r.game_date): (int(r.relievers_used), float(r.bullpen_ip or 0))
            for r in bp_daily.itertuples()}


def _bullpen_features(spine: pd.DataFrame, lut: dict) -> pd.DataFrame:
    """Opposing bullpen usage over strictly PRIOR days."""
    dates = pd.to_datetime(spine['game_date']).dt.date
    rel1, ip2, ip3 = [], [], []
    for opp, d in zip(spine['opp_team_id'], dates):
        opp = int(opp)
        prior = [lut.get((opp, d - timedelta(days=k)), (0, 0.0)) for k in (1, 2, 3)]
        rel1.append(prior[0][0])
        ip2.append(prior[0][1] + prior[1][1])
        ip3.append(prior[0][1] + prior[1][1] + prior[2][1])
    return pd.DataFrame({
        'game_id': spine['game_id'].values, 'player_id': spine['player_id'].values,
        'ctx_opp_bullpen_relievers_1d': rel1,
        'ctx_opp_bullpen_ip_2d': ip2,
        'ctx_opp_bullpen_ip_3d': ip3,
    })


def _rest_features(spine: pd.DataFrame) -> pd.DataFrame:
    """Batter rest: days since previous game (capped), games in prior 7 days."""
    df = spine[['game_id', 'player_id', 'game_date']].copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['player_id', 'game_date'])
    df['prev'] = df.groupby('player_id')['game_date'].shift(1)
    df['ctx_batter_rest_days'] = ((df['game_date'] - df['prev']).dt.days
                                  .clip(upper=REST_CAP).fillna(REST_CAP))
    # games in strictly-prior 7 days, closed-left rolling count
    df['one'] = 1
    counts = []
    for pid, grp in df.groupby('player_id', sort=False):
        c = grp.set_index('game_date')['one'].rolling('7D', closed='left').sum()
        counts.append(c.reset_index(drop=True))
    df['ctx_batter_games_7d'] = pd.concat(counts, ignore_index=True).fillna(0).values
    return df[['game_id', 'player_id', 'ctx_batter_rest_days', 'ctx_batter_games_7d']]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def build_training_set(start_date: date, end_date: date) -> pd.DataFrame:
    """One row per (game_id, player_id) with all 8 context features."""
    pull_start = start_date - timedelta(days=370)   # ump window needs history
    print(f"[game_context] pulling raw data {pull_start} -> {end_date}...")
    ump_games = _pull_ump_games(pull_start, end_date + timedelta(days=1))
    bp_daily  = _pull_bullpen_daily(start_date - timedelta(days=4), end_date + timedelta(days=1))
    # spine pulled with a 10-day buffer so rest/games_7d see prior appearances at the
    # window edge; trimmed back to [start, end] after feature computation.
    spine     = _pull_spine(start_date - timedelta(days=10), end_date)
    print(f"  ump games: {len(ump_games):,} | bullpen team-days: {len(bp_daily):,} | "
          f"spine(+buffer): {len(spine):,} batter-games")

    ump = _ump_features(ump_games)
    out = spine.merge(ump, on='game_id', how='left')
    out = out.merge(_bullpen_features(spine, _bullpen_lookup(bp_daily)),
                    on=['game_id', 'player_id'], how='left')
    out = out.merge(_rest_features(spine), on=['game_id', 'player_id'], how='left')

    out = out[pd.to_datetime(out['game_date']) >= pd.Timestamp(start_date)]
    feats = [c for c in out.columns if c.startswith('ctx_')]
    out = out[['game_id', 'player_id'] + feats].drop_duplicates(['game_id', 'player_id'])
    print(f"[game_context] done: {len(out):,} rows x {len(feats)} features")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end",   default="2025-06-15")
    args = ap.parse_args()
    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end,   "%Y-%m-%d").date()
    df = build_training_set(s, e)
    feats = [c for c in df.columns if c.startswith('ctx_')]
    print("\n=== coverage / distribution ===")
    for c in feats:
        col = df[c].astype(float)
        print(f"  {c:32s} nn={col.notna().mean()*100:5.1f}%  "
              f"mean={col.mean():7.3f}  p10={col.quantile(.1):7.3f}  p90={col.quantile(.9):7.3f}")
