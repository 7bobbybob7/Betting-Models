"""
models/mlb/feature_sets.py — SINGLE SOURCE OF TRUTH for the accepted Leg-1 feature stack.

Production (train_v3, v3_tracker) and research scripts both import from here, so the
daily cron never depends on one-shot experiment files. When a research batch is ACCEPTED
by its gate, its feature list (and any builder production needs) gets promoted into this
module; the gate script that proved it stays frozen in models/mlb/research/.

Acceptance history (gates in models/mlb/research/):
    ADV_FEATS  — Attack 3 (spray/bat-tracking/framing), accepted on TB both directions
    BATCH1     — pull-air / smash factor / platoon pull / speed-vs-95, accepted (v4)
    LUCK       — actual-minus-deserved x-stat gaps, accepted on TB (v5)
    FULL_CACHE — same features rebuilt at FULL 2019-2026 coverage, accepted BOTH targets (v6)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from datetime import date, timedelta

import numpy as np
import pandas as pd

from db.db import query

# ---------------------------------------------------------------------------
# Dataset caches + model hyperparameters (shared by every gate + trainer)
# ---------------------------------------------------------------------------
TRAIN_PQ = "models/mlb/cache/train_2019_2024.parquet"
BTEST_PQ = "models/mlb/cache/backtest_2025_2026.parquet"
ADV_CACHE = "models/mlb/cache/adv_profile_2024_2026.parquet"      # legacy 2024+ coverage
FULL_CACHE = "models/mlb/cache/adv_profile_2019_2026.parquet"     # v6 full coverage
CTX_CACHE = "models/mlb/cache/game_context_2019_2026.parquet"     # Attack-1 context feats
XGB_PARAMS = dict(max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=50, random_state=42, verbosity=0)

# ---------------------------------------------------------------------------
# Accepted feature lists
# ---------------------------------------------------------------------------
ADV_FEATS = ['bat_pull_rate_120d', 'bat_oppo_rate_120d', 'bat_bat_speed_120d',
             'bat_swing_len_120d', 'bat_attack_angle_120d', 'bat_fast_swing_rate_120d',
             'ctx_catcher_framing_120d']
BATCH1 = ['bat_pull_air_rate_120d', 'bat_smash_factor_120d',
          'bat_pull_rate_vs_L_120d', 'bat_pull_rate_vs_R_120d', 'bat_speed_vs95_120d']
LUCK = ['luck_ba_60d', 'luck_slg_60d', 'luck_slg_120d']
ADV_ALL = ADV_FEATS + BATCH1 + LUCK

# Attack-1 game-context features (REJECTED by residual gate; kept for reference/reruns)
CTX_FEATS = ['ctx_ump_k_rate_365d', 'ctx_ump_bb_rate_365d', 'ctx_ump_runs_pg_365d',
             'ctx_opp_bullpen_relievers_1d', 'ctx_opp_bullpen_ip_2d', 'ctx_opp_bullpen_ip_3d',
             'ctx_batter_rest_days', 'ctx_batter_games_7d']


# ---------------------------------------------------------------------------
# Luck-gap builder (promoted from research/luck_gap_gate.py — batch 3, accepted)
# ---------------------------------------------------------------------------
def build_luck(start, end):
    """Rolling actual-minus-deserved contact quality per (game_id, player_id)."""
    from models.mlb.features.advanced_profile_features import _roll_rate
    x = query("""
        SELECT p.batter_id AS player_id, g.game_date,
               COUNT(*) AS bip, SUM(p.xba) AS xba_s, SUM(p.xslg) AS xslg_s
        FROM mlb_pitches p JOIN games g ON p.game_id = g.game_id
        WHERE p.is_in_play = true AND p.xba IS NOT NULL AND g.sport_id = 2
          AND g.game_date >= %(s)s AND g.game_date < %(e)s
        GROUP BY 1, 2""", params={'s': start - timedelta(days=130), 'e': end + timedelta(days=1)})
    a = query("""
        SELECT bg.player_id, g.game_date, bg.hits AS h,
               bg.hits + bg.doubles + 2*bg.triples + 3*bg.hr AS tb_a, bg.game_id
        FROM mlb_batting_game bg JOIN games g ON bg.game_id = g.game_id
        WHERE g.sport_id = 2 AND g.status = 'final' AND bg.pa > 0
          AND g.game_date >= %(s)s AND g.game_date <= %(e)s""",
        params={'s': start - timedelta(days=130), 'e': end})
    for d in (x, a):
        d['game_date'] = pd.to_datetime(d['game_date'])
    m = a.merge(x, on=['player_id', 'game_date'], how='left')
    for c in ('bip', 'xba_s', 'xslg_s'):
        m[c] = m[c].fillna(0.0).astype(float)
    m['h'] = m['h'].astype(float); m['tb_a'] = m['tb_a'].astype(float)
    out = m[['game_id', 'player_id', 'game_date']].copy()
    for win, tag, mn in [('60D', '60d', 25), ('120D', '120d', 50)]:
        r = _roll_rate(m, ['h', 'tb_a', 'bip', 'xba_s', 'xslg_s'], win)
        r = r.drop_duplicates(['player_id', 'game_date'])
        r = m[['player_id', 'game_date']].merge(r, on=['player_id', 'game_date'], how='left')
        ok = r['bip'] >= mn
        out[f'luck_ba_{tag}'] = np.where(ok, (r['h'] - r['xba_s']) / r['bip'], np.nan)
        out[f'luck_slg_{tag}'] = np.where(ok, (r['tb_a'] - r['xslg_s']) / r['bip'], np.nan)
    return out[['game_id', 'player_id'] + LUCK].drop_duplicates(['game_id', 'player_id'])
