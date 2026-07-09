"""
models/mlb/swing_delta_gate.py — BATCH 2: within-season swing-CHANGE deltas.

Declared before testing (2026-07-05): d_bat_speed, d_attack_angle, d_pull_air,
d_fast_swing = rolling 30d mean minus rolling 120d mean (both closed='left').
Thesis: books anchor HR/TB prices on season-level profiles; recent-vs-baseline
divergence flags swing changes before outcomes accumulate. Gate: control=v4,
candidate=v4+deltas, HR (Novig anchor) + TB (UD anchor), both directions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import contextlib, io as _io
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from db.db import query
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 _build_player_match, TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.research.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr
from models.mlb.features.advanced_profile_features import (_pull_batted_balls, _pull_swings,
                                                  _pull_spine, _spray_labels, _roll_rate)

DELTAS = ['d_bat_speed', 'd_attack_angle', 'd_pull_air', 'd_fast_swing']


def build_deltas(start, end):
    bb = _pull_batted_balls(start - timedelta(days=130), end + timedelta(days=1))
    sw = _pull_swings(start - timedelta(days=130), end + timedelta(days=1))
    spine = _pull_spine(start, end)
    for d in (bb, sw, spine):
        d['game_date'] = pd.to_datetime(d['game_date'])
    sd = spine[['player_id', 'game_date']].drop_duplicates()

    sw['n'] = 1.0
    sw['bs'] = sw['bat_speed'].astype(float)
    sw['aa'] = sw['attack_angle'].astype(float)
    sw['fast'] = (sw['bs'] >= 75).astype(float)
    out = sd.copy()
    for win, tag in [('30D', 's'), ('120D', 'l')]:
        r = _roll_rate(sw, ['bs', 'aa', 'fast', 'n'], win)
        r = r.drop_duplicates(['player_id', 'game_date'])   # same-date events share window
        r = sd.merge(r, on=['player_id', 'game_date'], how='left')
        ok = r['n'] >= (10 if tag == 's' else 40)
        out[f'bs_{tag}'] = np.where(ok, r['bs'] / r['n'], np.nan)
        out[f'aa_{tag}'] = np.where(ok, r['aa'] / r['n'], np.nan)
        out[f'fw_{tag}'] = np.where(ok, r['fast'] / r['n'], np.nan)
    bb = _spray_labels(bb); bb['bip'] = 1.0
    la = bb['launch_angle'].astype(float)
    bb['pa_'] = (bb['pulled'].astype(bool) & (la > 20)).astype(float)
    for win, tag, mn in [('30D', 's', 8), ('120D', 'l', 30)]:
        r = _roll_rate(bb, ['pa_', 'bip'], win)
        r = r.drop_duplicates(['player_id', 'game_date'])
        r = sd.merge(r, on=['player_id', 'game_date'], how='left')
        out[f'pa_{tag}'] = np.where(r['bip'] >= mn, r['pa_'] / r['bip'], np.nan)
    out['d_bat_speed'] = out['bs_s'] - out['bs_l']
    out['d_attack_angle'] = out['aa_s'] - out['aa_l']
    out['d_fast_swing'] = out['fw_s'] - out['fw_l']
    out['d_pull_air'] = out['pa_s'] - out['pa_l']
    out = spine[['game_id', 'player_id', 'game_date']].merge(
        out[['player_id', 'game_date'] + DELTAS], on=['player_id', 'game_date'], how='left')
    return out[['game_id', 'player_id'] + DELTAS].drop_duplicates(['game_id', 'player_id'])


def main():
    print("building swing deltas 2024-2026...")
    dl = build_deltas(date(2024, 1, 1), date(2026, 12, 31))
    print(f"  {len(dl):,} rows; delta stds: "
          f"{ {c: round(float(dl[c].std()),3) for c in DELTAS} }")
    adv = pd.read_parquet(ADV_CACHE)
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(dl, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(dl, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])

    for target in ['hr', 'tb']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb', Path('models/mlb/saved'))['features']
        v4 = base + ADV_FEATS + BATCH1; v7 = v4 + DELTAS
        m4 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m4.fit(trt[v4].values, y, verbose=False)
        m7 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m7.fit(trt[v7].values, y, verbose=False)
        btt = btt.copy()
        btt['p_v4'] = m4.predict(btt[v4].values); btt['p_v7'] = m7.predict(btt[v7].values)
        if target == 'tb':
            with contextlib.redirect_stdout(_io.StringIO()):
                J = attach_odds(btt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
        else:
            odds = query("""SELECT prop_date AS game_date, bp_player_id, over_odds, under_odds
                FROM bettingpros_props WHERE book_id=%(b)s AND market_id=%(m)s AND over_line=0.5
                AND over_odds IS NOT NULL AND under_odds IS NOT NULL""",
                params={'b': NOVIG_BOOK_ID, 'm': HR_MARKET_ID})
            odds['game_date'] = pd.to_datetime(odds['game_date'])
            with contextlib.redirect_stdout(_io.StringIO()):
                mt = _build_player_match(date(2025, 1, 1), date(2026, 12, 31))
            odds = odds.merge(mt[mt['player_id'].notna()][['bp_player_id', 'player_id']], on='bp_player_id', how='inner')
            J = btt.merge(odds, on=['game_date', 'player_id'], how='inner')
        io_ = 1 / J['over_odds'].apply(american_to_decimal); iu_ = 1 / J['under_odds'].apply(american_to_decimal)
        J['p_mkt'] = io_ / (io_ + iu_); J['y'] = J[label].astype(int); J['yr'] = J['game_date'].dt.year
        print(f"\n===== SWING-DELTA GATE [{target.upper()}] (control = v4) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            aa = {}
            for kk, cols in [('A', ['p_mkt']), ('V4', ['p_mkt', 'p_v4']), ('V7', ['p_mkt', 'p_v7'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                aa[kk] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa['A']:.4f} V4={aa['V4']:.4f} "
                  f"V7={aa['V7']:.4f}  V7-V4={aa['V7']-aa['V4']:+.4f}")
    print("\n(ACCEPT iff V7>V4 both directions on either target.)")


if __name__ == "__main__":
    main()
