"""
models/mlb/luck_gap_gate.py — BATCH 3: "luck gap" (actual minus deserved outcomes).

Thesis: books anchor prices on ACTUAL results; Statcast x-stats measure DESERVED contact
quality. rolling(actual - expected) < 0 -> underperforming skill -> market underprices.
Declared 2026-07-05: luck_ba_60d, luck_slg_60d, luck_slg_120d. Full 2019+ coverage
(unlike bat tracking) so all six training seasons carry signal.
Gate: control=v4, candidate=v4+luck, HR (Novig) + TB (UD), both directions.
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
from models.mlb.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 _build_player_match, TARGET_TO_MARKET)
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.leg1_attack3_test import ADV_CACHE, ADV_FEATS
from models.mlb.leg1_batch_gate import BATCH1
from models.mlb.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr
from models.mlb.advanced_profile_features import _roll_rate

LUCK = ['luck_ba_60d', 'luck_slg_60d', 'luck_slg_120d']


def build_luck(start, end):
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


def main():
    print("building luck-gap features 2019-2026...")
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    print(f"  {len(lk):,} rows; stds: { {c: round(float(lk[c].std()),4) for c in LUCK} }")
    adv = pd.read_parquet(ADV_CACHE)
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"  luck coverage train: {tr[LUCK].notna().mean().round(2).to_dict()}")

    for target in ['hr', 'tb']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb', Path('models/mlb/saved'))['features']
        v4 = base + ADV_FEATS + BATCH1; v8 = v4 + LUCK
        m4 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m4.fit(trt[v4].values, y, verbose=False)
        m8 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m8.fit(trt[v8].values, y, verbose=False)
        btt = btt.copy()
        btt['p_v4'] = m4.predict(btt[v4].values); btt['p_v8'] = m8.predict(btt[v8].values)
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
        print(f"\n===== LUCK-GAP GATE [{target.upper()}] (control = v4) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            aa = {}
            for kk, cols in [('A', ['p_mkt']), ('V4', ['p_mkt', 'p_v4']), ('V8', ['p_mkt', 'p_v8'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                aa[kk] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa['A']:.4f} V4={aa['V4']:.4f} "
                  f"V8={aa['V8']:.4f}  V8-V4={aa['V8']-aa['V4']:+.4f}")
    print("\n(ACCEPT iff V8>V4 both directions on either target.)")


if __name__ == "__main__":
    main()
