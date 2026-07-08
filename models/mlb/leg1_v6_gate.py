"""
models/mlb/leg1_v6_gate.py — v6 candidate: SAME accepted features, FULL training coverage.

The accepted spray features (pull, oppo, pull-air, platoon pull, smash) existed in only
1 of 6 training seasons (extras backfilled 2024+). Spray/catcher now backfilled to 2019.
Lesson from luck-gap batch: full-coverage features punch harder. This gate tests identical
feature NAMES with 6-season coverage (v6) vs 1-season coverage (v5) as control.
Bat-tracking dims stay NaN pre-2024 (the tech didn't exist) — XGB routes them natively.

Gate: TB (UD anchor) + HR (Novig anchor), both directions. ACCEPT iff V6>V5 both
directions on either target.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import contextlib, io as _io
from datetime import date
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
from models.mlb.luck_gap_gate import build_luck, LUCK
from models.mlb.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr
from models.mlb.advanced_profile_features import build_training_set as build_adv

FULL_CACHE = "models/mlb/cache/adv_profile_2019_2026.parquet"


def main():
    if os.path.exists(FULL_CACHE):
        adv_full = pd.read_parquet(FULL_CACHE)
        print(f"loaded full-coverage cache: {len(adv_full):,}")
    else:
        print("building FULL-WINDOW advanced features 2019-2026 (one-time)...")
        adv_full = build_adv(date(2019, 3, 1), date(2026, 12, 31))
        adv_full.to_parquet(FULL_CACHE, index=False)
    adv_old = pd.read_parquet(ADV_CACHE)      # 2024+ only (v5's coverage)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))

    tr0 = pd.read_parquet(TRAIN_PQ); bt0 = pd.read_parquet(BTEST_PQ)
    frames = {}
    for tag, adv in [('v5', adv_old), ('v6', adv_full)]:
        tr = _attach_hr(tr0).merge(adv, on=['game_id', 'player_id'], how='left') \
             .merge(lk, on=['game_id', 'player_id'], how='left')
        bt = _attach_hr(bt0).merge(adv, on=['game_id', 'player_id'], how='left') \
             .merge(lk, on=['game_id', 'player_id'], how='left')
        for df in (tr, bt):
            df['game_date'] = pd.to_datetime(df['game_date'])
        frames[tag] = (tr, bt)
    cov5 = frames['v5'][0]['bat_pull_rate_120d'].notna().mean()
    cov6 = frames['v6'][0]['bat_pull_rate_120d'].notna().mean()
    print(f"train pull-rate coverage: v5 {cov5:.2f} -> v6 {cov6:.2f}")

    for target in ['tb', 'hr']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        preds = {}
        for tag in ('v5', 'v6'):
            tr, bt = frames[tag]
            trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
            base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb',
                               Path('models/mlb/saved'))['features']
            feats = base + ADV_FEATS + BATCH1 + LUCK
            m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
            m.fit(trt[feats].values, trt[label].astype(int).values, verbose=False)
            p = pd.Series(m.predict(btt[feats].values), index=btt.set_index(
                ['game_id', 'player_id']).index)
            preds[tag] = p
        tr, bt = frames['v6']
        btt = bt[bt[label].notna()].copy()
        idx = btt.set_index(['game_id', 'player_id']).index
        btt['p_v5'] = preds['v5'].reindex(idx).values
        btt['p_v6'] = preds['v6'].reindex(idx).values

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
            odds = odds.merge(mt[mt['player_id'].notna()][['bp_player_id', 'player_id']],
                              on='bp_player_id', how='inner')
            J = btt.merge(odds, on=['game_date', 'player_id'], how='inner')
        io_ = 1 / J['over_odds'].apply(american_to_decimal)
        iu_ = 1 / J['under_odds'].apply(american_to_decimal)
        J['p_mkt'] = io_ / (io_ + iu_); J['y'] = J[label].astype(int)
        J['yr'] = J['game_date'].dt.year
        print(f"\n===== V6 GATE [{target.upper()}] (control = v5 coverage) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            aa = {}
            for kk, col in [('A', ['p_mkt']), ('V5', ['p_mkt', 'p_v5']), ('V6', ['p_mkt', 'p_v6'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[col], f['y'])
                aa[kk] = roc_auc_score(t['y'], lm.predict_proba(t[col])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa['A']:.4f} V5={aa['V5']:.4f} "
                  f"V6={aa['V6']:.4f}  V6-V5={aa['V6']-aa['V5']:+.4f}  V6-A={aa['V6']-aa['A']:+.4f}")
    print("\n(ACCEPT iff V6>V5 both directions on either target.)")


if __name__ == "__main__":
    main()
