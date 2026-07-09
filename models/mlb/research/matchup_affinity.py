"""
models/mlb/matchup_affinity.py — bilinear batter x pitcher matchup affinity (embeddings v1).

The n=7 problem: hard segment buckets (RHB pull vs sinker LHP) have no sample. Fix:
continuous interaction — every batter-pitcher pair contributes, weighted by profile
similarity. v1 is an explicit bilinear model: z-scored batter skill vector (8d) crossed
with pitcher arsenal vector (8d) -> 64 products -> L1 logistic on game-level outcomes.
The cross-only logit ("affinity") becomes ONE feature per target, then must pass the
standard batch gate (control = current v4) to earn its place. XGB gets main effects
already; this adds the explicit product structure trees approximate poorly in sparse cells.

Usage: python -m models.mlb.matchup_affinity     (fits affinities + runs TB and HR gates)
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
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 _build_player_match, TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.research.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr

BAT_DIMS = ['bat_xwoba_vs_FB_90d', 'bat_whiff_rate_vs_FB_90d', 'bat_xwoba_vs_SL_90d',
            'bat_whiff_rate_vs_SL_90d', 'bat_xwoba_vs_CH_90d',
            'bat_pull_air_rate_120d', 'bat_bat_speed_120d', 'bat_attack_angle_120d']
PIT_HAND_DIMS = ['FF', 'SI', 'SL', 'CH', 'CU']          # usage vs batter hand
PIT_GLOB_DIMS = ['pit_k_rate_szn', 'pit_whiff_rate_90d', 'pit_hr_per_9_szn']


def build_vectors(df):
    B = df[BAT_DIMS].astype(float).copy()
    P = pd.DataFrame(index=df.index)
    for pt in PIT_HAND_DIMS:
        P[f'use_{pt}'] = np.where(df['bat_hand'] == 'R',
                                  df[f'pit_pct_{pt}_vs_RHB_30d'], df[f'pit_pct_{pt}_vs_LHB_30d'])
    for c in PIT_GLOB_DIMS:
        P[c] = df[c].astype(float)
    return B, P


def cross_features(B, P, mu_b, sd_b, mu_p, sd_p):
    Bz = ((B - mu_b) / sd_b).fillna(0.0).values
    Pz = ((P - mu_p) / sd_p).fillna(0.0).values
    n = Bz.shape[0]
    X = np.einsum('ni,nj->nij', Bz, Pz).reshape(n, -1)
    return X


def main():
    adv = pd.read_parquet(ADV_CACHE)
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])

    Btr, Ptr = build_vectors(tr)
    mu_b, sd_b = Btr.mean(), Btr.std().replace(0, 1)
    mu_p, sd_p = Ptr.mean(), Ptr.std().replace(0, 1)
    Xtr = cross_features(Btr, Ptr, mu_b, sd_b, mu_p, sd_p)
    Bbt, Pbt = build_vectors(bt)
    Xbt = cross_features(Bbt, Pbt, mu_b, sd_b, mu_p, sd_p)
    print(f"cross features: {Xtr.shape[1]} products, train {Xtr.shape[0]:,}")

    for target, label in [('hr', 'lbl_hr'), ('tb', 'lbl_tb_over_15')]:
        m = tr[label].notna()
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.05, max_iter=2000)
        lr.fit(Xtr[m.values], tr.loc[m, label].astype(int))
        nz = (lr.coef_ != 0).sum()
        tr[f'aff_{target}'] = Xtr @ lr.coef_[0]
        bt[f'aff_{target}'] = Xbt @ lr.coef_[0]
        print(f"aff_{target}: {nz} nonzero of {Xtr.shape[1]} crosses")

    AFF = ['aff_hr', 'aff_tb']

    # ---- gate on TB (UD anchor) and HR (Novig anchor), control = current v4 ----
    for target in ['tb', 'hr']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb',
                           Path('models/mlb/saved'))['features']
        v4 = base + ADV_FEATS + BATCH1
        v5 = v4 + AFF
        m4 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
        m4.fit(trt[v4].values, y, verbose=False)
        m5 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
        m5.fit(trt[v5].values, y, verbose=False)
        btt = btt.copy()
        btt['p_v4'] = m4.predict(btt[v4].values)
        btt['p_v5'] = m5.predict(btt[v5].values)

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
        J['p_mkt'] = io_ / (io_ + iu_)
        J['y'] = J[label].astype(int)
        J['yr'] = J['game_date'].dt.year

        print(f"\n===== AFFINITY GATE [{target.upper()}] (control = v4) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            a = {}
            for k, cols in [('A', ['p_mkt']), ('V4', ['p_mkt', 'p_v4']), ('V5', ['p_mkt', 'p_v5'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={a['A']:.4f} V4={a['V4']:.4f} "
                  f"V5={a['V5']:.4f}  V5-V4={a['V5']-a['V4']:+.4f}")
    print("\n(ACCEPT affinity iff V5>V4 both directions on either target.)")


if __name__ == "__main__":
    main()
