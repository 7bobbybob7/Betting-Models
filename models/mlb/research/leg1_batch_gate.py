"""
models/mlb/leg1_batch_gate.py — feature-batch iteration gate for the v3 model.

Tests a pre-declared BATCH of candidate features against the CURRENT v3 feature set as
control, on the same protocol that validated Attack 3 (TB, blend residual, both time
directions). Accept the batch only if V4 beats V3 in BOTH directions.

Batch 1 (declared 2026-07-05, before testing):
    bat_pull_air_rate_120d    pulled balls in the air (LA>20) — the XBH/HR signal
    bat_smash_factor_120d     EV / bat speed on contact — contact-quality efficiency
    bat_pull_rate_vs_L_120d   platoon-split pull (vs LHP)
    bat_pull_rate_vs_R_120d   platoon-split pull (vs RHP)
    bat_speed_vs95_120d       bat speed vs premium velocity (>=95)

Usage: python -m models.mlb.leg1_batch_gate       (rebuilds adv cache first)
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

from models.mlb.features.advanced_profile_features import build_training_set as build_adv
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS

BATCH1 = ['bat_pull_air_rate_120d', 'bat_smash_factor_120d',
          'bat_pull_rate_vs_L_120d', 'bat_pull_rate_vs_R_120d', 'bat_speed_vs95_120d']


def main():
    print("rebuilding adv cache with batch-1 columns...")
    adv = build_adv(date(2024, 1, 1), date(2026, 12, 31))
    adv.to_parquet(ADV_CACHE, index=False)

    cfg = TARGET_TO_MARKET['tb']
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr[cfg['label_col']].notna()]; bt = bt[bt[cfg['label_col']].notna()]
    ytr = tr[cfg['label_col']].astype(int).values

    base = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    v3_feats = base + ADV_FEATS
    v4_feats = base + ADV_FEATS + BATCH1
    print(f"training V3 control ({len(v3_feats)}) and V4 candidate ({len(v4_feats)}) "
          f"on {len(tr):,} rows...")
    m3 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m3.fit(tr[v3_feats].values, ytr, verbose=False)
    m4 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m4.fit(tr[v4_feats].values, ytr, verbose=False)

    bt = bt.copy()
    bt['p_v3'] = m3.predict(bt[v3_feats].values)
    bt['p_v4'] = m4.predict(bt[v4_feats].values)

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[cfg['label_col']].astype(int)
    B['yr'] = B['game_date'].dt.year

    print("\n========= BATCH 1 GATE (TB, both directions) =========")
    passes = []
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = B[B['yr'] == fy], B[B['yr'] == ty]
        aucs = {}
        for k, cols in [('A', ['p_mkt']), ('V3', ['p_mkt', 'p_v3']), ('V4', ['p_mkt', 'p_v4'])]:
            lm = LogisticRegression().fit(f[cols], f['y'])
            aucs[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        d43 = aucs['V4'] - aucs['V3']
        passes.append(d43 > 0)
        print(f"  fit {fy} -> test {ty} (n={len(t):,}): "
              f"A={aucs['A']:.4f}  V3={aucs['V3']:.4f}  V4={aucs['V4']:.4f}  "
              f"V4-V3={d43:+.4f}  V4-A={aucs['V4']-aucs['A']:+.4f}")
    verdict = "ACCEPT batch 1 -> retrain bundle as v4" if all(passes) else \
              "REJECT batch 1 -> keep v3 bundle (batch stays in cache, unused)"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
