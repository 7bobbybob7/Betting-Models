"""
models/mlb/leg1_attack4_test.py — LEG 1 v2 / Attack 4: train on the betting population.

Hypothesis (PRD H4): v1 trained on ALL starters, learning easy scrub-vs-star separation,
but bets only inside the market's 40-60% coin-flip clump. Restricting training to the
clump may sharpen discrimination exactly where bets happen.

Operationalization: no odds exist pre-2024, so the clump is proxied by the control model's
own predicted prob p in [0.35, 0.62] (HRR base rate ~0.44). In-sample prediction is used
ONLY for row selection, never as a feature/label.

Arms (blend fit 2025 -> test 2026, same as Attack 1):
    A: market alone                          (reference: 0.5587 from Attack 1 run)
    D: market + control-feats, clump-trained
    E: market + v2-feats,      clump-trained

Pass iff D or E beats A (and beats the full-trained equivalents from Attack 1).
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

from models.mlb.hitter.backtest import load_bundle, attach_odds, american_to_decimal, TARGET_TO_MARKET
from models.mlb.feature_sets import CTX_CACHE, TRAIN_PQ, BTEST_PQ, XGB_PARAMS, CTX_FEATS

TARGET = 'hrr'
CLUMP = (0.35, 0.62)


def main():
    cfg = TARGET_TO_MARKET[TARGET]
    ctx = pd.read_parquet(CTX_CACHE)
    tr = pd.read_parquet(TRAIN_PQ).merge(ctx, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(ctx, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[(tr['lbl_hrr_valid'] == True) & tr[cfg['label_col']].notna()]
    bt = bt[(bt['lbl_hrr_valid'] == True) & bt[cfg['label_col']].notna()]
    ytr = tr[cfg['label_col']].astype(int).values

    base_feats = load_bundle(TARGET, 'xgb', Path('models/mlb/saved'))['features']
    v2_feats = base_feats + CTX_FEATS

    # selector model (full population) -> clump filter
    print(f"fitting selector on {len(tr):,} rows...")
    sel = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    sel.fit(tr[base_feats].values, ytr, verbose=False)
    p_sel = sel.predict(tr[base_feats].values)
    clump = (p_sel >= CLUMP[0]) & (p_sel <= CLUMP[1])
    print(f"clump rows: {clump.sum():,} / {len(tr):,} ({clump.mean()*100:.0f}%)  "
          f"[p in {CLUMP}]")

    print("training clump-restricted models (control feats + v2 feats)...")
    m_d = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_d.fit(tr.loc[clump, base_feats].values, ytr[clump], verbose=False)
    m_e = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_e.fit(tr.loc[clump, v2_feats].values, ytr[clump], verbose=False)

    bt = bt.copy()
    bt['p_d'] = m_d.predict(bt[base_feats].values)
    bt['p_e'] = m_e.predict(bt[v2_feats].values)

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, TARGET, date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[cfg['label_col']].astype(int)
    B['yr'] = B['game_date'].dt.year
    fit, te = B[B['yr'] == 2025], B[B['yr'] == 2026]

    print("\n========= ATTACK 4 RESIDUAL TEST (fit 2025 -> test 2026) =========")
    for name, cols in [('A: market alone', ['p_mkt']),
                       ('D: market + ctl-feats clump-trained', ['p_mkt', 'p_d']),
                       ('E: market + v2-feats  clump-trained', ['p_mkt', 'p_e'])]:
        lm = LogisticRegression().fit(fit[cols], fit['y'])
        auc = roc_auc_score(te['y'], lm.predict_proba(te[cols])[:, 1])
        coef = "" if len(cols) == 1 else f"  model_coef={lm.coef_[0][1]:+.3f}"
        print(f"  {name:38s} AUC={auc:.4f}{coef}")
    print("\n  (Attack 1 reference: A=0.5587, full-trained B=0.5538, C=0.5568)")


if __name__ == "__main__":
    main()
