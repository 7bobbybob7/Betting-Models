"""
models/mlb/research/tb_line_conditional_gate.py — line-conditional TB architecture gate.

Transfer of the outs-model invention back to hitters: one model with THE LINE as input,
trained on every batter-game x rungs (0.5/1.5/2.5/3.5) = ~1.4M rows, learning the whole
survival curve P(TB > x). vs v6 (fixed 1.5-line) as control.

Evaluations:
  1. FAIR: both models on 1.5-line props only (v6's home turf) — does rung-sharing help?
  2. EXPANSION: line-conditional on ALL-rung props (0.5/2.5 — invisible to v6).
ACCEPT for production iff (1) LC >= v6 both directions AND (2) all-rung blend > market.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import contextlib, io as _io
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.hitter.backtest import (load_bundle, american_to_decimal,
                                        _build_player_match, TARGET_TO_MARKET)
from models.mlb.feature_sets import (TRAIN_PQ, BTEST_PQ, FULL_CACHE, XGB_PARAMS,
                                     ADV_ALL, build_luck)

RUNGS = [0.5, 1.5, 2.5, 3.5]
TB_MKT, UD = 293, 36


def main():
    adv = pd.read_parquet(FULL_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr['lbl_tb'].notna()]; bt = bt[bt['lbl_tb'].notna()]

    base = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    feats = base + ADV_ALL

    # control: v6 fixed-line
    y15 = (tr['lbl_tb'] > 1.5).astype(int).values
    m6 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m6.fit(tr[feats].values, y15, verbose=False)
    bt = bt.copy(); bt['p_v6'] = m6.predict(bt[feats].values)

    # candidate: line-conditional
    med = tr[feats].median()
    aug = []
    for r in RUNGS:
        a = tr[feats].fillna(med).copy(); a['line'] = r
        a['y'] = (tr['lbl_tb'].values > r).astype(int); aug.append(a)
    A = pd.concat(aug, ignore_index=True)
    print(f"line-conditional training rows: {len(A):,}")
    mlc = xgb.XGBClassifier(**XGB_PARAMS)
    mlc.fit(A[feats + ['line']].values, A['y'].values)

    # props at ALL lines (UD book)
    odds = query("""SELECT prop_date AS game_date, bp_player_id, over_line, over_odds, under_odds
        FROM bettingpros_props WHERE book_id=%(b)s AND market_id=%(m)s
        AND over_odds IS NOT NULL AND under_odds IS NOT NULL""", params={'b': UD, 'm': TB_MKT})
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    with contextlib.redirect_stdout(_io.StringIO()):
        mt = _build_player_match(date(2025, 1, 1), date(2026, 12, 31))
    odds = odds.merge(mt[mt['player_id'].notna()][['bp_player_id', 'player_id']],
                      on='bp_player_id', how='inner')
    J = bt.merge(odds, on=['game_date', 'player_id'], how='inner')
    X = J[feats].fillna(med).copy(); X['line'] = J['over_line'].values
    J['p_lc'] = mlc.predict_proba(X[feats + ['line']].values)[:, 1]
    io_ = 1 / J['over_odds'].apply(american_to_decimal); iu_ = 1 / J['under_odds'].apply(american_to_decimal)
    J['p_mkt'] = io_ / (io_ + iu_)
    J['y'] = (J['lbl_tb'] > J['over_line']).astype(int)
    J['yr'] = J['game_date'].dt.year

    for tag, sub in [("FAIR (1.5-line only)", J[J['over_line'] == 1.5]),
                     ("EXPANSION (all rungs)", J)]:
        print(f"\n===== {tag}: n={len(sub):,} "
              f"(rungs: {sub['over_line'].value_counts().to_dict()})")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = sub[sub['yr'] == fy], sub[sub['yr'] == ty]
            if len(f) < 300 or len(t) < 300:
                print(f"  {fy}->{ty}: thin"); continue
            arms = [('A', ['p_mkt']), ('LC', ['p_mkt', 'p_lc'])]
            if '1.5' in tag or 'FAIR' in tag:
                arms.insert(1, ('V6', ['p_mkt', 'p_v6']))
            a = {}
            for k, cols in arms:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            extra = f" V6={a['V6']:.4f} LC-V6={a['LC']-a['V6']:+.4f}" if 'V6' in a else ""
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={a['A']:.4f}{extra} "
                  f"LC={a['LC']:.4f} LC-A={a['LC']-a['A']:+.4f}")
    print("\n(ACCEPT iff FAIR: LC>=V6 both dirs AND EXPANSION: LC>A both dirs.)")


if __name__ == "__main__":
    main()
