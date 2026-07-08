"""
models/mlb/blend_walkforward.py — ADAPTIVE BLEND backtest (walk-forward, leak-free).

Instead of fitting the market+model combiner on one YEAR and betting another (which
imports a stale regime: the over-shade drifted 2.3 -> 1.5 -> 0.6 pts across 2024-26 and
caused the 95%-unders / 86%-overs side floods), refit the 3-parameter logistic every
bet date on the TRAILING 90 days only. Always centered on the current regime; live-
deployable by construction. Purely additive experiment — touches no production files.

Burn-in: first ~45 days of 2025 have no trailing window and are skipped.
Usage: python -m models.mlb.blend_walkforward
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
from models.mlb.backtest import (load_bundle, attach_odds, american_to_decimal, TARGET_TO_MARKET)
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.leg1_attack3_test import ADV_FEATS
from models.mlb.leg1_v6_gate import FULL_CACHE as ADV_CACHE   # v6: full-coverage features
from models.mlb.leg1_batch_gate import BATCH1
from models.mlb.luck_gap_gate import build_luck, LUCK

WINDOW_D = 90
MIN_FIT = 500


def main():
    cfg = TARGET_TO_MARKET['tb']; label = cfg['label_col']
    adv = pd.read_parquet(ADV_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr[label].notna()]; bt = bt[bt[label].notna()]
    base = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    feats = base + ADV_FEATS + BATCH1 + LUCK
    m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m.fit(tr[feats].values, tr[label].astype(int).values, verbose=False)
    bt = bt.copy(); bt['p_v5'] = m.predict(bt[feats].values)

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[label].astype(int); B['yr'] = B['game_date'].dt.year
    B['dec_o'] = B['over_odds'].apply(american_to_decimal)
    B['dec_u'] = B['under_odds'].apply(american_to_decimal)
    B = B.sort_values('game_date')

    B['p_blend'] = np.nan
    fitted = skipped = 0
    for d in sorted(B['game_date'].unique()):
        lo = d - np.timedelta64(WINDOW_D, 'D')
        w = B[(B['game_date'] >= lo) & (B['game_date'] < d)]
        rows = B['game_date'] == d
        if len(w) < MIN_FIT or w['y'].nunique() < 2:
            skipped += rows.sum(); continue
        lm = LogisticRegression().fit(w[['p_mkt', 'p_v5']], w['y'])
        B.loc[rows, 'p_blend'] = lm.predict_proba(B.loc[rows, ['p_mkt', 'p_v5']])[:, 1]
        fitted += rows.sum()
    print(f"walk-forward: {fitted:,} scored, {skipped:,} skipped (burn-in/thin window)")

    D = B[B['p_blend'].notna()].copy()
    p = D['p_blend']
    evo = p * (D['dec_o'] - 1) - (1 - p); evu = (1 - p) * (D['dec_u'] - 1) - p
    over = evo >= evu
    D['side'] = np.where(over, 'over', 'under')
    D['ev'] = np.where(over, evo, evu)
    D['won'] = np.where(over, D['y'] == 1, D['y'] == 0)
    D['profit'] = np.where(D['won'], np.where(over, D['dec_o'], D['dec_u']) - 1, -1.0)

    print(f"\n===== ADAPTIVE BLEND ({WINDOW_D}d trailing) standalone @ UD odds =====")
    print(f"{'year':>6} {'thr':>5} {'n':>6} {'hit':>6} {'ROI':>8} {'±2SE':>7} {'over%':>6}")
    for yr in (2025, 2026):
        d0 = D[D['yr'] == yr]
        for thr in (0.0, 0.02, 0.04, 0.06):
            x = d0[d0['ev'] > thr]
            if len(x) < 25:
                print(f"{yr:>6} {thr:>5.2f} {len(x):>6,}   (too small)"); continue
            se = x['profit'].std() / np.sqrt(len(x))
            print(f"{yr:>6} {thr:>5.2f} {len(x):>6,} {x['won'].mean():>6.3f} "
                  f"{x['profit'].mean():>+8.4f} {2*se:>7.3f} {(x['side']=='over').mean():>6.2f}")
    print("\n(Year-swap reference: 2025 +0.5/+5.9/+14.7%; 2026 -0.5/-1.0/-0.2%;")
    print(" side floods 95%U/86%O. Success = balanced sides + both years >= 0.)")


if __name__ == "__main__":
    main()
