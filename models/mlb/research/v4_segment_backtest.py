"""
models/mlb/v4_segment_backtest.py — can v4 beat Underdog STANDALONE anywhere?

Extends v4_backtest (TB overall: FAIL, sign flip) to the remaining untested cells:
  - HRR standalone (overall)
  - Standalone WITHIN pre-registered segments (fixed list, declared before running):
      S-A  RHB pull-hitter (top-tercile TRUE pull%) vs sinker-heavy LHP  [user archetype]
      S-B  pull-hitter vs sinker-heavy, any hands
      S-C  high-K pitcher              [K-suppression family, thrice-confirmed]
      S-D  high-whiff bat x high-whiff pitcher  [K-family]

Protocol identical to v4_backtest: blend fit on the OTHER year, bet at UD odds when
blend EV > threshold. PASS requires positive ROI in BOTH years within the same segment.

Usage: python -m models.mlb.v4_segment_backtest
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

from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.feature_sets import build_luck, LUCK


def prep_target(target, tr_all, bt_all):
    cfg = TARGET_TO_MARKET[target]
    label = cfg['label_col']
    tr = tr_all[tr_all[label].notna()]
    bt = bt_all[bt_all[label].notna()]
    if target == 'hrr':
        tr = tr[tr['lbl_hrr_valid'] == True]; bt = bt[bt['lbl_hrr_valid'] == True]
    base = load_bundle(target, 'xgb', Path('models/mlb/saved'))['features']
    feats = base + ADV_FEATS + BATCH1 + LUCK   # v5 feature set
    m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m.fit(tr[feats].values, tr[label].astype(int).values, verbose=False)
    bt = bt.copy()
    bt['p_v4'] = m.predict(bt[feats].values)

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, target, date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[label].astype(int)
    B['yr'] = B['game_date'].dt.year
    B['dec_o'] = B['over_odds'].apply(american_to_decimal)
    B['dec_u'] = B['under_odds'].apply(american_to_decimal)
    for fy in (2025, 2026):
        f = B[B['yr'] == fy]
        if len(f) < 300:
            continue
        lm = LogisticRegression().fit(f[['p_mkt', 'p_v4']], f['y'])
        oth = B['yr'] != fy
        B.loc[oth, 'p_blend'] = lm.predict_proba(B.loc[oth, ['p_mkt', 'p_v4']])[:, 1]
    p = B['p_blend']
    B['ev_o'] = p * (B['dec_o'] - 1) - (1 - p)
    B['ev_u'] = (1 - p) * (B['dec_u'] - 1) - p
    over = B['ev_o'] >= B['ev_u']
    B['ev'] = np.where(over, B['ev_o'], B['ev_u'])
    B['won'] = np.where(over, B['y'] == 1, B['y'] == 0)
    B['payout'] = np.where(over, B['dec_o'], B['dec_u'])
    B['profit'] = np.where(B['won'], B['payout'] - 1, -1.0)
    return B


def segments(B):
    R = B['bat_hand'] == 'R'
    pL = B['pit_throws'] == 'L'
    sink = np.where(B['bat_hand'] == 'R', B['pit_pct_SI_vs_RHB_30d'], B['pit_pct_SI_vs_LHB_30d'])
    sink = pd.Series(sink, index=B.index)
    pull_hi = B['bat_pull_rate_120d'] >= B['bat_pull_rate_120d'].quantile(2/3)
    k_hi = B['pit_k_rate_szn'] >= B['pit_k_rate_szn'].quantile(2/3)
    bwh = B['bat_whiff_rate_vs_FB_90d'] >= B['bat_whiff_rate_vs_FB_90d'].quantile(2/3)
    pwh = B['pit_whiff_rate_90d'] >= B['pit_whiff_rate_90d'].quantile(2/3)
    return {
        'S-A RHB pull vs sinker LHP': R & pL & pull_hi & (sink >= .25),
        'S-B pull vs sinker (any)':   pull_hi & (sink >= .25),
        'S-C high-K pitcher':         k_hi,
        'S-D whiff x whiff':          bwh & pwh,
    }


def main():
    adv = pd.read_parquet(ADV_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr_all = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
             .merge(lk, on=['game_id', 'player_id'], how='left')
    bt_all = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
             .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr_all, bt_all):
        df['game_date'] = pd.to_datetime(df['game_date'])

    for target in ['tb', 'hrr']:
        print(f"\n################ {target.upper()} ################")
        B = prep_target(target, tr_all, bt_all)
        B = B[B['p_blend'].notna()]
        segs = {'OVERALL': pd.Series(True, index=B.index), **segments(B)}
        print(f"{'segment':30s} {'yr':>5s} {'thr':>5s} {'n':>6s} {'hit':>6s} {'ROI':>8s} {'±2SE':>7s}")
        for name, mask in segs.items():
            for yr in (2025, 2026):
                for thr in (0.02, 0.04):
                    d = B[mask & (B['yr'] == yr) & (B['ev'] > thr)]
                    if len(d) < 25:
                        print(f"{name:30s} {yr} {thr:>5.2f} {len(d):>6,}   (too small)")
                        continue
                    se = d['profit'].std() / np.sqrt(len(d))
                    print(f"{name:30s} {yr} {thr:>5.2f} {len(d):>6,} {d['won'].mean():>6.3f} "
                          f"{d['profit'].mean():>+8.4f} {2*se:>7.3f}")
    print("\n(PASS bar: same segment, positive ROI in BOTH years at the same threshold.)")


if __name__ == "__main__":
    main()
