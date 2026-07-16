"""
models/mlb/hitter/train_production.py — PRODUCTION retrain through current season (v7).

The research bundles freeze training at 2024 for honest backtesting. The LIVE bundle has
no such constraint: this trains on everything through 2026-04-30 (adds 1.5 seasons — the
only bat-tracking-rich ones; coverage of the accepted features goes 1/6 -> 2.5/7.5
seasons), isotonic-calibrates on 2026-05-01..06-30, and saves:

    hitter_tb_xgb_v3.pkl   (the current-bundle artifact the tracker loads; version=v7)
    hitter_hr_xgb_v7.pkl   (HR passed 4/4 gates but never had a production artifact)

July 2026+ remains untouched -> the forward record stays honest OOS.

Usage: python -m models.mlb.hitter.train_production
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from db.db import query
from models.mlb.hitter.backtest import load_bundle, TARGET_TO_MARKET
from models.mlb.feature_sets import (TRAIN_PQ, BTEST_PQ, FULL_CACHE, XGB_PARAMS,
                                     ADV_ALL, build_luck)

SAVED = Path("models/mlb/saved")
BASE_END = pd.Timestamp("2026-04-30")
CAL_END = pd.Timestamp("2026-06-30")
VERSION = "v7"


def hr_label(df):
    hr = query("SELECT game_id, player_id, hr FROM mlb_batting_game WHERE hr IS NOT NULL")
    df = df.merge(hr, on=['game_id', 'player_id'], how='left')
    df['lbl_hr'] = (df['hr'] >= 1).astype(float).where(df['hr'].notna())
    return df


def main():
    adv = pd.read_parquet(FULL_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    full = pd.concat([pd.read_parquet(TRAIN_PQ), pd.read_parquet(BTEST_PQ)], ignore_index=True)
    full = hr_label(full).merge(adv, on=['game_id', 'player_id'], how='left') \
                         .merge(lk, on=['game_id', 'player_id'], how='left')
    full['game_date'] = pd.to_datetime(full['game_date'])

    jobs = [('tb', TARGET_TO_MARKET['tb']['label_col'], 'tb', 'hitter_tb_xgb_v3.pkl'),
            ('hr', 'lbl_hr', 'hrr', 'hitter_hr_xgb_v7.pkl')]
    for tgt, label, base_key, out in jobs:
        d = full[full[label].notna()]
        base_feats = load_bundle(base_key, 'xgb', SAVED)['features']
        feats = base_feats + ADV_ALL
        tr = d[d['game_date'] <= BASE_END]
        cal = d[(d['game_date'] > BASE_END) & (d['game_date'] <= CAL_END)]
        print(f"[{tgt}] base {len(tr):,} rows (2019 -> {BASE_END.date()}) | calib {len(cal):,}")
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(tr[feats].values, tr[label].astype(int).values, verbose=False)
        raw = m.predict_proba(cal[feats].values)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw, cal[label].astype(int).values)
        print(f"[{tgt}] Brier {brier_score_loss(cal[label].astype(int), raw):.5f} -> "
              f"{brier_score_loss(cal[label].astype(int), iso.predict(raw)):.5f}")
        bundle = {'target': tgt, 'cfg': TARGET_TO_MARKET.get(tgt, {'label_col': label}),
                  'model_type': out.replace('hitter_', '').replace('.pkl', '').replace(f'{tgt}_', ''),
                  'params': XGB_PARAMS, 'model': m, 'calibrator': iso,
                  'calib_season': '2026-05/06', 'features': feats, 'adv_features': ADV_ALL,
                  'trained_window': f'2019 -> {BASE_END.date()}', 'version': VERSION,
                  'provenance': 'production retrain: research protocol froze 2024; live uses all data'}
        with open(SAVED / out, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[{tgt}] saved -> {SAVED / out} (version {VERSION})")


if __name__ == "__main__":
    main()
