"""
models/mlb/hr_gate.py — does the v4 feature stack beat the HOME RUN market?

Mechanism (why HR is the best-aligned market for the new data): books anchor HR prices
on trailing HR counts and preseason projections; bat speed, pull-air rate, and attack
angle are LEADING indicators of HR-rate change. If a hitter's swing profile shifts,
our features see it before his HR count does.

Anchor: Novig (book 60) HR 0.5 — 11.3K scored props 2025, 18.8K 2026 (Underdog rows are
too sparse in bettingpros for this market). Protocol identical to the Attack 3 gate:
    A: outcome ~ p_mkt          B: p_mkt + p_v4(base+adv)     C: p_mkt + p_control(base)
both time directions. PASS iff B > A and B > C in both.

Usage: python -m models.mlb.hr_gate
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
from models.mlb.hitter.backtest import (load_bundle, american_to_decimal, _build_player_match)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1

HR_MARKET_ID = 299
NOVIG_BOOK_ID = 60


def _attach_hr(df: pd.DataFrame) -> pd.DataFrame:
    """Join HR counts from box scores -> lbl_hr (>=1 HR)."""
    hr = query("""SELECT game_id, player_id, hr FROM mlb_batting_game WHERE hr IS NOT NULL""")
    df = df.merge(hr, on=['game_id', 'player_id'], how='left')
    df['lbl_hr'] = (df['hr'] >= 1).astype(float).where(df['hr'].notna())
    return df


def main():
    adv = pd.read_parquet(ADV_CACHE)
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr['lbl_hr'].notna()]; bt = bt[bt['lbl_hr'].notna()]
    ytr = tr['lbl_hr'].astype(int).values
    print(f"train {len(tr):,} (HR rate {ytr.mean():.3f}) | backtest {len(bt):,}")

    base = load_bundle('hrr', 'xgb', Path('models/mlb/saved'))['features']
    v4 = base + ADV_FEATS + BATCH1
    print(f"fitting CONTROL ({len(base)}) and V4 ({len(v4)})...")
    m_c = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_c.fit(tr[base].values, ytr, verbose=False)
    m_v = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_v.fit(tr[v4].values, ytr, verbose=False)
    bt = bt.copy()
    bt['p_ctl'] = m_c.predict(bt[base].values)
    bt['p_v4'] = m_v.predict(bt[v4].values)

    odds = query("""
        SELECT prop_date AS game_date, bp_player_id, over_odds, under_odds
        FROM bettingpros_props
        WHERE book_id = %(b)s AND market_id = %(m)s AND over_line = 0.5
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """, params={'b': NOVIG_BOOK_ID, 'm': HR_MARKET_ID})
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    with contextlib.redirect_stdout(_io.StringIO()):
        match = _build_player_match(date(2025, 1, 1), date(2026, 12, 31))
    odds = odds.merge(match[match['player_id'].notna()][['bp_player_id', 'player_id']],
                      on='bp_player_id', how='inner')
    B = bt.merge(odds, on=['game_date', 'player_id'], how='inner')
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B['lbl_hr'].astype(int)
    B['yr'] = B['game_date'].dt.year
    print(f"joined props: 2025={len(B[B.yr==2025]):,}  2026={len(B[B.yr==2026]):,} "
          f"(HR rate {B['y'].mean():.3f}, mkt implied {B['p_mkt'].mean():.3f})")

    print("\n=========== HR GATE (Novig anchor, both directions) ===========")
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = B[B['yr'] == fy], B[B['yr'] == ty]
        aucs = {}
        for k, cols in [('A', ['p_mkt']), ('B', ['p_mkt', 'p_v4']), ('C', ['p_mkt', 'p_ctl'])]:
            lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
            aucs[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        print(f"  fit {fy} -> test {ty} (n={len(t):,}): "
              f"A={aucs['A']:.4f}  B={aucs['B']:.4f}  C={aucs['C']:.4f}  "
              f"B-A={aucs['B']-aucs['A']:+.4f}  B-C={aucs['B']-aucs['C']:+.4f}")
    print("\n  PASS iff B-A>0 AND B-C>0 in BOTH directions.")


if __name__ == "__main__":
    main()
