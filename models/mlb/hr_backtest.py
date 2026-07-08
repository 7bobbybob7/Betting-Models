"""
models/mlb/hr_backtest.py — betting sim for the HR market (gate PASSED both directions).

Bets at Novig's actual quoted prices (book 60, HR 0.5) — Novig is an exchange that takes
singles, so unlike the TB sim this is an executable strategy if it survives the spread.
Blend fit on the OTHER year (honest), bet when blend EV > threshold, settle on box scores.

Also saves the scored frame to cache for downstream analysis (soft-book HR line-shop
with v4 filter, once the 5-book backfill lands).

Usage: python -m models.mlb.hr_backtest
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

from db.db import query
from models.mlb.backtest import load_bundle, american_to_decimal, _build_player_match
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.leg1_attack3_test import ADV_CACHE, ADV_FEATS
from models.mlb.leg1_batch_gate import BATCH1
from models.mlb.luck_gap_gate import build_luck, LUCK
from models.mlb.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr

SCORED_CACHE = "models/mlb/cache/hr_scored_2025_2026.parquet"


def main():
    adv = pd.read_parquet(ADV_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr['lbl_hr'].notna()]; bt = bt[bt['lbl_hr'].notna()]

    base = load_bundle('hrr', 'xgb', Path('models/mlb/saved'))['features']
    v4 = base + ADV_FEATS + BATCH1 + LUCK   # v5 feature set
    print(f"fitting v4-HR ({len(v4)} feats) on {len(tr):,} rows...")
    m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m.fit(tr[v4].values, tr['lbl_hr'].astype(int).values, verbose=False)
    bt = bt.copy()
    bt['p_v4'] = m.predict(bt[v4].values)

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
    B['spread_vig'] = io_ + iu_ - 1.0
    B['y'] = B['lbl_hr'].astype(int)
    B['yr'] = B['game_date'].dt.year
    B['dec_o'] = B['over_odds'].apply(american_to_decimal)
    B['dec_u'] = B['under_odds'].apply(american_to_decimal)
    print(f"props: 2025={len(B[B.yr==2025]):,} 2026={len(B[B.yr==2026]):,} | "
          f"avg two-sided overround {B['spread_vig'].mean()*100:.1f}%")

    for fy in (2025, 2026):
        f = B[B['yr'] == fy]
        lm = LogisticRegression(max_iter=1000).fit(f[['p_mkt', 'p_v4']], f['y'])
        oth = B['yr'] != fy
        B.loc[oth, 'p_blend'] = lm.predict_proba(B.loc[oth, ['p_mkt', 'p_v4']])[:, 1]

    p = B['p_blend']
    B['ev_o'] = p * (B['dec_o'] - 1) - (1 - p)
    B['ev_u'] = (1 - p) * (B['dec_u'] - 1) - p
    over = B['ev_o'] >= B['ev_u']
    B['side'] = np.where(over, 'over', 'under')
    B['ev'] = np.where(over, B['ev_o'], B['ev_u'])
    B['won'] = np.where(over, B['y'] == 1, B['y'] == 0)
    B['payout'] = np.where(over, B['dec_o'], B['dec_u'])
    B['profit'] = np.where(B['won'], B['payout'] - 1, -1.0)

    B.to_parquet(SCORED_CACHE, index=False)
    print(f"scored frame cached -> {SCORED_CACHE}")

    print("\n===== HR standalone sim @ Novig quoted prices (blend fit other year) =====")
    print(f"{'year':>6s} {'thr':>5s} {'n':>6s} {'hit':>6s} {'ROI':>8s} {'±2SE':>7s} {'over%':>6s}")
    for yr in (2025, 2026):
        d0 = B[B['yr'] == yr]
        for thr in (0.0, 0.02, 0.04, 0.06):
            d = d0[d0['ev'] > thr]
            if len(d) < 25:
                print(f"{yr:>6} {thr:>5.2f} {len(d):>6,}   (too small)"); continue
            se = d['profit'].std() / np.sqrt(len(d))
            print(f"{yr:>6} {thr:>5.2f} {len(d):>6,} {d['won'].mean():>6.3f} "
                  f"{d['profit'].mean():>+8.4f} {2*se:>7.3f} "
                  f"{(d['side']=='over').mean():>6.2f}")


if __name__ == "__main__":
    main()
