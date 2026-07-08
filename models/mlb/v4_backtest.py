"""
models/mlb/v4_backtest.py — betting backtests for the v4 TB model (Leg 1).

Converts the validated AUC edge (PRD §10: blend beats market both directions) into
simulated betting P&L on historical odds. Two questions:

  1. STANDALONE: bet TB 1.5 at Underdog's own odds whenever the market+v4 blend says
     EV > threshold. Does the informational edge clear Underdog's vig by itself?
     (Blend fit on the OTHER year; both directions; dedup = one row per player-game.)

  2. FILTER VALUE (the production claim): take Leg 2 line-shopping bets — UD price vs
     Novig de-vigged fair, EV > 2% — and split by whether the v4 blend AGREES with the
     bet side. If agree-ROI > disagree-ROI in both years, "v4 filters Leg 2" is
     historically demonstrated, not just asserted.

Usage: python -m models.mlb.v4_backtest
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
from models.mlb.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 TARGET_TO_MARKET, _build_player_match)
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.leg1_attack3_test import ADV_CACHE, ADV_FEATS
from models.mlb.leg1_batch_gate import BATCH1
from models.mlb.luck_gap_gate import build_luck, LUCK
from datetime import date as _date

NOVIG_BOOK_ID = 60
LS_EV_MIN = 0.02          # line-shopping entry threshold (matches paper-trade)


def _roi_row(d, label=""):
    if len(d) == 0:
        return f"  {label:34s}      0        -         -"
    se = d['profit'].std() / np.sqrt(len(d)) if len(d) > 1 else 0
    return (f"  {label:34s} {len(d):>6,}  hit={d['won'].mean():.3f}  "
            f"ROI={d['profit'].mean():+.4f} (±{2*se:.3f})")


def main():
    cfg = TARGET_TO_MARKET['tb']
    label = cfg['label_col']

    adv = pd.read_parquet(ADV_CACHE)
    lk = build_luck(_date(2019, 3, 1), _date(2026, 12, 31))
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    tr = tr[tr[label].notna()]; bt = bt[bt[label].notna()]
    ytr = tr[label].astype(int).values

    base = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    feats = base + ADV_FEATS + BATCH1 + LUCK   # v5 feature set
    print(f"fitting v4 ({len(feats)} feats) on {len(tr):,} rows...")
    m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m.fit(tr[feats].values, ytr, verbose=False)
    bt = bt.copy()
    bt['p_v4'] = m.predict(bt[feats].values)

    # UD odds
    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[label].astype(int)
    B['yr'] = B['game_date'].dt.year
    # Side-shade correction (books load vig onto overs; midpoint devig overstates
    # P(over)). delta = TRAILING-year measured shade at UD TB: 2024 -> 0.023 corrects
    # 2025; 2025 -> 0.015 corrects 2026. No lookahead. See calibration audit 2026-07-05.
    B['p_mkt'] = B['p_mkt'] - B['yr'].map({2025: 0.023, 2026: 0.015}).fillna(0.0)
    B['dec_o'] = B['over_odds'].apply(american_to_decimal)
    B['dec_u'] = B['under_odds'].apply(american_to_decimal)

    # Novig odds (same market/line) -> de-vigged fair
    nv = query("""
        SELECT prop_date AS game_date, bp_player_id,
               over_odds AS nv_o, under_odds AS nv_u
        FROM bettingpros_props
        WHERE book_id = %(b)s AND market_id = %(m)s AND over_line = %(l)s
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """, params={'b': NOVIG_BOOK_ID, 'm': cfg['market_id'], 'l': cfg['over_line']})
    nv['game_date'] = pd.to_datetime(nv['game_date'])
    with contextlib.redirect_stdout(_io.StringIO()):
        match = _build_player_match(date(2025, 1, 1), date(2026, 12, 31))
    nv = nv.merge(match[match['player_id'].notna()][['bp_player_id', 'player_id']],
                  on='bp_player_id', how='inner')
    B = B.merge(nv[['game_date', 'player_id', 'nv_o', 'nv_u']],
                on=['game_date', 'player_id'], how='left')
    nio = 1 / B['nv_o'].apply(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    niu = 1 / B['nv_u'].apply(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    B['nv_fair'] = nio / (nio + niu)
    print(f"props: {len(B):,} UD-quoted | {B['nv_fair'].notna().sum():,} also Novig-quoted")

    # blend per direction (fit on the OTHER year)
    for fy in (2025, 2026):
        lm = LogisticRegression().fit(B.loc[B['yr'] == fy, ['p_mkt', 'p_v4']],
                                      B.loc[B['yr'] == fy, 'y'])
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

    print("\n===== 1. STANDALONE: blend EV vs Underdog odds (blend fit on other year) =====")
    for yr in (2025, 2026):
        d = B[B['yr'] == yr]
        print(f" bet year {yr}:")
        for thr in [0.0, 0.02, 0.04, 0.06]:
            print(_roi_row(d[d['ev'] > thr], f"ev>{thr:.2f}"))

    print("\n===== 2. FILTER VALUE: Leg 2 line-shop bets (UD vs Novig fair, EV>2%) =====")
    N = B[B['nv_fair'].notna()].copy()
    pf = N['nv_fair']
    N['ls_ev_o'] = pf * (N['dec_o'] - 1) - (1 - pf)
    N['ls_ev_u'] = (1 - pf) * (N['dec_u'] - 1) - pf
    ls_over = N['ls_ev_o'] >= N['ls_ev_u']
    N['ls_side'] = np.where(ls_over, 'over', 'under')
    N['ls_ev'] = np.where(ls_over, N['ls_ev_o'], N['ls_ev_u'])
    N = N[N['ls_ev'] > LS_EV_MIN].copy()
    N['won'] = np.where(N['ls_side'] == 'over', N['y'] == 1, N['y'] == 0)
    N['payout'] = np.where(N['ls_side'] == 'over', N['dec_o'], N['dec_u'])
    N['profit'] = np.where(N['won'], N['payout'] - 1, -1.0)
    N['v4_agrees'] = N['ls_side'] == N['side']       # blend's preferred side == bet side
    for yr in (2025, 2026):
        d = N[N['yr'] == yr]
        print(f" bet year {yr}:")
        print(_roi_row(d, "ALL line-shop bets"))
        print(_roi_row(d[d['v4_agrees']], "v4 AGREES"))
        print(_roi_row(d[~d['v4_agrees']], "v4 disagrees"))
    print("\n(Filter thesis holds iff AGREE-ROI > disagree-ROI in BOTH years.)")


if __name__ == "__main__":
    main()
