"""
models/mlb/archetype_audit.py — continuous (kernel-weighted) archetype audit.

Instead of hard buckets (n=7), weight EVERY prop by profile similarity to a prototype
matchup (e.g. Isaac Paredes vs Framber Valdez), then compute similarity-weighted betting
ROI and market gap. Embeddings as an audit lens, not model features. Effective sample
n_eff = (sum w)^2 / sum w^2.

Usage: python -m models.mlb.archetype_audit
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
from models.mlb.backtest import (load_bundle, attach_odds, american_to_decimal, TARGET_TO_MARKET)
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.leg1_attack3_test import ADV_CACHE, ADV_FEATS
from models.mlb.leg1_batch_gate import BATCH1
from models.mlb.luck_gap_gate import build_luck, LUCK

# similarity dims: batter side + pitcher side (z-scored)
BDIMS = ['bat_pull_rate_120d', 'bat_attack_angle_120d', 'bat_gb_rate_90d']
PDIMS_HAND = 'SI'                    # sinker share vs batter hand
PDIMS = ['pit_k_rate_szn', 'pit_whiff_rate_90d']


def main():
    adv = pd.read_parquet(ADV_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    cfg = TARGET_TO_MARKET['tb']; label = cfg['label_col']
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
    io_ = 1 / B['over_odds'].apply(american_to_decimal); iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_); B['y'] = B[label].astype(int); B['yr'] = B['game_date'].dt.year
    B['dec_o'] = B['over_odds'].apply(american_to_decimal); B['dec_u'] = B['under_odds'].apply(american_to_decimal)
    for fy in (2025, 2026):
        f = B[B['yr'] == fy]
        lm = LogisticRegression().fit(f[['p_mkt', 'p_v5']], f['y'])
        oth = B['yr'] != fy
        B.loc[oth, 'p_blend'] = lm.predict_proba(B.loc[oth, ['p_mkt', 'p_v5']])[:, 1]
    p = B['p_blend']
    evo = p * (B['dec_o'] - 1) - (1 - p); evu = (1 - p) * (B['dec_u'] - 1) - p
    over = evo >= evu
    B['ev'] = np.where(over, evo, evu)
    B['won'] = np.where(over, B['y'] == 1, B['y'] == 0)
    B['profit'] = np.where(B['won'], np.where(over, B['dec_o'], B['dec_u']) - 1, -1.0)

    # prototype = Paredes (batter dims) x Valdez (pitcher dims), from their own 2026 rows
    pid = query("SELECT player_id, full_name FROM players WHERE full_name IN "
                "('Isaac Paredes','Framber Valdez')")
    ip = pid.set_index('full_name')['player_id'].to_dict()
    sin = pd.Series(np.where(B['bat_hand'] == 'R', B['pit_pct_SI_vs_RHB_30d'],
                             B['pit_pct_SI_vs_LHB_30d']), index=B.index)
    B['_sin'] = sin
    dims = BDIMS + ['_sin'] + PDIMS
    Z = B[dims].astype(float)
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    Zz = ((Z - mu) / sd).fillna(0)
    proto_b = B[B['player_id'] == ip.get('Isaac Paredes', -1)][BDIMS].astype(float).mean()
    fram = query("""SELECT pg.game_id FROM mlb_pitching_game pg
                    WHERE pg.player_id = %(p)s AND pg.is_starter = true""",
                 params={'p': int(ip.get('Framber Valdez', -1))})
    pv = B[B['game_id'].isin(fram['game_id'])][['_sin'] + PDIMS].astype(float).mean()
    proto = pd.concat([proto_b, pv])
    proto_z = ((proto - mu) / sd).fillna(0)
    hand_match = ((B['bat_hand'] == 'R') & (B['pit_throws'] == 'L')).astype(float)
    d2 = ((Zz - proto_z.values) ** 2).sum(axis=1)
    print(f"prototype (z): {proto_z.round(2).to_dict()}")
    print(f"\n=== kernel-weighted audit: 'like Paredes vs like Framber' (TB, v5) ===")
    print(f"{'sigma':>6} {'yr':>5} {'n_eff':>7} {'w-ROI(ev>2%)':>13} {'w-gap':>7}")
    for sig in (1.0, 1.5, 2.5):
        w_all = np.exp(-d2 / (2 * sig ** 2)) * (0.25 + 0.75 * hand_match)
        for yr in (2025, 2026):
            msk = (B['yr'] == yr) & (B['ev'] > 0.02)
            w = w_all[msk]
            if w.sum() == 0: continue
            neff = w.sum() ** 2 / (w ** 2).sum()
            roi = np.average(B.loc[msk, 'profit'], weights=w)
            mg = (B['yr'] == yr)
            wg = w_all[mg]
            gap = np.average(B.loc[mg, 'y'] - B.loc[mg, 'p_mkt'], weights=wg)
            print(f"{sig:>6.1f} {yr:>5} {neff:>7.0f} {roi:>+13.4f} {gap:>+7.3f}")
    print("\n(w-ROI: similarity-weighted ROI of v5-blend bets. w-gap: weighted actual-minus-")
    print(" market — negative = unders beat market near this archetype. n_eff = effective n.)")


if __name__ == "__main__":
    main()
