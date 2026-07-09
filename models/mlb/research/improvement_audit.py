"""
models/mlb/improvement_audit.py — is the v4->v5 TB improvement real model skill?

Rules out two artifacts:
  1. SIDE SKEW: per version/year — over%, ROI on overs vs unders separately. If the
     "improvement" is a side-mix drift into the side that happened to win in 2026,
     it's regime luck, not skill.
  2. FIT JITTER (placebo): v4 + 3 pure-noise columns ("v4n") through the identical
     pipeline. Any new column changes XGB tree structure; if noise "improves" 2026 ROI
     like the luck features did, the delta is jitter, not information.
  3. PAIRED DECOMPOSITION: 2026 bets split into common (same prop+side), side-flipped,
     v4-only (dropped by v5), v5-only (added). Where does the ROI delta come from?

Usage: python -m models.mlb.improvement_audit
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
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal, TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.feature_sets import build_luck, LUCK

THR = 0.02


def bets_for(B, pcol):
    p = B[pcol]
    evo = p * (B['dec_o'] - 1) - (1 - p); evu = (1 - p) * (B['dec_u'] - 1) - p
    over = evo >= evu
    d = B.copy()
    d['side'] = np.where(over, 'over', 'under')
    d['ev'] = np.where(over, evo, evu)
    d['won'] = np.where(over, d['y'] == 1, d['y'] == 0)
    d['profit'] = np.where(d['won'], np.where(over, d['dec_o'], d['dec_u']) - 1, -1.0)
    return d[d['ev'] > THR]


def side_table(d, tag):
    for yr in (2025, 2026):
        x = d[d['yr'] == yr]
        if not len(x): continue
        ov = x[x['side'] == 'over']; un = x[x['side'] == 'under']
        print(f"  {tag} {yr}: n={len(x):>5,} over%={len(ov)/len(x):.2f} ROI={x['profit'].mean():+.4f}"
              f" | overs ROI={ov['profit'].mean() if len(ov) else 0:+.4f} (n={len(ov):,})"
              f" | unders ROI={un['profit'].mean() if len(un) else 0:+.4f} (n={len(un):,})")


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
    y = tr[label].astype(int).values
    rng = np.random.default_rng(7)
    NOISE = ['nz1', 'nz2', 'nz3']
    for i, c in enumerate(NOISE):
        tr[c] = rng.normal(size=len(tr)); bt[c] = rng.normal(size=len(bt))

    base = load_bundle('tb', 'xgb', Path('models/mlb/saved'))['features']
    sets = {'v4': base + ADV_FEATS + BATCH1,
            'v5': base + ADV_FEATS + BATCH1 + LUCK,
            'v4n': base + ADV_FEATS + BATCH1 + NOISE}
    bt = bt.copy()
    for name, fs in sets.items():
        m = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
        m.fit(tr[fs].values, y, verbose=False)
        bt[f'p_{name}'] = m.predict(bt[fs].values)
        print(f"fitted {name} ({len(fs)} feats)")

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal); iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_); B['y'] = B[label].astype(int); B['yr'] = B['game_date'].dt.year
    B['dec_o'] = B['over_odds'].apply(american_to_decimal); B['dec_u'] = B['under_odds'].apply(american_to_decimal)
    for name in sets:
        for fy in (2025, 2026):
            f = B[B['yr'] == fy]
            lm = LogisticRegression().fit(f[['p_mkt', f'p_{name}']], f['y'])
            oth = B['yr'] != fy
            B.loc[oth, f'pb_{name}'] = lm.predict_proba(B.loc[oth, ['p_mkt', f'p_{name}']])[:, 1]

    print(f"\n===== 1+2. SIDE MIX + PLACEBO (bets at ev>{THR}) =====")
    D = {}
    for name in sets:
        D[name] = bets_for(B.assign(pcol=B[f'pb_{name}']).rename(columns={}), f'pb_{name}')
        side_table(D[name], name)

    print("\n===== 3. PAIRED DECOMPOSITION (2026, v4 vs v5) =====")
    k = ['game_date', 'player_id']
    a = D['v4'][D['v4']['yr'] == 2026][k + ['side', 'profit']].rename(
        columns={'side': 's4', 'profit': 'pr4'})
    b = D['v5'][D['v5']['yr'] == 2026][k + ['side', 'profit']].rename(
        columns={'side': 's5', 'profit': 'pr5'})
    j = a.merge(b, on=k, how='outer', indicator=True)
    common = j[(j['_merge'] == 'both') & (j['s4'] == j['s5'])]
    flipped = j[(j['_merge'] == 'both') & (j['s4'] != j['s5'])]
    only4 = j[j['_merge'] == 'left_only']; only5 = j[j['_merge'] == 'right_only']
    print(f"  common same-side: n={len(common):,}  ROI={common['pr5'].mean():+.4f}")
    print(f"  side-flipped:     n={len(flipped):,}  v4 ROI={flipped['pr4'].mean() if len(flipped) else 0:+.4f} -> v5 ROI={flipped['pr5'].mean() if len(flipped) else 0:+.4f}")
    print(f"  v4-only (dropped): n={len(only4):,}  their ROI was {only4['pr4'].mean() if len(only4) else 0:+.4f}")
    print(f"  v5-only (added):   n={len(only5):,}  their ROI is  {only5['pr5'].mean() if len(only5) else 0:+.4f}")
    print("\n(Real improvement = dropped bets were bad / added bets are good / flips won,")
    print(" with side mix stable. Regime luck = side mix shifted toward 2026's winning side.)")


if __name__ == "__main__":
    main()
