"""
models/mlb/segment_skill_scan.py — per-matchup-segment model-vs-market skill scan.

Question (user's): even though the model loses to the market OVERALL, does it beat the
market WITHIN specific matchup archetypes? If yes (consistently), filter bets to those.

Discipline against multiple comparisons (~15 segments => expect 1-2 false positives):
  - Fixed segment taxonomy, declared below, no post-hoc additions.
  - Blend lift must be POSITIVE IN BOTH time directions:
        fit blend on 2025 -> eval per-segment on 2026, AND fit 2026 -> eval 2025.
  - A segment "passes" only if delta_blend > 0 in both directions (and n >= 150 each).

Segments (pre-registered):
  hand matchups: RHB v LHP, RHB v RHP, LHB v LHP, LHB v RHP
  user's example (proxy): GB-prone RHB v sinker-heavy LHP; also GB-prone x sinker-heavy (any hands)
  FB-prone batter v 4seam-heavy pitcher
  high-K pitcher; low-K pitcher
  high-whiff batter v high-whiff pitcher
  platoon adv; no platoon adv
  (pull proxied by bat_gb_rate_90d until hc_x/hc_y spray data is pulled)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import contextlib, io as _io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.mlb.hitter.backtest import (load_bundle, predict_proba, attach_odds,
                                 american_to_decimal, TARGET_TO_MARKET)

BTEST_PQ = "models/mlb/cache/backtest_2025_2026.parquet"
MIN_N = 150


def segments(B):
    R, L = B['bat_hand'] == 'R', B['bat_hand'].isin(['L', 'S'])
    pR, pL = B['pit_throws'] == 'R', B['pit_throws'] == 'L'
    sink = np.where(B['bat_hand'] == 'R', B['pit_pct_SI_vs_RHB_30d'], B['pit_pct_SI_vs_LHB_30d'])
    ff   = np.where(B['bat_hand'] == 'R', B['pit_pct_FF_vs_RHB_30d'], B['pit_pct_FF_vs_LHB_30d'])
    sink = pd.Series(sink, index=B.index); ff = pd.Series(ff, index=B.index)
    gb_hi = B['bat_gb_rate_90d'] >= B['bat_gb_rate_90d'].quantile(2/3)
    fb_hi = B['bat_fb_rate_90d'] >= B['bat_fb_rate_90d'].quantile(2/3)
    k_hi  = B['pit_k_rate_szn'] >= B['pit_k_rate_szn'].quantile(2/3)
    k_lo  = B['pit_k_rate_szn'] <= B['pit_k_rate_szn'].quantile(1/3)
    bwh   = B['bat_whiff_rate_vs_FB_90d'] >= B['bat_whiff_rate_vs_FB_90d'].quantile(2/3)
    pwh   = B['pit_whiff_rate_90d'] >= B['pit_whiff_rate_90d'].quantile(2/3)
    return {
        'RHB v LHP': R & pL, 'RHB v RHP': R & pR,
        'LHB v LHP': L & pL, 'LHB v RHP': L & pR,
        'GB-prone RHB v sinker-heavy LHP': R & pL & gb_hi & (sink >= .25),
        'GB-prone bat v sinker-heavy (any)': gb_hi & (sink >= .25),
        'FB-prone bat v 4seam-heavy': fb_hi & (ff >= .45),
        'high-K pitcher': k_hi, 'low-K pitcher': k_lo,
        'high-whiff bat v high-whiff pit': bwh & pwh,
        'platoon adv': B['mu_platoon_advantage'] == 1,
        'no platoon adv': B['mu_platoon_advantage'] == 0,
    }


def main():
    bt = pd.read_parquet(BTEST_PQ)
    bt['game_date'] = pd.to_datetime(bt['game_date'])

    for target in ['tb', 'rbi', 'hrr']:
        cfg = TARGET_TO_MARKET[target]
        d = bt[bt[cfg['label_col']].notna()].copy()
        if target == 'hrr':
            d = d[d['lbl_hrr_valid'] == True]
        bundle = load_bundle(target, 'xgb', Path('models/mlb/saved'))
        d['p_model'] = predict_proba(bundle, d[bundle['features']])
        with contextlib.redirect_stdout(_io.StringIO()):
            B = attach_odds(d, target, date(2025, 1, 1), date(2026, 12, 31))
        io_ = 1 / B['over_odds'].apply(american_to_decimal)
        iu_ = 1 / B['under_odds'].apply(american_to_decimal)
        B['p_mkt'] = io_ / (io_ + iu_)
        B['y'] = B[cfg['label_col']].astype(int)
        B['yr'] = B['game_date'].dt.year
        segs = segments(B)

        print(f"\n================ {target.upper()} — per-segment blend lift ================")
        print(f"{'segment':36s} | {'25->26: n':>9s} {'dAUC':>7s} | {'26->25: n':>9s} {'dAUC':>7s} | pass")
        print("-" * 92)
        for name, mask in segs.items():
            deltas, ns = [], []
            ok = True
            for fit_yr, te_yr in [(2025, 2026), (2026, 2025)]:
                f, t = B[B['yr'] == fit_yr], B[mask & (B['yr'] == te_yr)]
                if len(f) < 300 or len(t) < MIN_N or t['y'].nunique() < 2:
                    ok = False; deltas.append(np.nan); ns.append(len(t)); continue
                lm2 = LogisticRegression().fit(f[['p_mkt', 'p_model']], f['y'])
                a_blend = roc_auc_score(t['y'], lm2.predict_proba(t[['p_mkt', 'p_model']])[:, 1])
                a_mkt = roc_auc_score(t['y'], t['p_mkt'])
                deltas.append(a_blend - a_mkt); ns.append(len(t))
            passed = ok and all(x > 0 for x in deltas if not np.isnan(x)) and not any(np.isnan(deltas))
            flag = '  << PASS' if passed else ''
            d1 = f"{deltas[0]:+.4f}" if not np.isnan(deltas[0]) else "   n/a"
            d2 = f"{deltas[1]:+.4f}" if not np.isnan(deltas[1]) else "   n/a"
            print(f"{name:36s} | {ns[0]:>9,} {d1:>7s} | {ns[1]:>9,} {d2:>7s} |{flag}")
    print("\n(dAUC = AUC(market+model blend) - AUC(market) within segment, on held-out year.")
    print(" PASS requires positive in BOTH directions. ~15 segments x 3 targets => expect ~2")
    print(" chance passes; treat any pass as a candidate for forward confirmation, not truth.)")


if __name__ == "__main__":
    main()
