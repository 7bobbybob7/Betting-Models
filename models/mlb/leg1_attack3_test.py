"""
models/mlb/leg1_attack3_test.py — LEG 1 v2 / Attack 3: do the NEW Statcast datasets
(true pull%/spray, bat tracking, catcher framing) give the model information the market
line lacks? Same decisive gate as Attack 1 (PRD §6), plus a segment scan built on REAL
pull% (the user's RHB-pull-hitter vs sinker-heavy-LHP archetype, finally measurable).

    A: outcome ~ p_mkt                       (market alone)
    B: outcome ~ p_mkt + p_model_v3          (base feats + advanced-profile feats)
    C: outcome ~ p_mkt + p_model_control     (base feats only, identical training)

New features exist only 2024+, so they are NaN across most of the 2019-2024 training set.
XGBoost routes NaN natively; the blend-fit (2025) and test (2026) years are fully covered,
so the model learns to use them precisely in the era where they exist.

PASS iff AUC_B > AUC_A (target +0.005) AND AUC_B > AUC_C.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse, contextlib, io as _io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.mlb.advanced_profile_features import build_training_set as build_adv
from models.mlb.backtest import (load_bundle, attach_odds, american_to_decimal, TARGET_TO_MARKET)
from models.mlb.leg1_v2_test import TRAIN_PQ, BTEST_PQ, XGB_PARAMS

ADV_CACHE = "models/mlb/cache/adv_profile_2024_2026.parquet"
ADV_FEATS = ['bat_pull_rate_120d', 'bat_oppo_rate_120d', 'bat_bat_speed_120d',
             'bat_swing_len_120d', 'bat_attack_angle_120d', 'bat_fast_swing_rate_120d',
             'ctx_catcher_framing_120d']


def _adv_features() -> pd.DataFrame:
    if os.path.exists(ADV_CACHE):
        print(f"loading cached advanced features: {ADV_CACHE}")
        return pd.read_parquet(ADV_CACHE)
    print("building advanced features 2024-01-01 -> 2026-12-31 (one-time, ~minutes)...")
    adv = build_adv(date(2024, 1, 1), date(2026, 12, 31))
    adv.to_parquet(ADV_CACHE, index=False)
    print(f"cached -> {ADV_CACHE}")
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGET_TO_MARKET.keys()), default="hrr")
    args = ap.parse_args()
    cfg = TARGET_TO_MARKET[args.target]

    adv = _adv_features()
    tr = pd.read_parquet(TRAIN_PQ).merge(adv, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(adv, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    cov = bt[ADV_FEATS].notna().mean()
    print(f"  train {len(tr):,} | backtest {len(bt):,} | adv coverage in backtest: "
          f"min={cov.min():.2f} mean={cov.mean():.2f}")

    if args.target == 'hrr':
        tr = tr[tr['lbl_hrr_valid'] == True]; bt = bt[bt['lbl_hrr_valid'] == True]
    tr = tr[tr[cfg['label_col']].notna()]; bt = bt[bt[cfg['label_col']].notna()]
    ytr = tr[cfg['label_col']].astype(int).values

    base_feats = load_bundle(args.target, 'xgb', Path('models/mlb/saved'))['features']
    v3_feats = base_feats + ADV_FEATS
    print(f"training CONTROL ({len(base_feats)}) and V3 ({len(v3_feats)}) on {len(tr):,} rows...")
    m_ctl = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_ctl.fit(tr[base_feats].values, ytr, verbose=False)
    m_v3 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_v3.fit(tr[v3_feats].values, ytr, verbose=False)

    bt = bt.copy()
    bt['p_ctl'] = m_ctl.predict(bt[base_feats].values)
    bt['p_v3']  = m_v3.predict(bt[v3_feats].values)

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, args.target, date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[cfg['label_col']].astype(int)
    B['yr'] = B['game_date'].dt.year
    fit, te = B[B['yr'] == 2025], B[B['yr'] == 2026]
    print(f"  blend-fit 2025: {len(fit):,} | test 2026: {len(te):,}")

    print("\n============ ATTACK 3 RESIDUAL TEST (fit 2025 -> test 2026) ============")
    res = {}
    for name, cols in [('A: market alone', ['p_mkt']),
                       ('B: market + v3', ['p_mkt', 'p_v3']),
                       ('C: market + control', ['p_mkt', 'p_ctl'])]:
        lm = LogisticRegression().fit(fit[cols], fit['y'])
        auc = roc_auc_score(te['y'], lm.predict_proba(te[cols])[:, 1])
        res[name[0]] = auc
        coef = "" if len(cols) == 1 else f"  model_coef={lm.coef_[0][1]:+.3f}"
        print(f"  {name:22s} AUC={auc:.4f}{coef}")
    dBA, dBC = res['B'] - res['A'], res['B'] - res['C']
    print(f"\n  B-A = {dBA:+.4f} (needs >0, target +0.005) | B-C = {dBC:+.4f} (needs >0)")
    print(f"  VERDICT: {'PASS' if (dBA>0 and dBC>0) else 'FAIL'}"
          f"{' (meets +0.005)' if dBA>=0.005 else ''}")

    # ---- REAL-pull segment scan (both time directions, like segment_skill_scan) ----
    print("\n============ REAL-PULL SEGMENTS (market gap, per year) ============")
    pull_hi = B['bat_pull_rate_120d'] >= B['bat_pull_rate_120d'].quantile(2/3)
    R = B['bat_hand'] == 'R'
    pL = B['pit_throws'] == 'L'
    sink = np.where(B['bat_hand'] == 'R', B['pit_pct_SI_vs_RHB_30d'], B['pit_pct_SI_vs_LHB_30d'])
    sink = pd.Series(sink, index=B.index)
    fram_neg = B['ctx_catcher_framing_120d'] >= B['ctx_catcher_framing_120d'].quantile(2/3)
    segs = {
        'RHB pull-hitter vs sinker-heavy LHP': R & pL & pull_hi & (sink >= .25),
        'pull-hitter vs sinker-heavy (any)':   pull_hi & (sink >= .25),
        'vs elite-framing catcher (pred: unders)': fram_neg,
    }
    print(f"{'segment':42s} | {'yr':>4s} {'n':>6s} {'gap':>7s} {'z':>6s}")
    print("-" * 74)
    for name, mask in segs.items():
        for yr in (2025, 2026):
            d = B[mask & (B['yr'] == yr)].dropna(subset=['p_mkt'])
            if len(d) < 60:
                print(f"{name:42s} | {yr} {len(d):>6,}  (too small)"); continue
            act = d['y'].mean(); gap = act - d['p_mkt'].mean()
            se = np.sqrt(act * (1 - act) / len(d)); z = gap / se if se > 0 else 0
            print(f"{name:42s} | {yr} {len(d):>6,} {gap:>+7.3f} {z:>+6.2f}")
    print("\n(gap = actual_over - market_implied; consistency across BOTH years is the bar.)")


if __name__ == "__main__":
    main()
