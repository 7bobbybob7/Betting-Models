"""
models/mlb/leg1_v2_test.py — LEG 1 v2 / Attack 1 decisive test (docs/LEG1_MODEL_V2_PRD.md §6).

Does adding fresh game-context features (umpire / bullpen fatigue / batter rest) give our
model information the market line lacks?

    A: outcome ~ p_mkt                       (market alone)
    B: outcome ~ p_mkt + p_model_v2          (112 + 8 ctx features)
    C: outcome ~ p_mkt + p_model_control     (112 features, same training window/params)

PASS iff AUC_B > AUC_A (target +0.005) AND AUC_B > AUC_C. Control isolates the feature
delta: both models trained on identical rows (2019-2024), identical hyperparameters.

Also reports the PRE-REGISTERED segments (S1 ump x K-pitcher, S2 gassed pen, S3 no-rest
batter) as direct market-gap tests: actual_over - market_implied within segment, per year.

Usage:
    python -m models.mlb.leg1_v2_test --target hrr
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse, contextlib, io as _io
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.mlb.features.game_context_features import build_training_set as build_ctx
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 TARGET_TO_MARKET)

CTX_CACHE = "models/mlb/cache/game_context_2019_2026.parquet"
TRAIN_PQ  = "models/mlb/cache/train_2019_2024.parquet"
BTEST_PQ  = "models/mlb/cache/backtest_2025_2026.parquet"
XGB_PARAMS = dict(max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=50, random_state=42, verbosity=0)
CTX_FEATS = ['ctx_ump_k_rate_365d', 'ctx_ump_bb_rate_365d', 'ctx_ump_runs_pg_365d',
             'ctx_opp_bullpen_relievers_1d', 'ctx_opp_bullpen_ip_2d', 'ctx_opp_bullpen_ip_3d',
             'ctx_batter_rest_days', 'ctx_batter_games_7d']


def _ctx_features() -> pd.DataFrame:
    if os.path.exists(CTX_CACHE):
        print(f"loading cached context features: {CTX_CACHE}")
        return pd.read_parquet(CTX_CACHE)
    ctx = build_ctx(date(2019, 4, 1), date(2026, 12, 31))
    ctx.to_parquet(CTX_CACHE, index=False)
    print(f"cached -> {CTX_CACHE}")
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGET_TO_MARKET.keys()), default="hrr")
    args = ap.parse_args()
    cfg = TARGET_TO_MARKET[args.target]

    ctx = _ctx_features()

    print("loading cached datasets + merging context features...")
    tr = pd.read_parquet(TRAIN_PQ).merge(ctx, on=['game_id', 'player_id'], how='left')
    bt = pd.read_parquet(BTEST_PQ).merge(ctx, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    cov = tr[CTX_FEATS].notna().mean()
    print(f"  train {len(tr):,} rows | backtest {len(bt):,} rows | "
          f"ctx coverage train: min={cov.min():.2f} mean={cov.mean():.2f}")

    # target filter
    if args.target == 'hrr':
        tr = tr[tr['lbl_hrr_valid'] == True]; bt = bt[bt['lbl_hrr_valid'] == True]
    tr = tr[tr[cfg['label_col']].notna()]; bt = bt[bt[cfg['label_col']].notna()]
    ytr = tr[cfg['label_col']].astype(int).values

    base_feats = load_bundle(args.target, 'xgb', Path('models/mlb/saved'))['features']
    v2_feats = base_feats + CTX_FEATS

    print(f"training CONTROL ({len(base_feats)} feats) and V2 ({len(v2_feats)} feats) "
          f"on {len(tr):,} rows, identical params...")
    m_ctl = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_ctl.fit(tr[base_feats].values, ytr, verbose=False)
    m_v2 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS)
    m_v2.fit(tr[v2_feats].values, ytr, verbose=False)

    bt = bt.copy()
    bt['p_ctl'] = m_ctl.predict(bt[base_feats].values)
    bt['p_v2']  = m_v2.predict(bt[v2_feats].values)

    print("attaching Underdog odds...")
    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(bt, args.target, date(2025, 1, 1), date(2026, 12, 31))
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['y'] = B[cfg['label_col']].astype(int)
    B['yr'] = B['game_date'].dt.year
    fit, te = B[B['yr'] == 2025], B[B['yr'] == 2026]
    print(f"  blend-fit 2025: {len(fit):,} | test 2026: {len(te):,}")

    # -------- residual test --------
    print("\n================ RESIDUAL TEST (fit 2025 -> test 2026) ================")
    res = {}
    for name, cols in [('A: market alone', ['p_mkt']),
                       ('B: market + v2', ['p_mkt', 'p_v2']),
                       ('C: market + control', ['p_mkt', 'p_ctl'])]:
        lm = LogisticRegression().fit(fit[cols], fit['y'])
        auc = roc_auc_score(te['y'], lm.predict_proba(te[cols])[:, 1])
        res[name[0]] = auc
        coef = "" if len(cols) == 1 else f"  model_coef={lm.coef_[0][1]:+.3f}"
        print(f"  {name:22s} AUC={auc:.4f}{coef}")
    dBA, dBC = res['B'] - res['A'], res['B'] - res['C']
    print(f"\n  B-A = {dBA:+.4f} (pass needs >0, target +0.005) | B-C = {dBC:+.4f} (needs >0)")
    verdict = "PASS" if (dBA > 0 and dBC > 0) else "FAIL"
    strong = " (meets +0.005 target)" if dBA >= 0.005 else ""
    print(f"  VERDICT: {verdict}{strong}")

    # -------- pre-registered segment market-gap tests --------
    print("\n================ PRE-REGISTERED SEGMENTS (market gap) ================")
    ump_hi = B['ctx_ump_k_rate_365d'] >= B['ctx_ump_k_rate_365d'].quantile(2/3)
    pit_hi = B['pit_k_rate_szn'] >= B['pit_k_rate_szn'].quantile(2/3)
    segs = {
        'S1 big-zone ump x high-K pitcher (pred: unders)': ump_hi & pit_hi,
        'S2 gassed opp pen ip_2d>=7 (pred: overs)':        B['ctx_opp_bullpen_ip_2d'] >= 7,
        'S3 no-rest batter games_7d>=7 (pred: unders)':    B['ctx_batter_games_7d'] >= 7,
    }
    print(f"{'segment':48s} | {'yr':>4s} {'n':>6s} {'gap':>7s} {'z':>6s}")
    print("-" * 80)
    for name, mask in segs.items():
        for yr in (2025, 2026):
            d = B[mask & (B['yr'] == yr)].dropna(subset=['p_mkt'])
            if len(d) < 80:
                print(f"{name:48s} | {yr} {len(d):>6,}  (too small)"); continue
            act = d['y'].mean(); gap = act - d['p_mkt'].mean()
            se = np.sqrt(act * (1 - act) / len(d)); z = gap / se if se > 0 else 0
            print(f"{name:48s} | {yr} {len(d):>6,} {gap:>+7.3f} {z:>+6.2f}")
    print("\n(gap = actual_over - market_implied. Negative = unders beat the market there.)")


if __name__ == "__main__":
    main()
