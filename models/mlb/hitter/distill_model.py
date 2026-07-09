"""
models/mlb/distill_model.py — LEG 3: distill the sharp market into our own model.

Instead of training on noisy game outcomes (leg 1, hitter_prop_model.py), we train on
Novig's de-vigged price as a SOFT TARGET — i.e. learn to reproduce what the sharp market
would price, using our own features. The result is an owned, Novig-independent model that
inherits Novig's sharpness (capped at Novig's level, limited by what our features capture).

    leg 1:  features -> P(did it go over?)        target = 0/1 outcome   (high variance)
    leg 3:  features -> P(Novig's fair price)     target = 0.47          (sharp, low variance)

Teacher labels: Novig de-vigged probability from bettingpros_props (book_id=60), historical.
Features: the same 112-feature stack, loaded from the cached parquets.

Key question this answers: does mimicking the sharp line land SHARPER (higher outcome-AUC)
than training on outcomes? If yes, the distilled model is the better owned asset.

Usage:
    python -m models.mlb.hitter.distill_model --target hrr
    python -m models.mlb.hitter.distill_model --target all
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import argparse, pickle
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_absolute_error
import contextlib, io as _io

from db.db import query
from models.mlb.hitter.backtest import (load_bundle, predict_proba, american_to_decimal,
                                 _build_player_match, TARGET_TO_MARKET)

NOVIG_BOOK = 60
TRAIN_PARQUETS = ["models/mlb/cache/train_2019_2024.parquet",
                  "models/mlb/cache/backtest_2025_2026.parquet"]


def _novig_prob(market_id: int, line: float, start="2024-01-01") -> pd.DataFrame:
    """Novig de-vigged OVER probability per (player_id, game_date) for a market+line."""
    odds = query("""
        SELECT prop_date AS game_date, bp_player_id, over_odds, under_odds
        FROM bettingpros_props
        WHERE book_id=%(b)s AND market_id=%(m)s AND over_line=%(l)s
          AND prop_date >= %(s)s AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """, params={'b': NOVIG_BOOK, 'm': market_id, 'l': line, 's': start})
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    io_ = 1 / odds['over_odds'].apply(american_to_decimal)
    iu_ = 1 / odds['under_odds'].apply(american_to_decimal)
    odds['novig_prob'] = io_ / (io_ + iu_)
    # map bp_player_id -> our player_id
    with contextlib.redirect_stdout(_io.StringIO()):
        m = _build_player_match(pd.Timestamp(start).date(), pd.Timestamp('2026-12-31').date())
    m = m[m['player_id'].notna()][['bp_player_id', 'player_id']]
    odds = odds.merge(m, on='bp_player_id', how='inner')
    # one row per (player_id, game_date): average if dupes
    return (odds.groupby(['player_id', 'game_date'], as_index=False)['novig_prob'].mean())


def _underdog_odds(market_id: int, line: float, start="2024-01-01") -> pd.DataFrame:
    """Underdog over/under American odds per (player_id, game_date) — for ROI eval."""
    odds = query("""
        SELECT prop_date AS game_date, bp_player_id, over_odds, under_odds
        FROM bettingpros_props
        WHERE book_id=36 AND market_id=%(m)s AND over_line=%(l)s
          AND prop_date >= %(s)s AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """, params={'m': market_id, 'l': line, 's': start})
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    with contextlib.redirect_stdout(_io.StringIO()):
        m = _build_player_match(pd.Timestamp(start).date(), pd.Timestamp('2026-12-31').date())
    m = m[m['player_id'].notna()][['bp_player_id', 'player_id']]
    odds = odds.merge(m, on='bp_player_id', how='inner')
    return (odds.groupby(['player_id', 'game_date'], as_index=False)
                .agg(ud_over=('over_odds', 'mean'), ud_under=('under_odds', 'mean')))


def _load_features() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in TRAIN_PARQUETS]
    df = pd.concat(parts, ignore_index=True)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df


def build_distill_set(target: str, feat_cols: list) -> pd.DataFrame:
    cfg = TARGET_TO_MARKET[target]
    feats = _load_features()
    if target == 'hrr':
        feats = feats[feats['lbl_hrr_valid'] == True]
    feats = feats[feats[cfg['label_col']].notna()]
    nv = _novig_prob(cfg['market_id'], cfg['over_line'])
    df = feats.merge(nv, on=['player_id', 'game_date'], how='inner')
    df['year'] = df['game_date'].dt.year
    keep = ['player_id', 'game_date', 'year', 'novig_prob', cfg['label_col']] + feat_cols
    return df[keep].copy()


def train_distill(target: str, output_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print(f"  LEG 3 DISTILLATION: {target.upper()} — learn Novig's price")
    print("=" * 70)

    # Reuse the exact feature list the outcome model uses, for an apples-to-apples compare
    leg1 = load_bundle(target, 'xgb', output_dir)
    feat_cols = leg1['features']

    df = build_distill_set(target, feat_cols)
    print(f"  distill pairs (features + Novig price): {len(df):,}")
    print(f"  years: {sorted(df['year'].unique())}")

    # Time split: train on <2026, test on 2026
    tr, te = df[df['year'] < 2026], df[df['year'] == 2026]
    print(f"  train (<2026): {len(tr):,} | test (2026): {len(te):,}")
    if len(te) < 300:
        print("  WARNING: small test set")

    Xtr, Xte = tr[feat_cols], te[feat_cols]
    ytr_soft = tr['novig_prob'].values            # SOFT target = Novig's price
    yte_soft = te['novig_prob'].values
    yte_out  = te[TARGET_TO_MARKET[target]['label_col']].astype(int).values  # real outcomes

    # reg:logistic — predict a probability, logistic loss against the soft target
    model = xgb.XGBRegressor(objective='reg:logistic', eval_metric='logloss',
                             max_depth=4, learning_rate=0.05, n_estimators=400,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                             random_state=42, verbosity=0)
    model.fit(Xtr.values, ytr_soft, verbose=False)
    pred = model.predict(Xte.values)

    # --- How well do we reproduce Novig? ---
    mae = mean_absolute_error(yte_soft, pred)
    corr = np.corrcoef(pred, yte_soft)[0, 1]

    # --- The decisive metric: outcome-AUC. Sharper teacher learning => higher AUC ---
    auc_distill = roc_auc_score(yte_out, pred)             # leg 3 (distilled)
    auc_novig   = roc_auc_score(yte_out, yte_soft)         # the teacher (ceiling)
    p_leg1 = predict_proba(leg1, te[feat_cols])            # leg 1 (outcome-trained)
    auc_leg1 = roc_auc_score(yte_out, p_leg1)

    print(f"\n  Reproducing Novig:  MAE={mae:.4f}  corr={corr:.3f}")
    print(f"\n  Outcome-AUC on 2026 (higher = sharper):")
    print(f"    Leg 1  (trained on outcomes):   {auc_leg1:.4f}")
    print(f"    Leg 3  (distilled from Novig):  {auc_distill:.4f}")
    print(f"    Novig  (the teacher / ceiling): {auc_novig:.4f}")
    verdict = ("distillation SHARPER than outcome model" if auc_distill > auc_leg1 + 0.002
               else "no improvement over outcome model" if auc_distill < auc_leg1 - 0.002
               else "≈ tie with outcome model")
    print(f"    -> {verdict}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {'target': target, 'model_type': 'distill', 'model': model,
              'features': feat_cols, 'cfg': TARGET_TO_MARKET[target],
              'test_auc_distill': float(auc_distill), 'test_auc_leg1': float(auc_leg1),
              'test_auc_novig': float(auc_novig), 'novig_mae': float(mae)}
    with open(output_dir / f"hitter_{target}_distill.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print(f"  saved -> {output_dir}/hitter_{target}_distill.pkl")
    return bundle


def _fit_xgb(X, y):
    m = xgb.XGBRegressor(objective='reg:logistic', eval_metric='logloss',
                         max_depth=4, learning_rate=0.05, n_estimators=400,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                         random_state=42, verbosity=0)
    m.fit(X.values, y, verbose=False)
    return m


def walk_forward_cv(target: str, output_dir: Path, embargo_days=5, min_train=800, min_test=120):
    """Monthly expanding-window CV with an embargo gap. Per fold: train on everything
    before (test month start − embargo), test on the month. Reports AUC + MAE-to-Novig
    + ROI-vs-Underdog distribution."""
    print("\n" + "=" * 78)
    print(f"  WALK-FORWARD CV (distill): {target.upper()}  (embargo={embargo_days}d)")
    print("=" * 78)
    feat_cols = load_bundle(target, 'xgb', output_dir)['features']
    cfg = TARGET_TO_MARKET[target]

    feats = _load_features()
    if target == 'hrr':
        feats = feats[feats['lbl_hrr_valid'] == True]
    feats = feats[feats[cfg['label_col']].notna()]
    df = (feats.merge(_novig_prob(cfg['market_id'], cfg['over_line']), on=['player_id', 'game_date'])
               .merge(_underdog_odds(cfg['market_id'], cfg['over_line']), on=['player_id', 'game_date']))
    df['ym'] = df['game_date'].dt.to_period('M')
    months = sorted(df['ym'].unique())

    rows = []
    for test_ym in months[2:]:                     # need a couple months of history first
        test_start = test_ym.to_timestamp()
        cutoff = test_start - pd.Timedelta(days=embargo_days)
        tr = df[df['game_date'] < cutoff]
        te = df[df['ym'] == test_ym]
        y = te[cfg['label_col']].astype(int).values
        if len(tr) < min_train or len(te) < min_test or y.min() == y.max():
            continue
        model = _fit_xgb(tr[feat_cols], tr['novig_prob'].values)
        pred = model.predict(te[feat_cols].values)

        auc = roc_auc_score(y, pred)
        auc_nv = roc_auc_score(y, te['novig_prob'].values)
        mae = mean_absolute_error(te['novig_prob'].values, pred)

        # ROI vs Underdog: bet the +EV side using distill pred as P_true
        do = te['ud_over'].apply(american_to_decimal).values
        du = te['ud_under'].apply(american_to_decimal).values
        ev_o = pred * (do - 1) - (1 - pred)
        ev_u = (1 - pred) * (du - 1) - pred
        bet_over = ev_o >= ev_u
        ev_best = np.where(bet_over, ev_o, ev_u)
        won = np.where(bet_over, y == 1, y == 0)
        payout = np.where(bet_over, do, du)
        profit = np.where(won, payout - 1.0, -1.0)
        gate = ev_best > 0                          # all +EV bets (high-volume view)
        roi = profit[gate].mean() if gate.sum() else np.nan
        rows.append({'month': str(test_ym), 'train_n': len(tr), 'test_n': len(te),
                     'auc': auc, 'auc_novig': auc_nv, 'mae': mae,
                     'bets': int(gate.sum()), 'roi': roi})

    if not rows:
        print("  insufficient data for any fold"); return None
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    print(f"\n  AUC:  mean={R.auc.mean():.4f} std={R.auc.std():.4f}  "
          f"(Novig ceiling mean={R.auc_novig.mean():.4f})  folds={len(R)}")
    print(f"  MAE to Novig: mean={R.mae.mean():.4f}")
    print(f"  Monthly ROI vs Underdog (+EV gate): mean={R.roi.mean():+.4f} "
          f"std={R.roi.std():.4f} worst={R.roi.min():+.4f} best={R.roi.max():+.4f}")
    print(f"  total bets across folds: {R.bets.sum():,}")
    pos = (R.roi > 0).sum()
    print(f"  profitable months: {pos}/{len(R)}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGET_TO_MARKET.keys()) + ["all"], default="all")
    ap.add_argument("--mode", choices=["fit", "cv"], default="fit")
    ap.add_argument("--output-dir", default="models/mlb/saved")
    args = ap.parse_args()
    out = Path(args.output_dir)
    targets = list(TARGET_TO_MARKET.keys()) if args.target == "all" else [args.target]

    if args.mode == "cv":
        for t in targets:
            walk_forward_cv(t, out)
        return

    results = {t: train_distill(t, out) for t in targets}
    print("\n" + "=" * 70)
    print("  SUMMARY — outcome-AUC (2026): leg1 vs leg3-distill vs Novig ceiling")
    print("=" * 70)
    for t, b in results.items():
        print(f"  {t:4s}  leg1={b['test_auc_leg1']:.4f}  distill={b['test_auc_distill']:.4f}  "
              f"novig={b['test_auc_novig']:.4f}")


if __name__ == "__main__":
    main()
