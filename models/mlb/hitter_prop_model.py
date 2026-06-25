"""
models/mlb/hitter_prop_model.py — Train v1 hitter prop binary classifiers.

Trains three target markets in parallel using the same feature stack:
    - HRR > 1.5   (primary; H + R + RBI)
    - TB  > 1.5   (companion; matchup-friendly market for 2024-2025 backtest)
    - RBI > 0.5   (companion; +197 avg over odds — stresses lineup-OBP feature)

Two model families per target:
    - L1-regularized Logistic Regression (auto-sparsifies the 112-feature set)
    - XGBoost (captures nonlinear interactions; uses native NaN handling)

CROSS-VALIDATION
    Expanding-by-season for hyperparameter selection:
        Train 2019      → val 2020
        Train 2019-2020 → val 2021
        Train 2019-2021 → val 2022
        Train 2019-2022 → val 2023
        Train 2019-2023 → val 2024
    Pick hyperparameters by average validation Brier score across folds.

    Then refit final production model on full train range with locked hyperparameters,
    save bundle. The backtest script will do per-month evaluation on 2025 (and ROI vs
    actual Underdog odds where available).

METRICS REPORTED
    Per fold: Brier score, log loss, ROC-AUC
    Pooled across folds: calibration table (10 deciles), direction-segmented hit rate

NO LEAKAGE INVARIANT
    Median imputation values are computed on TRAIN ONLY, applied to val/test.
    StandardScaler (for LR) fit on TRAIN ONLY.
    Per-fold sample weights = 1 (no class balancing for v1 — base rates already balanced).

Usage:
    python -m models.mlb.hitter_prop_model --target hrr  --start 2019-01-01 --end 2024-12-31
    python -m models.mlb.hitter_prop_model --target all  --start 2019-01-01 --end 2024-12-31
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb

from models.mlb.hitter_prop_dataset import build_dataset


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

TARGETS = {
    'hrr': {
        'label_col':     'lbl_hrr_over_15',
        'valid_col':     'lbl_hrr_valid',  # HRR depends on runs column — backfill in progress
        'line':          1.5,
        'description':   'HRR (Hits + Runs + RBIs) > 1.5',
    },
    'tb': {
        'label_col':     'lbl_tb_over_15',
        'valid_col':     None,  # always valid (no runs dependency)
        'line':          1.5,
        'description':   'Total Bases > 1.5',
    },
    'rbi': {
        'label_col':     'lbl_rbi_over_05',
        'valid_col':     None,
        'line':          0.5,
        'description':   'RBIs > 0.5',
    },
}


# L1 LogReg hyperparameter grid (C = inverse regularization strength).
# Trimmed: C=0.01 won cleanly for all 3 targets in the full sweep; keep it + one looser.
LR_C_GRID = [0.01, 0.1]

# XGBoost grid — trimmed to the proven winner (d4/lr0.05/n400) + a faster alternative.
# Full 4-config sweep picked d4/lr0.05/n400 for all 3 targets; widen again if features change.
XGB_GRID = [
    {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 50},
    {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 200, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 50},
]


# ----------------------------------------------------------------------------
# Feature / label preparation
# ----------------------------------------------------------------------------

def _feature_columns(df: pd.DataFrame) -> list:
    """Every numeric column starting with bat_, pit_, ctx_, mu_.
    Excludes string metadata like bat_hand / pit_throws (those drive matchup features
    upstream but aren't model inputs themselves — handedness info is already baked into
    pit_pct_*_vs_{R,L}HB_30d and mu_platoon_advantage)."""
    cols = [c for c in df.columns if c.startswith(('bat_', 'pit_', 'ctx_', 'mu_'))]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _prepare_X_y(df: pd.DataFrame, target: str) -> tuple:
    """Filter to valid rows for this target, return (X, y, season, game_date)."""
    cfg = TARGETS[target]
    sub = df[df[cfg['label_col']].notna()].copy()
    if cfg['valid_col']:
        sub = sub[sub[cfg['valid_col']] == True].copy()

    X = sub[_feature_columns(sub)].copy().reset_index(drop=True)  # positional alignment with y/season
    y = sub[cfg['label_col']].astype(int).to_numpy()
    season = sub['game_date'].dt.year.to_numpy()
    game_date = sub['game_date'].to_numpy()
    return X, y, season, game_date


def _fit_impute_scale(X_train: pd.DataFrame) -> tuple:
    """Compute train-only medians + StandardScaler for LR.
    Columns entirely NaN in train get median=0 fallback (LR can't handle remaining NaN)."""
    medians = X_train.median(numeric_only=True)
    medians = medians.fillna(0.0)  # for columns that were 100% NaN in train
    X_imp = X_train.fillna(medians).fillna(0.0)  # belt-and-suspenders
    scaler = StandardScaler().fit(X_imp.values)
    return medians, scaler


def _apply_impute_scale(X: pd.DataFrame, medians, scaler) -> np.ndarray:
    return scaler.transform(X.fillna(medians).fillna(0.0).values)


# ----------------------------------------------------------------------------
# Expanding-season cross-validation
# ----------------------------------------------------------------------------

def expanding_season_folds(season: np.ndarray) -> list:
    """Yield (train_idx, val_idx, train_label, val_label) per fold."""
    uniq = sorted(set(int(s) for s in season))
    folds = []
    for i in range(1, len(uniq)):
        train_seasons = uniq[:i]
        val_season = uniq[i]
        train_idx = np.where(np.isin(season, train_seasons))[0]
        val_idx   = np.where(season == val_season)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        train_label = f"{min(train_seasons)}-{max(train_seasons)}"
        val_label = str(val_season)
        folds.append((train_idx, val_idx, train_label, val_label))
    return folds


# ----------------------------------------------------------------------------
# Single-fold trainers
# ----------------------------------------------------------------------------

def _train_lr_l1(X_tr, y_tr, X_val, y_val, C: float) -> dict:
    """Train LR-L1 at given C, return val metrics."""
    medians, scaler = _fit_impute_scale(X_tr)
    X_tr_s  = _apply_impute_scale(X_tr,  medians, scaler)
    X_val_s = _apply_impute_scale(X_val, medians, scaler)
    model = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                                max_iter=2000, random_state=42)
    model.fit(X_tr_s, y_tr)
    p_val = model.predict_proba(X_val_s)[:, 1]
    n_active = int((model.coef_[0] != 0).sum())
    return {
        'val_brier': brier_score_loss(y_val, p_val),
        'val_logloss': log_loss(y_val, np.clip(p_val, 1e-7, 1 - 1e-7)),
        'val_auc': roc_auc_score(y_val, p_val),
        'n_features_active': n_active,
        '_model': model,
        '_medians': medians,
        '_scaler': scaler,
        '_preds': p_val,
    }


def _train_xgb(X_tr, y_tr, X_val, y_val, params: dict) -> dict:
    """Train XGBoost with given params, return val metrics."""
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
        **params,
    )
    # XGBoost handles NaN natively — pass raw values
    model.fit(X_tr.values, y_tr,
              eval_set=[(X_val.values, y_val)],
              verbose=False)
    p_val = model.predict_proba(X_val.values)[:, 1]
    return {
        'val_brier': brier_score_loss(y_val, p_val),
        'val_logloss': log_loss(y_val, np.clip(p_val, 1e-7, 1 - 1e-7)),
        'val_auc': roc_auc_score(y_val, p_val),
        '_model': model,
        '_preds': p_val,
    }


# ----------------------------------------------------------------------------
# Hyperparameter selection — average val Brier across expanding-season folds
# ----------------------------------------------------------------------------

def select_lr_hyperparameter(X: pd.DataFrame, y: np.ndarray, season: np.ndarray) -> tuple:
    folds = expanding_season_folds(season)
    print(f"  LR-L1: {len(folds)} folds")
    results = []
    for C in LR_C_GRID:
        fold_briers = []
        for tr_idx, val_idx, tr_lbl, val_lbl in folds:
            r = _train_lr_l1(X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx], C)
            fold_briers.append(r['val_brier'])
        avg = float(np.mean(fold_briers))
        print(f"    C={C:>5}: avg val Brier = {avg:.4f}  (per-fold: {[f'{b:.4f}' for b in fold_briers]})")
        results.append({'C': C, 'avg_brier': avg, 'fold_briers': fold_briers})
    best = min(results, key=lambda r: r['avg_brier'])
    print(f"  LR-L1 best: C={best['C']} (Brier={best['avg_brier']:.4f})")
    return best['C'], results


def select_xgb_hyperparameter(X: pd.DataFrame, y: np.ndarray, season: np.ndarray) -> tuple:
    folds = expanding_season_folds(season)
    print(f"  XGBoost: {len(folds)} folds, {len(XGB_GRID)} param sets")
    results = []
    for params in XGB_GRID:
        fold_briers = []
        for tr_idx, val_idx, tr_lbl, val_lbl in folds:
            r = _train_xgb(X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx], params)
            fold_briers.append(r['val_brier'])
        avg = float(np.mean(fold_briers))
        param_str = f"d={params['max_depth']} lr={params['learning_rate']} n={params['n_estimators']}"
        print(f"    {param_str}: avg val Brier = {avg:.4f}")
        results.append({'params': params, 'avg_brier': avg, 'fold_briers': fold_briers})
    best = min(results, key=lambda r: r['avg_brier'])
    print(f"  XGBoost best: {best['params']} (Brier={best['avg_brier']:.4f})")
    return best['params'], results


# ----------------------------------------------------------------------------
# Final fit on full window + isotonic calibration on held-out latest season
# ----------------------------------------------------------------------------
#
# We hold out the most recent training season (e.g. 2024) as a calibration set:
# the base model trains on the earlier seasons, then an isotonic regression learns
# the mapping raw_proba → true_rate on the unseen calibration season. This corrects
# the high-decile overconfidence that was manufacturing false +EV bets. The held-out
# season is genuinely unseen by the base model, so calibration respects time order.

def _isotonic_report(raw, cal, y) -> str:
    """One-line before/after Brier on the calibration set."""
    return (f"calib-set Brier raw={brier_score_loss(y, raw):.4f} "
            f"→ cal={brier_score_loss(y, cal):.4f}")


def fit_final_lr(X: pd.DataFrame, y: np.ndarray, season: np.ndarray, C: float) -> dict:
    """Train LR base on seasons < latest; isotonic-calibrate on the latest season."""
    calib_season = int(season.max())
    base_m  = season < calib_season
    calib_m = season == calib_season

    medians, scaler = _fit_impute_scale(X[base_m])
    Xb = _apply_impute_scale(X[base_m], medians, scaler)
    model = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                                max_iter=2000, random_state=42)
    model.fit(Xb, y[base_m])

    Xc = _apply_impute_scale(X[calib_m], medians, scaler)
    raw_calib = model.predict_proba(Xc)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_calib, y[calib_m])
    cal_calib = calibrator.predict(raw_calib)
    print(f"    LR  {_isotonic_report(raw_calib, cal_calib, y[calib_m])} "
          f"(base={base_m.sum():,} on <{calib_season}, calib={calib_m.sum():,} on {calib_season})")

    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
    active = coefs[coefs != 0]
    return {
        'model': model, 'medians': medians, 'scaler': scaler, 'calibrator': calibrator,
        'features': list(X.columns), 'n_active': len(active),
        'top_coefs': active.head(20).to_dict(), 'calib_season': calib_season,
    }


def fit_final_xgb(X: pd.DataFrame, y: np.ndarray, season: np.ndarray, params: dict) -> dict:
    """Train XGB base on seasons < latest; isotonic-calibrate on the latest season."""
    calib_season = int(season.max())
    base_m  = season < calib_season
    calib_m = season == calib_season

    model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        random_state=42, verbosity=0, **params,
    )
    model.fit(X[base_m].values, y[base_m], verbose=False)

    raw_calib = model.predict_proba(X[calib_m].values)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_calib, y[calib_m])
    cal_calib = calibrator.predict(raw_calib)
    print(f"    XGB {_isotonic_report(raw_calib, cal_calib, y[calib_m])} "
          f"(base={base_m.sum():,} on <{calib_season}, calib={calib_m.sum():,} on {calib_season})")

    importance = dict(zip(X.columns, model.feature_importances_))
    top = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:20])
    return {
        'model': model, 'calibrator': calibrator, 'features': list(X.columns),
        'top_importance': top, 'params': params, 'calib_season': calib_season,
    }


def calibration_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """Per-decile: count, mean predicted P, actual rate."""
    bins = np.linspace(0, 1, n_bins + 1)
    df = pd.DataFrame({'y': y_true, 'p': y_pred})
    df['bin'] = pd.cut(df['p'], bins=bins, include_lowest=True)
    out = df.groupby('bin', observed=True).agg(
        n=('y', 'size'),
        mean_pred=('p', 'mean'),
        actual_rate=('y', 'mean'),
    ).reset_index()
    out['diff'] = out['actual_rate'] - out['mean_pred']
    return out


# ----------------------------------------------------------------------------
# Orchestrator: train one target end-to-end
# ----------------------------------------------------------------------------

def train_target(df: pd.DataFrame, target: str, output_dir: Path) -> dict:
    cfg = TARGETS[target]
    print("\n" + "=" * 72)
    print(f"  TRAINING: {cfg['description']}")
    print("=" * 72)

    X, y, season, gdates = _prepare_X_y(df, target)
    print(f"  Total rows: {len(X):,}  (positive class: {y.mean():.3f})")
    print(f"  Seasons:    {sorted(set(season))}")
    print(f"  Features:   {len(_feature_columns(df))}")

    folds = expanding_season_folds(season)
    if len(folds) < 2:
        print(f"  WARNING: only {len(folds)} CV folds — need ≥2 for meaningful HP selection")

    # --- Hyperparameter selection ---
    print("\n[HP selection]")
    best_C, lr_grid = select_lr_hyperparameter(X, y, season)
    best_xgb, xgb_grid = select_xgb_hyperparameter(X, y, season)

    # --- Pooled CV diagnostics (use last fold's val as pseudo-test) ---
    print("\n[Pooled-fold diagnostics]")
    lr_preds_all, xgb_preds_all, y_all = [], [], []
    for tr_idx, val_idx, tr_lbl, val_lbl in folds:
        r_lr  = _train_lr_l1(X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx], best_C)
        r_xgb = _train_xgb(X.iloc[tr_idx], y[tr_idx], X.iloc[val_idx], y[val_idx], best_xgb)
        lr_preds_all.extend(r_lr['_preds'])
        xgb_preds_all.extend(r_xgb['_preds'])
        y_all.extend(y[val_idx])
    y_all = np.array(y_all)
    lr_preds_all = np.array(lr_preds_all)
    xgb_preds_all = np.array(xgb_preds_all)

    print(f"  LR-L1 pooled: Brier={brier_score_loss(y_all, lr_preds_all):.4f}  "
          f"LogLoss={log_loss(y_all, np.clip(lr_preds_all, 1e-7, 1-1e-7)):.4f}  "
          f"AUC={roc_auc_score(y_all, lr_preds_all):.4f}")
    print(f"  XGB   pooled: Brier={brier_score_loss(y_all, xgb_preds_all):.4f}  "
          f"LogLoss={log_loss(y_all, np.clip(xgb_preds_all, 1e-7, 1-1e-7)):.4f}  "
          f"AUC={roc_auc_score(y_all, xgb_preds_all):.4f}")

    cal_lr = calibration_table(y_all, lr_preds_all)
    print("\n  LR-L1 calibration (deciles):")
    print(cal_lr.to_string(index=False))
    cal_xgb = calibration_table(y_all, xgb_preds_all)
    print("\n  XGBoost calibration (deciles):")
    print(cal_xgb.to_string(index=False))

    # --- Final models: base fit + isotonic calibration on held-out latest season ---
    print("\n[Final fit + isotonic calibration]")
    lr_final  = fit_final_lr(X, y, season, best_C)
    xgb_final = fit_final_xgb(X, y, season, best_xgb)

    print(f"  LR-L1 active features: {lr_final['n_active']} / {len(X.columns)}")
    print(f"  LR-L1 top 10 coefficients:")
    for f, c in list(lr_final['top_coefs'].items())[:10]:
        print(f"    {f:42s}  {c:+.4f}")
    print(f"  XGBoost top 10 features by importance:")
    for f, imp in list(xgb_final['top_importance'].items())[:10]:
        print(f"    {f:42s}  {imp:.4f}")

    # --- Save bundles (model + calibrator) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_bundle = {
        'target': target, 'cfg': cfg,
        'model_type': 'lr_l1', 'C': best_C,
        'model': lr_final['model'], 'medians': lr_final['medians'], 'scaler': lr_final['scaler'],
        'calibrator': lr_final['calibrator'], 'calib_season': lr_final['calib_season'],
        'features': lr_final['features'], 'n_active': lr_final['n_active'],
        'top_coefs': lr_final['top_coefs'],
    }
    xgb_bundle = {
        'target': target, 'cfg': cfg,
        'model_type': 'xgb', 'params': best_xgb,
        'model': xgb_final['model'], 'calibrator': xgb_final['calibrator'],
        'calib_season': xgb_final['calib_season'],
        'features': xgb_final['features'], 'top_importance': xgb_final['top_importance'],
    }
    with open(output_dir / f'hitter_{target}_lr_l1.pkl', 'wb') as f:
        pickle.dump(lr_bundle, f)
    with open(output_dir / f'hitter_{target}_xgb.pkl', 'wb') as f:
        pickle.dump(xgb_bundle, f)
    print(f"  Bundles saved to {output_dir}/hitter_{target}_*.pkl")

    return {
        'target': target,
        'rows': len(X), 'positive_rate': float(y.mean()),
        'best_C': best_C, 'best_xgb_params': best_xgb,
        'pooled_brier_lr':  float(brier_score_loss(y_all, lr_preds_all)),
        'pooled_brier_xgb': float(brier_score_loss(y_all, xgb_preds_all)),
        'pooled_auc_lr':    float(roc_auc_score(y_all, lr_preds_all)),
        'pooled_auc_xgb':   float(roc_auc_score(y_all, xgb_preds_all)),
        'n_active_lr_features': lr_final['n_active'],
    }


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=list(TARGETS.keys()) + ['all'], default='all')
    parser.add_argument('--start',  default='2019-01-01')
    parser.add_argument('--end',    default='2024-12-31')
    parser.add_argument('--output-dir', default='models/mlb/saved')
    parser.add_argument('--dataset-parquet', default=None,
                        help="Path to a prebuilt dataset parquet. If it exists, load it "
                             "(skips the ~30min rebuild). If it doesn't, build + save it there.")
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').date()
    end   = datetime.strptime(args.end,   '%Y-%m-%d').date()
    output_dir = Path(args.output_dir)

    if args.dataset_parquet and os.path.exists(args.dataset_parquet):
        print(f"Loading cached dataset from {args.dataset_parquet}...")
        df = pd.read_parquet(args.dataset_parquet)
        print(f"Dataset: {len(df):,} rows x {df.shape[1]} cols (cached)")
    else:
        print(f"Building dataset {start} → {end}...")
        df = build_dataset(start, end)
        print(f"Dataset: {len(df):,} rows x {df.shape[1]} cols")
        if args.dataset_parquet:
            df.to_parquet(args.dataset_parquet, index=False)
            print(f"Cached dataset to {args.dataset_parquet}")

    targets = list(TARGETS.keys()) if args.target == 'all' else [args.target]
    all_results = {}
    for t in targets:
        all_results[t] = train_target(df, t, output_dir)

    # Summary
    print("\n" + "=" * 72)
    print("  TRAINING SUMMARY")
    print("=" * 72)
    summary = pd.DataFrame(all_results).T
    print(summary.to_string())

    # Save summary
    with open(output_dir / 'training_summary.json', 'w') as f:
        # Convert numpy types for JSON serialization
        for k in all_results:
            for kk, vv in all_results[k].items():
                if hasattr(vv, 'item'):
                    all_results[k][kk] = vv.item()
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
