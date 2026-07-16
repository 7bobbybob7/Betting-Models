"""
models/mlb/pitcher/outs_gate.py — pitcher OUTS-RECORDED model, iteration 1.

Strategic reframe (from the venue sweep): the K market is SHARP (model can't beat it),
but OUTS/IP is SOFT — DK misprices it +14.8% vs Novig, mechanism = starter innings have
trended DOWN (quick hooks) and lines lag. A soft market means a model has room. So we
build a model of "how deep does this starter go" and gate it against the outs market the
same way we gated hitters.

Target: outs recorded (= IP in outs; ip 5.2 -> 17 outs). Predict via Poisson.
Durability/efficiency feature set (deep-outing drivers), gated vs Novig outs market both
directions, plus does model beat market standalone (easier here than K if market is soft).

ACCEPT iff model+market blend > market alone in both directions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
import xgboost as xgb

from db.db import query
from models.mlb.hitter.backtest import american_to_decimal
from models.mlb.feature_sets import XGB_PARAMS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
OUTS_MKT, NOVIG, DK = 405, 60, 12

# durability / efficiency features (drivers of how deep a starter goes)
OUTS_FEATS = ['ip_per_start_5', 'ip_per_start_szn', 'pitches_per_start_5',
              'k_per_start_5', 'k_per_start_szn', 'sc_swstr_rate_5', 'sc_fb_velo_5',
              'opp_k_rate_15', 'opp_k_rate_30']


def ip_to_outs(ip):
    f = np.floor(ip)
    return (f * 3 + np.round((ip - f) * 10)).astype(int)


def _load(name, rel, reg=None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); sys.modules[reg or name] = m
    spec.loader.exec_module(m); return m


def main():
    _load("models.mlb.statcast_features", "archive/models/mlb/statcast_features.py")
    km = _load("k_model_arch", "archive/models/mlb/k_model.py")
    print("building dataset...")
    df = km.build_k_dataset()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df[df['actual_ip'].notna() & (df['actual_ip'] > 0)]
    df['actual_outs'] = ip_to_outs(df['actual_ip'].values)
    feats = [c for c in OUTS_FEATS if c in df.columns]
    print(f"outs range: {df['actual_outs'].min()}-{df['actual_outs'].max()}, "
          f"mean {df['actual_outs'].mean():.1f} | {len(feats)} feats")

    tr = df[df['season'] <= 2024]; te = df[df['season'] >= 2025].copy()
    m = xgb.XGBRegressor(objective='count:poisson', **XGB_PARAMS)
    med = tr[feats].median()
    m.fit(tr[feats].fillna(med).values, tr['actual_outs'].values)
    te['lam'] = m.predict(te[feats].fillna(med).values)
    print(f"OOS outs MAE: {mean_absolute_error(te['actual_outs'], te['lam']):.2f} "
          f"(naive szn-mean MAE: {mean_absolute_error(te['actual_outs'], [tr['actual_outs'].mean()]*len(te)):.2f})")

    pl = query("SELECT player_id, LOWER(full_name) fn FROM players")
    P = query("""SELECT prop_date AS game_date, over_line, over_odds, under_odds, actual, is_scored,
                 LOWER(player_first_name||' '||player_last_name) fn
                 FROM bettingpros_props WHERE market_id=%(m)s AND book_id=%(b)s
                 AND over_odds IS NOT NULL AND under_odds IS NOT NULL""", params={'m': OUTS_MKT, 'b': NOVIG})
    P['game_date'] = pd.to_datetime(P['game_date'])
    P = P.merge(pl, on='fn', how='inner').rename(columns={'player_id': 'pitcher_id'})
    P = P.merge(te[['game_date', 'pitcher_id', 'lam']], on=['game_date', 'pitcher_id'], how='inner')
    io_ = 1 / P['over_odds'].apply(american_to_decimal); iu_ = 1 / P['under_odds'].apply(american_to_decimal)
    P['p_mkt'] = io_ / (io_ + iu_)
    P['p_model'] = poisson.sf(np.floor(P['over_line']), P['lam'])
    P['y'] = (P['actual'].astype(float) > P['over_line']).astype(int); P['yr'] = P['game_date'].dt.year

    print(f"\nmatched Novig outs props: 2025={len(P[P.yr==2025]):,} 2026={len(P[P.yr==2026]):,}")
    print(f"model-alone AUC {roc_auc_score(P['y'],P['p_model']):.4f} | market AUC {roc_auc_score(P['y'],P['p_mkt']):.4f}")
    print("\n===== OUTS MODEL GATE (Novig anchor, both directions) =====")
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = P[P['yr'] == fy], P[P['yr'] == ty]
        if len(f) < 200 or len(t) < 200:
            print(f"  {fy}->{ty}: thin ({len(f)}/{len(t)})"); continue
        a = {}
        for k, cols in [('mkt', ['p_mkt']), ('B', ['p_mkt', 'p_model'])]:
            lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
            a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        print(f"  fit {fy}->test {ty} (n={len(t):,}): mkt={a['mkt']:.4f} B={a['B']:.4f}  "
              f"B-mkt={a['B']-a['mkt']:+.4f}")
    print("\n(ACCEPT iff B>mkt both directions. Standalone-vs-market gap tells if model competitive.)")


if __name__ == "__main__":
    main()
