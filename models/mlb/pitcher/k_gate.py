"""
models/mlb/pitcher_k_gate.py — PITCHER STRIKEOUT market gate (market 285).

The Poisson K model (Phase 2; 1.83 MAE) has never seen prop odds. It produces a full
distribution, so VARYING lines (4.5..8.5) are handled natively: P(K > L) = PoissonSF.
Mechanism prior: K-suppression is the market's most-confirmed soft spot (4 independent
confirmations on hitter markets).

Protocol:
  - Rebuild the archived K dataset (leak-safe prior-only rolling features).
  - RETRAIN Poisson on seasons <= 2024 in-script (don't trust old pickle's window) ->
    2025/2026 predictions are honestly OOS.
  - Gate (Novig anchor, both directions): A = market devig alone vs B = market + P_model.
  - Betting sim @ Underdog odds with the 90d walk-forward blend (the honest protocol).

Archived modules are injected under their legacy import paths (models.mlb.statcast_features).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import importlib.util, contextlib, io as _io
from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from db.db import query
from models.mlb.hitter.backtest import american_to_decimal

ROOT = os.path.join(os.path.dirname(__file__), "../../..")


def _load_archived(name, relpath, register_as=None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register_as or name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    # archived statcast_features must be importable under its legacy name first
    _load_archived("models.mlb.statcast_features", "archive/models/mlb/statcast_features.py",
                   register_as="models.mlb.statcast_features")
    km = _load_archived("k_model_arch", "archive/models/mlb/k_model.py")

    print("building K dataset (archived builder, prior-only features)...")
    df = km.build_k_dataset()
    df['game_date'] = pd.to_datetime(df['game_date'])
    feats = [c for c in km.K_FEATURES if c in df.columns]
    tr = df[df['season'] <= 2024].copy()
    te = df[df['season'] >= 2025].copy()
    print(f"train {len(tr):,} starts (<=2024) | score {len(te):,} (2025-26) | {len(feats)} feats")

    med = tr[feats].median()
    Xtr = tr[feats].fillna(med); Xte = te[feats].fillna(med)
    sc = StandardScaler().fit(Xtr)
    m = PoissonRegressor(alpha=1.0, max_iter=1000)
    m.fit(sc.transform(Xtr), tr['actual_k'])
    te['lam'] = m.predict(sc.transform(Xte))
    mae = np.abs(te['lam'] - te['actual_k']).mean()
    print(f"OOS 2025-26 MAE: {mae:.3f} (phase-2 reference: 1.83)")

    # ---- props: match pitchers by full name ----
    pl = query("SELECT player_id, LOWER(full_name) fn FROM players")
    def props_for(book):
        p = query("""SELECT prop_date AS game_date, over_line, over_odds, under_odds, actual,
                     is_scored, LOWER(player_first_name || ' ' || player_last_name) fn
                     FROM bettingpros_props WHERE market_id=285 AND book_id=%(b)s
                     AND over_odds IS NOT NULL AND under_odds IS NOT NULL""",
                  params={'b': book})
        p['game_date'] = pd.to_datetime(p['game_date'])
        p = p.merge(pl, on='fn', how='inner').rename(columns={'player_id': 'pitcher_id'})
        return p

    T = te[['game_date', 'pitcher_id', 'lam', 'actual_k']]
    for book, tag in [(60, 'NOVIG'), (36, 'UD')]:
        P = props_for(book).merge(T, on=['game_date', 'pitcher_id'], how='inner')
        io_ = 1 / P['over_odds'].apply(american_to_decimal)
        iu_ = 1 / P['under_odds'].apply(american_to_decimal)
        P['p_mkt'] = io_ / (io_ + iu_)
        P['p_model'] = poisson.sf(np.floor(P['over_line']), P['lam'])
        P['y'] = (P['actual_k'] > P['over_line']).astype(int)
        P['yr'] = P['game_date'].dt.year
        print(f"\n===== {tag} matched props: 2025={len(P[P.yr==2025]):,} 2026={len(P[P.yr==2026]):,} "
              f"| model-alone AUC {roc_auc_score(P['y'], P['p_model']):.4f} "
              f"| market AUC {roc_auc_score(P['y'], P['p_mkt']):.4f}")
        if tag == 'NOVIG':
            print("===== K GATE (Novig anchor, both directions) =====")
            for fy, ty in [(2025, 2026), (2026, 2025)]:
                f, t = P[P['yr'] == fy], P[P['yr'] == ty]
                if len(f) < 300 or len(t) < 300:
                    print(f"  fit {fy}->test {ty}: too thin ({len(f)}/{len(t)})"); continue
                aa = {}
                for kk, cols in [('A', ['p_mkt']), ('B', ['p_mkt', 'p_model'])]:
                    lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                    aa[kk] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
                print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa['A']:.4f} B={aa['B']:.4f} "
                      f"B-A={aa['B']-aa['A']:+.4f}")
        else:
            # walk-forward blend betting sim @ UD odds
            P = P.sort_values('game_date')
            P['dec_o'] = P['over_odds'].apply(american_to_decimal)
            P['dec_u'] = P['under_odds'].apply(american_to_decimal)
            P['pb'] = np.nan
            for d0 in sorted(P['game_date'].unique()):
                w = P[(P['game_date'] >= d0 - pd.Timedelta(days=90)) & (P['game_date'] < d0)]
                if len(w) < 300 or w['y'].nunique() < 2: continue
                lm = LogisticRegression(max_iter=1000).fit(w[['p_mkt', 'p_model']], w['y'])
                r = P['game_date'] == d0
                P.loc[r, 'pb'] = lm.predict_proba(P.loc[r, ['p_mkt', 'p_model']])[:, 1]
            D = P[P['pb'].notna() & P['is_scored']].copy()
            pv = D['pb']
            evo = pv * (D['dec_o'] - 1) - (1 - pv); evu = (1 - pv) * (D['dec_u'] - 1) - pv
            over = evo >= evu
            D['is_over'] = over.astype(float)
            D['ev'] = np.where(over, evo, evu)
            D['won'] = np.where(over, D['y'] == 1, D['y'] == 0)
            D['profit'] = np.where(D['won'], np.where(over, D['dec_o'], D['dec_u']) - 1, -1.0)
            print("===== K standalone sim @ UD odds (90d walk-forward blend) =====")
            print(f"{'yr':>5} {'thr':>5} {'n':>5} {'hit':>6} {'ROI':>8} {'±2SE':>7} {'over%':>6}")
            for yr in (2025, 2026):
                d0_ = D[D['yr'] == yr]
                for thr in (0.02, 0.04, 0.06):
                    x = d0_[d0_['ev'] > thr]
                    if len(x) < 25:
                        print(f"{yr:>5} {thr:>5.2f} {len(x):>5,}  (too small)"); continue
                    se = x['profit'].std() / np.sqrt(len(x))
                    print(f"{yr:>5} {thr:>5.2f} {len(x):>5,} {x['won'].mean():>6.3f} "
                          f"{x['profit'].mean():>+8.4f} {2*se:>7.3f} {x['is_over'].mean():>6.2f}")


if __name__ == "__main__":
    main()
