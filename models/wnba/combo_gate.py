"""
models/wnba/combo_gate.py — untested COMBO markets: PRA (396) and R+A (398).
Same v2 feature stack + line-conditional; labels = component sums. Books compound
component errors in combos -> plausibly softer than the parts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.feature_sets import XGB_PARAMS
from models.mlb.hitter.backtest import american_to_decimal
from models.wnba.prop_model_v2_gate import build_dataset, V1, B2, norm

CMKTS = {396: ('pra', np.arange(10.5, 36.6, 2.0)), 398: ('ra', np.arange(3.5, 17.6, 1.0)),
         395: ('pr', np.arange(8.5, 32.6, 2.0)), 394: ('pa', np.arange(8.5, 30.6, 2.0))}

def main():
    d = build_dataset()
    d['pra'] = d['points'] + d['rebounds'] + d['assists']
    d['ra'] = d['rebounds'] + d['assists']
    d['pr'] = d['points'] + d['rebounds']
    d['pa'] = d['points'] + d['assists']
    d['fn'] = d['full_name'].map(norm)
    tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
    feats = V1 + B2
    for mid, (stat, rungs) in CMKTS.items():
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        med = trm[feats].median()
        aug = []
        for r in rungs:
            a = trm[feats].fillna(med).copy(); a['line'] = r
            a['y'] = (y > r).astype(int); aug.append(a)
        A = pd.concat(aug, ignore_index=True)
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(A[feats + ['line']].values, A['y'].values)
        P = query("""SELECT prop_date d2, over_line ln, over_odds o, under_odds u, actual,
            LOWER(player_first_name||' '||player_last_name) nm
            FROM bettingpros_props WHERE book_id=60 AND market_id=%(m)s
            AND over_odds IS NOT NULL AND under_odds IS NOT NULL
            AND ABS(over_odds)<=2000 AND ABS(under_odds)<=2000
            AND is_scored AND actual IS NOT NULL""", params={'m': mid})
        P['fn'] = P['nm'].map(norm); P['d2'] = pd.to_datetime(P['d2'])
        te2 = te[['fn', 'game_date'] + feats].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{stat} m{mid}] thin: {len(J)}"); continue
        X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
        J['p_model'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{stat} m{mid}] n={len(J):,} | model={roc_auc_score(J['y'],J['p_model']):.4f} "
              f"mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 150 or len(t) < 150: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('B', ['p_mkt', 'p_model'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} B={a['B']:.4f} B-A={a['B']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff B>A both directions. Also validates 394/395 stat identity via AUC sanity.)")

if __name__ == "__main__":
    main()
