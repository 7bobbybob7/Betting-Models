"""
models/wnba/batch45_gate.py — combined B4 (opponent-matchup) + B5 (PBP tendencies).

Rationale: both families independently lifted REBOUNDS vs control in both directions
(b4: +0.0008/+0.0039, b5: +0.0002/+0.0077) but each alone stayed a hair under the
market. Combining two declared, individually-consistent families is the natural next
step. Gate: control = v2, candidate = v2+B4+B5, all four markets, both directions.
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
from models.wnba.prop_model_v2_gate import build_dataset, V1, B2, MKTS, STAT, RUNGS, norm
from models.wnba.batch4_gate import build_b4, B4
from models.wnba.batch5_gate import build_b5, B5

def main():
    print("building v2 + B4 + B5...")
    d = build_dataset()
    d = build_b4(d)                      # adds fn column + B4
    P5 = build_b5()
    for shift in (0, 1):
        p = P5.copy(); p['game_date'] = p['pbp_date'] + pd.Timedelta(days=shift)
        d = d.merge(p.drop(columns='pbp_date'), on=['fn', 'game_date'], how='left',
                    suffixes=('', f'_s{shift}'))
    for c in B5:
        if f'{c}_s1' in d:
            d[c] = d[c].fillna(d[f'{c}_s1']); d.drop(columns=[f'{c}_s1'], inplace=True)
    tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
    NEW = B4 + B5
    print(f"coverage: {te[NEW].notna().mean().round(2).min():.2f} min across {len(NEW)} feats")

    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        preds = {}
        for tag, feats in [('ctl', V1 + B2), ('cmb', V1 + B2 + NEW)]:
            med = trm[feats].median()
            aug = []
            for r_ in RUNGS[mid]:
                a = trm[feats].fillna(med).copy(); a['line'] = r_
                a['y'] = (y > r_).astype(int); aug.append(a)
            A = pd.concat(aug, ignore_index=True)
            m = xgb.XGBClassifier(**XGB_PARAMS)
            m.fit(A[feats + ['line']].values, A['y'].values)
            preds[tag] = (m, med, feats)
        P = query("""SELECT prop_date d2, over_line ln, over_odds o, under_odds u, actual,
            LOWER(player_first_name||' '||player_last_name) nm
            FROM bettingpros_props WHERE book_id=60 AND market_id=%(m)s
            AND over_odds IS NOT NULL AND under_odds IS NOT NULL
            AND ABS(over_odds)<=2000 AND ABS(under_odds)<=2000
            AND is_scored AND actual IS NOT NULL""", params={'m': mid})
        P['fn'] = P['nm'].map(norm); P['d2'] = pd.to_datetime(P['d2'])
        cols = list(dict.fromkeys(V1 + B2 + NEW))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{mname}] thin: {len(J)}"); continue
        for tag in ('ctl', 'cmb'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] n={len(J):,} | ctl={roc_auc_score(J['y'],J['p_ctl']):.4f} "
              f"cmb={roc_auc_score(J['y'],J['p_cmb']):.4f} mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('M', ['p_mkt', 'p_cmb'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"cmb={a['M']:.4f}  cmb-ctl={a['M']-a['C']:+.4f}  cmb-A={a['M']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff cmb>ctl AND cmb>A both directions.)")

if __name__ == "__main__":
    main()
