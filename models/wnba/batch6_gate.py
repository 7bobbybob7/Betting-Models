"""
models/wnba/batch6_gate.py — batch 6: player-vs-TEAM head-to-head interaction.

Why this isn't MLB's rejected embeddings: MLB pairs had 2-6 PAs (chemistry = noise);
WNBA has 13 teams -> 10-15 career meetings per (player, opponent). Realized H2H residual
(stat vs own rolling form, against THIS opponent), EB-shrunk (k=8), strictly trailing:
    h2h_resid  = sum(stat - form) / (n_meetings + 8)   [closed-left]
    h2h_n      = prior meetings count
Gate: control = v2, candidate = v2 + [h2h_resid, h2h_n] per market, both directions.
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

K_SHRINK = 8.0

def add_h2h(d, stat):
    form = d[f'{stat}_15']
    d = d.copy()
    d['_resid'] = d[stat].astype(float) - form
    d = d.sort_values('game_date')
    ch = []
    for (fn, opp), g in d.groupby(['fn', 'opp'], sort=False):
        g = g.sort_values('game_date').copy()
        ok = g['_resid'].notna().astype(float)
        g['h2h_n'] = ok.shift(1).fillna(0).cumsum()
        g['h2h_resid'] = (g['_resid'].fillna(0) * ok).shift(1).fillna(0).cumsum() / (g['h2h_n'] + K_SHRINK)
        ch.append(g)
    return pd.concat(ch)

def main():
    print("building v2 dataset...")
    base = build_dataset()
    base['fn'] = base['full_name'].map(norm)
    B6 = ['h2h_resid', 'h2h_n']
    for mid, mname in MKTS.items():
        stat = STAT[mid]
        d = add_h2h(base[base[stat].notna()], stat)
        tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
        y = tr[stat].astype(float).values
        preds = {}
        for tag, feats in [('ctl', V1 + B2), ('h2h', V1 + B2 + B6)]:
            med = tr[feats].median()
            aug = []
            for r_ in RUNGS[mid]:
                a = tr[feats].fillna(med).copy(); a['line'] = r_
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
        cols = list(dict.fromkeys(V1 + B2 + B6))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{mname}] thin: {len(J)}"); continue
        for tag in ('ctl', 'h2h'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] n={len(J):,} | ctl={roc_auc_score(J['y'],J['p_ctl']):.4f} "
              f"h2h={roc_auc_score(J['y'],J['p_h2h']):.4f} mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('H', ['p_mkt', 'p_h2h'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"h2h={a['H']:.4f}  h2h-ctl={a['H']-a['C']:+.4f}  h2h-A={a['H']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff h2h>ctl AND h2h>A both directions.)")

if __name__ == "__main__":
    main()
