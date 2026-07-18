"""
models/wnba/batch4_gate.py — batch 4: OPPONENT-LINEUP DEFENSIVE PROFILE (matchup family).

The fingerprint's unique asset: DEFENSIVE net-points by play type per player. Aggregate
the opposing team's trailing usual lineup (top-6 by prior 30d minutes, leak-safe) using
PRIOR-season fingerprints -> opponent defense profile + explicit size-mismatch diffs.
    opp_front_height   avg height of opp's 3 tallest usual starters
    opp_reb_d_pp, opp_rim_d_pp, opp_corner_d_pp, opp_3pt_d_pp  (lineup d-netpts / poss)
    ht_vs_front        my height - opp_front_height (rebound mismatch prior)
Gate: control = v2, candidate = v2+B4; rebounds/points/threes/assists, both directions.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.feature_sets import XGB_PARAMS
from models.mlb.hitter.backtest import american_to_decimal
from models.wnba.prop_model_v2_gate import build_dataset, V1, B2, MKTS, STAT, RUNGS, norm

B4 = ['opp_front_height', 'opp_reb_d_pp', 'opp_rim_d_pp', 'opp_corner_d_pp',
      'opp_3pt_d_pp', 'ht_vs_front', 'my_height']


def build_b4(d):
    fp = query("SELECT wnba_player_id, season, display_name, height_in, tposs, data FROM wnba_fingerprint")
    fp['fn'] = fp['display_name'].map(norm)
    pp = fp['tposs'].astype(float).clip(lower=1)
    def jx(s, k):
        try: return json.loads(s).get(k) or 0.0
        except Exception: return 0.0
    for col, key in [('reb_d', 'rebound_dNetPts'), ('rim_d', 'rim_dNetPts'),
                     ('corner_d', 'corner_dNetPts'), ('p3_d', '3pt_dNetPts')]:
        fp[col] = [jx(s, key) / p for s, p in zip(fp['data'], pp)]
    fpp = fp[['fn', 'season', 'height_in', 'reb_d', 'rim_d', 'corner_d', 'p3_d']]

    d['fn'] = d['full_name'].map(norm)
    d['prior_szn'] = d['game_date'].dt.year - 1
    d = d.merge(fpp.rename(columns={'season': 'prior_szn', 'height_in': 'my_height',
                                    'reb_d': '_rd', 'rim_d': '_rim', 'corner_d': '_c',
                                    'p3_d': '_p3'}), on=['fn', 'prior_szn'], how='left')
    # usual lineup: per (team, date) top-6 by trailing minutes (minutes_15 already built)
    lineup = d[d['minutes_15'].notna()].copy()
    lineup['rk'] = lineup.groupby(['team_id', 'game_date'])['minutes_15'].rank(ascending=False)
    top6 = lineup[lineup['rk'] <= 6]
    prof = top6.groupby(['team_id', 'game_date']).agg(
        opp_reb_d_pp=('_rd', 'mean'), opp_rim_d_pp=('_rim', 'mean'),
        opp_corner_d_pp=('_c', 'mean'), opp_3pt_d_pp=('_p3', 'mean'),
        opp_front_height=('my_height', lambda s: s.nlargest(3).mean())).reset_index()
    d = d.merge(prof.rename(columns={'team_id': 'opp'}), on=['opp', 'game_date'], how='left')
    d['ht_vs_front'] = d['my_height'] - d['opp_front_height']
    return d


def main():
    print("building v2 + batch4 (opponent defense profile)...")
    d = build_dataset()
    d = build_b4(d)
    tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
    print(f"b4 coverage: {te[B4].notna().mean().round(2).to_dict()}")
    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        preds = {}
        for tag, feats in [('ctl', V1 + B2), ('b4', V1 + B2 + B4)]:
            med = trm[feats].median()
            aug = []
            for r in RUNGS[mid]:
                a = trm[feats].fillna(med).copy(); a['line'] = r
                a['y'] = (y > r).astype(int); aug.append(a)
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
        cols = list(dict.fromkeys(V1 + B2 + B4))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{mname}] thin: {len(J)}"); continue
        for tag in ('ctl', 'b4'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] n={len(J):,} | ctl={roc_auc_score(J['y'],J['p_ctl']):.4f} "
              f"b4={roc_auc_score(J['y'],J['p_b4']):.4f} mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('B', ['p_mkt', 'p_b4'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"b4={a['B']:.4f}  b4-ctl={a['B']-a['C']:+.4f}  b4-A={a['B']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff b4>ctl AND b4>A both directions.)")

if __name__ == "__main__":
    main()
