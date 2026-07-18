"""
models/wnba/batch3_gate.py — batch 3: shot-profile + prior-season fingerprint features.

    Shots (rolling 10, closed-left):  fg3a_share, corner3_share, rim_share, avg_dist
    Fingerprint (PRIOR season only — season-aggregate leak rule): height_in,
    avg_position, offensive_usage, assisted_rate, rebound/putback/3pt/corner
    net-points per possession.

Targets: the markets v2 didn't crack (rebounds, threes) + points/assists for completeness.
Gate: control = v2, candidate = v2+B3, both period-directions vs Novig.
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

B3 = ['fg3a_share_10', 'corner3_share_10', 'rim_share_10', 'avg_dist_10',
      'fp_height_in', 'fp_avg_position', 'fp_usage', 'fp_assisted_rate',
      'fp_reb_pts_pp', 'fp_putback_pp', 'fp_3pt_pp', 'fp_corner_pp']


def build_b3(d):
    # id -> normalized name map from fingerprint
    fp = query("SELECT wnba_player_id, season, display_name, height_in, avg_position, offensive_usage, assisted_rate, tposs, data FROM wnba_fingerprint")
    fp['fn'] = fp['display_name'].map(norm)
    # shot-profile rolling per (player, date)
    sh = query("SELECT wnba_player_id, game_date, zone, area, dist FROM wnba_shots")
    sh = sh.merge(fp[['wnba_player_id', 'fn']].drop_duplicates('wnba_player_id'), on='wnba_player_id')
    sh['game_date'] = pd.to_datetime(sh['game_date'])
    sh['is3'] = sh['zone'].str.contains('3', na=False).astype(float)
    sh['isc3'] = ((sh['is3'] > 0) & sh['area'].str.contains('Corner|Side', case=False, na=False)).astype(float)
    sh['rim'] = (sh['dist'] <= 4).astype(float)
    g = sh.groupby(['fn', 'game_date']).agg(n=('dist', 'size'), s3=('is3', 'sum'),
                                            c3=('isc3', 'sum'), rm=('rim', 'sum'),
                                            dsum=('dist', 'sum')).reset_index().sort_values('game_date')
    ch = []
    for f_, gg in g.groupby('fn', sort=False):
        gg = gg.sort_values('game_date').copy()
        for c in ['n', 's3', 'c3', 'rm', 'dsum']:
            gg[f'r_{c}'] = gg[c].shift(1).rolling(10, min_periods=4).sum()
        ch.append(gg)
    G = pd.concat(ch)
    G['fg3a_share_10'] = G['r_s3'] / G['r_n']
    G['corner3_share_10'] = G['r_c3'] / G['r_n'].clip(lower=1)
    G['rim_share_10'] = G['r_rm'] / G['r_n']
    G['avg_dist_10'] = G['r_dsum'] / G['r_n']
    sf = ['fg3a_share_10', 'corner3_share_10', 'rim_share_10', 'avg_dist_10']
    # dual-date join to our spine (their dates may differ by a day from ours)
    d['fn'] = d['full_name'].map(norm)
    for shift in (0, 1):
        gg = G[['fn', 'game_date'] + sf].copy()
        gg['game_date'] = gg['game_date'] + pd.Timedelta(days=shift)
        d = d.merge(gg, on=['fn', 'game_date'], how='left', suffixes=('', f'_s{shift}'))
    for c in sf:
        if f'{c}_s1' in d:
            d[c] = d[c].fillna(d[f'{c}_s1']); d.drop(columns=[f'{c}_s1'], inplace=True)
    # prior-season fingerprint
    fp['pp'] = fp['tposs'].astype(float).clip(lower=1)
    def jx(row, key):
        try: return json.loads(row) .get(key)
        except Exception: return None
    fp['fp_reb_pts_pp'] = [ (jx(r, 'rebound_oNetPts') or 0)/p for r, p in zip(fp['data'], fp['pp'])]
    fp['fp_putback_pp'] = [ (jx(r, 'putback_oNetPts') or 0)/p for r, p in zip(fp['data'], fp['pp'])]
    fp['fp_3pt_pp'] = [ (jx(r, '3pt_oNetPts') or 0)/p for r, p in zip(fp['data'], fp['pp'])]
    fp['fp_corner_pp'] = [ (jx(r, 'corner_oNetPts') or 0)/p for r, p in zip(fp['data'], fp['pp'])]
    fpp = fp.rename(columns={'height_in': 'fp_height_in', 'avg_position': 'fp_avg_position',
                             'offensive_usage': 'fp_usage', 'assisted_rate': 'fp_assisted_rate'})
    fpp = fpp[['fn', 'season', 'fp_height_in', 'fp_avg_position', 'fp_usage', 'fp_assisted_rate',
               'fp_reb_pts_pp', 'fp_putback_pp', 'fp_3pt_pp', 'fp_corner_pp']]
    d['prior_szn'] = d['game_date'].dt.year - 1
    d = d.merge(fpp, left_on=['fn', 'prior_szn'], right_on=['fn', 'season'], how='left')
    return d


def main():
    print("building v2 dataset + batch3...")
    d = build_dataset()
    d = build_b3(d)
    tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
    print(f"b3 coverage (score rows): {te[B3].notna().mean().round(2).to_dict()}")
    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        preds = {}
        for tag, feats in [('ctl', V1 + B2), ('b3', V1 + B2 + B3)]:
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
        cols = list(dict.fromkeys(V1 + B2 + B3))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{mname}] thin: {len(J)}"); continue
        for tag in ('ctl', 'b3'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] n={len(J):,} | ctl={roc_auc_score(J['y'],J['p_ctl']):.4f} "
              f"b3={roc_auc_score(J['y'],J['p_b3']):.4f} mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('B', ['p_mkt', 'p_b3'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"b3={a['B']:.4f}  b3-ctl={a['B']-a['C']:+.4f}  b3-A={a['B']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff b3>ctl AND b3>A both directions.)")

if __name__ == "__main__":
    main()
