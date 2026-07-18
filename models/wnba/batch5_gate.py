"""
models/wnba/batch5_gate.py — batch 5: PLAY-BY-PLAY tendencies (rotation/foul/game-state).

Source: shufinskiy/nba_data wnba_nbastats_{yr} (events w/ subs, fouls, score margin).
All features are TRAILING tendencies (closed-left) — pregame-knowable:
    b5_foul36_10        fouls per 36 (trailing 10 games)
    b5_earlyfoul_10     share of games with 2+ fouls in H1 (foul-trouble propensity)
    b5_firstsub_10      avg elapsed seconds before first sub-out (coach's leash)
    b5_q4close_10       share of games appearing in Q4 with |margin|<=8 (closer role)
    b5_gtshare_10       share of scoring events in garbage time (|margin|>15) (mop-up role)
    b5_cleanrate_10     scoring events per game EXCLUDING garbage time (blowout-cleaned form)
Gate: control = v2, candidate = v2+B5; 4 markets, both period-directions vs Novig.
"""
import sys, os, io, tarfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd, requests, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.feature_sets import XGB_PARAMS
from models.mlb.hitter.backtest import american_to_decimal
from models.wnba.prop_model_v2_gate import build_dataset, V1, B2, MKTS, STAT, RUNGS, norm

CACHE = os.path.join(os.path.dirname(__file__), "cache/pbp")
YEARS = list(range(2020, 2027))
B5 = ['b5_foul36_10', 'b5_earlyfoul_10', 'b5_firstsub_10', 'b5_q4close_10',
      'b5_gtshare_10', 'b5_cleanrate_10']


def load_pbp():
    frames = []
    for y in YEARS:
        for tag in (f"wnba_nbastats_{y}", f"wnba_nbastats_po_{y}"):
            fp = os.path.join(CACHE, f"{tag}.csv")
            if not os.path.exists(fp):
                url = f"https://github.com/shufinskiy/nba_data/raw/main/datasets/{tag}.tar.xz"
                r = requests.get(url, timeout=120)
                if r.status_code != 200 or len(r.content) < 5000:
                    continue
                with tarfile.open(fileobj=io.BytesIO(r.content), mode='r:xz') as t:
                    t.extractall(CACHE)
                time.sleep(0.5)
            if os.path.exists(fp):
                frames.append(pd.read_csv(fp, usecols=[
                    'GAME_ID', 'EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING', 'SCOREMARGIN',
                    'PLAYER1_ID', 'PLAYER1_NAME']))
    d = pd.concat(frames, ignore_index=True)
    print(f"PBP events: {len(d):,} across {d.GAME_ID.nunique():,} games")
    return d


def game_dates():
    from nba_api.stats.endpoints import leaguegamelog
    maps = []
    for y in YEARS:
        try:
            gl = leaguegamelog.LeagueGameLog(league_id='10', season=str(y)).get_data_frames()[0]
            maps.append(gl[['GAME_ID', 'GAME_DATE']].drop_duplicates())
            time.sleep(0.8)
        except Exception as e:
            print(f"  gamelog {y} failed: {e}")
    m = pd.concat(maps).drop_duplicates('GAME_ID')
    m['GAME_DATE'] = pd.to_datetime(m['GAME_DATE'])
    m['GAME_ID'] = m['GAME_ID'].astype(int)
    return m


def build_b5():
    d = load_pbp()
    gd = game_dates()
    d = d.merge(gd, on='GAME_ID', how='inner')
    d['fn'] = d['PLAYER1_NAME'].map(norm)
    d = d[d['fn'] != '']
    mm = pd.to_numeric(d['PCTIMESTRING'].str.split(':').str[0], errors='coerce')
    d['elapsed_in_p'] = (10 - mm.clip(upper=10)) * 60
    d['margin'] = pd.to_numeric(d['SCOREMARGIN'].replace('TIE', 0), errors='coerce')
    d['margin'] = d.groupby('GAME_ID')['margin'].ffill().fillna(0).abs()

    per = []
    fouls = d[d.EVENTMSGTYPE == 6]
    f1 = fouls.groupby(['fn', 'GAME_ID', 'GAME_DATE']).agg(
        n_fouls=('EVENTMSGTYPE', 'size'),
        h1_fouls=('PERIOD', lambda s: (s <= 2).sum())).reset_index()
    subs_out = d[(d.EVENTMSGTYPE == 8)]
    so = subs_out.groupby(['fn', 'GAME_ID', 'GAME_DATE']).agg(
        first_sub=('elapsed_in_p', 'min')).reset_index()
    score = d[d.EVENTMSGTYPE.isin([1, 3])]
    sc = score.groupby(['fn', 'GAME_ID', 'GAME_DATE']).agg(
        n_sc=('EVENTMSGTYPE', 'size'),
        gt_sc=('margin', lambda s: (s > 15).sum())).reset_index()
    q4c = d[(d.PERIOD >= 4) & (d.margin <= 8)].groupby(
        ['fn', 'GAME_ID', 'GAME_DATE']).size().rename('q4close').reset_index()

    pg = f1.merge(so, on=['fn', 'GAME_ID', 'GAME_DATE'], how='outer') \
           .merge(sc, on=['fn', 'GAME_ID', 'GAME_DATE'], how='outer') \
           .merge(q4c, on=['fn', 'GAME_ID', 'GAME_DATE'], how='outer')
    for c in ['n_fouls', 'h1_fouls', 'n_sc', 'gt_sc', 'q4close']:
        pg[c] = pg[c].fillna(0)
    pg = pg.sort_values('GAME_DATE')
    ch = []
    for f_, g in pg.groupby('fn', sort=False):
        g = g.sort_values('GAME_DATE').copy()
        r = lambda s: s.shift(1).rolling(10, min_periods=4).mean()
        g['b5_foul36_10'] = r(g['n_fouls'])
        g['b5_earlyfoul_10'] = r((g['h1_fouls'] >= 2).astype(float))
        g['b5_firstsub_10'] = r(g['first_sub'])
        g['b5_q4close_10'] = r((g['q4close'] > 0).astype(float))
        g['b5_gtshare_10'] = r(g['gt_sc'] / g['n_sc'].clip(lower=1))
        g['b5_cleanrate_10'] = r(g['n_sc'] - g['gt_sc'])
        ch.append(g)
    P5 = pd.concat(ch)[['fn', 'GAME_DATE'] + B5].rename(columns={'GAME_DATE': 'pbp_date'})
    return P5


def main():
    print("building v2 dataset...")
    d = build_dataset()
    d['fn'] = d['full_name'].map(norm)
    print("building batch-5 PBP tendencies...")
    P5 = build_b5()
    # dual-date join (our dates vs stats dates may shift a day)
    for shift in (0, 1):
        p = P5.copy(); p['game_date'] = p['pbp_date'] + pd.Timedelta(days=shift)
        d = d.merge(p.drop(columns='pbp_date'), on=['fn', 'game_date'], how='left',
                    suffixes=('', f'_s{shift}'))
    for c in B5:
        if f'{c}_s1' in d:
            d[c] = d[c].fillna(d[f'{c}_s1']); d.drop(columns=[f'{c}_s1'], inplace=True)
    tr = d[d['game_date'] < '2025-01-01']; te = d[d['game_date'] >= '2025-01-01'].copy()
    print(f"b5 coverage: {te[B5].notna().mean().round(2).to_dict()}")

    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        preds = {}
        for tag, feats in [('ctl', V1 + B2), ('b5', V1 + B2 + B5)]:
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
        cols = list(dict.fromkeys(V1 + B2 + B5))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd2'})
        te2s = te2.copy(); te2s['d2'] = te2s['d2'] - pd.Timedelta(days=1)
        J = pd.concat([P.merge(te2, on=['fn', 'd2']), P.merge(te2s, on=['fn', 'd2'])]) \
              .drop_duplicates(['fn', 'd2', 'ln'])
        if len(J) < 300: print(f"[{mname}] thin: {len(J)}"); continue
        for tag in ('ctl', 'b5'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d2'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] n={len(J):,} | ctl={roc_auc_score(J['y'],J['p_ctl']):.4f} "
              f"b5={roc_auc_score(J['y'],J['p_b5']):.4f} mkt={roc_auc_score(J['y'],J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('B', ['p_mkt', 'p_b5'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"b5={a['B']:.4f}  b5-ctl={a['B']-a['C']:+.4f}  b5-A={a['B']-a['A']:+.4f}")
    print("\n(ACCEPT per market iff b5>ctl AND b5>A both directions.)")

if __name__ == "__main__":
    main()
