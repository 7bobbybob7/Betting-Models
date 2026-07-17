"""
models/wnba/prop_model_gate.py — WNBA prop model v1: assembler + line-conditional gate.

Market bar is only ~0.558 AUC (flat hierarchy, no sharp end) — softest target measured.
Model: line-conditional XGB (the outs-architecture: line as input), minutes-first
features. Train on 2020-2024 outcomes ONLY -> both blend directions over the props era
(2025H2 fit -> 2026H1 test, and reverse) are honest.
Role if it passes: rank the Novig unders (structural trade) + standalone candidate.
"""
import sys, os, re, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.feature_sets import XGB_PARAMS
from models.mlb.hitter.backtest import american_to_decimal

MKTS = {393: 'points', 397: 'rebounds', 391: 'assists', 390: 'threes'}
STAT = {393: 'points', 397: 'rebounds', 391: 'assists', 390: 'fg3m'}
RUNGS = {393: np.arange(6.5, 26.6, 2.0), 397: np.arange(2.5, 12.6, 1.0),
         391: np.arange(1.5, 8.6, 1.0), 390: np.arange(0.5, 3.6, 1.0)}

def norm(s):
    if not s: return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', '', s.lower()).strip()

def build_dataset():
    d = query("""SELECT w.game_id, w.player_id, w.team_id, w.minutes, w.points,
        w.orb+w.drb rebounds, w.assists, w.fg3m, w.fga, w.fta, w.turnovers tov,
        g.game_date, g.home_team_id, g.away_team_id, COALESCE(p.full_name, p.name) AS full_name
        FROM wnba_player_game w JOIN games g ON w.game_id=g.game_id
        JOIN players p ON w.player_id=p.player_id
        WHERE g.status='final' AND w.minutes > 0""")
    d['game_date'] = pd.to_datetime(d['game_date'])
    d['home'] = (d['team_id'] == d['home_team_id']).astype(int)
    d['opp'] = np.where(d['home'] == 1, d['away_team_id'], d['home_team_id'])
    d = d.sort_values('game_date')
    d['usage'] = (d['fga'] + 0.44*d['fta'] + d['tov'])
    parts = []
    for pid, g in d.groupby('player_id', sort=False):
        g = g.sort_values('game_date').copy()
        for c in ['minutes', 'points', 'rebounds', 'assists', 'fg3m', 'usage']:
            g[f'{c}_5'] = g[c].shift(1).rolling(5, min_periods=3).mean()
            g[f'{c}_15'] = g[c].shift(1).rolling(15, min_periods=5).mean()
        for c in ['points', 'rebounds', 'assists', 'fg3m']:
            g[f'{c}_pm'] = (g[c].shift(1).rolling(10, min_periods=5).sum()
                            / g['minutes'].shift(1).rolling(10, min_periods=5).sum())
        g['rest'] = g['game_date'].diff().dt.days.clip(upper=10)
        parts.append(g)
    d = pd.concat(parts)
    # opponent allowed per game rolling 10 (closed-left)
    ta = d.groupby(['opp', 'game_date'])[['points', 'rebounds', 'assists', 'fg3m']].sum().reset_index()
    ch = []
    for t, g in ta.groupby('opp', sort=False):
        g = g.sort_values('game_date').copy()
        for c in ['points', 'rebounds', 'assists', 'fg3m']:
            g[f'opp_{c}_allowed'] = g[c].shift(1).rolling(10, min_periods=3).mean()
        ch.append(g[['opp', 'game_date'] + [f'opp_{c}_allowed' for c in ['points','rebounds','assists','fg3m']]])
    d = d.merge(pd.concat(ch), on=['opp', 'game_date'], how='left')
    return d

FEATS = (['minutes_5','minutes_15','usage_5','usage_15','rest','home'] +
         [f'{c}_5' for c in ['points','rebounds','assists','fg3m']] +
         [f'{c}_15' for c in ['points','rebounds','assists','fg3m']] +
         [f'{c}_pm' for c in ['points','rebounds','assists','fg3m']] +
         [f'opp_{c}_allowed' for c in ['points','rebounds','assists','fg3m']])

def main():
    print("building dataset...")
    d = build_dataset()
    d['fn'] = d['full_name'].map(norm)
    tr = d[d['game_date'] < '2025-01-01']
    te = d[d['game_date'] >= '2025-01-01'].copy()
    print(f"train {len(tr):,} (2020-24) | score {len(te):,} (2025-26) | {len(FEATS)} feats")

    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        med = trm[FEATS].median()
        aug = []
        for r in RUNGS[mid]:
            a = trm[FEATS].fillna(med).copy(); a['line'] = r
            a['y'] = (y > r).astype(int); aug.append(a)
        A = pd.concat(aug, ignore_index=True)
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(A[FEATS + ['line']].values, A['y'].values)

        P = query("""SELECT prop_date d, over_line ln, over_odds o, under_odds u, actual,
            LOWER(player_first_name||' '||player_last_name) nm
            FROM bettingpros_props WHERE book_id=60 AND market_id=%(m)s
            AND over_odds IS NOT NULL AND under_odds IS NOT NULL
            AND ABS(over_odds)<=2000 AND ABS(under_odds)<=2000
            AND is_scored AND actual IS NOT NULL""", params={'m': mid})
        P['fn'] = P['nm'].map(norm)
        P['d'] = pd.to_datetime(P['d'])
        te2 = te[['fn', 'game_date'] + FEATS].rename(columns={'game_date': 'd'})
        J = P.merge(te2, on=['fn', 'd'], how='inner')
        if len(J) < 300:
            print(f"\n[{mname}] name-match FAILED: {len(J)} of {len(P):,}")
            print("  bp sample:", P['fn'].dropna().unique()[:3])
            print("  db sample:", te['fn'].dropna().unique()[:3])
            continue
        X = J[FEATS].fillna(med).copy(); X['line'] = J['ln'].values
        J['p_model'] = m.predict_proba(X[FEATS + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = np.where(J['d'] < '2026-01-01', 0, 1)
        print(f"\n[{mname}] matched {len(J):,}/{len(P):,} | model AUC "
              f"{roc_auc_score(J['y'], J['p_model']):.4f} | market {roc_auc_score(J['y'], J['p_mkt']):.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, cols in [('A', ['p_mkt']), ('B', ['p_mkt', 'p_model'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit per{f_}->test per{t_} (n={len(t):,}): A={a['A']:.4f} B={a['B']:.4f} B-A={a['B']-a['A']:+.4f}")

if __name__ == "__main__":
    main()
