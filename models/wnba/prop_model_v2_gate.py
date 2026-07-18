"""
models/wnba/prop_model_v2_gate.py — WNBA model batch 2: the MINUTES/ROTATION decision
family (the "leash" analog that gave MLB's outs model its +0.032 jump), plus fixes:
rebounds-constant bug diagnostics and improved name matching.

Batch 2 features (declared before testing):
    started_last, starter_rate_10        rotation role (is_starter is in the box scores)
    min_trend (min_3 - min_15)           role changing right now
    min_vol_10                           minutes volatility (blowout/rotation risk)
    b2b                                  back-to-back (rest == 1 day)
    teammates_out_top5                   how many of team's top-5 (by prior-season+szn
                                         minutes) are ABSENT this game -> usage/minutes
                                         redistribution. Knowable pregame in reality
                                         (injury reports), same class as ump identity.
    net_rtg_10, opp_net_rtg_10           blowout risk (garbage-time minutes)
    opp_pace_10                          possessions proxy -> opportunity volume

Gate: control = v1 features, candidate = v1 + batch2. Both period-directions
(2025H2 <-> 2026H1) vs Novig. ACCEPT iff cand > ctl AND cand > market both directions.
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

V1 = (['minutes_5', 'minutes_15', 'usage_5', 'usage_15', 'rest', 'home'] +
      [f'{c}_5' for c in ['points', 'rebounds', 'assists', 'fg3m']] +
      [f'{c}_15' for c in ['points', 'rebounds', 'assists', 'fg3m']] +
      [f'{c}_pm' for c in ['points', 'rebounds', 'assists', 'fg3m']] +
      [f'opp_{c}_allowed' for c in ['points', 'rebounds', 'assists', 'fg3m']])
B2 = ['started_last', 'starter_rate_10', 'min_trend', 'min_vol_10', 'b2b',
      'teammates_out_top5', 'net_rtg_10', 'opp_net_rtg_10', 'opp_pace_10']


def norm(s):
    if not s: return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', s)
    return re.sub(r'[^a-z ]', '', s).strip()


def build_dataset():
    d = query("""SELECT w.game_id, w.player_id, w.team_id, w.is_starter, w.minutes,
        w.points, COALESCE(w.orb,0)+w.drb rebounds, w.assists, w.fg3m, w.fga, w.fta, w.orb,
        w.turnovers tov, g.game_date, g.home_team_id, g.away_team_id,
        g.home_score, g.away_score, COALESCE(p.full_name, p.name) AS full_name
        FROM wnba_player_game w JOIN games g ON w.game_id=g.game_id
        JOIN players p ON w.player_id=p.player_id
        WHERE g.status='final' AND w.minutes > 0""")
    d['game_date'] = pd.to_datetime(d['game_date'])
    d['home'] = (d['team_id'] == d['home_team_id']).astype(int)
    d['opp'] = np.where(d['home'] == 1, d['away_team_id'], d['home_team_id'])
    d['usage'] = d['fga'] + 0.44 * d['fta'] + d['tov']
    d['poss'] = d['fga'] + 0.44 * d['fta'] - d['orb'] + d['tov']
    d = d.sort_values('game_date')

    # per-player rolling (all closed-left via shift)
    parts = []
    for pid, g in d.groupby('player_id', sort=False):
        g = g.sort_values('game_date').copy()
        for c in ['minutes', 'points', 'rebounds', 'assists', 'fg3m', 'usage']:
            g[f'{c}_5'] = g[c].shift(1).rolling(5, min_periods=3).mean()
            g[f'{c}_15'] = g[c].shift(1).rolling(15, min_periods=5).mean()
        g['minutes_3'] = g['minutes'].shift(1).rolling(3, min_periods=2).mean()
        g['min_vol_10'] = g['minutes'].shift(1).rolling(10, min_periods=4).std()
        for c in ['points', 'rebounds', 'assists', 'fg3m']:
            g[f'{c}_pm'] = (g[c].shift(1).rolling(10, min_periods=5).sum()
                            / g['minutes'].shift(1).rolling(10, min_periods=5).sum())
        g['rest'] = g['game_date'].diff().dt.days.clip(upper=10)
        g['b2b'] = (g['rest'] == 1).astype(float)
        g['started_last'] = g['is_starter'].astype(float).shift(1)
        g['starter_rate_10'] = g['is_starter'].astype(float).shift(1).rolling(10, min_periods=3).mean()
        parts.append(g)
    d = pd.concat(parts)
    d['min_trend'] = d['minutes_3'] - d['minutes_15']

    # team-game aggregates for net rating, pace, opp-allowed
    tg = d.groupby(['team_id', 'game_id', 'game_date', 'opp']).agg(
        t_pts=('points', 'sum'), t_poss=('poss', 'sum')).reset_index()
    sc = query("SELECT game_id, home_team_id, away_team_id, home_score, away_score FROM games WHERE sport_id=3")
    tg = tg.merge(sc, on='game_id', how='left')
    tg['scored'] = np.where(tg['team_id'] == tg['home_team_id'], tg['home_score'], tg['away_score'])
    tg['allowed'] = np.where(tg['team_id'] == tg['home_team_id'], tg['away_score'], tg['home_score'])
    tg['net'] = tg['scored'] - tg['allowed']
    ch = []
    for t, g in tg.groupby('team_id', sort=False):
        g = g.sort_values('game_date').copy()
        g['net_rtg_10'] = g['net'].shift(1).rolling(10, min_periods=3).mean()
        g['pace_10'] = g['t_poss'].shift(1).rolling(10, min_periods=3).mean()
        ch.append(g[['team_id', 'game_date', 'net_rtg_10', 'pace_10']])
    tr_ = pd.concat(ch).drop_duplicates(['team_id', 'game_date'])
    d = d.merge(tr_, on=['team_id', 'game_date'], how='left')
    d = d.merge(tr_.rename(columns={'team_id': 'opp', 'net_rtg_10': 'opp_net_rtg_10',
                                    'pace_10': 'opp_pace_10'}), on=['opp', 'game_date'], how='left')

    # opp allowed per stat (rolling 10)
    ta = d.groupby(['opp', 'game_date'])[['points', 'rebounds', 'assists', 'fg3m']].sum().reset_index()
    ch = []
    for t, g in ta.groupby('opp', sort=False):
        g = g.sort_values('game_date').copy()
        for c in ['points', 'rebounds', 'assists', 'fg3m']:
            g[f'opp_{c}_allowed'] = g[c].shift(1).rolling(10, min_periods=3).mean()
        ch.append(g[['opp', 'game_date'] + [f'opp_{c}_allowed' for c in ['points', 'rebounds', 'assists', 'fg3m']]])
    d = d.merge(pd.concat(ch), on=['opp', 'game_date'], how='left')

    # teammates_out_top5: of the team's top-5 by trailing-20-game minutes, how many absent
    d['szn'] = d['game_date'].dt.year
    top = d.copy()
    top['min20'] = top.groupby('player_id')['minutes'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    ranks = top.groupby(['team_id', 'game_id'])['min20'].rank(ascending=False)
    # roster present per game; absence of top players = compare to team's prior top-5 set
    # approx: for each (team, game), count how many of team's top-5 min20 across the
    # TRAILING 30 days are not in today's box
    d['teammates_out_top5'] = np.nan
    for (t, szn), g in top.groupby(['team_id', 'szn'], sort=False):
        dates = sorted(g['game_date'].unique())
        for dt_ in dates:
            hist = g[(g['game_date'] < dt_) & (g['game_date'] >= dt_ - pd.Timedelta(days=30))]
            if hist.empty: continue
            top5 = hist.groupby('player_id')['minutes'].mean().nlargest(5).index
            today = set(g.loc[g['game_date'] == dt_, 'player_id'])
            n_out = sum(1 for p in top5 if p not in today)
            d.loc[(d['team_id'] == t) & (d['game_date'] == dt_), 'teammates_out_top5'] = n_out
    return d


def main():
    print("building dataset (v2)...")
    d = build_dataset()
    d['fn'] = d['full_name'].map(norm)
    tr = d[d['game_date'] < '2025-01-01']
    te = d[d['game_date'] >= '2025-01-01'].copy()
    print(f"train {len(tr):,} | score {len(te):,} | b2 coverage: "
          f"{tr[B2].notna().mean().round(2).to_dict()}")

    for mid, mname in MKTS.items():
        stat = STAT[mid]
        trm = tr[tr[stat].notna()]
        y = trm[stat].astype(float).values
        results = {}
        preds = {}
        for tag, feats in [('ctl', V1), ('v2', V1 + B2)]:
            med = trm[feats].median()
            aug = []
            for r in RUNGS[mid]:
                a = trm[feats].fillna(med).copy(); a['line'] = r
                a['y'] = (y > r).astype(int); aug.append(a)
            A = pd.concat(aug, ignore_index=True)
            m = xgb.XGBClassifier(**XGB_PARAMS)
            m.fit(A[feats + ['line']].values, A['y'].values)
            preds[tag] = (m, med, feats)

        P = query("""SELECT prop_date d, over_line ln, over_odds o, under_odds u, actual,
            LOWER(player_first_name||' '||player_last_name) nm
            FROM bettingpros_props WHERE book_id=60 AND market_id=%(m)s
            AND over_odds IS NOT NULL AND under_odds IS NOT NULL
            AND ABS(over_odds)<=2000 AND ABS(under_odds)<=2000
            AND is_scored AND actual IS NOT NULL""", params={'m': mid})
        P['fn'] = P['nm'].map(norm)
        P['d'] = pd.to_datetime(P['d'])
        cols = list(dict.fromkeys(V1 + B2))
        te2 = te[['fn', 'game_date'] + cols].rename(columns={'game_date': 'd'})
        te2s = te2.copy(); te2s['d'] = te2s['d'] - pd.Timedelta(days=1)   # UTC-shifted evening games
        J = pd.concat([P.merge(te2, on=['fn', 'd'], how='inner'),
                       P.merge(te2s, on=['fn', 'd'], how='inner')]
                      ).drop_duplicates(['fn', 'd', 'ln'])
        if len(J) < 300:
            print(f"\n[{mname}] thin match: {len(J)}/{len(P):,}"); continue
        for tag in ('ctl', 'v2'):
            m, med, feats = preds[tag]
            X = J[feats].fillna(med).copy(); X['line'] = J['ln'].values
            J[f'p_{tag}'] = m.predict_proba(X[feats + ['line']].values)[:, 1]
        io_ = 1/J['o'].apply(american_to_decimal); iu_ = 1/J['u'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_)
        J['y'] = (J['actual'].astype(float) > J['ln']).astype(int)
        J['per'] = (J['d'] >= '2026-01-01').astype(int)
        print(f"\n[{mname}] matched {len(J):,}/{len(P):,} | "
              f"AUC ctl={roc_auc_score(J['y'], J['p_ctl']):.4f} "
              f"v2={roc_auc_score(J['y'], J['p_v2']):.4f} "
              f"mkt={roc_auc_score(J['y'], J['p_mkt']):.4f} "
              f"| p_v2 std={J['p_v2'].std():.4f}")
        for f_, t_ in [(0, 1), (1, 0)]:
            f = J[J['per'] == f_]; t = J[J['per'] == t_]
            if len(f) < 200 or len(t) < 200: continue
            a = {}
            for k, cols in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('V', ['p_mkt', 'p_v2'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit per{f_}->per{t_} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"v2={a['V']:.4f}  v2-ctl={a['V']-a['C']:+.4f}  v2-A={a['V']-a['A']:+.4f}")
    print("\n(ACCEPT iff v2>ctl AND v2>A both directions on a market.)")


if __name__ == "__main__":
    main()
