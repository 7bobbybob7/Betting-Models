"""
models/mlb/pitcher/outs_v2_gate.py — OUTS model iteration 2: decision-side features +
line-conditional architecture.

Insight: outs recorded is half performance, half MANAGER DECISION (leash). v1 had only
performance features -> 0.52 vs market 0.58. v2 adds the decision family:
    team_hook_outs_60d, own_pen_ip_2d, own_pen_relievers_1d (gassed pen -> longer leash),
    pitches_per_out_5, max_pitches_5 (revealed budget), rest_days, cum_ip_szn, opp_ppa_30d
And fixes the model class: managers pull at inning boundaries (spikes at 15/18/21 outs),
so instead of Poisson-SF we train a LINE-CONDITIONAL classifier: every start x rungs
13.5..21.5 with the line as a feature -> P(outs > line) learns the hook cliffs directly.

Gate: A=market, B=market+v1(poisson), C=market+v2(line-conditional+decision feats).
ACCEPT iff C>B and C>A both directions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.hitter.backtest import american_to_decimal
from models.mlb.feature_sets import XGB_PARAMS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
OUTS_MKT, NOVIG = 405, 60
V1_FEATS = ['ip_per_start_5', 'ip_per_start_szn', 'pitches_per_start_5',
            'k_per_start_5', 'k_per_start_szn', 'sc_swstr_rate_5', 'sc_fb_velo_5',
            'opp_k_rate_15', 'opp_k_rate_30']
DECISION = ['team_hook_outs_60d', 'own_pen_ip_2d', 'own_pen_rel_1d', 'pitches_per_out_5',
            'max_pitches_5', 'rest_days', 'cum_ip_szn', 'opp_ppa_30d']
RUNGS = [13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def ip_to_outs(ip):
    f = np.floor(ip); return (f * 3 + np.round((ip - f) * 10)).astype(int)


def build_decision(df):
    """Decision-side features per (game_id, pitcher_id). df: game_id, pitcher_id,
    game_date, team_id, opp_team_id, actual_ip, pitches, season."""
    df = df.sort_values('game_date').copy()
    df['outs'] = ip_to_outs(df['actual_ip'].values)
    if 'pitches' not in df.columns:
        px = query("""SELECT game_id, player_id AS pitcher_id, pitches
                      FROM mlb_pitching_game WHERE is_starter=true""")
        df = df.merge(px, on=['game_id', 'pitcher_id'], how='left')

    # team hook: rolling mean starter outs per team, closed-left 60d
    th = df[['team_id', 'game_date', 'outs']].copy()
    ch = []
    for t, g in th.groupby('team_id', sort=False):
        r = g.set_index('game_date')['outs'].rolling('60D', closed='left').mean()
        ch.append(pd.DataFrame({'team_id': t, 'game_date': g['game_date'].values,
                                'team_hook_outs_60d': r.values}))
    df = df.merge(pd.concat(ch).drop_duplicates(['team_id', 'game_date']),
                  on=['team_id', 'game_date'], how='left')

    # own bullpen usage prior 1-2 days
    bp = query("""SELECT pg.team_id, g.game_date, COUNT(*) rel, SUM(pg.ip) bip
        FROM mlb_pitching_game pg JOIN games g ON pg.game_id=g.game_id
        WHERE g.sport_id=2 AND g.status='final' AND pg.is_starter=false
        GROUP BY 1,2""")
    bp['game_date'] = pd.to_datetime(bp['game_date']).dt.date
    lut = {(int(r.team_id), r.game_date): (int(r.rel), float(r.bip or 0)) for r in bp.itertuples()}
    from datetime import timedelta
    dts = pd.to_datetime(df['game_date']).dt.date
    df['own_pen_rel_1d'] = [lut.get((int(t), d - timedelta(days=1)), (0, 0))[0]
                            for t, d in zip(df['team_id'], dts)]
    df['own_pen_ip_2d'] = [lut.get((int(t), d - timedelta(days=1)), (0, 0))[1] +
                           lut.get((int(t), d - timedelta(days=2)), (0, 0))[1]
                           for t, d in zip(df['team_id'], dts)]

    # pitcher-level: efficiency, budget, rest, cumulative load (prior-only)
    parts = []
    for pid, g in df.groupby('pitcher_id', sort=False):
        g = g.sort_values('game_date').copy()
        po = g['pitches'] / g['outs'].clip(lower=1)
        g['pitches_per_out_5'] = po.shift(1).rolling(5, min_periods=3).mean()
        g['max_pitches_5'] = g['pitches'].shift(1).rolling(5, min_periods=3).max()
        g['rest_days'] = pd.to_datetime(g['game_date']).diff().dt.days.clip(upper=15)
        g['cum_ip_szn'] = g.groupby('season')['actual_ip'].transform(
            lambda s: s.shift(1).fillna(0).cumsum())
        parts.append(g)
    df = pd.concat(parts)

    # opposing team pitches-per-PA, rolling 30d closed-left (patience burns pitch counts)
    ppa = query("""SELECT p.game_id, g.game_date,
          CASE WHEN p.top_bottom='top' THEN g.away_team_id ELSE g.home_team_id END bat_team,
          COUNT(*) pitches, COUNT(DISTINCT p.at_bat_number||'-'||p.top_bottom||'-'||p.inning) pas
        FROM mlb_pitches p JOIN games g ON p.game_id=g.game_id
        WHERE g.sport_id=2 GROUP BY 1,2,3""")
    ppa['game_date'] = pd.to_datetime(ppa['game_date'])
    ch = []
    for t, g in ppa.groupby('bat_team', sort=False):
        g = g.sort_values('game_date')
        r = g.set_index('game_date')[['pitches', 'pas']].rolling('30D', closed='left').sum()
        ch.append(pd.DataFrame({'opp_team_id': t, 'game_date': g['game_date'].values,
                                'opp_ppa_30d': (r['pitches'] / r['pas'].clip(lower=1)).values}))
    df = df.merge(pd.concat(ch).drop_duplicates(['opp_team_id', 'game_date']),
                  on=['opp_team_id', 'game_date'], how='left')
    return df


def main():
    _load("models.mlb.statcast_features", "archive/models/mlb/statcast_features.py")
    km = _load("k_model_arch", "archive/models/mlb/k_model.py")
    print("building dataset...")
    df = km.build_k_dataset()
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df[df['actual_ip'].notna() & (df['actual_ip'] > 0)]
    gm = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id=2")
    pg = query("SELECT game_id, player_id AS pitcher_id, team_id FROM mlb_pitching_game WHERE is_starter=true")
    df = df.merge(pg, on=['game_id', 'pitcher_id'], how='left').merge(gm, on='game_id', how='left')
    df['opp_team_id'] = np.where(df['team_id'] == df['home_team_id'], df['away_team_id'], df['home_team_id'])
    print("building decision features...")
    df = build_decision(df)
    dec = [c for c in DECISION if c in df.columns]
    v1 = [c for c in V1_FEATS if c in df.columns]
    print(f"decision coverage 2025+: {df[df.season>=2025][dec].notna().mean().round(2).to_dict()}")

    tr = df[df['season'] <= 2024]; te = df[df['season'] >= 2025].copy()
    med1 = tr[v1].median(); med2 = tr[v1 + dec].median()

    # v1: poisson on outs
    m1 = xgb.XGBRegressor(objective='count:poisson', **XGB_PARAMS)
    m1.fit(tr[v1].fillna(med1).values, tr['outs'].values)
    te['lam1'] = m1.predict(te[v1].fillna(med1).values)

    # v2: line-conditional classifier (start x rung augmentation)
    aug = []
    for r in RUNGS:
        a = tr[v1 + dec].fillna(med2).copy()
        a['line'] = r; a['y'] = (tr['outs'].values > r).astype(int)
        aug.append(a)
    A = pd.concat(aug, ignore_index=True)
    print(f"line-conditional training rows: {len(A):,}")
    m2 = xgb.XGBClassifier(**XGB_PARAMS)
    m2.fit(A[v1 + dec + ['line']].values, A['y'].values)

    pl = query("SELECT player_id, LOWER(full_name) fn FROM players")
    P = query("""SELECT prop_date AS game_date, over_line, over_odds, under_odds, actual, is_scored,
                 LOWER(player_first_name||' '||player_last_name) fn
                 FROM bettingpros_props WHERE market_id=%(m)s AND book_id=%(b)s
                 AND over_odds IS NOT NULL AND under_odds IS NOT NULL""",
              params={'m': OUTS_MKT, 'b': NOVIG})
    P['game_date'] = pd.to_datetime(P['game_date'])
    P = P.merge(pl, on='fn', how='inner').rename(columns={'player_id': 'pitcher_id'})
    P = P.merge(te[['game_date', 'pitcher_id', 'lam1'] + v1 + dec], on=['game_date', 'pitcher_id'], how='inner')
    X2 = P[v1 + dec].fillna(med2).copy(); X2['line'] = P['over_line'].values
    P['p_v2'] = m2.predict_proba(X2[v1 + dec + ['line']].values)[:, 1]
    P['p_v1'] = poisson.sf(np.floor(P['over_line']), P['lam1'])
    io_ = 1 / P['over_odds'].apply(american_to_decimal); iu_ = 1 / P['under_odds'].apply(american_to_decimal)
    P['p_mkt'] = io_ / (io_ + iu_)
    P['y'] = (P['actual'].astype(float) > P['over_line']).astype(int)
    P['yr'] = P['game_date'].dt.year

    print(f"\nmatched props: 2025={len(P[P.yr==2025]):,} 2026={len(P[P.yr==2026]):,}")
    print(f"standalone AUC: v1={roc_auc_score(P['y'],P['p_v1']):.4f} "
          f"v2={roc_auc_score(P['y'],P['p_v2']):.4f} market={roc_auc_score(P['y'],P['p_mkt']):.4f}")
    print("\n===== OUTS v2 GATE (both directions) =====")
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = P[P['yr'] == fy], P[P['yr'] == ty]
        if len(f) < 200 or len(t) < 200: continue
        a = {}
        for k, cols in [('A', ['p_mkt']), ('B', ['p_mkt', 'p_v1']), ('C', ['p_mkt', 'p_v2'])]:
            lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
            a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        print(f"  fit {fy}->test {ty} (n={len(t):,}): A={a['A']:.4f} B={a['B']:.4f} C={a['C']:.4f}  "
              f"C-B={a['C']-a['B']:+.4f}  C-A={a['C']-a['A']:+.4f}")
    print("\n(ACCEPT iff C>B and C>A both directions.)")


if __name__ == "__main__":
    main()
