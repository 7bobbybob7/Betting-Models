"""
models/mlb/pitcher/outs_v3_gate.py — OUTS batch 2 (declared 2026-07-11):
    gs_run_diff_own_30d / gs_run_diff_opp_30d  — game-script/blowout risk
    sched_days_to_off                          — bullpen-rest planning changes the hook
    ump_k_rate_365d                            — big zone -> faster outs -> deeper outings
    tto_decay_szn                              — xwOBA vs 3rd time through minus 1st (hook driver)

Control = v2 (decision feats + line-conditional). ACCEPT iff v3>v2 and v3>market both dirs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from db.db import query
from models.mlb.hitter.backtest import american_to_decimal
from models.mlb.feature_sets import XGB_PARAMS
from models.mlb.pitcher.outs_v2_gate import (_load, ip_to_outs, build_decision,
                                             V1_FEATS, DECISION, RUNGS, OUTS_MKT, NOVIG, ROOT)

B2 = ['gs_run_diff_own_30d', 'gs_run_diff_opp_30d', 'sched_days_to_off',
      'ump_k_rate_365d', 'tto_decay_szn']


def build_b2(df):
    g = query("""SELECT game_id, game_date, home_team_id, away_team_id, home_score, away_score
                 FROM games WHERE sport_id=2 AND status='final'""")
    g['game_date'] = pd.to_datetime(g['game_date'])
    # team run diff 30d closed-left
    t1 = g.rename(columns={'home_team_id': 'team_id'}); t1['rd'] = t1['home_score'] - t1['away_score']
    t2 = g.rename(columns={'away_team_id': 'team_id'}); t2['rd'] = t2['away_score'] - t2['home_score']
    td = pd.concat([t1[['team_id', 'game_date', 'rd']], t2[['team_id', 'game_date', 'rd']]]).sort_values('game_date')
    ch = []
    for t, gr in td.groupby('team_id', sort=False):
        r = gr.set_index('game_date')['rd'].rolling('30D', closed='left').mean()
        ch.append(pd.DataFrame({'team_id': t, 'game_date': gr['game_date'].values, 'rd30': r.values}))
    rd = pd.concat(ch).drop_duplicates(['team_id', 'game_date'])
    df = df.merge(rd.rename(columns={'rd30': 'gs_run_diff_own_30d'}), on=['team_id', 'game_date'], how='left')
    df = df.merge(rd.rename(columns={'team_id': 'opp_team_id', 'rd30': 'gs_run_diff_opp_30d'}),
                  on=['opp_team_id', 'game_date'], how='left')
    # days to next off-day (all games incl scheduled)
    ag = query("SELECT game_id, game_date, home_team_id, away_team_id FROM games WHERE sport_id=2")
    ag['game_date'] = pd.to_datetime(ag['game_date'])
    sched = pd.concat([ag[['home_team_id', 'game_date']].rename(columns={'home_team_id': 'team_id'}),
                       ag[['away_team_id', 'game_date']].rename(columns={'away_team_id': 'team_id'})]).drop_duplicates()
    sset = set(zip(sched['team_id'], sched['game_date'].dt.date))
    from datetime import timedelta
    dts = df['game_date'].dt.date
    df['sched_days_to_off'] = [next((k for k in range(1, 8)
                                     if (int(t), d + timedelta(days=k)) not in sset), 7)
                               for t, d in zip(df['team_id'], dts)]
    # ump K rate 365d closed-left
    u = query("""SELECT g.game_id, g.game_date, gi.umpire_hp, bg.so_t, bg.pa_t
        FROM games g JOIN mlb_game_info gi ON g.game_id=gi.game_id
        JOIN (SELECT game_id, SUM(so) so_t, SUM(pa) pa_t FROM mlb_batting_game GROUP BY game_id) bg
        ON g.game_id=bg.game_id WHERE g.sport_id=2 AND g.status='final' AND gi.umpire_hp IS NOT NULL""")
    u['game_date'] = pd.to_datetime(u['game_date']); u = u.sort_values(['umpire_hp', 'game_date'])
    ch = []
    for _, gr in u.groupby('umpire_hp', sort=False):
        r = gr.set_index('game_date')[['so_t', 'pa_t']].rolling('365D', closed='left').sum()
        ch.append(pd.DataFrame({'game_id': gr['game_id'].values,
                                'ump_k_rate_365d': np.where(r['pa_t'] >= 2000, r['so_t'] / r['pa_t'], np.nan)}))
    df = df.merge(pd.concat(ch), on='game_id', how='left')
    # TTO decay: xwoba 3rd-time-through minus 1st, season rolling (prior-only)
    ab = query("""SELECT p.game_id, p.pitcher_id, p.at_bat_number, MAX(p.xwoba) xw
        FROM mlb_pitches p JOIN games g ON p.game_id=g.game_id
        WHERE g.sport_id=2 AND p.xwoba IS NOT NULL GROUP BY 1,2,3""")
    ab = ab.sort_values(['game_id', 'pitcher_id', 'at_bat_number'])
    ab['seq'] = ab.groupby(['game_id', 'pitcher_id']).cumcount() + 1
    per = ab.groupby(['game_id', 'pitcher_id']).apply(
        lambda x: pd.Series({'x1': x[x.seq <= 9]['xw'].mean(),
                             'x3': x[x.seq >= 19]['xw'].mean()})).reset_index()
    per = per.merge(df[['game_id', 'pitcher_id', 'game_date', 'season']], on=['game_id', 'pitcher_id'], how='inner')
    per = per.sort_values('game_date')
    ch = []
    for pid, gr in per.groupby('pitcher_id', sort=False):
        gr = gr.copy()
        gr['tto_decay_szn'] = (gr['x3'] - gr['x1']).shift(1).expanding(min_periods=5).mean()
        ch.append(gr[['game_id', 'pitcher_id', 'tto_decay_szn']])
    df = df.merge(pd.concat(ch), on=['game_id', 'pitcher_id'], how='left')
    return df


def main():
    _load("models.mlb.statcast_features", "archive/models/mlb/statcast_features.py")
    km = _load("k_model_arch", "archive/models/mlb/k_model.py")
    print("building dataset...")
    df = km.build_k_dataset(); df['game_date'] = pd.to_datetime(df['game_date'])
    df = df[df['actual_ip'].notna() & (df['actual_ip'] > 0)]
    gm = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id=2")
    pg = query("SELECT game_id, player_id AS pitcher_id, team_id FROM mlb_pitching_game WHERE is_starter=true")
    df = df.merge(pg, on=['game_id', 'pitcher_id'], how='left').merge(gm, on='game_id', how='left')
    df['opp_team_id'] = np.where(df['team_id'] == df['home_team_id'], df['away_team_id'], df['home_team_id'])
    df = build_decision(df)
    print("building batch-2 features...")
    df = build_b2(df)
    dec = [c for c in DECISION if c in df.columns]; b2 = [c for c in B2 if c in df.columns]
    v1 = [c for c in V1_FEATS if c in df.columns]
    print(f"b2 coverage 2025+: {df[df.season>=2025][b2].notna().mean().round(2).to_dict()}")

    tr = df[df['season'] <= 2024]; te = df[df['season'] >= 2025].copy()
    def fit_lc(feats):
        med = tr[feats].median()
        aug = []
        for r in RUNGS:
            a = tr[feats].fillna(med).copy(); a['line'] = r
            a['y'] = (tr['outs'].values > r).astype(int); aug.append(a)
        A = pd.concat(aug, ignore_index=True)
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(A[feats + ['line']].values, A['y'].values)
        return m, med
    m2, med2 = fit_lc(v1 + dec)
    m3, med3 = fit_lc(v1 + dec + b2)

    pl = query("SELECT player_id, LOWER(full_name) fn FROM players")
    P = query("""SELECT prop_date AS game_date, over_line, over_odds, under_odds, actual,
                 LOWER(player_first_name||' '||player_last_name) fn
                 FROM bettingpros_props WHERE market_id=%(m)s AND book_id=%(b)s
                 AND over_odds IS NOT NULL AND under_odds IS NOT NULL""",
              params={'m': OUTS_MKT, 'b': NOVIG})
    P['game_date'] = pd.to_datetime(P['game_date'])
    P = P.merge(pl, on='fn', how='inner').rename(columns={'player_id': 'pitcher_id'})
    P = P.merge(te[['game_date', 'pitcher_id'] + v1 + dec + b2], on=['game_date', 'pitcher_id'], how='inner')
    for tag, m, med, fs in [('p_v2', m2, med2, v1 + dec), ('p_v3', m3, med3, v1 + dec + b2)]:
        X = P[fs].fillna(med).copy(); X['line'] = P['over_line'].values
        P[tag] = m.predict_proba(X[fs + ['line']].values)[:, 1]
    io_ = 1 / P['over_odds'].apply(american_to_decimal); iu_ = 1 / P['under_odds'].apply(american_to_decimal)
    P['p_mkt'] = io_ / (io_ + iu_)
    P['y'] = (P['actual'].astype(float) > P['over_line']).astype(int); P['yr'] = P['game_date'].dt.year
    print(f"\nstandalone AUC: v2={roc_auc_score(P['y'],P['p_v2']):.4f} "
          f"v3={roc_auc_score(P['y'],P['p_v3']):.4f} market={roc_auc_score(P['y'],P['p_mkt']):.4f}")
    print("===== OUTS v3 GATE =====")
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = P[P['yr'] == fy], P[P['yr'] == ty]
        if len(f) < 200 or len(t) < 200: continue
        a = {}
        for k, cols in [('A', ['p_mkt']), ('V2', ['p_mkt', 'p_v2']), ('V3', ['p_mkt', 'p_v3'])]:
            lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
            a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        print(f"  fit {fy}->test {ty} (n={len(t):,}): A={a['A']:.4f} V2={a['V2']:.4f} V3={a['V3']:.4f} "
              f"V3-V2={a['V3']-a['V2']:+.4f} V3-A={a['V3']-a['A']:+.4f}")
    print("(ACCEPT iff V3>V2 and V3>A both directions.)")


if __name__ == "__main__":
    main()
