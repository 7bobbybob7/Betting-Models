"""
models/mlb/research/travel_gate.py — MLB batch: circadian/travel features (declared 7/28).

From circadian_test.py: west batters on eastern trips run ~1.4pts under implied (excess
of control), consistent 2025+2026, day AND night -> eastward-jetlag mechanism.
    trav_body_lag        visitor_tz - host_tz (0 at home; -3 = west team in east)
    trav_east_trip       body_lag <= -2 flag
    trav_night           host-local first pitch >= 18:00
    trav_days_into_trip  consecutive games at |lag|>=2 (jetlag decays ~1 day/hour)
Gate: control = v7 stack, candidate = +travel; TB (UD) + HR (Novig), both directions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import contextlib, io as _io
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from db.db import query
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                        _build_player_match, TARGET_TO_MARKET)
from models.mlb.feature_sets import (TRAIN_PQ, BTEST_PQ, FULL_CACHE, XGB_PARAMS,
                                     ADV_ALL, build_luck)
from models.mlb.research.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr
from models.mlb.research.circadian_test import TZ

TRAV = ['trav_body_lag', 'trav_east_trip', 'trav_night', 'trav_days_into_trip']


def build_travel():
    g = query("""SELECT g.game_id, g.game_date, g.game_time, g.home_team_id, g.away_team_id,
        th.name hn, ta.name an FROM games g
        JOIN teams th ON g.home_team_id=th.team_id JOIN teams ta ON g.away_team_id=ta.team_id
        WHERE g.sport_id=2 AND g.status='final' AND g.game_time IS NOT NULL""")
    g['h_tz'] = g['hn'].map(TZ); g['a_tz'] = g['an'].map(TZ)
    g = g.dropna(subset=['h_tz', 'a_tz'])
    g['local_hr'] = (pd.to_datetime(g['game_time'], utc=True).dt.hour + g['h_tz']) % 24
    g['night'] = (g['local_hr'] >= 18).astype(float)
    rows = []
    for _, r in g.iterrows():
        rows.append((r['game_id'], r['game_date'], r['away_team_id'],
                     r['a_tz'] - r['h_tz'], r['night']))
        rows.append((r['game_id'], r['game_date'], r['home_team_id'], 0.0, r['night']))
    t = pd.DataFrame(rows, columns=['game_id', 'game_date', 'team_id', 'trav_body_lag', 'trav_night'])
    t = t.sort_values('game_date')
    ch = []
    for tid, gr in t.groupby('team_id', sort=False):
        gr = gr.sort_values('game_date').copy()
        big = (gr['trav_body_lag'].abs() >= 2).astype(int)
        streak, out = 0, []
        for b in big:
            streak = streak + 1 if b else 0
            out.append(streak)
        gr['trav_days_into_trip'] = out
        ch.append(gr)
    t = pd.concat(ch)
    t['trav_east_trip'] = (t['trav_body_lag'] <= -2).astype(float)
    return t[['game_id', 'team_id'] + TRAV]


def main():
    trav = build_travel()
    print(f"travel rows: {len(trav):,} | east-trip share: {trav['trav_east_trip'].mean():.3f}")
    adv = pd.read_parquet(FULL_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left') \
         .merge(trav, left_on=['game_id', 'batter_team_id'], right_on=['game_id', 'team_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left') \
         .merge(trav, left_on=['game_id', 'batter_team_id'], right_on=['game_id', 'team_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"travel coverage: train {tr[TRAV].notna().mean().min():.2f} bt {bt[TRAV].notna().mean().min():.2f}")

    for target in ['tb', 'hr']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb', Path('models/mlb/saved'))['features']
        ctl = base + ADV_ALL; cand = ctl + TRAV
        mC = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); mC.fit(trt[ctl].values, y, verbose=False)
        mT = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); mT.fit(trt[cand].values, y, verbose=False)
        btt = btt.copy()
        btt['p_ctl'] = mC.predict(btt[ctl].values); btt['p_trv'] = mT.predict(btt[cand].values)
        if target == 'tb':
            with contextlib.redirect_stdout(_io.StringIO()):
                J = attach_odds(btt, 'tb', date(2025, 1, 1), date(2026, 12, 31))
        else:
            odds = query("""SELECT prop_date AS game_date, bp_player_id, over_odds, under_odds
                FROM bettingpros_props WHERE book_id=%(b)s AND market_id=%(m)s AND over_line=0.5
                AND over_odds IS NOT NULL AND under_odds IS NOT NULL""",
                params={'b': NOVIG_BOOK_ID, 'm': HR_MARKET_ID})
            odds['game_date'] = pd.to_datetime(odds['game_date'])
            with contextlib.redirect_stdout(_io.StringIO()):
                mt = _build_player_match(date(2025, 1, 1), date(2026, 12, 31))
            odds = odds.merge(mt[mt['player_id'].notna()][['bp_player_id', 'player_id']],
                              on='bp_player_id', how='inner')
            J = btt.merge(odds, on=['game_date', 'player_id'], how='inner')
        io_ = 1/J['over_odds'].apply(american_to_decimal); iu_ = 1/J['under_odds'].apply(american_to_decimal)
        J['p_mkt'] = io_/(io_+iu_); J['y'] = J[label].astype(int); J['yr'] = J['game_date'].dt.year
        print(f"\n===== TRAVEL GATE [{target.upper()}] =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            if len(f) < 300 or len(t) < 300: continue
            a = {}
            for k, c_ in [('A', ['p_mkt']), ('C', ['p_mkt', 'p_ctl']), ('T', ['p_mkt', 'p_trv'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[c_], f['y'])
                a[k] = roc_auc_score(t['y'], lm.predict_proba(t[c_])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={a['A']:.4f} ctl={a['C']:.4f} "
                  f"trv={a['T']:.4f}  trv-ctl={a['T']-a['C']:+.4f}  trv-A={a['T']-a['A']:+.4f}")
    print("\n(ACCEPT iff trv>ctl AND trv>A both directions on either target.)")


if __name__ == "__main__":
    main()
