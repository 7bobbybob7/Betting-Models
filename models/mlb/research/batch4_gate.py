"""
models/mlb/batch4_gate.py — BATCH 4: swing-path + arm-angle features (declared 2026-07-08).

    bat_attack_dir_120d   mean attack direction (horizontal swing-path angle), 120d
    bat_tilt_120d         mean swing path tilt, 120d
    pit_arm_angle_365d    opposing starter's mean release arm angle, 365d (99% coverage)

Gate: control = v6 (full-coverage stack), candidate = v6 + batch4. TB (UD) + HR (Novig),
both directions. ACCEPT iff V7>V6 both directions on either target.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import contextlib, io as _io
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from db.db import query
from models.mlb.hitter.backtest import (load_bundle, attach_odds, american_to_decimal,
                                 _build_player_match, TARGET_TO_MARKET)
from models.mlb.feature_sets import TRAIN_PQ, BTEST_PQ, XGB_PARAMS
from models.mlb.feature_sets import ADV_FEATS
from models.mlb.feature_sets import FULL_CACHE
from models.mlb.feature_sets import BATCH1
from models.mlb.feature_sets import build_luck, LUCK
from models.mlb.research.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr
from models.mlb.features.advanced_profile_features import _roll_rate, _pull_spine

BATCH4 = ['bat_attack_dir_120d', 'bat_tilt_120d', 'pit_arm_angle_365d']


def build_batch4(start, end):
    sw = query("""
        SELECT p.batter_id AS player_id, g.game_date,
               e.attack_direction AS ad, e.swing_path_tilt AS tl
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e ON e.game_id=p.game_id AND e.at_bat_number=p.at_bat_number
         AND e.pitch_number=p.pitch_number
        JOIN games g ON p.game_id=g.game_id
        WHERE e.attack_direction IS NOT NULL
          AND g.game_date >= %(s)s AND g.game_date < %(e)s""",
        params={'s': start - timedelta(days=130), 'e': end + timedelta(days=1)})
    aa = query("""
        SELECT p.pitcher_id, g.game_date, AVG(e.arm_angle) AS ang, COUNT(*) AS n
        FROM mlb_pitches p
        JOIN mlb_pitch_extras e ON e.game_id=p.game_id AND e.at_bat_number=p.at_bat_number
         AND e.pitch_number=p.pitch_number
        JOIN games g ON p.game_id=g.game_id
        WHERE e.arm_angle IS NOT NULL AND g.game_date >= %(s)s AND g.game_date < %(e)s
        GROUP BY 1, 2""",
        params={'s': start - timedelta(days=370), 'e': end + timedelta(days=1)})
    spine = _pull_spine(start, end)
    for d in (sw, aa, spine):
        d['game_date'] = pd.to_datetime(d['game_date'])
    sd = spine[['player_id', 'game_date']].drop_duplicates()

    sw['n'] = 1.0; sw['ad'] = sw['ad'].astype(float); sw['tl'] = sw['tl'].astype(float)
    r = _roll_rate(sw, ['ad', 'tl', 'n'], '120D').drop_duplicates(['player_id', 'game_date'])
    r = sd.merge(r, on=['player_id', 'game_date'], how='left')
    out = sd.copy()
    ok = r['n'] >= 40
    out['bat_attack_dir_120d'] = np.where(ok, r['ad'] / r['n'], np.nan)
    out['bat_tilt_120d'] = np.where(ok, r['tl'] / r['n'], np.nan)

    # opposing starter's rolling arm angle
    aa['w'] = aa['ang'] * aa['n']
    ar = _roll_rate(aa.rename(columns={'pitcher_id': 'player_id'}), ['w', 'n'], '365D') \
         .drop_duplicates(['player_id', 'game_date']).rename(columns={'player_id': 'pitcher_id'})
    ar['pit_arm_angle_365d'] = np.where(ar['n'] >= 200, ar['w'] / ar['n'], np.nan)
    st = query("SELECT game_id, team_id, player_id AS starter_id FROM mlb_pitching_game WHERE is_starter=true")
    gm = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id=2")
    sp = spine.merge(gm, on='game_id', how='left')
    sp['opp'] = np.where(sp['batter_team_id'] == sp['home_team_id'] if 'batter_team_id' in sp
                         else False, sp['away_team_id'], sp['home_team_id'])
    # spine from advanced_profile has opp_team_id already
    sp['opp'] = spine['opp_team_id'].values
    sp = sp.merge(st.rename(columns={'team_id': 'opp'}), on=['game_id', 'opp'], how='left')
    sp = sp.merge(ar[['pitcher_id', 'game_date', 'pit_arm_angle_365d']],
                  left_on=['starter_id', 'game_date'], right_on=['pitcher_id', 'game_date'], how='left')
    out = out.merge(sp[['player_id', 'game_date', 'pit_arm_angle_365d']].drop_duplicates(
        ['player_id', 'game_date']), on=['player_id', 'game_date'], how='left')
    out = spine[['game_id', 'player_id', 'game_date']].merge(
        out, on=['player_id', 'game_date'], how='left')
    return out[['game_id', 'player_id'] + BATCH4].drop_duplicates(['game_id', 'player_id'])


def main():
    print("building batch-4 features 2019-2026...")
    b4 = build_batch4(date(2019, 3, 1), date(2026, 12, 31))
    print(f"  {len(b4):,} rows; coverage: {b4[BATCH4].notna().mean().round(2).to_dict()}")
    adv = pd.read_parquet(FULL_CACHE)
    lk = build_luck(date(2019, 3, 1), date(2026, 12, 31))
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left').merge(b4, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left') \
         .merge(lk, on=['game_id', 'player_id'], how='left').merge(b4, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])

    for target in ['tb', 'hr']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb', Path('models/mlb/saved'))['features']
        v6 = base + ADV_FEATS + BATCH1 + LUCK; v7 = v6 + BATCH4
        m6 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m6.fit(trt[v6].values, y, verbose=False)
        m7 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m7.fit(trt[v7].values, y, verbose=False)
        btt = btt.copy()
        btt['p_v6'] = m6.predict(btt[v6].values); btt['p_v7'] = m7.predict(btt[v7].values)
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
            odds = odds.merge(mt[mt['player_id'].notna()][['bp_player_id', 'player_id']], on='bp_player_id', how='inner')
            J = btt.merge(odds, on=['game_date', 'player_id'], how='inner')
        io_ = 1 / J['over_odds'].apply(american_to_decimal); iu_ = 1 / J['under_odds'].apply(american_to_decimal)
        J['p_mkt'] = io_ / (io_ + iu_); J['y'] = J[label].astype(int); J['yr'] = J['game_date'].dt.year
        print(f"\n===== BATCH-4 GATE [{target.upper()}] (control = v6) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            aa_ = {}
            for kk, col in [('A', ['p_mkt']), ('V6', ['p_mkt', 'p_v6']), ('V7', ['p_mkt', 'p_v7'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[col], f['y'])
                aa_[kk] = roc_auc_score(t['y'], lm.predict_proba(t[col])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa_['A']:.4f} V6={aa_['V6']:.4f} "
                  f"V7={aa_['V7']:.4f}  V7-V6={aa_['V7']-aa_['V6']:+.4f}")
    print("\n(ACCEPT iff V7>V6 both directions on either target.)")


if __name__ == "__main__":
    main()
