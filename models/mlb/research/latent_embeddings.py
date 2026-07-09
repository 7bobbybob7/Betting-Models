"""
models/mlb/latent_embeddings.py — embeddings v2: latent batter/pitcher factors via
logistic matrix factorization on PA-level outcomes (2019-2024 ONLY -> no leakage).

s_ij = mu + a_i + b_j + u_i . v_j   per outcome (HR-in-PA, K-in-PA), rank 8.
Biases soak up main effects (already features); u.v is pure interaction. Features
emitted: emb_hr_dot, emb_k_dot per (batter, opposing starter). Gated vs v4 on HR + TB.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import contextlib, io as _io
from datetime import date
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
from models.mlb.feature_sets import ADV_CACHE, ADV_FEATS
from models.mlb.feature_sets import BATCH1
from models.mlb.research.hr_gate import HR_MARKET_ID, NOVIG_BOOK_ID, _attach_hr

RANK, EPOCHS, LR_, REG = 8, 30, 0.05, 0.02


def fit_mf(pairs, kcol):
    bats = {p: i for i, p in enumerate(pairs['batter_id'].unique())}
    pits = {p: i for i, p in enumerate(pairs['pitcher_id'].unique())}
    bi = pairs['batter_id'].map(bats).values
    pj = pairs['pitcher_id'].map(pits).values
    n = pairs['n_pa'].values.astype(float)
    k = pairs[kcol].values.astype(float)
    rng = np.random.default_rng(42)
    U = rng.normal(0, .01, (len(bats), RANK)); V = rng.normal(0, .01, (len(pits), RANK))
    a = np.zeros(len(bats)); b = np.zeros(len(pits))
    mu = np.log(k.sum() / (n.sum() - k.sum()))
    w = np.sqrt(np.minimum(n, 25.0))                     # capped pair weight
    cnt_a = np.zeros(len(bats)); np.add.at(cnt_a, bi, w); cnt_a = np.maximum(cnt_a, 1)
    cnt_b = np.zeros(len(pits)); np.add.at(cnt_b, pj, w); cnt_b = np.maximum(cnt_b, 1)
    for ep in range(EPOCHS):
        s = np.clip(mu + a[bi] + b[pj] + (U[bi] * V[pj]).sum(1), -10, 10)
        p = 1 / (1 + np.exp(-s))
        g = w * (k / n - p)                              # rate residual, bounded
        ga = np.zeros_like(a); np.add.at(ga, bi, g)
        gb = np.zeros_like(b); np.add.at(gb, pj, g)
        gU = np.zeros_like(U); np.add.at(gU, bi, g[:, None] * V[pj])
        gV = np.zeros_like(V); np.add.at(gV, pj, g[:, None] * U[bi])
        a += LR_ * (ga / cnt_a - REG * a); b += LR_ * (gb / cnt_b - REG * b)
        U += LR_ * (gU / cnt_a[:, None] - REG * U); V += LR_ * (gV / cnt_b[:, None] - REG * V)
    return bats, pits, U, V


def main():
    pairs = query("""
      SELECT p.batter_id, p.pitcher_id, COUNT(*) n_pa,
             COUNT(*) FILTER (WHERE p.result ILIKE '%%home_run%%' OR p.result ILIKE '%%home run%%') n_hr,
             COUNT(*) FILTER (WHERE p.result ILIKE '%%strikeout%%') n_k
      FROM mlb_pitches p JOIN games g ON p.game_id = g.game_id
      WHERE g.game_date < '2025-01-01' AND g.sport_id = 2
        AND p.result IS NOT NULL AND p.result <> ''
      GROUP BY 1, 2""")
    print(f"pairs: {len(pairs):,} | PAs {pairs['n_pa'].sum():,} | "
          f"HR rate {pairs['n_hr'].sum()/pairs['n_pa'].sum():.4f} | "
          f"K rate {pairs['n_k'].sum()/pairs['n_pa'].sum():.4f}")
    embs = {}
    for kcol, name in [('n_hr', 'hr'), ('n_k', 'k')]:
        embs[name] = fit_mf(pairs, kcol)
        print(f"emb_{name}: fitted")

    adv = pd.read_parquet(ADV_CACHE)
    tr = _attach_hr(pd.read_parquet(TRAIN_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    bt = _attach_hr(pd.read_parquet(BTEST_PQ)).merge(adv, on=['game_id', 'player_id'], how='left')
    for df in (tr, bt):
        df['game_date'] = pd.to_datetime(df['game_date'])

    # opposing starter per row: parquets carry pitcher context via game — need starter id
    st = query("""SELECT pg.game_id, pg.team_id, pg.player_id AS starter_id
                  FROM mlb_pitching_game pg WHERE pg.is_starter = true""")
    gm = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id=2")
    for df_name in ('tr', 'bt'):
        df = locals()[df_name]
        d = df.merge(gm, on='game_id', how='left')
        opp = np.where(d['batter_team_id'] == d['home_team_id'], d['away_team_id'], d['home_team_id'])
        d['opp_team_id2'] = opp
        d = d.merge(st.rename(columns={'team_id': 'opp_team_id2'}), on=['game_id', 'opp_team_id2'], how='left')
        for name in ('hr', 'k'):
            bats, pits, U, V = embs[name]
            bix = d['player_id'].map(bats); pix = d['starter_id'].map(pits)
            ok = bix.notna() & pix.notna()
            dot = np.zeros(len(d))
            dot[ok.values] = (U[bix[ok].astype(int)] * V[pix[ok].astype(int)]).sum(1)
            df[f'emb_{name}_dot'] = dot
    EMB = ['emb_hr_dot', 'emb_k_dot']
    print(f"emb feature stds: tr {tr[EMB].std().round(4).to_dict()}")

    for target in ['tb', 'hr']:
        label = 'lbl_hr' if target == 'hr' else TARGET_TO_MARKET['tb']['label_col']
        trt = tr[tr[label].notna()]; btt = bt[bt[label].notna()]
        y = trt[label].astype(int).values
        base = load_bundle('hrr' if target == 'hr' else 'tb', 'xgb', Path('models/mlb/saved'))['features']
        v4 = base + ADV_FEATS + BATCH1; v6 = v4 + EMB
        m4 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m4.fit(trt[v4].values, y, verbose=False)
        m6 = xgb.XGBRegressor(objective='binary:logistic', **XGB_PARAMS); m6.fit(trt[v6].values, y, verbose=False)
        btt = btt.copy()
        btt['p_v4'] = m4.predict(btt[v4].values); btt['p_v6'] = m6.predict(btt[v6].values)
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
        print(f"\n===== LATENT EMB GATE [{target.upper()}] (control = v4) =====")
        for fy, ty in [(2025, 2026), (2026, 2025)]:
            f, t = J[J['yr'] == fy], J[J['yr'] == ty]
            aa = {}
            for kk, cols in [('A', ['p_mkt']), ('V4', ['p_mkt', 'p_v4']), ('V6', ['p_mkt', 'p_v6'])]:
                lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
                aa[kk] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
            print(f"  fit {fy}->test {ty} (n={len(t):,}): A={aa['A']:.4f} V4={aa['V4']:.4f} "
                  f"V6={aa['V6']:.4f}  V6-V4={aa['V6']-aa['V4']:+.4f}")
    print("\n(ACCEPT iff V6>V4 both directions on either target.)")


if __name__ == "__main__":
    main()
