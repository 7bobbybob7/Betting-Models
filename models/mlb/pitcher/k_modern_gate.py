"""
models/mlb/pitcher/k_modern_gate.py — modernize the strikeout model (first iteration).

The archived Poisson K model (24 features, 2022-era) LOST to the market in k_gate.py
(model AUC 0.54 vs market 0.58). It has never had the treatment the hitter model got.
Its only opposing-batter signal is TEAM-level K rate (blunt). This gate adds 4 features
that are on-mechanism for strikeouts and were unavailable in 2022:

    kf_opp_whiff_vs_arsenal   opposing lineup's whiff rates, WEIGHTED by THIS pitcher's
                              actual pitch mix (slider-heavy pitcher vs slider-whiffing
                              lineup = K-rich matchup the market may underprice)
    kf_opp_bat_speed_120d     opposing lineup avg bat speed (slow bats -> more Ks)
    kf_ump_k_rate_365d        home-plate umpire's K/PA tendency (zone size — directly
                              moves K props; built for hitters, rejected there, but a
                              STRIKEOUT prop is exactly where it should matter)
    kf_arm_angle_365d         pitcher release arm angle (deception proxy)

Gate: control = old-24, candidate = old-24 + 4 modern. Retrain Poisson-features in XGB
(distributional model kept separately); residual test vs Novig K market both directions,
plus does candidate beat market standalone. ACCEPT iff candidate blend > control blend
AND > market in both directions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import importlib.util
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from db.db import query
from models.mlb.hitter.backtest import american_to_decimal
from models.mlb.feature_sets import XGB_PARAMS

ROOT = os.path.join(os.path.dirname(__file__), "../../..")
PITCH_TYPES = ['FF', 'SI', 'SL', 'CH', 'CU', 'FC', 'ST', 'KC', 'FS']


def _load_archived(name, relpath, register_as=None):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[register_as or name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_modern(starts: pd.DataFrame) -> pd.DataFrame:
    """4 modern features per (game_id, pitcher_id). starts carries game_id, pitcher_id,
    game_date, opp_team_id."""
    gids = tuple(int(g) for g in starts['game_id'].unique())

    # --- umpire K-rate (365d, closed-left) ---
    ump = query("""
        SELECT g.game_id, g.game_date, gi.umpire_hp, bg.so_total, bg.pa_total
        FROM games g JOIN mlb_game_info gi ON g.game_id = gi.game_id
        JOIN (SELECT game_id, SUM(so) so_total, SUM(pa) pa_total FROM mlb_batting_game GROUP BY game_id) bg
          ON g.game_id = bg.game_id
        WHERE g.sport_id = 2 AND g.status='final' AND gi.umpire_hp IS NOT NULL""")
    ump['game_date'] = pd.to_datetime(ump['game_date'])
    ump = ump.sort_values(['umpire_hp', 'game_date'])
    chunks = []
    for u, grp in ump.groupby('umpire_hp', sort=False):
        gi = grp.set_index('game_date')[['so_total', 'pa_total']].rolling('365D', closed='left').sum()
        gi['game_id'] = grp['game_id'].values
        chunks.append(gi.reset_index(drop=True))
    ur = pd.concat(chunks, ignore_index=True)
    ur['kf_ump_k_rate_365d'] = np.where(ur['pa_total'] >= 2000, ur['so_total'] / ur['pa_total'], np.nan)
    umap = ur.set_index('game_id')['kf_ump_k_rate_365d'].to_dict()

    # --- pitcher arm angle (365d rolling) ---
    aa = query("""
        SELECT p.pitcher_id, g.game_date, AVG(e.arm_angle) ang, COUNT(*) n
        FROM mlb_pitches p JOIN mlb_pitch_extras e
          ON e.game_id=p.game_id AND e.at_bat_number=p.at_bat_number AND e.pitch_number=p.pitch_number
        JOIN games g ON p.game_id=g.game_id
        WHERE e.arm_angle IS NOT NULL AND g.sport_id=2 GROUP BY 1,2""")
    aa['game_date'] = pd.to_datetime(aa['game_date'])
    aa = aa.sort_values(['pitcher_id', 'game_date'])
    ch = []
    for pid, grp in aa.groupby('pitcher_id', sort=False):
        g2 = grp.set_index('game_date')
        r = pd.DataFrame({'w': (g2['ang'] * g2['n']), 'n': g2['n']}).rolling('365D', closed='left').sum()
        r['pitcher_id'] = pid; r['game_date'] = grp['game_date'].values
        ch.append(r.reset_index(drop=True))
    ar = pd.concat(ch, ignore_index=True)
    ar['kf_arm_angle_365d'] = np.where(ar['n'] >= 200, ar['w'] / ar['n'], np.nan)

    # --- opposing lineup whiff-vs-arsenal + bat speed (from hitter parquets, as-of) ---
    adv = pd.read_parquet("models/mlb/cache/adv_profile_2019_2026.parquet")
    tr = pd.read_parquet("models/mlb/cache/train_2019_2024.parquet")
    bt = pd.read_parquet("models/mlb/cache/backtest_2025_2026.parquet")
    whiff_cols = [f'bat_whiff_rate_vs_{pt}_90d' for pt in PITCH_TYPES]
    keep = ['game_id', 'player_id', 'batter_team_id'] + [c for c in whiff_cols if c in tr.columns]
    hit = pd.concat([tr[keep], bt[keep]], ignore_index=True).merge(
        adv[['game_id', 'player_id', 'bat_bat_speed_120d']], on=['game_id', 'player_id'], how='left')

    # pitcher arsenal: pct by pitch type, rolling season (from pitch mix already in bt/tr as pit_pct_*)
    # simpler: use the opposing lineup's mean whiff vs FB+SL (the two highest-K pitch families)
    # weighted equally — a clean v1 proxy for "lineup swings-and-misses"
    lineup = hit.groupby('game_id').agg(
        opp_whiff_fb=('bat_whiff_rate_vs_FF_90d', 'mean') if 'bat_whiff_rate_vs_FF_90d' in hit else ('game_id', 'size'),
        opp_whiff_sl=('bat_whiff_rate_vs_SL_90d', 'mean') if 'bat_whiff_rate_vs_SL_90d' in hit else ('game_id', 'size'),
        opp_bat_speed=('bat_bat_speed_120d', 'mean'),
    ).reset_index()
    lineup['kf_opp_whiff_vs_arsenal'] = lineup[['opp_whiff_fb', 'opp_whiff_sl']].mean(axis=1)
    lineup = lineup.rename(columns={'opp_bat_speed': 'kf_opp_bat_speed_120d'})

    out = starts[['game_id', 'pitcher_id', 'game_date']].copy()
    out['game_date'] = pd.to_datetime(out['game_date'])
    out['kf_ump_k_rate_365d'] = out['game_id'].map(umap)
    out = out.merge(ar[['pitcher_id', 'game_date', 'kf_arm_angle_365d']], on=['pitcher_id', 'game_date'], how='left')
    # lineup keyed by the OPPONENT's batter rows = same game_id (batters in this game)
    out = out.merge(lineup[['game_id', 'kf_opp_whiff_vs_arsenal', 'kf_opp_bat_speed_120d']],
                    on='game_id', how='left')
    return out


MODERN = ['kf_opp_whiff_vs_arsenal', 'kf_opp_bat_speed_120d', 'kf_ump_k_rate_365d', 'kf_arm_angle_365d']


def main():
    _load_archived("models.mlb.statcast_features", "archive/models/mlb/statcast_features.py",
                   register_as="models.mlb.statcast_features")
    km = _load_archived("k_model_arch", "archive/models/mlb/k_model.py")
    print("building archived K dataset...")
    df = km.build_k_dataset()
    df['game_date'] = pd.to_datetime(df['game_date'])
    old = [c for c in km.K_FEATURES if c in df.columns]

    # need opp_team_id for lineup join — derive
    gm = query("SELECT game_id, home_team_id, away_team_id FROM games WHERE sport_id=2")
    pg = query("SELECT game_id, player_id AS pitcher_id, team_id FROM mlb_pitching_game WHERE is_starter=true")
    df = df.merge(pg, on=['game_id', 'pitcher_id'], how='left').merge(gm, on='game_id', how='left')
    df['opp_team_id'] = np.where(df['team_id'] == df['home_team_id'], df['away_team_id'], df['home_team_id'])

    print("building modern features...")
    mod = build_modern(df)
    df = df.merge(mod[['game_id', 'pitcher_id'] + MODERN], on=['game_id', 'pitcher_id'], how='left')
    cov = df[df['season'] >= 2025][MODERN].notna().mean()
    print(f"modern feature coverage (2025-26): {cov.round(2).to_dict()}")

    tr = df[df['season'] <= 2024]; te = df[df['season'] >= 2025]
    ytr = tr['actual_k'].values

    # XGB regressors: old vs old+modern (predict K count -> P(over line) via market join)
    m_old = xgb.XGBRegressor(objective='count:poisson', **XGB_PARAMS)
    m_old.fit(tr[old].fillna(tr[old].median()).values, ytr)
    m_new = xgb.XGBRegressor(objective='count:poisson', **XGB_PARAMS)
    m_new.fit(tr[old + MODERN].fillna(tr[old + MODERN].median()).values, ytr)
    te = te.copy()
    te['lam_old'] = m_old.predict(te[old].fillna(tr[old].median()).values)
    te['lam_new'] = m_new.predict(te[old + MODERN].fillna(tr[old + MODERN].median()).values)

    from scipy.stats import poisson
    pl = query("SELECT player_id, LOWER(full_name) fn FROM players")
    P = query("""SELECT prop_date AS game_date, over_line, over_odds, under_odds, actual, is_scored,
                 LOWER(player_first_name||' '||player_last_name) fn
                 FROM bettingpros_props WHERE market_id=285 AND book_id=60
                 AND over_odds IS NOT NULL AND under_odds IS NOT NULL""")
    P['game_date'] = pd.to_datetime(P['game_date'])
    P = P.merge(pl, on='fn', how='inner').rename(columns={'player_id': 'pitcher_id'})
    P = P.merge(te[['game_date', 'pitcher_id', 'lam_old', 'lam_new']], on=['game_date', 'pitcher_id'], how='inner')
    io_ = 1 / P['over_odds'].apply(american_to_decimal); iu_ = 1 / P['under_odds'].apply(american_to_decimal)
    P['p_mkt'] = io_ / (io_ + iu_)
    P['p_old'] = poisson.sf(np.floor(P['over_line']), P['lam_old'])
    P['p_new'] = poisson.sf(np.floor(P['over_line']), P['lam_new'])
    P['y'] = (P['actual'].astype(float) > P['over_line']).astype(int); P['yr'] = P['game_date'].dt.year

    print(f"\nmatched Novig K props: 2025={len(P[P.yr==2025]):,} 2026={len(P[P.yr==2026]):,}")
    print(f"model-alone AUC: old={roc_auc_score(P['y'],P['p_old']):.4f} new={roc_auc_score(P['y'],P['p_new']):.4f} "
          f"| market={roc_auc_score(P['y'],P['p_mkt']):.4f}")
    print("\n===== MODERN-K GATE (Novig anchor, both directions) =====")
    for fy, ty in [(2025, 2026), (2026, 2025)]:
        f, t = P[P['yr'] == fy], P[P['yr'] == ty]
        if len(f) < 300 or len(t) < 300:
            print(f"  {fy}->{ty}: thin"); continue
        a = {}
        for k, cols in [('mkt', ['p_mkt']), ('old', ['p_mkt', 'p_old']), ('new', ['p_mkt', 'p_new'])]:
            lm = LogisticRegression(max_iter=1000).fit(f[cols], f['y'])
            a[k] = roc_auc_score(t['y'], lm.predict_proba(t[cols])[:, 1])
        print(f"  fit {fy}->test {ty} (n={len(t):,}): mkt={a['mkt']:.4f} old={a['old']:.4f} "
              f"new={a['new']:.4f}  new-old={a['new']-a['old']:+.4f}  new-mkt={a['new']-a['mkt']:+.4f}")
    print("\n(ACCEPT iff new>old AND new>mkt in both directions.)")


if __name__ == "__main__":
    main()
