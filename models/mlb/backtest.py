"""
models/mlb/backtest.py — Score trained hitter-prop models against real Underdog odds.

Pipeline:
    1. Load model bundle (LR-L1 or XGBoost) saved by hitter_prop_model.py
    2. Build dataset over backtest window (same path as training)
    3. Score every row → P_model
    4. Join Underdog odds from bettingpros_props (book_id=36) on (player, date, market, line)
    5. Compute EV for over and under sides; take the side with higher EV
    6. Sweep EV thresholds (0%, 0.5%, 1%, 2%, 3%, 5%)
    7. Per threshold: bets placed, hit rate, ROI per bet, daily Sharpe, max drawdown

PUSH HANDLING
    All v1 lines (HRR 1.5, TB 1.5, RBI 0.5) are half-points → integer outcomes can't tie.
    No push handling needed.

PLAYER MATCHING
    BettingPros stores (player_first_name, player_last_name, player_team). Our players
    table stores name as "Last, F" and has no team abbreviation. We use a hardcoded
    MLB team abbreviation → team_id map + name-and-team matching to bridge the two.

Usage:
    python -m models.mlb.backtest --target hrr --model lr_l1 --start 2025-04-01 --end 2025-10-31
    python -m models.mlb.backtest --target all --model both
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pickle
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from db.db import query
from models.mlb.hitter_prop_dataset import build_dataset


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

UNDERDOG_BOOK_ID = 36

# Target → (market_id in bettingpros_props, over_line, label column)
TARGET_TO_MARKET = {
    'hrr': {'market_id': 403, 'over_line': 1.5, 'label_col': 'lbl_hrr_over_15'},
    'tb':  {'market_id': 293, 'over_line': 1.5, 'label_col': 'lbl_tb_over_15'},
    'rbi': {'market_id': 289, 'over_line': 0.5, 'label_col': 'lbl_rbi_over_05'},
}

# EV thresholds to sweep (fraction of stake, so 0.01 = need ≥1% edge)
EV_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

# Hardcoded MLB team abbreviation → team_id lookup (loaded lazily from DB)
_team_abbr_cache = None


def _team_abbr_map() -> dict:
    """Build {abbr → team_id} from our DB. Names are full ("Arizona Diamondbacks") so
    we map by canonical abbreviation rules."""
    global _team_abbr_cache
    if _team_abbr_cache is not None:
        return _team_abbr_cache

    # MLB standard abbreviations (BettingPros uses these)
    abbr_to_name = {
        'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves',
        'BAL': 'Baltimore Orioles',    'BOS': 'Boston Red Sox',
        'CHC': 'Chicago Cubs',         'CWS': 'Chicago White Sox',
        'CHW': 'Chicago White Sox',                          # legacy abbr
        'CIN': 'Cincinnati Reds',      'CLE': 'Cleveland Guardians',
        'COL': 'Colorado Rockies',     'DET': 'Detroit Tigers',
        'HOU': 'Houston Astros',       'KC':  'Kansas City Royals',
        'KCR': 'Kansas City Royals',
        'LAA': 'Los Angeles Angels',   'LAD': 'Los Angeles Dodgers',
        'MIA': 'Miami Marlins',        'MIL': 'Milwaukee Brewers',
        'MIN': 'Minnesota Twins',      'NYM': 'New York Mets',
        'NYY': 'New York Yankees',
        'OAK': 'Oakland Athletics',    'ATH': 'Athletics',  # 2025+ rebrand
        'PHI': 'Philadelphia Phillies','PIT': 'Pittsburgh Pirates',
        'SD':  'San Diego Padres',     'SDP': 'San Diego Padres',
        'SEA': 'Seattle Mariners',
        'SF':  'San Francisco Giants', 'SFG': 'San Francisco Giants',
        'STL': 'St. Louis Cardinals',  'TB':  'Tampa Bay Rays',
        'TBR': 'Tampa Bay Rays',
        'TEX': 'Texas Rangers',        'TOR': 'Toronto Blue Jays',
        'WSH': 'Washington Nationals', 'WAS': 'Washington Nationals',
    }
    teams = query("SELECT team_id, name FROM teams WHERE sport_id = 2")
    name_to_id = dict(zip(teams['name'], teams['team_id']))
    _team_abbr_cache = {abbr: name_to_id[name]
                        for abbr, name in abbr_to_name.items()
                        if name in name_to_id}
    return _team_abbr_cache


# ----------------------------------------------------------------------------
# Model loading + prediction
# ----------------------------------------------------------------------------

def load_bundle(target: str, model_type: str, output_dir: Path):
    path = output_dir / f"hitter_{target}_{model_type}.pkl"
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle


def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Apply the saved model to the feature DataFrame. Returns calibrated P(over)
    when the bundle carries an isotonic calibrator, else the raw model probability."""
    # Reorder columns to match training
    X_aligned = X.reindex(columns=bundle['features'])
    if bundle['model_type'] == 'lr_l1':
        X_imp = X_aligned.fillna(bundle['medians']).fillna(0.0)
        Xs = bundle['scaler'].transform(X_imp.values)
        raw = bundle['model'].predict_proba(Xs)[:, 1]
    else:  # xgb — handles NaN natively
        raw = bundle['model'].predict_proba(X_aligned.values)[:, 1]

    calibrator = bundle.get('calibrator')
    if calibrator is not None:
        return calibrator.predict(raw)
    return raw


# ----------------------------------------------------------------------------
# Player-name → our player_id mapping (BettingPros side)
# ----------------------------------------------------------------------------

import unicodedata


def _normalize_name(s: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace, drop suffixes.
    'Pete Crow-Armstrong' → 'pete crowarmstrong'; 'José Ramírez Jr.' → 'jose ramirez'."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    for suf in (' jr', ' sr', ' ii', ' iii', ' iv'):
        if s.endswith(suf):
            s = s[: -len(suf)]
    # Remove punctuation (periods, apostrophes, hyphens, etc.) and collapse whitespace
    s = ''.join(ch if ch.isalnum() or ch.isspace() else '' for ch in s)
    return ' '.join(s.split())


def _build_player_match(start: date, end: date) -> pd.DataFrame:
    """Match BettingPros players to our player_id via normalized full name.

    Our players.full_name was backfilled from the MLB API (canonical "First Last").
    BettingPros gives first/last separately. We normalize both (accent-strip, lowercase,
    drop punctuation/suffixes) and join. Duplicate normalized names (rare — e.g. two
    'Will Smith') are flagged and dropped rather than guessed."""
    bp = query("""
        SELECT DISTINCT bp_player_id, player_first_name, player_last_name, player_team
        FROM bettingpros_props
        WHERE prop_date >= %(start)s AND prop_date <= %(end)s
          AND book_id = %(book)s
    """, params={'start': start, 'end': end, 'book': UNDERDOG_BOOK_ID})

    ours = query("""
        SELECT player_id, full_name FROM players
        WHERE sport_id = 2 AND full_name IS NOT NULL
    """)
    ours['_norm'] = ours['full_name'].map(_normalize_name)

    # Collapse our side to unique normalized name → player_id. Drop ambiguous duplicates.
    name_counts = ours.groupby('_norm')['player_id'].nunique()
    unique_names = set(name_counts[name_counts == 1].index)
    ambiguous_ours = set(name_counts[name_counts > 1].index)
    ours_unique = ours[ours['_norm'].isin(unique_names)].drop_duplicates('_norm')

    bp['_norm'] = (bp['player_first_name'].fillna('') + ' ' +
                   bp['player_last_name'].fillna('')).map(_normalize_name)

    merged = bp.merge(ours_unique[['_norm', 'player_id']], on='_norm', how='left')

    n_matched = merged['player_id'].notna().sum()
    n_ambig = merged['_norm'].isin(ambiguous_ours).sum()
    print(f"  [match] {n_matched:,} / {len(bp):,} BP players matched by full name "
          f"({100*n_matched/max(len(bp),1):.0f}%); {n_ambig} hit ambiguous duplicate names")
    return merged[['bp_player_id', 'player_id']]


# ----------------------------------------------------------------------------
# Odds attachment
# ----------------------------------------------------------------------------

def attach_odds(dataset: pd.DataFrame, target: str, start: date, end: date) -> pd.DataFrame:
    """Join model rows to Underdog odds. Returns one row per (player, game_date) where
    Underdog offered the specific (market, over_line) we care about."""
    cfg = TARGET_TO_MARKET[target]
    print(f"  pulling Underdog odds for market {cfg['market_id']} line {cfg['over_line']}...")
    odds = query("""
        SELECT prop_date AS game_date, bp_player_id,
               over_line, over_odds, under_odds, actual, is_scored
        FROM bettingpros_props
        WHERE book_id = %(book)s
          AND market_id = %(mkt)s
          AND over_line = %(line)s
          AND prop_date >= %(start)s AND prop_date <= %(end)s
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
    """, params={'book': UNDERDOG_BOOK_ID, 'mkt': cfg['market_id'],
                 'line': cfg['over_line'], 'start': start, 'end': end})
    odds['game_date'] = pd.to_datetime(odds['game_date'])
    print(f"  {len(odds):,} Underdog prop rows")

    # Build player matching for the window
    match = _build_player_match(start, end)
    match_keep = match[match['player_id'].notna()][['bp_player_id', 'player_id']]
    odds = odds.merge(match_keep, on='bp_player_id', how='inner')
    print(f"  {len(odds):,} odds rows after player match")

    # Join to model dataset on (player_id, game_date)
    out = dataset.merge(odds, on=['player_id', 'game_date'], how='inner')
    print(f"  {len(out):,} (model row × Underdog prop) joined rows for backtest")
    return out


# ----------------------------------------------------------------------------
# EV computation + bet selection
# ----------------------------------------------------------------------------

def american_to_decimal(american_odds: float) -> float:
    """+150 → 2.50, -120 → 1.833. Decimal odds = total payout per $1 stake (incl. stake)."""
    if american_odds > 0:
        return american_odds / 100.0 + 1.0
    else:
        return 100.0 / abs(american_odds) + 1.0


def compute_bet_ev(df: pd.DataFrame, p_col: str = 'p_model') -> pd.DataFrame:
    """For each row, compute EV for over and under sides. Returns df with added cols:
    dec_over, dec_under, ev_over, ev_under, best_side ('over'/'under'), ev_best, P_bet, payout_dec."""
    df = df.copy()
    df['dec_over']  = df['over_odds'].apply(american_to_decimal)
    df['dec_under'] = df['under_odds'].apply(american_to_decimal)
    p = df[p_col].astype(float)
    df['ev_over']   = p * (df['dec_over']  - 1.0) - (1.0 - p) * 1.0
    df['ev_under']  = (1.0 - p) * (df['dec_under'] - 1.0) - p * 1.0
    df['best_side'] = np.where(df['ev_over'] >= df['ev_under'], 'over', 'under')
    df['ev_best']   = np.maximum(df['ev_over'], df['ev_under'])
    df['P_bet']     = np.where(df['best_side'] == 'over', p, 1.0 - p)
    df['payout_dec'] = np.where(df['best_side'] == 'over', df['dec_over'], df['dec_under'])
    return df


def compute_bet_outcome(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """For each row, compute the actual outcome of the bet (1 = won, 0 = lost).
    Uses the dataset's label column (which we computed from box scores).
    Returns df with: actual_over (1/0 from box scores), bet_won (1/0)."""
    df = df.copy()
    label_col = TARGET_TO_MARKET[target]['label_col']
    df['actual_over'] = df[label_col].astype(int)
    df['bet_won'] = np.where(df['best_side'] == 'over', df['actual_over'], 1 - df['actual_over'])
    # Profit per $1 bet: win → payout_dec - 1, lose → -1
    df['profit'] = np.where(df['bet_won'] == 1, df['payout_dec'] - 1.0, -1.0)
    return df


# ----------------------------------------------------------------------------
# Threshold sweep + summary
# ----------------------------------------------------------------------------

def backtest_at_threshold(bets: pd.DataFrame, threshold: float) -> dict:
    """Filter to bets where ev_best > threshold; return stats."""
    sub = bets[bets['ev_best'] > threshold].copy()
    n = len(sub)
    if n == 0:
        return {'threshold': threshold, 'n_bets': 0}
    hit_rate = sub['bet_won'].mean()
    roi_per_bet = sub['profit'].mean()
    total_profit = sub['profit'].sum()

    # Daily aggregation for Sharpe
    daily = sub.groupby('game_date')['profit'].agg(['sum', 'count']).reset_index()
    mean_daily = daily['sum'].mean()
    std_daily  = daily['sum'].std()
    sharpe = mean_daily / std_daily if std_daily > 0 else np.nan

    # Max drawdown — running min of cumulative profit
    daily_sorted = daily.sort_values('game_date').reset_index(drop=True)
    daily_sorted['cum'] = daily_sorted['sum'].cumsum()
    daily_sorted['peak'] = daily_sorted['cum'].cummax()
    daily_sorted['dd'] = daily_sorted['cum'] - daily_sorted['peak']
    max_dd = daily_sorted['dd'].min()

    return {
        'threshold': threshold,
        'n_bets': n,
        'bets_per_day': n / max(daily['game_date'].nunique(), 1),
        'hit_rate': hit_rate,
        'roi_per_bet': roi_per_bet,
        'total_profit': total_profit,
        'mean_daily_pl': mean_daily,
        'std_daily_pl': std_daily,
        'sharpe_daily': sharpe,
        'max_drawdown': max_dd,
        'over_pct': (sub['best_side'] == 'over').mean(),
    }


def threshold_sweep(bets: pd.DataFrame, thresholds=EV_THRESHOLDS) -> pd.DataFrame:
    rows = [backtest_at_threshold(bets, t) for t in thresholds]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------------------

def backtest_one(target: str, model_type: str, start: date, end: date,
                 output_dir: Path, dataset: pd.DataFrame = None) -> dict:
    print("\n" + "=" * 72)
    print(f"  BACKTEST: {target.upper()} ({model_type})  window {start} → {end}")
    print("=" * 72)

    print("  loading model bundle...")
    bundle = load_bundle(target, model_type, output_dir)

    if dataset is None:
        print("  building dataset...")
        dataset = build_dataset(start, end)

    # Filter dataset to rows where this target's label is valid
    cfg = TARGET_TO_MARKET[target]
    if target == 'hrr':
        dataset = dataset[dataset['lbl_hrr_valid'] == True].copy()
    dataset = dataset[dataset[cfg['label_col']].notna()].copy()

    print("  scoring with model...")
    feature_cols = bundle['features']
    X = dataset[feature_cols].copy()
    dataset['p_model'] = predict_proba(bundle, X)

    print("  attaching Underdog odds...")
    bets = attach_odds(dataset, target, start, end)
    if len(bets) == 0:
        print("  ⚠️  No betting rows after odds join — aborting backtest")
        return {'target': target, 'model_type': model_type, 'n_bets': 0}

    print("  computing EV + outcomes...")
    bets = compute_bet_ev(bets, p_col='p_model')
    bets = compute_bet_outcome(bets, target)

    print("\n  EV threshold sweep:")
    sweep = threshold_sweep(bets)
    sweep_disp = sweep.copy()
    for c in ['hit_rate', 'roi_per_bet', 'sharpe_daily', 'over_pct']:
        if c in sweep_disp.columns:
            sweep_disp[c] = sweep_disp[c].round(4)
    for c in ['mean_daily_pl', 'std_daily_pl', 'max_drawdown', 'total_profit', 'bets_per_day']:
        if c in sweep_disp.columns:
            sweep_disp[c] = sweep_disp[c].round(2)
    print(sweep_disp.to_string(index=False))

    return {
        'target': target, 'model_type': model_type,
        'window': f"{start} → {end}",
        'n_bets_total': len(bets),
        'sweep': sweep.to_dict('records'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=list(TARGET_TO_MARKET.keys()) + ['all'], default='all')
    parser.add_argument('--model',  choices=['lr_l1', 'xgb', 'both'], default='both')
    parser.add_argument('--start',  default='2025-04-01')
    parser.add_argument('--end',    default='2025-10-31')
    parser.add_argument('--output-dir', default='models/mlb/saved')
    parser.add_argument('--dataset-parquet', default=None,
                        help="Path to a prebuilt dataset parquet. If it exists, load it "
                             "(skips rebuild). If it doesn't, build + save it there.")
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').date()
    end   = datetime.strptime(args.end,   '%Y-%m-%d').date()
    output_dir = Path(args.output_dir)

    if args.dataset_parquet and os.path.exists(args.dataset_parquet):
        print(f"Loading cached dataset from {args.dataset_parquet}...")
        dataset = pd.read_parquet(args.dataset_parquet)
        print(f"Dataset: {len(dataset):,} rows x {dataset.shape[1]} cols (cached)")
    else:
        print(f"Building dataset {start} → {end} (used by all backtest targets)...")
        dataset = build_dataset(start, end)
        print(f"Dataset: {len(dataset):,} rows x {dataset.shape[1]} cols")
        if args.dataset_parquet:
            dataset.to_parquet(args.dataset_parquet, index=False)
            print(f"Cached dataset to {args.dataset_parquet}")

    targets = list(TARGET_TO_MARKET.keys()) if args.target == 'all' else [args.target]
    models = ['lr_l1', 'xgb'] if args.model == 'both' else [args.model]

    all_results = {}
    for t in targets:
        for m in models:
            try:
                all_results[f"{t}_{m}"] = backtest_one(t, m, start, end, output_dir, dataset)
            except FileNotFoundError as e:
                print(f"  ⚠️  Bundle not found for {t}_{m}: {e}")

    print("\n" + "=" * 72)
    print("  BACKTEST SUMMARY (pick threshold by sharpe_daily)")
    print("=" * 72)
    for k, v in all_results.items():
        if v.get('n_bets_total', 0) == 0:
            continue
        # Pick best Sharpe row from sweep
        best = max(v['sweep'], key=lambda r: r.get('sharpe_daily') or -999)
        print(f"\n  {k}: {v['n_bets_total']:,} potential bets")
        print(f"    best Sharpe at threshold={best['threshold']:.3f}: "
              f"{best['n_bets']:,} bets, "
              f"hit_rate={best['hit_rate']:.3f}, "
              f"ROI/bet={best['roi_per_bet']:+.4f}, "
              f"daily Sharpe={best['sharpe_daily']:.3f}")


if __name__ == "__main__":
    main()
