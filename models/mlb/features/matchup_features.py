"""
models/mlb/matchup_features.py — Cross-pollinates batter × pitcher features into
matchup-level signals. This is the bridge module that takes batter arsenal features
(per pitch type) and pitcher arsenal features (mix per pitch type per batter hand)
and produces the actually-predictive matchup metrics.

Unlike the arsenal modules, this is NOT an independent data puller — it operates on
a DataFrame that already has both batter_* and pitcher_* features joined onto the
same row (which the dataset assembler does). It exposes pure computation functions.

FEATURES PRODUCED
    mu_xwoba_expected         — Σ pit_pct(pt | batter_hand) × bat_xwoba_vs(pt)
    mu_whiff_expected         — same formula but for whiff rate
    mu_platoon_advantage      — 1 if batter is opposite hand of pitcher, else 0
    mu_starter_pa_expected    — expected PAs vs starting pitcher (uses pit_ip_per_start)
    mu_total_pa_expected      — expected total PAs (position-adjusted)
    mu_bullpen_pa_share       — fraction of expected PAs vs the bullpen (not starter)

INPUT REQUIREMENTS (columns on the input DataFrame)
    bat_hand        — 'L', 'R', or 'S' (switch)
    pit_throws      — 'L' or 'R'
    ctx_batter_order_position — 1-9
    pit_ip_per_start_30d
    bat_xwoba_vs_{FF, SI, FC, SL, ST, SV, CU, KC, CH, FS, FO}_90d (some may be NaN)
    bat_xwoba_vs_{FB, BR, OS}_90d (bucket fallbacks)
    bat_whiff_rate_vs_{...}_90d (same shape as xwoba)
    pit_pct_{pt}_vs_{R,L}HB_30d
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import pandas as pd


# Must stay in sync with batter_arsenal_features.py / pitcher_arsenal_features.py
INDIVIDUAL_PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'SV', 'CU', 'KC', 'CH', 'FS', 'FO']
BUCKETS = {
    'FB': ['FF', 'SI'],
    'BR': ['SL', 'ST', 'SV', 'CU', 'KC', 'FC'],
    'OS': ['CH', 'FS', 'FO'],
}
PITCH_TO_BUCKET = {pt: bk for bk, pts in BUCKETS.items() for pt in pts}


# ----------------------------------------------------------------------------
# Effective batter hand (switch hitters bat opposite the pitcher's throwing hand)
# ----------------------------------------------------------------------------

def effective_batter_hand(bat_hand: str, pit_throws: str) -> str:
    """Returns 'L' or 'R'. Switch hitters always bat opposite the pitcher."""
    if bat_hand == 'S':
        return 'L' if pit_throws == 'R' else 'R'
    return bat_hand


# ----------------------------------------------------------------------------
# Weighted expected xwOBA / whiff via pitcher mix × batter per-type performance
# ----------------------------------------------------------------------------

def _weighted_batter_metric(row: pd.Series, eff_hand: str, metric_prefix: str) -> float:
    """For a single (batter, pitcher) row, compute:
        Σ over pitch type pt:  pit_pct(pt | eff_hand) × batter_metric_vs(pt)
    falling back from individual pitch type → bucket when the batter has thin sample.
    Returns weighted average, NaN if no usable data."""
    total = 0.0
    weight_sum = 0.0
    for pt in INDIVIDUAL_PITCH_TYPES:
        pit_pct = row.get(f'pit_pct_{pt}_vs_{eff_hand}HB_30d')
        if pit_pct is None or pd.isna(pit_pct) or pit_pct <= 0:
            continue
        # Try individual pitch type first
        bat_val = row.get(f'{metric_prefix}_{pt}_90d')
        if bat_val is None or pd.isna(bat_val):
            # Fall back to bucket
            bucket = PITCH_TO_BUCKET[pt]
            bat_val = row.get(f'{metric_prefix}_{bucket}_90d')
        if bat_val is None or pd.isna(bat_val):
            continue  # neither individual nor bucket has data
        total += float(pit_pct) * float(bat_val)
        weight_sum += float(pit_pct)
    return (total / weight_sum) if weight_sum > 0 else np.nan


def mu_xwoba_expected(row: pd.Series) -> float:
    eff = effective_batter_hand(row.get('bat_hand'), row.get('pit_throws'))
    if eff not in ('L', 'R'):
        return np.nan
    return _weighted_batter_metric(row, eff, 'bat_xwoba_vs')


def mu_whiff_expected(row: pd.Series) -> float:
    eff = effective_batter_hand(row.get('bat_hand'), row.get('pit_throws'))
    if eff not in ('L', 'R'):
        return np.nan
    return _weighted_batter_metric(row, eff, 'bat_whiff_rate_vs')


# ----------------------------------------------------------------------------
# Simple matchup features
# ----------------------------------------------------------------------------

def mu_platoon_advantage(bat_hand: str, pit_throws: str) -> int:
    """1 if opposite-hand matchup (favorable to batter), 0 otherwise.
    Switch hitters always have the platoon advantage."""
    if bat_hand == 'S':
        return 1
    if bat_hand not in ('L', 'R') or pit_throws not in ('L', 'R'):
        return 0
    return 1 if bat_hand != pit_throws else 0


# Expected PA constants. A 9-inning game faces ~38 batters total (~4.2 per slot).
# Modern starters average ~5.0-5.5 IP per start → roughly 3 trips through the order.
_PA_BY_ORDER = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2,
                6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8}


def mu_total_pa_expected(order_position: int) -> float:
    """Total expected PAs for the game, by lineup position (1-9)."""
    return _PA_BY_ORDER.get(int(order_position), np.nan) if pd.notna(order_position) else np.nan


def mu_starter_pa_expected(ip_per_start: float) -> float:
    """Expected PAs the batter sees against the starting pitcher specifically.
    Starter faces roughly 4 batters per inning (3 outs + ~1 baserunner).
    Divided across 9 lineup slots."""
    if ip_per_start is None or pd.isna(ip_per_start):
        return np.nan
    return float(ip_per_start) * 4.0 / 9.0


def mu_bullpen_pa_share(starter_pa: float, total_pa: float) -> float:
    """Fraction of expected PAs vs bullpen (1 - starter share)."""
    if pd.isna(starter_pa) or pd.isna(total_pa) or total_pa <= 0:
        return np.nan
    bullpen = max(0.0, float(total_pa) - float(starter_pa))
    return bullpen / float(total_pa)


# ----------------------------------------------------------------------------
# Vectorized assembler — apply all matchup features to a joined DataFrame
# ----------------------------------------------------------------------------

def compute_all(joined: pd.DataFrame) -> pd.DataFrame:
    """Given a DataFrame with both batter and pitcher features (plus bat_hand,
    pit_throws, ctx_batter_order_position), return the same frame with the 6
    new mu_* columns added."""
    out = joined.copy()

    # Row-wise computations for the weighted matchup metrics
    out['mu_xwoba_expected'] = out.apply(mu_xwoba_expected, axis=1)
    out['mu_whiff_expected'] = out.apply(mu_whiff_expected, axis=1)

    # Vectorized scalar features
    out['mu_platoon_advantage'] = [
        mu_platoon_advantage(b, p) for b, p in zip(out['bat_hand'], out['pit_throws'])
    ]
    out['mu_total_pa_expected'] = out['ctx_batter_order_position'].apply(mu_total_pa_expected)
    out['mu_starter_pa_expected'] = out['pit_ip_per_start_30d'].apply(mu_starter_pa_expected)
    out['mu_bullpen_pa_share'] = [
        mu_bullpen_pa_share(s, t) for s, t in
        zip(out['mu_starter_pa_expected'], out['mu_total_pa_expected'])
    ]

    return out


# ----------------------------------------------------------------------------
# Smoke test — synthesize a Walker (RHB) vs Framber (LHP) row by hand
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a minimal test row with the values we've already seen are real
    test_row = pd.Series({
        # Walker (RHB) batter side
        'bat_hand': 'R',
        'bat_xwoba_vs_SI_90d': 0.295,   # made up — RHB vs SI is usually meh
        'bat_xwoba_vs_FF_90d': 0.385,
        'bat_xwoba_vs_CU_90d': 0.350,
        'bat_xwoba_vs_CH_90d': 0.320,
        'bat_xwoba_vs_FB_90d': 0.350,   # bucket fallback
        'bat_xwoba_vs_BR_90d': 0.320,
        'bat_xwoba_vs_OS_90d': 0.310,
        'bat_whiff_rate_vs_SI_90d': 0.18,
        'bat_whiff_rate_vs_FF_90d': 0.22,
        'bat_whiff_rate_vs_CU_90d': 0.30,
        'bat_whiff_rate_vs_CH_90d': 0.28,
        'bat_whiff_rate_vs_FB_90d': 0.20,
        'bat_whiff_rate_vs_BR_90d': 0.28,
        'bat_whiff_rate_vs_OS_90d': 0.27,
        # Framber (LHP) pitcher side — mix to RHB (from earlier validation)
        'pit_throws': 'L',
        'pit_pct_SI_vs_RHB_30d': 0.460,
        'pit_pct_CU_vs_RHB_30d': 0.327,
        'pit_pct_CH_vs_RHB_30d': 0.197,
        'pit_pct_FF_vs_RHB_30d': 0.000,
        'pit_pct_FC_vs_RHB_30d': 0.000,
        'pit_pct_SL_vs_RHB_30d': 0.000,
        'pit_pct_ST_vs_RHB_30d': 0.000,
        'pit_pct_SV_vs_RHB_30d': 0.000,
        'pit_pct_KC_vs_RHB_30d': 0.000,
        'pit_pct_FS_vs_RHB_30d': 0.000,
        'pit_pct_FO_vs_RHB_30d': 0.000,
        'pit_ip_per_start_30d': 7.4,
        # Context
        'ctx_batter_order_position': 3,
    })

    print("=== Matchup features: Walker (RHB cleanup hitter) vs Framber (LHP starter) ===")
    print(f"  effective_batter_hand:    {effective_batter_hand(test_row['bat_hand'], test_row['pit_throws'])}")
    print(f"  mu_xwoba_expected:        {mu_xwoba_expected(test_row):.4f}")
    print(f"  mu_whiff_expected:        {mu_whiff_expected(test_row):.4f}")
    print(f"  mu_platoon_advantage:     {mu_platoon_advantage(test_row['bat_hand'], test_row['pit_throws'])}")
    print(f"  mu_total_pa_expected:     {mu_total_pa_expected(test_row['ctx_batter_order_position']):.2f}")
    print(f"  mu_starter_pa_expected:   {mu_starter_pa_expected(test_row['pit_ip_per_start_30d']):.2f}")
    print(f"  mu_bullpen_pa_share:      "
          f"{mu_bullpen_pa_share(mu_starter_pa_expected(test_row['pit_ip_per_start_30d']),  mu_total_pa_expected(test_row['ctx_batter_order_position'])):.3f}")

    print()
    print("=== Sanity check: same batter vs a generic 4-seam-heavy RHP (Strider-like) ===")
    strider_row = test_row.copy()
    strider_row['pit_throws'] = 'R'
    strider_row['pit_pct_SI_vs_RHB_30d'] = 0.0
    strider_row['pit_pct_CU_vs_RHB_30d'] = 0.0
    strider_row['pit_pct_CH_vs_RHB_30d'] = 0.0
    strider_row['pit_pct_FF_vs_RHB_30d'] = 0.541
    strider_row['pit_pct_SL_vs_RHB_30d'] = 0.387
    strider_row['pit_ip_per_start_30d'] = 5.5
    print(f"  effective_batter_hand:    {effective_batter_hand(strider_row['bat_hand'], strider_row['pit_throws'])}")
    print(f"  mu_xwoba_expected:        {mu_xwoba_expected(strider_row):.4f}")
    print(f"  mu_whiff_expected:        {mu_whiff_expected(strider_row):.4f}")
    print(f"  mu_platoon_advantage:     {mu_platoon_advantage(strider_row['bat_hand'], strider_row['pit_throws'])}")
    print(f"  mu_starter_pa_expected:   {mu_starter_pa_expected(strider_row['pit_ip_per_start_30d']):.2f}")
