"""
models/mlb/line_shopping.py — Sharp-vs-soft line discrepancy backtest.

A SECOND strategy alongside our own model (models/mlb/hitter_prop_model.py). Instead of
predicting outcomes ourselves, we treat a sharp book's de-vigged price as the "fair"
probability and bet Underdog props where Underdog's price lags that fair value.

    sharp (Novig, book_id=60)  → de-vig → fair probability  [the "truth"]
    soft  (Underdog, book_id=36) → the odds we actually bet at

Edge thesis: Underdog is a recreational DFS app; its lines can lag the sharp market.
When the sharp fair prob implies Underdog's price is off by more than Underdog's vig,
that side is +EV. This needs NO predictive model — only that the two books disagree.

Entirely within bettingpros_props (both books share bp_player_id + market + line + date),
so no player-name matching or feature dataset needed. Outcome comes from the `actual`
field BettingPros joins to each prop.

Usage:
    python -m models.mlb.line_shopping                       # all hitter markets
    python -m models.mlb.line_shopping --sharp 0             # use Consensus instead of Novig
    python -m models.mlb.line_shopping --markets 403,293,289 # specific markets
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import numpy as np
import pandas as pd

from db.db import query


SOFT_BOOK = 36   # Underdog — where we bet
SHARP_DEFAULT = 60  # Novig — the fair-price reference

MARKET_NAMES = {
    287: 'hits', 288: 'runs', 289: 'rbi', 291: 'doubles', 292: 'triples',
    293: 'total-bases', 294: 'steals', 295: 'singles', 299: 'homeruns', 403: 'runs-hits-rbis',
}
HITTER_MARKETS = [403, 293, 289, 288, 287, 295, 299, 291, 294]

EDGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]


def american_to_decimal(a: float) -> float:
    return a / 100.0 + 1.0 if a > 0 else 100.0 / abs(a) + 1.0


def _pull_overlap(market_id: int, sharp_book: int) -> pd.DataFrame:
    """One row per prop where both soft(Underdog) and sharp books quote the same
    (market, line, player, date). Includes soft odds, sharp odds, actual outcome."""
    sql = """
        SELECT
            ud.prop_date, ud.bp_player_id, ud.over_line,
            ud.over_odds  AS ud_over,  ud.under_odds  AS ud_under,
            nv.over_odds  AS sh_over,  nv.under_odds  AS sh_under,
            ud.actual, ud.is_push
        FROM bettingpros_props ud
        JOIN bettingpros_props nv
          ON ud.prop_date = nv.prop_date AND ud.market_id = nv.market_id
         AND ud.over_line = nv.over_line AND ud.bp_player_id = nv.bp_player_id
        WHERE ud.book_id = %(soft)s AND nv.book_id = %(sharp)s
          AND ud.market_id = %(mkt)s
          AND ud.is_scored = true AND ud.actual IS NOT NULL
          AND ud.over_odds IS NOT NULL AND ud.under_odds IS NOT NULL
          AND nv.over_odds IS NOT NULL AND nv.under_odds IS NOT NULL
    """
    return query(sql, params={'soft': SOFT_BOOK, 'sharp': sharp_book, 'mkt': market_id})


def _evaluate_market(market_id: int, sharp_book: int) -> pd.DataFrame:
    df = _pull_overlap(market_id, sharp_book)
    if len(df) == 0:
        return pd.DataFrame()

    # Sharp fair probability for the OVER (de-vigged)
    sh_io = 1.0 / df['sh_over'].apply(american_to_decimal)
    sh_iu = 1.0 / df['sh_under'].apply(american_to_decimal)
    df['fair_over'] = sh_io / (sh_io + sh_iu)

    # Underdog decimal payouts
    df['ud_dec_over']  = df['ud_over'].apply(american_to_decimal)
    df['ud_dec_under'] = df['ud_under'].apply(american_to_decimal)

    # EV of each side at Underdog odds, using sharp fair prob as truth
    df['ev_over']  = df['fair_over'] * (df['ud_dec_over'] - 1) - (1 - df['fair_over'])
    df['ev_under'] = (1 - df['fair_over']) * (df['ud_dec_under'] - 1) - df['fair_over']
    df['best_side'] = np.where(df['ev_over'] >= df['ev_under'], 'over', 'under')
    df['ev_best']   = np.maximum(df['ev_over'], df['ev_under'])
    df['payout_dec'] = np.where(df['best_side'] == 'over', df['ud_dec_over'], df['ud_dec_under'])

    # Outcome: half-point lines → over wins iff actual > line
    df['over_won'] = (df['actual'].astype(float) > df['over_line'].astype(float)).astype(int)
    df['bet_won']  = np.where(df['best_side'] == 'over', df['over_won'], 1 - df['over_won'])
    df['profit']   = np.where(df['bet_won'] == 1, df['payout_dec'] - 1.0, -1.0)
    df['market_id'] = market_id
    return df


def _sweep(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in EDGE_THRESHOLDS:
        sub = bets[bets['ev_best'] > thr]
        if len(sub) == 0:
            rows.append({'edge_thr': thr, 'n_bets': 0})
            continue
        daily = sub.groupby('prop_date')['profit'].sum()
        rows.append({
            'edge_thr': thr,
            'n_bets': len(sub),
            'hit_rate': sub['bet_won'].mean(),
            'roi_per_bet': sub['profit'].mean(),
            'total_profit': sub['profit'].sum(),
            'sharpe_daily': daily.mean() / daily.std() if daily.std() > 0 else np.nan,
            'over_pct': (sub['best_side'] == 'over').mean(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sharp', type=int, default=SHARP_DEFAULT,
                        help=f"Sharp reference book_id (default {SHARP_DEFAULT}=Novig; 0=Consensus)")
    parser.add_argument('--markets', type=str, default=None,
                        help="Comma-separated market_ids (default: all hitter markets)")
    args = parser.parse_args()

    markets = [int(m) for m in args.markets.split(',')] if args.markets else HITTER_MARKETS
    sharp_name = {60: 'Novig', 0: 'Consensus'}.get(args.sharp, f'book_{args.sharp}')
    print(f"Line-shopping backtest: bet Underdog where {sharp_name} implies value\n")

    all_bets = []
    for mkt in markets:
        bets = _evaluate_market(mkt, args.sharp)
        if len(bets) == 0:
            continue
        all_bets.append(bets)
        sweep = _sweep(bets)
        disp = sweep.copy()
        for c in ['hit_rate', 'roi_per_bet', 'sharpe_daily', 'over_pct']:
            if c in disp: disp[c] = disp[c].round(4)
        if 'total_profit' in disp: disp['total_profit'] = disp['total_profit'].round(1)
        print(f"=== {MARKET_NAMES.get(mkt, mkt)} (market {mkt}) — {len(bets):,} overlapping scored props ===")
        print(disp.to_string(index=False))
        print()

    if all_bets:
        combined = pd.concat(all_bets, ignore_index=True)
        print("=" * 70)
        print(f"  ALL HITTER MARKETS COMBINED — {len(combined):,} props")
        print("=" * 70)
        print(_sweep(combined).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
