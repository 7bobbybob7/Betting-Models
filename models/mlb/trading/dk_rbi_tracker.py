"""
models/mlb/trading/dk_rbi_tracker.py — forward tracker for the venue-sweep survivor:
DraftKings RBI 0.5 singles vs Novig de-vigged fair.

Backtest (Apr-Jun 2026): +8.5% ROI on 243 EV>2% bets, positive all 3 months — the only
book x market that survived the cross-book sweep vs Novig.

Both DK (book 12) and Novig (book 60) land via the daily BettingPros pull under the same
bp_player_id, so this is a retrospective + leak-free tracker: EV from lines that were live
pre-game, settled from BettingPros' own `actual`. No live capture, no name matching.

Usage:
    python -m models.mlb.trading.dk_rbi_tracker log --days 3     # cron: settle recent
    python -m models.mlb.trading.dk_rbi_tracker backfill          # re-log Apr-Jun baseline
    python -m models.mlb.trading.dk_rbi_tracker report
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import argparse
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd

from db.db import query, execute
from models.mlb.hitter.backtest import american_to_decimal

DK_BOOK, NOVIG_BOOK, RBI_MKT, LINE = 12, 60, 289, 0.5
MIN_EV = 0.02
FORWARD_START = date(2026, 7, 9)      # rows before this are backfill baseline


def _score_range(start: date, end: date) -> pd.DataFrame:
    """Join DK RBI 0.5 odds to Novig fair for [start, end], compute side/EV/outcome."""
    dk = query("""
        SELECT prop_date, bp_player_id,
               player_first_name || ' ' || player_last_name AS player_name,
               over_odds, under_odds, actual, is_scored
        FROM bettingpros_props
        WHERE book_id=%(b)s AND market_id=%(m)s AND over_line=%(l)s
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND prop_date >= %(s)s AND prop_date <= %(e)s""",
        params={'b': DK_BOOK, 'm': RBI_MKT, 'l': LINE, 's': start, 'e': end})
    nv = query("""
        SELECT prop_date, bp_player_id, over_odds AS nv_o, under_odds AS nv_u
        FROM bettingpros_props
        WHERE book_id=%(b)s AND market_id=%(m)s AND over_line=%(l)s
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND prop_date >= %(s)s AND prop_date <= %(e)s""",
        params={'b': NOVIG_BOOK, 'm': RBI_MKT, 'l': LINE, 's': start, 'e': end})
    j = dk.merge(nv, on=['prop_date', 'bp_player_id'], how='inner')
    if j.empty:
        return j
    nio = j['nv_o'].apply(american_to_decimal); niu = j['nv_u'].apply(american_to_decimal)
    fo = (1 / nio) / (1 / nio + 1 / niu)              # Novig fair P(over)
    do = j['over_odds'].apply(american_to_decimal); du = j['under_odds'].apply(american_to_decimal)
    ev_o = fo * (do - 1) - (1 - fo); ev_u = (1 - fo) * (du - 1) - fo
    over = ev_o >= ev_u
    j['side'] = np.where(over, 'OVER', 'UNDER')
    j['dk_odds'] = np.where(over, j['over_odds'], j['under_odds'])
    j['payout_dec'] = np.where(over, do, du)
    j['novig_fair'] = np.where(over, fo, 1 - fo)
    j['ev'] = np.where(over, ev_o, ev_u)
    a = j['actual'].astype(float)
    scored = j['is_scored'] & j['actual'].notna()
    won = np.where(over, a > LINE, a < LINE)
    j['won'] = np.where(scored, won, None)
    j['profit'] = np.where(scored, np.where(won, j['payout_dec'] - 1, -1.0), np.nan)
    j['actual'] = np.where(scored, a, np.nan)
    return j[j['ev'] > MIN_EV].copy()


def _upsert(df: pd.DataFrame, backfill: bool):
    n = 0
    for r in df.itertuples():
        won = None if pd.isna(r.won) else bool(r.won)
        profit = None if pd.isna(r.profit) else float(r.profit)
        actual = None if pd.isna(r.actual) else float(r.actual)
        execute("""
            INSERT INTO dk_rbi_signals (prop_date, bp_player_id, player_name, dk_line, side,
                dk_odds, payout_dec, novig_fair, ev, actual, won, profit, is_backfill)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (prop_date, bp_player_id) DO UPDATE
              SET actual=EXCLUDED.actual, won=EXCLUDED.won, profit=EXCLUDED.profit
        """, (r.prop_date, int(r.bp_player_id), r.player_name, LINE, r.side,
              int(r.dk_odds), float(r.payout_dec), round(float(r.novig_fair), 5),
              round(float(r.ev), 4), actual, won, profit, backfill))
        n += 1
    return n


def report():
    d = query("SELECT * FROM dk_rbi_signals WHERE won IS NOT NULL")
    if d.empty:
        print("no settled DK RBI signals yet"); return
    d['profit'] = d['profit'].astype(float); d['ev'] = d['ev'].astype(float)
    d['prop_date'] = pd.to_datetime(d['prop_date'])
    print("=== DK RBI 0.5 singles vs Novig fair — realized ===")
    for tag, mask in [('backfill (Apr-Jun baseline)', d['is_backfill']),
                      ('FORWARD (>= 2026-07-09)', ~d['is_backfill'])]:
        s = d[mask]
        if s.empty:
            print(f"  {tag}: none yet"); continue
        for thr in (0.02, 0.04):
            x = s[s['ev'] > thr]
            if len(x) < 5: continue
            se = x['profit'].std() / np.sqrt(len(x))
            print(f"  {tag:32s} ev>{thr:.2f}: n={len(x):>4} hit={x['won'].mean():.3f} "
                  f"ROI={x['profit'].mean():+.4f} (±{2*se:.3f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('log'); lg.add_argument('--days', type=int, default=3)
    sub.add_parser('backfill'); sub.add_parser('report')
    args = ap.parse_args()
    if args.cmd == 'report':
        report(); return
    if args.cmd == 'backfill':
        df = _score_range(date(2026, 4, 1), FORWARD_START - timedelta(days=1))
        print(f"backfill baseline: {_upsert(df, backfill=True):,} EV>2% signals logged")
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=args.days - 1)
        df = _score_range(start, end)
        df = df[pd.to_datetime(df['prop_date']).dt.date >= FORWARD_START]
        print(f"forward: {_upsert(df, backfill=False):,} signals logged/settled "
              f"({start}..{end})")
    report()


if __name__ == "__main__":
    main()
