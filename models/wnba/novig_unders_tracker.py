"""
models/wnba/novig_unders_tracker.py — blanket-unders at Novig: threes/assists/points.
See db/migrate_wnba_unders_signals.sql for thesis. Retrospective + leak-free (BettingPros
actuals). Usage: backfill | log --days 3 | report
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import argparse
from datetime import date, timedelta
import numpy as np, pandas as pd
from db.db import query, execute

MKTS = {390: 'threes', 391: 'assists', 393: 'points'}
FORWARD_START = date(2026, 7, 12)

def dec(o): return np.where(o > 0, o/100+1, 100/np.abs(o)+1)

def score(start, end):
    d = query("""SELECT prop_date, bp_player_id, market_id, over_line ln, under_odds,
        actual, is_scored FROM bettingpros_props
        WHERE book_id=60 AND market_id IN (390,391,393)
        AND under_odds IS NOT NULL AND ABS(under_odds) <= 2000
        AND prop_date >= %(s)s AND prop_date <= %(e)s""", params={'s': start, 'e': end})
    if d.empty: return d
    d['payout_dec'] = dec(d['under_odds'].values)
    scored = d['is_scored'] & d['actual'].notna()
    won = d['actual'].astype(float) < d['ln']
    d['won'] = np.where(scored, won, None)
    d['profit'] = np.where(scored, np.where(won, d['payout_dec']-1, -1.0), np.nan)
    return d

def upsert(d, backfill):
    n = 0
    for r in d.itertuples():
        execute("""INSERT INTO wnba_unders_signals
            (prop_date, bp_player_id, market_id, line, under_odds, payout_dec,
             actual, won, profit, is_backfill)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (prop_date, bp_player_id, market_id) DO UPDATE
              SET actual=EXCLUDED.actual, won=EXCLUDED.won, profit=EXCLUDED.profit""",
            (r.prop_date, int(r.bp_player_id), int(r.market_id), float(r.ln),
             int(r.under_odds), float(r.payout_dec),
             None if pd.isna(r.actual) else float(r.actual),
             None if pd.isna(r.won) else bool(r.won),
             None if pd.isna(r.profit) else float(r.profit), backfill))
        n += 1
    return n

def report():
    d = query("SELECT * FROM wnba_unders_signals WHERE won IS NOT NULL")
    if d.empty: print("no settled signals"); return
    d['profit'] = d['profit'].astype(float)
    print("=== WNBA Novig blanket-unders ===")
    for mid, nm in MKTS.items():
        for tag, mask in [('backfill', d['is_backfill']), ('FORWARD', ~d['is_backfill'])]:
            s = d[(d['market_id'] == mid) & mask]
            if len(s) < 5:
                print(f"  {nm:>8} {tag:>9}: n={len(s)}"); continue
            se = s['profit'].std()/np.sqrt(len(s))
            print(f"  {nm:>8} {tag:>9}: n={len(s):>5,} hit={s['won'].mean():.3f} "
                  f"ROI={s['profit'].mean():+.4f} (±{2*se:.3f})")

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('log'); lg.add_argument('--days', type=int, default=3)
    sub.add_parser('backfill'); sub.add_parser('report')
    a = ap.parse_args()
    if a.cmd == 'report': report(); return
    if a.cmd == 'backfill':
        d = score(date(2025, 7, 15), FORWARD_START - timedelta(days=1))
        print(f"backfill: {upsert(d, True):,} logged")
    else:
        end = date.today(); start = end - timedelta(days=a.days)
        d = score(max(start, FORWARD_START), end)
        print(f"forward: {upsert(d, False):,} logged/settled")
    report()

if __name__ == "__main__":
    main()
