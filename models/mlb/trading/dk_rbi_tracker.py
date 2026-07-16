"""
models/mlb/trading/dk_rbi_tracker.py — forward tracker for DraftKings soft-line singles
vs Novig de-vigged fair. DK systematically lags Novig on LOW-LINE COUNTING props:

    RBI 0.5   backtest Apr-Jun +8.5% (n=243, 3/4 months positive)
    Runs 0.5  backtest Apr-Jun +11.9% (n=139, 4/4 months positive)

Cross-book check (2026-07): the edge does NOT replicate on Fliff (RBI −11%, hugs Novig),
so it's DK-specific pricing softness, not a market-wide RBI/runs mispricing — bettable at
DK only. Both DK (12) and Novig (60) arrive via the daily BettingPros pull under the same
bp_player_id, so scoring is retrospective + leak-free (EV from pre-game lines, settled
from BettingPros' `actual`). Exploratory sweep (~9 cells) => forward record is the arbiter.

Usage:
    python -m models.mlb.trading.dk_rbi_tracker log --days 3     # cron
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

DK_BOOK, NOVIG_BOOK = 12, 60
MIN_EV = 0.02
FORWARD_START = date(2026, 7, 9)      # rows before this are backfill baseline
# tracked DK soft markets: (market_id, fixed_line_or_None, label). None => varying line,
# matched DK<->Novig per row (e.g. outs-recorded ~16, strikeout-adjacent innings props).
MARKETS = [(289, 0.5, 'RBI'), (288, 0.5, 'RUNS'), (405, None, 'OUTS')]


def _score_market(mid: int, line, mkt: str, start: date, end: date) -> pd.DataFrame:
    """Join DK odds to Novig fair for one market over [start, end]. `line` fixed (filter)
    or None (varying — match DK<->Novig on the same over_line per row)."""
    lf = "" if line is None else "AND over_line=%(l)s"
    prm = {'b': DK_BOOK, 'm': mid, 's': start, 'e': end}
    if line is not None:
        prm['l'] = line
    dk = query(f"""
        SELECT prop_date, bp_player_id, over_line,
               player_first_name || ' ' || player_last_name AS player_name,
               over_odds, under_odds, actual, is_scored
        FROM bettingpros_props
        WHERE book_id=%(b)s AND market_id=%(m)s {lf}
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND prop_date >= %(s)s AND prop_date <= %(e)s""", params=prm)
    prm2 = {'b': NOVIG_BOOK, 'm': mid, 's': start, 'e': end}
    if line is not None:
        prm2['l'] = line
    nv = query(f"""
        SELECT prop_date, bp_player_id, over_line, over_odds AS nv_o, under_odds AS nv_u
        FROM bettingpros_props
        WHERE book_id=%(b)s AND market_id=%(m)s {lf}
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
          AND prop_date >= %(s)s AND prop_date <= %(e)s""", params=prm2)
    j = dk.merge(nv, on=['prop_date', 'bp_player_id', 'over_line'], how='inner')  # same line
    if j.empty:
        return j
    j['market'] = mkt; j['dk_line'] = j['over_line']
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
    ln = j['over_line'].astype(float)
    scored = j['is_scored'] & j['actual'].notna()
    won = np.where(over, a > ln, a < ln)
    j['won'] = np.where(scored, won, None)
    j['profit'] = np.where(scored, np.where(won, j['payout_dec'] - 1, -1.0), np.nan)
    j['actual'] = np.where(scored, a, np.nan)
    return j[j['ev'] > MIN_EV].copy()


def _score_range(start: date, end: date) -> pd.DataFrame:
    parts = [_score_market(mid, line, mkt, start, end) for mid, line, mkt in MARKETS]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _upsert(df: pd.DataFrame, backfill: bool):
    n = 0
    for r in df.itertuples():
        won = None if pd.isna(r.won) else bool(r.won)
        profit = None if pd.isna(r.profit) else float(r.profit)
        actual = None if pd.isna(r.actual) else float(r.actual)
        execute("""
            INSERT INTO dk_rbi_signals (prop_date, bp_player_id, market, player_name, dk_line,
                side, dk_odds, payout_dec, novig_fair, ev, actual, won, profit, is_backfill)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (prop_date, bp_player_id, market) DO UPDATE
              SET actual=EXCLUDED.actual, won=EXCLUDED.won, profit=EXCLUDED.profit
        """, (r.prop_date, int(r.bp_player_id), r.market, r.player_name, float(r.dk_line),
              r.side, int(r.dk_odds), float(r.payout_dec), round(float(r.novig_fair), 5),
              round(float(r.ev), 4), actual, won, profit, backfill))
        n += 1
    return n


def report():
    d = query("SELECT * FROM dk_rbi_signals WHERE won IS NOT NULL")
    if d.empty:
        print("no settled DK signals yet"); return
    d['profit'] = d['profit'].astype(float); d['ev'] = d['ev'].astype(float)
    print("=== DK soft-line singles vs Novig fair — realized ===")
    for mkt in sorted(d['market'].unique()):
        dm = d[d['market'] == mkt]
        for tag, mask in [('backfill', dm['is_backfill']), ('FORWARD', ~dm['is_backfill'])]:
            s = dm[mask]
            if s.empty:
                print(f"  {mkt:5s} {tag:9s}: none yet"); continue
            x = s[s['ev'] > 0.02]
            if len(x) < 5:
                print(f"  {mkt:5s} {tag:9s}: n={len(x)} (thin)"); continue
            se = x['profit'].std() / np.sqrt(len(x))
            print(f"  {mkt:5s} {tag:9s} ev>0.02: n={len(x):>4} hit={x['won'].mean():.3f} "
                  f"ROI={x['profit'].mean():+.4f} (±{2*se:.3f})")


def show():
    """Today's actionable DK board: +EV singles (RBI/RUNS/OUTS) vs Novig fair, pre-game.
    Relies on the morning BettingPros pull having captured today's lines."""
    today = date.today()
    df = _score_range(today, today)
    if df.empty:
        print("no +EV DK props for today (has the morning BettingPros pull run?)"); return
    df = df.sort_values('ev', ascending=False)
    print(f"DK singles board — {today} — {len(df)} props with EV>2% vs Novig fair")
    print(f"{'mkt':>5} {'player':<24} {'line':>5} {'side':>6} {'odds':>6} {'fair':>6} {'EV':>6}")
    for r in df.itertuples():
        print(f"{r.market:>5} {r.player_name:<24.24} {float(r.dk_line):>5.1f} {r.side:>6} "
              f"{int(r.dk_odds):>+6d} {float(r.novig_fair):>6.3f} {float(r.ev)*100:>5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('log'); lg.add_argument('--days', type=int, default=3)
    sub.add_parser('backfill'); sub.add_parser('report'); sub.add_parser('show')
    args = ap.parse_args()
    if args.cmd == 'show':
        show(); return
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
