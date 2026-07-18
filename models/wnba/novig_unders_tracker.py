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

def score_model(d):
    """Attach p_model (P(over) from the accepted points bundle) to market-393 rows."""
    import pickle
    path = os.path.join(os.path.dirname(__file__), "saved/wnba_points_lc.pkl")
    if not os.path.exists(path) or (d['market_id'] == 393).sum() == 0:
        d['p_model'] = np.nan; return d
    b = pickle.load(open(path, 'rb'))
    from models.wnba.prop_model_v2_gate import build_dataset, norm
    from models.wnba.batch6_gate import add_h2h
    import pandas as _pd
    ds = build_dataset()
    ds['fn'] = ds['full_name'].map(norm)
    ds = add_h2h(ds[ds['points'].notna()], 'points')
    feats = b['features']
    te = ds[['fn', 'game_date'] + feats].rename(columns={'game_date': 'gd'})
    tes = te.copy(); tes['gd'] = tes['gd'] - _pd.Timedelta(days=1)
    P = query("""SELECT prop_date, bp_player_id,
        LOWER(player_first_name||' '||player_last_name) nm
        FROM bettingpros_props WHERE book_id=60 AND market_id=393""")
    P['fn'] = P['nm'].map(norm); P['gd'] = _pd.to_datetime(P['prop_date'])
    M = _pd.concat([P.merge(te, on=['fn', 'gd']), P.merge(tes, on=['fn', 'gd'])])           .drop_duplicates(['prop_date', 'bp_player_id'])
    d = d.merge(M[['prop_date', 'bp_player_id'] + feats], on=['prop_date', 'bp_player_id'], how='left')
    mask = (d['market_id'] == 393) & d[feats[0]].notna()
    d['p_model'] = np.nan
    if mask.sum():
        X = d.loc[mask, feats].fillna(b['medians']).copy()
        X['line'] = d.loc[mask, 'ln'].values
        d.loc[mask, 'p_model'] = b['model'].predict_proba(X[feats + ['line']].values)[:, 1]
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
        pm = getattr(r, 'p_model', None)
        if pm is not None and not pd.isna(pm):
            execute("""UPDATE wnba_unders_signals SET p_model=%s, model_version='wnba-v2'
                WHERE prop_date=%s AND bp_player_id=%s AND market_id=%s""",
                (round(float(pm), 5), r.prop_date, int(r.bp_player_id), int(r.market_id)))
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

def report_ranked():
    d = query("""SELECT * FROM wnba_unders_signals WHERE won IS NOT NULL AND p_model IS NOT NULL
                 AND market_id=393""")
    if len(d) < 100: return
    d['profit'] = d['profit'].astype(float); d['p_model'] = d['p_model'].astype(float)
    d['agree'] = d['p_model'] < 0.5      # model also leans under
    print("\n=== points unders x model ranking ===")
    for tag, s_ in [('model AGREES (p<0.5)', d[d['agree']]), ('model disagrees', d[~d['agree']])]:
        if len(s_) < 20: continue
        se = s_['profit'].std()/np.sqrt(len(s_))
        print(f"  {tag:22s}: n={len(s_):>5,} ROI={s_['profit'].mean():+.4f} (±{2*se:.3f})")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('log'); lg.add_argument('--days', type=int, default=3)
    sub.add_parser('backfill'); sub.add_parser('report')
    a = ap.parse_args()
    if a.cmd == 'report': report(); report_ranked(); return
    if a.cmd == 'backfill':
        d = score_model(score(date(2025, 7, 15), FORWARD_START - timedelta(days=1)))
        print(f"backfill: {upsert(d, True):,} logged")
    else:
        end = date.today(); start = end - timedelta(days=a.days)
        d = score_model(score(max(start, FORWARD_START), end))
        print(f"forward: {upsert(d, False):,} logged/settled")
    report(); report_ranked()

if __name__ == "__main__":
    main()
