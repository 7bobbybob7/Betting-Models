"""
models/mlb/v3_tracker.py — forward out-of-sample tracking for the v3 TB model.

Retrospective daily scoring: the morning after games, build leak-safe features as-of each
game day, score the saved v3 bundle, compare to Underdog's de-vigged TB 1.5 price (the
same market anchor the validated residual test used), and log to v3_signals with the
realized outcome. No live infra needed; statistically identical to pre-game scoring
because every feature is strictly < game_date by construction.

2026 H1 was Attack 3's validation year — clean forward OOS starts 2026-07.
Production use of the signal is FILTERING Leg 2 bets, not standalone betting.

Usage:
    python -m models.mlb.v3_tracker log --start 2026-07-01 --end 2026-07-03
    python -m models.mlb.v3_tracker log --days 2          # cron: yesterday + safety overlap
    python -m models.mlb.v3_tracker report
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse, contextlib, io as _io
from datetime import date, timedelta, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from db.db import query, execute
from models.mlb.hitter_prop_dataset import build_dataset
from models.mlb.advanced_profile_features import build_training_set as build_adv
from models.mlb.backtest import (load_bundle, predict_proba, attach_odds,
                                 american_to_decimal, TARGET_TO_MARKET)

SAVED = Path("models/mlb/saved")


def log_range(start: date, end: date):
    cfg = TARGET_TO_MARKET['tb']
    line = float(cfg['over_line'])

    print(f"[v3_tracker] building features {start} -> {end}...")
    from models.mlb.luck_gap_gate import build_luck
    with contextlib.redirect_stdout(_io.StringIO()):
        ds = build_dataset(start, end)
        adv = build_adv(start, end)
        lk = build_luck(start, end)
    ds['game_date'] = pd.to_datetime(ds['game_date'])
    ds = ds.merge(adv, on=['game_id', 'player_id'], how='left') \
           .merge(lk, on=['game_id', 'player_id'], how='left')
    print(f"  {len(ds):,} batter-games")
    if ds.empty:
        print("  nothing to score"); return

    bundle = load_bundle('tb', 'xgb_v3', SAVED)
    ds['p_v3'] = predict_proba(bundle, ds[bundle['features']])

    with contextlib.redirect_stdout(_io.StringIO()):
        B = attach_odds(ds, 'tb', start, end)
    if B.empty:
        print("  no Underdog TB 1.5 props found for range"); return
    io_ = 1 / B['over_odds'].apply(american_to_decimal)
    iu_ = 1 / B['under_odds'].apply(american_to_decimal)
    B['p_mkt'] = io_ / (io_ + iu_)
    B['edge'] = B['p_v3'] - B['p_mkt']
    over = B['edge'] > 0
    B['side'] = np.where(over, 'OVER', 'UNDER')
    B['odds'] = np.where(over, B['over_odds'], B['under_odds'])
    B['payout_dec'] = B['odds'].apply(american_to_decimal)

    n_ins = 0
    for r in B.itertuples():
        actual = float(r.actual) if r.is_scored and pd.notna(r.actual) else None
        won = profit = None
        if actual is not None:
            won = (actual > line) if r.side == 'OVER' else (actual < line)
            profit = (float(r.payout_dec) - 1.0) if won else -1.0
        execute("""
            INSERT INTO v3_signals (game_date, game_id, player_id, line, p_v3, p_mkt,
                                    edge, side, odds, payout_dec, actual, won, profit,
                                    model_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (game_date, player_id) DO UPDATE
              SET p_v3 = EXCLUDED.p_v3, p_mkt = EXCLUDED.p_mkt, edge = EXCLUDED.edge,
                  side = EXCLUDED.side, odds = EXCLUDED.odds,
                  payout_dec = EXCLUDED.payout_dec, actual = EXCLUDED.actual,
                  won = EXCLUDED.won, profit = EXCLUDED.profit,
                  model_version = EXCLUDED.model_version
        """, (r.game_date.date(), int(r.game_id), int(r.player_id), line,
              round(float(r.p_v3), 5), round(float(r.p_mkt), 5), round(float(r.edge), 5),
              r.side, int(r.odds), float(r.payout_dec), actual, won, profit,
              bundle.get('version', 'v3')))
        n_ins += 1
    print(f"  logged/updated {n_ins:,} signals "
          f"({(B['edge'].abs() >= 0.03).sum()} with |edge|>=3%)")


def report():
    d = query("SELECT * FROM v3_signals WHERE won IS NOT NULL")
    if d.empty:
        print("no settled signals yet"); return
    d['profit'] = d['profit'].astype(float); d['edge'] = d['edge'].astype(float)
    print(f"=== v3 TB signal — forward OOS since {d['game_date'].min()} ===")
    print(f"{'filter':>12s} {'n':>6s} {'hit':>6s} {'ROI':>8s} {'±2SE':>7s}")
    for thr in [0.0, 0.02, 0.03, 0.05]:
        s = d[d['edge'].abs() >= thr]
        if len(s) < 5: continue
        se = s['profit'].std() / np.sqrt(len(s))
        print(f"  |edge|>={thr:.2f} {len(s):>6,} {s['won'].mean():>6.3f} "
              f"{s['profit'].mean():>+8.4f} {2*se:>7.3f}")
    # ---- walk-forward blend view: the pre-registered standalone candidate ----
    # Reconstructs "v6 + 90d trailing blend, ev>threshold" from stored signals.
    # June rows are warm-up (blend-fit only); the clean forward record starts 2026-07-01.
    from sklearn.linear_model import LogisticRegression
    from models.mlb.backtest import american_to_decimal as a2d
    H = query("""SELECT game_date, p_v3, p_mkt, odds, side, actual, won
                 FROM v3_signals WHERE won IS NOT NULL ORDER BY game_date""")
    if len(H) > 200:
        H['game_date'] = pd.to_datetime(H['game_date'])
        H['y'] = (H['actual'].astype(float) > 1.5).astype(int)
        for c in ('p_v3', 'p_mkt'):
            H[c] = H[c].astype(float)
        rows = []
        for d0 in sorted(H.loc[H['game_date'] >= '2026-07-01', 'game_date'].unique()):
            w = H[(H['game_date'] >= d0 - pd.Timedelta(days=90)) & (H['game_date'] < d0)]
            if len(w) < 200 or w['y'].nunique() < 2:
                continue
            lm = LogisticRegression(max_iter=1000).fit(w[['p_mkt', 'p_v3']], w['y'])
            t = H[H['game_date'] == d0].copy()
            t['pb'] = lm.predict_proba(t[['p_mkt', 'p_v3']])[:, 1]
            rows.append(t)
        if rows:
            T = pd.concat(rows)
            # EV vs the stored bet-side odds is only available for the logged side; use
            # symmetric approximation via p_mkt-implied fair for the other side.
            dec = T['odds'].astype(float).apply(a2d)
            p_side = np.where(T['side'] == 'OVER', T['pb'], 1 - T['pb'])
            T['ev_b'] = p_side * (dec - 1) - (1 - p_side)
            T['prof_b'] = np.where(
                (T['side'] == 'OVER') == (T['y'] == 1), dec - 1, -1.0)
            print("\n=== WALK-FORWARD BLEND (pre-registered standalone candidate, forward) ===")
            for thr in (0.02, 0.04):
                s = T[T['ev_b'] > thr]
                if len(s) < 5: continue
                print(f"  ev>{thr:.2f}: n={len(s):>4}  hit={(s['prof_b']>0).mean():.3f}  "
                      f"ROI={s['prof_b'].mean():+.4f}")

    # ---- filter view: paper-trade TB bets split by model agreement ----
    # (v4_backtest.py showed agree-ROI > disagree-ROI in BOTH backtest years;
    #  this tracks the same split on the live paper-trade, forward.)
    pb = query("""
        SELECT pb.game_date, pb.side AS bet_side, pb.profit, s.side AS model_side
        FROM paper_bets pb
        JOIN v3_signals s ON s.game_date = pb.game_date AND s.player_id = pb.player_id
        WHERE pb.market_type = 'TOTAL_BASES' AND pb.line = 1.5 AND pb.profit IS NOT NULL
    """)
    if not pb.empty:
        pb['profit'] = pb['profit'].astype(float)
        agree = pb['bet_side'].str.upper() == pb['model_side'].str.upper()
        print("\n=== paper-trade TB 1.5 bets x model agreement (forward) ===")
        for name, s in [('model AGREES', pb[agree]), ('model disagrees', pb[~agree])]:
            if len(s):
                print(f"  {name:16s} n={len(s):>4}  ROI={s['profit'].mean():+.4f}")
    print("\n(Signal validated on 2025/2026-H1; this table is the FORWARD confirmation."
          "\n Production use: filter Leg 2 line-shopping bets, not standalone.)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    lg = sub.add_parser('log')
    lg.add_argument('--start'); lg.add_argument('--end')
    lg.add_argument('--days', type=int, help="log the trailing N days (cron mode)")
    sub.add_parser('report')
    args = ap.parse_args()
    if args.cmd == 'log':
        if args.days:
            end = date.today() - timedelta(days=1)
            start = end - timedelta(days=args.days - 1)
        else:
            start = datetime.strptime(args.start, "%Y-%m-%d").date()
            end = datetime.strptime(args.end, "%Y-%m-%d").date()
        log_range(start, end)
    else:
        report()


if __name__ == "__main__":
    main()
