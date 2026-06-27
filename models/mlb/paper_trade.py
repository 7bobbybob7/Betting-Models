"""
models/mlb/paper_trade.py — forward paper-trade harness for the +EV line-shopping leg.

The decisive "is the edge tradeable" test. Three commands:

    log     — at the latest capture, find PRE-GAME props where Novig's de-vigged fair price
              implies Underdog's odds are +EV, and log them to paper_bets (no money at risk).
    settle  — fill in outcomes for past bets from mlb_batting_game; compute realized profit.
    report  — realized ROI segmented by edge size, market, and (last-pre-game) per-prop view.

Self-contained (only db + pandas + numpy + stdlib) so it runs in the slim capture cron.
PRE-GAME FILTER is baked in: only props with scheduled_start > capture_at are ever logged,
so live/in-progress prices never contaminate the log.

Usage:
    python -m models.mlb.paper_trade log         # flag + log pre-game +EV at latest capture
    python -m models.mlb.paper_trade settle      # settle finished games
    python -m models.mlb.paper_trade report
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse, unicodedata
from datetime import datetime, timezone
import numpy as np, pandas as pd
from db.db import query, get_conn

MIN_EV = 0.0          # log every +EV opportunity (high-volume view; filter later in report)
PAIR_WINDOW_MIN = 20  # max minutes between the Novig and Underdog captures to call them aligned

# Novig market_type -> Underdog stat_type
MKT_TO_UD = {
    'HITS_RUNS_RBIS': 'Hits + Runs + RBIs', 'TOTAL_BASES': 'Total Bases', 'RBIS': 'RBIs',
    'RUNS': 'Runs', 'HITS': 'Hits', 'HOME_RUNS': 'Home Runs',
    'BATTING_WALKS': 'Batter Walks', 'STOLEN_BASES': 'Stolen Bases',
}
# Novig market_type -> how to compute the realized stat from an mlb_batting_game row
def _actual_for(market_type, r):
    h, d, t, hr = r['hits'], r['doubles'], r['triples'], r['hr']
    return {
        'HITS_RUNS_RBIS': (r['hits'] or 0) + (r['runs'] or 0) + (r['rbi'] or 0),
        'TOTAL_BASES':    (h or 0) + (d or 0) + 2 * (t or 0) + 3 * (hr or 0),
        'RBIS':  r['rbi'], 'RUNS': r['runs'], 'HITS': r['hits'],
        'HOME_RUNS': r['hr'], 'BATTING_WALKS': r['bb'], 'STOLEN_BASES': r['sb'],
    }.get(market_type)


def _norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii').lower()
    for suf in (' jr', ' sr', ' ii', ' iii', ' iv'):
        if s.endswith(suf):
            s = s[:-len(suf)]
    s = ''.join(c if c.isalnum() or c.isspace() else '' for c in s)
    return ' '.join(s.split())


def _to_decimal(a):
    return a / 100.0 + 1.0 if a > 0 else 100.0 / abs(a) + 1.0


def _novig_name_to_player_id():
    """Map normalized Novig full name -> our player_id (drop ambiguous duplicates)."""
    p = query("SELECT player_id, full_name FROM players WHERE sport_id=2 AND full_name IS NOT NULL")
    p['norm'] = p['full_name'].map(_norm)
    counts = p.groupby('norm')['player_id'].nunique()
    uniq = p[p['norm'].isin(counts[counts == 1].index)].drop_duplicates('norm')
    return dict(zip(uniq['norm'], uniq['player_id']))


# ----------------------------------------------------------------------------
# log — flag + record pre-game +EV opportunities at the latest capture
# ----------------------------------------------------------------------------

def log_bets(min_ev=MIN_EV):
    nv_t = query("SELECT MAX(captured_at) m FROM novig_snapshots").iloc[0].m
    if nv_t is None:
        print("no novig snapshots"); return 0
    # Underdog snapshot closest to the novig capture (within the pair window)
    ud_row = query("""
        SELECT snapshot_ts FROM underdog_props
        WHERE ABS(EXTRACT(EPOCH FROM (snapshot_ts - %(t)s))) <= %(w)s
        ORDER BY ABS(EXTRACT(EPOCH FROM (snapshot_ts - %(t)s))) LIMIT 1
    """, params={'t': nv_t, 'w': PAIR_WINDOW_MIN * 60})
    if len(ud_row) == 0:
        print(f"no Underdog capture within {PAIR_WINDOW_MIN}min of novig {nv_t}"); return 0
    ud_t = ud_row.iloc[0].snapshot_ts
    print(f"pairing Novig {nv_t} with Underdog {ud_t}")

    # Novig: traded (last present) AND pre-game
    nv = query("""SELECT player_name, market_type, strike, over_last, under_last, volume, scheduled_start
                  FROM novig_snapshots
                  WHERE captured_at=%(t)s AND over_last IS NOT NULL AND scheduled_start > %(t)s""",
               params={'t': nv_t})
    if len(nv) == 0:
        print("no traded pre-game Novig props at latest capture"); return 0
    nv['ud_stat'] = nv['market_type'].map(MKT_TO_UD)
    nv = nv.dropna(subset=['ud_stat'])
    nv['key'] = nv['player_name'].map(_norm) + '|' + nv['ud_stat'] + '|' + nv['strike'].astype(float).astype(str)

    ud = query("""SELECT player_first_name, player_last_name, stat_type, stat_value, choice, american_price
                  FROM underdog_props WHERE snapshot_ts=%(t)s""", params={'t': ud_t})
    ud['key'] = ((ud['player_first_name'].fillna('') + ' ' + ud['player_last_name'].fillna('')).map(_norm)
                 + '|' + ud['stat_type'] + '|' + ud['stat_value'].astype(float).astype(str))
    piv = (ud.pivot_table(index='key', columns='choice', values='american_price', aggfunc='first')
             .dropna(subset=['higher', 'lower']).reset_index())

    m = nv.merge(piv, on='key', how='inner')
    if len(m) == 0:
        print("no matched pre-game props"); return 0

    m['od'] = m['higher'].apply(_to_decimal)
    m['ud'] = m['lower'].apply(_to_decimal)
    m['ev_o'] = m['over_last'] * (m['od'] - 1) - (1 - m['over_last'])
    m['ev_u'] = m['under_last'] * (m['ud'] - 1) - (1 - m['under_last'])
    over = m['ev_o'] >= m['ev_u']
    m['side'] = np.where(over, 'OVER', 'UNDER')
    m['ev'] = np.where(over, m['ev_o'], m['ev_u'])
    m['ud_odds'] = np.where(over, m['higher'], m['lower']).astype(int)
    m['payout'] = np.where(over, m['od'], m['ud'])
    m['fair'] = np.where(over, m['over_last'], m['under_last'])

    name2id = _novig_name_to_player_id()
    m['player_id'] = m['player_name'].map(_norm).map(name2id)
    bets = m[(m['ev'] > min_ev) & (m['player_id'].notna())].copy()
    print(f"matched={len(m)}  +EV={int((m['ev']>min_ev).sum())}  loggable (player matched)={len(bets)}")
    if len(bets) == 0:
        return 0

    rows = [(nv_t, r.scheduled_start.date(), r.scheduled_start, int(r.player_id), r.player_name,
             r.market_type, float(r.strike), r.side, int(r.ud_odds), float(r.payout),
             float(r.fair), float(r.ev), float(r.volume or 0)) for r in bets.itertuples()]
    sql = """INSERT INTO paper_bets
        (capture_at, game_date, scheduled_start, player_id, player_name, market_type, line, side,
         ud_odds, ud_payout_dec, novig_fair, ev, novig_volume)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (capture_at, player_id, market_type, line, side) DO NOTHING"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows); n = cur.rowcount
        conn.commit()
    print(f"logged {n} new paper bets")
    return n


# ----------------------------------------------------------------------------
# settle — fill outcomes for finished games
# ----------------------------------------------------------------------------

def settle():
    todo = query("""SELECT DISTINCT player_id, game_date FROM paper_bets
                    WHERE settled_at IS NULL AND game_date < (now() AT TIME ZONE 'UTC')::date""")
    if len(todo) == 0:
        print("nothing to settle"); return 0
    print(f"settling {len(todo)} player-games...")

    # Pull box scores for those player-games (aggregate handles doubleheaders)
    box = query("""
        SELECT bg.player_id, g.game_date,
               SUM(bg.hits) hits, SUM(bg.doubles) doubles, SUM(bg.triples) triples, SUM(bg.hr) hr,
               SUM(bg.rbi) rbi, SUM(bg.runs) runs, SUM(bg.bb) bb, SUM(bg.sb) sb
        FROM mlb_batting_game bg JOIN games g ON bg.game_id=g.game_id
        WHERE g.sport_id=2 AND g.status='final'
        GROUP BY bg.player_id, g.game_date""")
    box['game_date'] = pd.to_datetime(box['game_date']).dt.date
    boxidx = {(int(r.player_id), r.game_date): r for r in box.itertuples()}

    bets = query("SELECT * FROM paper_bets WHERE settled_at IS NULL")
    bets['game_date'] = pd.to_datetime(bets['game_date']).dt.date
    updates, unresolved = [], 0
    for b in bets.itertuples():
        key = (int(b.player_id), b.game_date)
        if key not in boxidx:
            unresolved += 1; continue   # box score not in yet (or player DNP)
        r = boxidx[key]._asdict()
        actual = _actual_for(b.market_type, r)
        if actual is None:
            unresolved += 1; continue
        won = (actual > float(b.line)) if b.side == 'OVER' else (actual < float(b.line))
        profit = (float(b.ud_payout_dec) - 1.0) if won else -1.0
        updates.append((float(actual), bool(won), float(profit),
                        b.capture_at, b.player_id, b.market_type, float(b.line), b.side))

    if updates:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany("""UPDATE paper_bets SET actual=%s, won=%s, profit=%s, settled_at=now()
                    WHERE capture_at=%s AND player_id=%s AND market_type=%s AND line=%s AND side=%s""", updates)
            conn.commit()
    print(f"settled {len(updates)} bets | {unresolved} still unresolved (box score pending / DNP)")
    return len(updates)


# ----------------------------------------------------------------------------
# report — realized ROI by edge bucket, market, and last-pre-game-per-prop
# ----------------------------------------------------------------------------

def report():
    s = query("SELECT * FROM paper_bets WHERE settled_at IS NOT NULL")
    print(f"settled paper bets: {len(s):,}")
    if len(s) == 0:
        pend = query("SELECT COUNT(*) n FROM paper_bets").iloc[0].n
        print(f"  ({pend} logged, none settled yet)"); return
    s['profit'] = s['profit'].astype(float); s['ev'] = s['ev'].astype(float)

    print(f"\n  Overall: bets={len(s):,}  hit={s.won.mean():.3f}  ROI/bet={s.profit.mean():+.4f}  "
          f"total_units={s.profit.sum():+.1f}")

    print("\n  By EV bucket at flag time:")
    s['evb'] = pd.cut(s['ev'], [0, .02, .04, .06, .10, 1], labels=['0-2%', '2-4%', '4-6%', '6-10%', '10%+'])
    g = s.groupby('evb', observed=True).agg(bets=('profit', 'size'), hit=('won', 'mean'),
                                            roi=('profit', 'mean'))
    print(g.round(4).to_string())

    print("\n  By market:")
    g2 = s.groupby('market_type').agg(bets=('profit', 'size'), hit=('won', 'mean'), roi=('profit', 'mean'))
    print(g2.round(4).to_string())

    # Realistic single-bet view: last pre-game capture per prop
    s['prop'] = s['player_id'].astype(str)+'|'+s['market_type']+'|'+s['line'].astype(str)+'|'+s['side']
    last = s.sort_values('capture_at').groupby('prop', as_index=False).last()
    print(f"\n  Last-pre-game-per-prop (one bet per prop): bets={len(last):,}  "
          f"hit={last.won.mean():.3f}  ROI/bet={last.profit.mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "settle", "report"])
    ap.add_argument("--min-ev", type=float, default=MIN_EV)
    args = ap.parse_args()
    if args.cmd == "log":
        log_bets(args.min_ev)
    elif args.cmd == "settle":
        settle()
    else:
        report()


if __name__ == "__main__":
    main()
