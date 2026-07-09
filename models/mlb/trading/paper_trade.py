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
    python -m models.mlb.trading.paper_trade log         # flag + log pre-game +EV at latest capture
    python -m models.mlb.trading.paper_trade settle      # settle finished games
    python -m models.mlb.trading.paper_trade report
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

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

    # Novig: PRE-GAME with a live TWO-SIDED order book (both available prices present).
    # Fair value = bid/ask midpoint of `available` — NOT the last-traded price, which can
    # be hours stale and manufactures false +EV. (Validated: last -> -12.6% ROI,
    # available -> +0.8%, mid -> +10.5% on the same data.) Two-sided requirement also
    # acts as a liquidity filter.
    nv = query("""SELECT player_name, market_type, strike, over_available, under_available,
                         volume, scheduled_start
                  FROM novig_snapshots
                  WHERE captured_at=%(t)s AND scheduled_start > %(t)s
                    AND over_available IS NOT NULL AND under_available IS NOT NULL""",
               params={'t': nv_t})
    if len(nv) == 0:
        print("no pre-game Novig props with a two-sided book at latest capture"); return 0
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
    # Fair P(over) = midpoint of Novig's two-sided book: (ask_over + (1 - ask_under)) / 2
    m['fair_over'] = (m['over_available'] + (1 - m['under_available'])) / 2
    m['ev_o'] = m['fair_over'] * (m['od'] - 1) - (1 - m['fair_over'])
    m['ev_u'] = (1 - m['fair_over']) * (m['ud'] - 1) - m['fair_over']
    over = m['ev_o'] >= m['ev_u']
    m['side'] = np.where(over, 'OVER', 'UNDER')
    m['ev'] = np.where(over, m['ev_o'], m['ev_u'])
    m['ud_odds'] = np.where(over, m['higher'], m['lower']).astype(int)
    m['payout'] = np.where(over, m['od'], m['ud'])
    m['fair'] = np.where(over, m['fair_over'], 1 - m['fair_over'])

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


# ----------------------------------------------------------------------------
# show — current actionable +EV bets (pre-game, freshest line per prop)
# ----------------------------------------------------------------------------

def show(min_ev=0.0, min_volume=0.0, top=30):
    rows = query("""SELECT player_name, market_type, line, side, ud_odds, novig_fair, ev,
                           novig_volume, scheduled_start, capture_at
                    FROM paper_bets
                    WHERE settled_at IS NULL AND scheduled_start > now()""")
    if len(rows) == 0:
        print("No upcoming pre-game +EV bets logged right now.")
        print("(Populated each capture tick — check after the next cron run, or run `log`.)")
        return
    # freshest line per prop (latest capture before game)
    rows = (rows.sort_values('capture_at')
                .groupby(['player_name', 'market_type', 'line', 'side'], as_index=False).last())
    rows = rows[(rows['ev'] >= min_ev) & (rows['novig_volume'] >= min_volume)]
    rows = rows.sort_values('ev', ascending=False)
    if len(rows) == 0:
        print(f"No upcoming bets pass ev>={min_ev:.0%}, volume>={min_volume:.0f}.")
        return

    now = pd.Timestamp.now(tz='UTC')
    print(f"Upcoming +EV bets — {len(rows)} (showing top {min(top,len(rows))}); "
          f"ev>={min_ev:.0%}, vol>={min_volume:.0f}\n")
    print(f"{'EV':>6s} {'side':>5s} {'player':22s} {'market':16s} {'ln':>4s} "
          f"{'UDodds':>7s} {'fair':>5s} {'vol':>7s} {'1st pitch ET':>12s} {'in':>5s}")
    print("-"*100)
    for r in rows.head(top).itertuples():
        sp = pd.Timestamp(r.scheduled_start)
        mins = int((sp - now).total_seconds() / 60)
        eta = f"{mins//60}h{mins%60:02d}m" if mins >= 60 else f"{mins}m"
        print(f"{r.ev*100:>5.1f}% {r.side:>5s} {r.player_name[:22]:22s} {r.market_type[:16]:16s} "
              f"{float(r.line):>4.1f} {int(r.ud_odds):>+7d} {float(r.novig_fair):>5.2f} "
              f"{float(r.novig_volume or 0):>7.0f} {sp.tz_convert('America/New_York').strftime('%m-%d %H:%M'):>12s} {eta:>5s}")


# ----------------------------------------------------------------------------
# reprocess — re-derive bets from stored snapshots under different fair-value
# sources (last vs available vs mid) + a liquidity filter, settle in-memory,
# and compare. Lets us test signal fixes on data already captured.
# ----------------------------------------------------------------------------

def _boxscore_lookup():
    box = query("""
        SELECT bg.player_id, g.game_date,
               SUM(bg.hits) hits, SUM(bg.doubles) doubles, SUM(bg.triples) triples, SUM(bg.hr) hr,
               SUM(bg.rbi) rbi, SUM(bg.runs) runs, SUM(bg.bb) bb, SUM(bg.sb) sb
        FROM mlb_batting_game bg JOIN games g ON bg.game_id=g.game_id
        WHERE g.sport_id=2 AND g.status='final'
        GROUP BY bg.player_id, g.game_date""")
    box['game_date'] = pd.to_datetime(box['game_date']).dt.date
    return {(int(r.player_id), r.game_date): r._asdict() for r in box.itertuples()}


def _fair_over(row, src):
    """Novig fair P(over) from the chosen source. Returns None if unavailable."""
    ol, ul = row.get('over_last'), row.get('under_last')
    oa, ua = row.get('over_available'), row.get('under_available')
    if src == 'last':
        return float(ol) if pd.notna(ol) else None
    if src == 'available':                       # two-sided live book required
        if pd.notna(oa) and pd.notna(ua) and (oa + ua) > 0:
            return float(oa) / float(oa + ua)
        return None
    if src == 'mid':                             # midpoint of best bid/ask
        if pd.notna(oa) and pd.notna(ua):
            return (float(oa) + (1 - float(ua))) / 2
        return None
    return None


def reprocess(src='available', min_volume=0.0, min_ev=MIN_EV):
    name2id = _novig_name_to_player_id()
    box = _boxscore_lookup()
    caps = query("SELECT DISTINCT captured_at FROM novig_snapshots ORDER BY captured_at")
    rows = []
    for cap in caps['captured_at']:
        ud = query("""SELECT snapshot_ts FROM underdog_props
            WHERE ABS(EXTRACT(EPOCH FROM (snapshot_ts-%(t)s))) <= %(w)s
            ORDER BY ABS(EXTRACT(EPOCH FROM (snapshot_ts-%(t)s))) LIMIT 1""",
            params={'t': cap, 'w': PAIR_WINDOW_MIN*60})
        if len(ud) == 0:
            continue
        ud_t = ud.iloc[0].snapshot_ts
        nv = query("""SELECT player_name, market_type, strike, game_date,
            over_last, under_last, over_available, under_available, volume
            FROM novig_snapshots WHERE captured_at=%(t)s AND scheduled_start > %(t)s""", params={'t': cap})
        if len(nv) == 0:
            continue
        nv['ud_stat'] = nv['market_type'].map(MKT_TO_UD); nv = nv.dropna(subset=['ud_stat'])
        nv['key'] = nv['player_name'].map(_norm)+'|'+nv['ud_stat']+'|'+nv['strike'].astype(float).astype(str)
        udp = query("""SELECT player_first_name,player_last_name,stat_type,stat_value,choice,american_price
            FROM underdog_props WHERE snapshot_ts=%(t)s""", params={'t': ud_t})
        udp['key'] = ((udp['player_first_name'].fillna('')+' '+udp['player_last_name'].fillna('')).map(_norm)
                      +'|'+udp['stat_type']+'|'+udp['stat_value'].astype(float).astype(str))
        piv = (udp.pivot_table(index='key', columns='choice', values='american_price', aggfunc='first')
                  .dropna(subset=['higher','lower']).reset_index())
        m = nv.merge(piv, on='key', how='inner')
        for r in m.itertuples():
            d = r._asdict()
            if (d.get('volume') or 0) < min_volume:
                continue
            fair = _fair_over(d, src)
            if fair is None:
                continue
            od, udd = _to_decimal(d['higher']), _to_decimal(d['lower'])
            ev_o = fair*(od-1) - (1-fair); ev_u = (1-fair)*(udd-1) - fair
            side = 'OVER' if ev_o >= ev_u else 'UNDER'
            ev = max(ev_o, ev_u)
            if ev <= min_ev:
                continue
            pid = name2id.get(_norm(d['player_name']))
            gd = pd.to_datetime(d['game_date']).date() if d.get('game_date') is not None else None
            if pid is None or (pid, gd) not in box:
                continue                          # unsettleable
            b = box[(pid, gd)]
            actual = _actual_for(d['market_type'], b)
            if actual is None:
                continue
            won = (actual > float(d['strike'])) if side == 'OVER' else (actual < float(d['strike']))
            payout = od if side == 'OVER' else udd
            rows.append({'capture_at': cap, 'game_date': gd, 'player_id': pid,
                         'market': d['market_type'], 'line': float(d['strike']), 'side': side,
                         'ev': ev, 'volume': float(d.get('volume') or 0),
                         'won': bool(won), 'profit': (payout-1.0) if won else -1.0})
    return pd.DataFrame(rows)


def compare_sources():
    box_n = len(_boxscore_lookup())
    print(f"(box-score player-games available: {box_n:,})\n")
    for src in ['last', 'available', 'mid']:
        d = reprocess(src)
        if len(d) == 0:
            print(f"{src:10s}: no settleable bets"); continue
        d['evb'] = pd.cut(d['ev'], [0,.02,.04,.06,1], labels=['0-2','2-4','4-6','6+'])
        by = d.groupby('evb', observed=True)['profit'].agg(['size','mean'])
        line = "  ".join(f"{b}:{r['size']:.0f}@{r['mean']:+.2f}" for b, r in by.iterrows())
        print(f"{src:10s}: n={len(d):4d}  hit={d.won.mean():.3f}  ROI={d.profit.mean():+.4f}  "
              f"units={d.profit.sum():+.1f}   [{line}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "settle", "report", "compare", "show"])
    ap.add_argument("--min-ev", type=float, default=MIN_EV)
    ap.add_argument("--min-volume", type=float, default=0.0)
    args = ap.parse_args()
    if args.cmd == "compare":
        compare_sources()
    elif args.cmd == "log":
        log_bets(args.min_ev)
    elif args.cmd == "settle":
        settle()
    elif args.cmd == "show":
        show(args.min_ev, args.min_volume)
    else:
        report()


if __name__ == "__main__":
    main()
