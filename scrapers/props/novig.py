"""
scrapers/props/novig.py — Capture Novig exchange MLB player-prop prices directly.

Novig's app talks to an unauthenticated Hasura GraphQL API (api.novig.us/v1/graphql).
Prices are de-vigged probabilities (over + under ≈ 1.0) — the cleanest "fair value"
reference for sharp-vs-soft line shopping against Underdog.

Captures upcoming (non-FINAL) MLB player-prop markets for the hitter markets we bet,
writing one row per market per run into novig_snapshots. captured_at is set once per
run so multiple runs/day give us intraday persistence tracking.

Price fields:
    last       = last-traded probability (may be stale)
    available  = best resting order currently takeable (often null pre-game)
    volume     = lifetime market volume ($) — liquidity filter

Usage:
    python -m scrapers.props.novig                 # capture now (DB)
    python -m scrapers.props.novig --dry-run       # print summary, no DB write
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from db.db import get_conn


GRAPHQL = "https://api.novig.us/v1/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Origin": "https://app.novig.us",
    "Referer": "https://app.novig.us/",
}
ET = ZoneInfo("America/New_York")

# Hitter markets we care about (incl. softer ones that showed line-shopping edge)
WNBA_TYPES = ['POINTS', 'REBOUNDS', 'ASSISTS', 'THREE_POINTERS_MADE',
              'POINTS_REBOUNDS_ASSISTS', 'REBOUNDS_ASSISTS']
MARKET_TYPES = [
    "HITS_RUNS_RBIS", "TOTAL_BASES", "RBIS", "RUNS", "HITS",
    "HOME_RUNS", "BATTING_WALKS", "STOLEN_BASES", "TOTAL_BASES",
]
PAGE = 200  # markets per GraphQL page

QUERY = """
query Markets($types: [String!], $limit: Int!, $offset: Int!, $league: String!) {
  market(
    where: {
      league: {_eq: $league},
      type: {_in: $types},
      status: {_eq: "OPEN"},
      playerId: {_is_null: false},
      event: {status: {_neq: "FINAL"}}
    },
    order_by: {id: asc},
    limit: $limit, offset: $offset
  ) {
    id
    type
    strike
    volume
    player { full_name }
    event { scheduled_start }
    outcomes { type last available }
  }
}
"""


def _post(payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(GRAPHQL, headers=HEADERS, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(data["errors"])
            return data["data"]
        except (requests.RequestException, RuntimeError) as e:
            wait = 2 ** (attempt + 1)
            print(f"  GraphQL error attempt {attempt+1}/{max_retries}: {e}; sleeping {wait}s")
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


def fetch_markets(league="MLB"):
    """Page through all upcoming OPEN MLB hitter-prop markets."""
    out, offset = [], 0
    types = sorted(set(WNBA_TYPES if league == "WNBA" else MARKET_TYPES))
    while offset < 20000:  # safety bound
        data = _post({"query": QUERY,
                      "variables": {"league": league, "types": types, "limit": PAGE, "offset": offset}})
        batch = data.get("market", [])
        out.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
        time.sleep(0.15)
    return out


def _side(outcomes, want):
    for o in outcomes:
        if (o.get("type") or "").lower() == want:
            return o
    return {}


def flatten(market, captured_at):
    """Market dict -> row tuple for novig_snapshots."""
    over = _side(market.get("outcomes", []), "over")
    under = _side(market.get("outcomes", []), "under")
    sched = (market.get("event") or {}).get("scheduled_start")
    game_date = None
    sched_ts = None
    if sched:
        dt = datetime.fromisoformat(sched)
        sched_ts = dt
        game_date = dt.astimezone(ET).date()  # ET date is the MLB "game date"
    return (
        captured_at,
        str(market.get("id")),
        game_date,
        sched_ts,
        market.get("type"),
        (market.get("player") or {}).get("full_name"),
        market.get("strike"),
        over.get("last"),
        under.get("last"),
        over.get("available"),
        under.get("available"),
        market.get("volume"),
    )


INSERT_COLS = [
    "captured_at", "novig_market_id", "game_date", "scheduled_start", "market_type",
    "player_name", "strike", "over_last", "under_last",
    "over_available", "under_available", "volume",
]


def insert_rows(rows, max_retries=3):
    if not rows:
        return 0
    cols = ", ".join(INSERT_COLS)
    ph = ", ".join(["%s"] * len(INSERT_COLS))
    sql = (f"INSERT INTO novig_snapshots ({cols}) VALUES ({ph}) "
           f"ON CONFLICT (captured_at, novig_market_id) DO NOTHING")
    for attempt in range(max_retries):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
                    n = cur.rowcount
                conn.commit()
            return n
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  DB error attempt {attempt+1}/{max_retries}: {type(e).__name__}; sleeping {wait}s")
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print summary, no DB write")
    parser.add_argument("--league", default="MLB")
    args = parser.parse_args()

    captured_at = datetime.now(timezone.utc)
    print(f"Novig capture @ {captured_at.isoformat()}")
    markets = fetch_markets(league=args.league)
    print(f"Fetched {len(markets):,} upcoming OPEN MLB hitter-prop markets")

    rows = [flatten(m, captured_at) for m in markets if m.get("id")]
    n_last  = sum(1 for r in rows if r[7] is not None or r[8] is not None)
    n_avail = sum(1 for r in rows if r[9] is not None or r[10] is not None)
    print(f"  with last price: {n_last:,} | with available (live order): {n_avail:,}")

    # Market-type breakdown
    from collections import Counter
    by_type = Counter(r[4] for r in rows)
    print(f"  by market: {dict(by_type)}")

    if args.dry_run:
        print("\n[dry-run] sample priced markets:")
        for r in [x for x in rows if x[7] is not None][:8]:
            print(f"   {r[5]:24s} {r[4]:16s} {r[6]}  over_last={r[7]} avail={r[9]} vol={r[11]}")
        return

    inserted = insert_rows(rows)
    print(f"Inserted {inserted:,} snapshot rows")


if __name__ == "__main__":
    main()
