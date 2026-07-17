"""
scrapers/props/bettingpros.py - Pull MLB player prop snapshots from BettingPros API.

The BettingPros /v3/props endpoint exposes per-book historical odds (filter by
book_id) including actual scored outcomes — perfect for backtesting hitter prop
models against real Underdog lines without needing to wait for forward capture.

Books we capture by default:
    Underdog (36)    — primary betting venue
    Consensus (0)    — average across all books for CLV cross-reference
    Novig (60)       — sharp/exchange book for "true" market reference

Markets pulled (all hitter + pitcher props):
    287 Hits, 288 Runs, 289 RBI, 291 Doubles, 292 Triples,
    293 Total Bases, 294 Steals, 295 Singles, 299 Home Runs, 403 HRR,
    285 Strikeouts(P), 290 ER(P), 404 Hits Allowed(P),
    405 Outs Recorded(P), 408 Walks Allowed(P)

Usage:
    # Single date (e.g. for daily cron)
    python -m scrapers.props.bettingpros --date 2026-06-21

    # Backfill a date range (default books)
    python -m scrapers.props.bettingpros --start 2024-04-01 --end 2026-06-22

    # Backfill specific books only
    python -m scrapers.props.bettingpros --start 2026-04-01 --end 2026-06-22 --books 36

    # Snapshot today
    python -m scrapers.props.bettingpros
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import requests
import psycopg2
from datetime import date, timedelta

from db.db import get_conn, query
import db.db as db_module


BP_API = "https://api.bettingpros.com/v3/props"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.bettingpros.com/",
}

# Default books to pull (book_id → readable name)
DEFAULT_BOOKS = {
    36: "Underdog",     # primary betting venue
    0:  "Consensus",    # average across all books
    60: "Novig",        # no-vig exchange (sharp reference)
}

# market_id → market_name lookup (discovered via API)
MARKET_NAMES = {
    285: "strikeouts",
    287: "hits",
    288: "runs",
    289: "rbi",
    290: "earned-runs-allowed",
    291: "doubles",
    292: "triples",
    293: "total-bases",
    294: "steals",
    295: "singles",
    299: "homeruns",
    403: "runs-hits-rbis",
    404: "hits-allowed",
    405: "outs-recorded",
    408: "walks-allowed",
}


# Insert columns must match the schema order exactly
INSERT_COLS = [
    "prop_date", "event_id", "market_id", "market_name", "book_id", "book_name",
    "bp_player_id", "player_first_name", "player_last_name",
    "player_team", "player_position", "player_slug",
    "opposing_pitcher", "in_lineup",
    "over_line", "over_odds", "over_consensus_line", "over_consensus_odds", "over_probability",
    "under_line", "under_odds", "under_consensus_line", "under_consensus_odds", "under_probability",
    "bp_projected_value", "bp_recommended_side", "bp_bet_rating",
    "actual", "is_scored", "is_push",
]


def fetch_props_for_date_book(prop_date, book_id, sleep_between_pages=0.15, max_retries=4, sport="MLB"):
    """Pull all pages of props for a (date, book_id) combination, retrying transient errors."""
    rows = []
    page = 1
    while page < 50:  # safety upper bound
        params = {
            "sport": sport,
            "date": prop_date,
            "book_id": book_id,
            "ev_threshold": "false",
            "limit": 200,
            "page": page,
        }
        r = None
        for attempt in range(max_retries):
            try:
                r = requests.get(BP_API, headers=HEADERS, params=params, timeout=45)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                wait = 2 ** (attempt + 1)
                if attempt == max_retries - 1:
                    print(f"    Request error on page {page} after {max_retries} retries: {e}")
                    return rows
                print(f"    Page {page} attempt {attempt + 1}/{max_retries} failed ({type(e).__name__}); sleeping {wait}s")
                time.sleep(wait)
        data = r.json()
        rows.extend(data.get("props", []))
        pg = data.get("_pagination", {})
        if page >= pg.get("total_pages", 0) or not data.get("props"):
            break
        page += 1
        time.sleep(sleep_between_pages)
    return rows


def _reset_pool():
    """Drop the connection pool so the next get_conn() opens a fresh one.
    Used to recover from 'server closed connection unexpectedly' on long jobs."""
    try:
        if db_module._pool and not db_module._pool.closed:
            db_module._pool.closeall()
    except Exception:
        pass
    db_module._pool = None


def already_loaded(prop_date, book_id):
    """Return True if this (date, book) already has rows — skip refetch on idempotent resumes."""
    df = query(
        "SELECT 1 FROM bettingpros_props WHERE prop_date = %(d)s AND book_id = %(b)s LIMIT 1",
        params={"d": str(prop_date), "b": book_id},
    )
    return len(df) > 0


def flatten_prop(prop, prop_date, book_id, book_name):
    """Flatten a BettingPros API prop dict into a tuple matching INSERT_COLS."""
    participant = prop.get("participant", {})
    player = participant.get("player", {})
    over = prop.get("over", {}) or {}
    under = prop.get("under", {}) or {}
    projection = prop.get("projection", {}) or {}
    extra = prop.get("extra", {}) or {}
    scoring = prop.get("scoring", {}) or {}
    market_id = prop.get("market_id")

    return (
        prop_date,
        prop.get("event_id"),
        market_id,
        MARKET_NAMES.get(market_id),
        book_id,
        book_name,
        str(participant.get("id")) if participant.get("id") else None,
        player.get("first_name"),
        player.get("last_name"),
        player.get("team"),
        player.get("position"),
        player.get("slug"),
        extra.get("opposing_pitcher"),
        extra.get("in_lineup"),
        over.get("line"),
        over.get("odds"),
        over.get("consensus_line"),
        over.get("consensus_odds"),
        over.get("probability"),
        under.get("line"),
        under.get("odds"),
        under.get("consensus_line"),
        under.get("consensus_odds"),
        under.get("probability"),
        projection.get("value"),
        projection.get("recommended_side"),
        projection.get("bet_rating"),
        scoring.get("actual"),
        scoring.get("is_scored"),
        scoring.get("is_push"),
    )


def insert_rows(rows, max_retries=3):
    """Bulk insert with conflict handling (idempotent). Reconnects on dropped pool conns."""
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(INSERT_COLS))
    cols = ", ".join(INSERT_COLS)
    sql = (
        f"INSERT INTO bettingpros_props ({cols}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (prop_date, market_id, book_id, bp_player_id) DO UPDATE "
        f"SET actual = EXCLUDED.actual, is_scored = EXCLUDED.is_scored, "
        f"is_push = EXCLUDED.is_push, "
        f"closing_over_odds = EXCLUDED.over_odds, "
        f"closing_under_odds = EXCLUDED.under_odds, "
        f"closing_line = EXCLUDED.over_line"
    )
    for attempt in range(max_retries):
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
                    inserted = cur.rowcount
                conn.commit()
            return inserted
        except psycopg2.OperationalError as e:
            wait = 2 ** (attempt + 1)
            print(f"    DB conn error attempt {attempt + 1}/{max_retries}: {e.__class__.__name__}; resetting pool, sleeping {wait}s")
            _reset_pool()
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


def pull_date(prop_date, book_ids, sleep_between_books=0.3, skip_existing=True, sport="MLB"):
    """Pull a single date for all requested books.
    Per-book failures are logged but don't abort the rest of the backfill."""
    total_inserted = 0
    total_fetched = 0
    for book_id in book_ids:
        book_name = DEFAULT_BOOKS.get(book_id, f"book_{book_id}")
        if skip_existing and already_loaded(prop_date, book_id):
            print(f"  {prop_date} | {book_name:12s} (id={book_id}): already loaded, skipping")
            continue
        try:
            raw = fetch_props_for_date_book(str(prop_date), book_id, sport=sport)
            if not raw:
                print(f"  {prop_date} | {book_name:12s} (id={book_id}): no props")
                continue
            rows = []
            for p in raw:
                try:
                    row = flatten_prop(p, str(prop_date), book_id, book_name)
                    if row[6] is not None:  # bp_player_id required
                        rows.append(row)
                except Exception as e:
                    print(f"    Skipped malformed prop: {e}")
            inserted = insert_rows(rows)
            total_inserted += inserted
            total_fetched += len(rows)
            print(f"  {prop_date} | {book_name:12s} (id={book_id}): fetched {len(rows):>4d}, inserted {inserted:>4d}")
        except Exception as e:
            print(f"  {prop_date} | {book_name:12s} (id={book_id}): FAILED {type(e).__name__}: {e}")
            _reset_pool()
        time.sleep(sleep_between_books)
    return total_inserted, total_fetched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Single date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for backfill (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for backfill (inclusive, YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--books", type=str, default=None,
                        help="Comma-separated book IDs (default: Underdog, Consensus, Novig)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass skip-existing (needed for WNBA over dates MLB already covers)")
    parser.add_argument("--sport", type=str, default="MLB",
                        help="Sport (MLB, WNBA, ...) — WNBA market_ids 390-398 don't collide with MLB")
    args = parser.parse_args()

    if args.books:
        book_ids = [int(b.strip()) for b in args.books.split(",")]
    else:
        book_ids = list(DEFAULT_BOOKS.keys())

    if args.start:
        start_dt = date.fromisoformat(args.start)
        end_dt = date.fromisoformat(args.end) if args.end else date.today()
        dates = [start_dt + timedelta(days=i) for i in range((end_dt - start_dt).days + 1)]
        print(f"Backfill: {start_dt} → {end_dt} ({len(dates)} days) × {len(book_ids)} books")
    elif args.date:
        dates = [date.fromisoformat(args.date)]
        print(f"Single date: {dates[0]} × {len(book_ids)} books")
    else:
        dates = [date.today()]
        print(f"Today: {dates[0]} × {len(book_ids)} books")

    print(f"Books: {[DEFAULT_BOOKS.get(b, b) for b in book_ids]}")
    print()

    grand_inserted = 0
    grand_fetched = 0
    for d in dates:
        ins, fetched = pull_date(d, book_ids, sport=args.sport, skip_existing=not args.force)
        grand_inserted += ins
        grand_fetched += fetched

    print()
    print(f"=== TOTAL: fetched {grand_fetched:,} props, inserted {grand_inserted:,} new rows ===")


if __name__ == "__main__":
    main()
