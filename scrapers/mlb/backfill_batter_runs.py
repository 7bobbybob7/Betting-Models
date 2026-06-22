"""
scrapers/mlb/backfill_batter_runs.py — Backfill the new mlb_batting_game.runs column
by re-fetching boxscores for every game and UPDATEing per (game_id, player_id).

Iterates games where ANY batter row has runs IS NULL. For each game, fetches the
boxscore via statsapi.boxscore_data() and bulk-UPDATEs runs for all batters.

Idempotent — only touches games with NULL rows; safely resumable mid-run.

Usage:
    python -m scrapers.mlb.backfill_batter_runs                # all games with NULL rows
    python -m scrapers.mlb.backfill_batter_runs --year 2025    # restrict to one season
    python -m scrapers.mlb.backfill_batter_runs --limit 100    # smoke test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import statsapi

from db.db import query, get_conn


def _games_needing_runs(year: int | None, limit: int | None):
    """Game (game_id, external_id) pairs that have NULL runs on any batter row."""
    where = "g.sport_id = 2 AND bg.runs IS NULL"
    params = []
    if year is not None:
        where += " AND EXTRACT(YEAR FROM g.game_date) = %s"
        params.append(year)
    sql = f"""
        SELECT DISTINCT g.game_id, g.external_id, g.game_date
        FROM games g
        JOIN mlb_batting_game bg ON g.game_id = bg.game_id
        WHERE {where}
        ORDER BY g.game_date
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return query(sql, params=params)


def _fetch_boxscore(game_pk: int, max_retries: int = 3):
    """statsapi.boxscore_data with retries."""
    for attempt in range(max_retries):
        try:
            return statsapi.boxscore_data(int(game_pk))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    boxscore_data({game_pk}) FAILED: {e}")
                return None
            time.sleep(2 ** (attempt + 1))


def _extract_runs(box):
    """Return list of (mlb_personId, runs) for both teams."""
    out = []
    for side_key in ("homeBatters", "awayBatters"):
        for batter in box.get(side_key, []):
            pid = batter.get("personId")
            r_val = batter.get("r")
            if not pid or pid == 0:
                continue
            try:
                r_int = int(r_val)
            except (ValueError, TypeError):
                continue
            out.append((str(pid), r_int))
    return out


def _update_game_runs(game_id: int, ext_to_runs: list):
    """Bulk UPDATE runs by joining external_id -> player_id."""
    if not ext_to_runs:
        return 0
    # Build a temp table approach or do per-row by external_id join
    sql = """
        UPDATE mlb_batting_game bg
        SET runs = data.runs
        FROM (VALUES %s) AS data(external_id, runs)
        JOIN players pl ON pl.external_id = data.external_id AND pl.sport_id = 2
        WHERE bg.game_id = %s AND bg.player_id = pl.player_id AND bg.runs IS NULL
    """
    # psycopg2 doesn't support %s expansion for VALUES like execute_values easily here;
    # simpler: build with mogrify or executemany of a simpler statement
    simple_sql = """
        UPDATE mlb_batting_game
        SET runs = %s
        WHERE game_id = %s
          AND player_id = (SELECT player_id FROM players WHERE sport_id = 2 AND external_id = %s LIMIT 1)
          AND runs IS NULL
    """
    rows = [(r, game_id, ext) for ext, r in ext_to_runs]
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(simple_sql, rows)
            n = cur.rowcount
        conn.commit()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.10, help="seconds between game fetches")
    args = parser.parse_args()

    games = _games_needing_runs(args.year, args.limit)
    print(f"Games needing runs backfill: {len(games):,}")
    if len(games) == 0:
        return

    total_updated = 0
    failed_games = 0
    t0 = time.time()
    for i, row in enumerate(games.itertuples()):
        ext_id = str(row.external_id)
        if not ext_id.isdigit():
            failed_games += 1
            continue
        box = _fetch_boxscore(ext_id)
        if box is None:
            failed_games += 1
            continue
        runs_data = _extract_runs(box)
        try:
            n = _update_game_runs(int(row.game_id), runs_data)
        except Exception as e:
            print(f"  game_id {row.game_id} ({row.game_date}): UPDATE failed: {e}")
            failed_games += 1
            time.sleep(1)
            continue
        total_updated += n
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(games) - i - 1) / rate / 60
            print(f"  [{i + 1:>5d}/{len(games)}] {row.game_date} | "
                  f"{total_updated:,} rows updated | "
                  f"{rate:.1f} games/sec | ETA {eta:.0f} min")
        time.sleep(args.sleep)

    print(f"\nDone in {(time.time() - t0)/60:.1f} min")
    print(f"  Total rows updated: {total_updated:,}")
    print(f"  Failed games:       {failed_games:,}")

    # Verify
    final = query("""
        SELECT COUNT(*) FILTER (WHERE runs IS NOT NULL) AS pop,
               COUNT(*) AS total
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        WHERE g.sport_id = 2
    """)
    print(f"\nFinal: {final.to_string(index=False)}")


if __name__ == "__main__":
    main()
