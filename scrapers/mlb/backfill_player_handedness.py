"""
scrapers/mlb/backfill_player_handedness.py — Populate players.bats / .throws / .position
from MLB Stats API for every MLB player in the players table.

Batched against /api/v1/people?personIds=... (up to 100 IDs per call). Idempotent —
re-runnable; only fetches rows where any of (bats, throws, position) is NULL.

Usage:
    python -m scrapers.mlb.backfill_player_handedness                 # all MLB players missing fields
    python -m scrapers.mlb.backfill_player_handedness --force-all      # re-fetch even if populated
    python -m scrapers.mlb.backfill_player_handedness --new-only       # only rows with NULL on all three
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import requests

from db.db import query, get_conn


API = "https://statsapi.mlb.com/api/v1/people"
BATCH = 100  # MLB API accepts up to ~100 personIds per call
PAUSE = 0.25  # seconds between batches


def _select_missing(force_all: bool, new_only: bool):
    """Return rows of (player_id, external_id) needing handedness data."""
    if force_all:
        where = "sport_id = 2 AND external_id IS NOT NULL"
    elif new_only:
        where = "sport_id = 2 AND external_id IS NOT NULL AND bats IS NULL AND throws IS NULL AND position IS NULL"
    else:
        where = "sport_id = 2 AND external_id IS NOT NULL AND (bats IS NULL OR throws IS NULL OR position IS NULL)"
    sql = f"SELECT player_id, external_id FROM players WHERE {where}"
    return query(sql)


def _fetch_batch(external_ids):
    """Hit /people for up to ~100 external_ids; return dict ext_id -> (bats, throws, position)."""
    ids_param = ",".join(external_ids)
    params = {"personIds": ids_param}
    out = {}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for person in data.get("people", []):
                pid = str(person.get("id"))
                bats   = (person.get("batSide") or {}).get("code")
                throws = (person.get("pitchHand") or {}).get("code")
                pos    = (person.get("primaryPosition") or {}).get("abbreviation")
                out[pid] = (bats, throws, pos)
            return out
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"    batch fetch error attempt {attempt + 1}: {e}; sleeping {wait}s")
            time.sleep(wait)
    print(f"    batch fetch FAILED after retries — skipping {len(external_ids)} players")
    return {}


def _update_rows(updates):
    """Bulk UPDATE — updates is list of (bats, throws, position, player_id)."""
    if not updates:
        return 0
    sql = "UPDATE players SET bats = %s, throws = %s, position = COALESCE(%s, position) WHERE player_id = %s"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, updates)
            n = cur.rowcount
        conn.commit()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true", help="Refetch every MLB player")
    parser.add_argument("--new-only", action="store_true", help="Only players with NULL on all fields")
    args = parser.parse_args()

    rows = _select_missing(args.force_all, args.new_only)
    print(f"Players needing handedness backfill: {len(rows):,}")
    if len(rows) == 0:
        return

    ext_to_pid = {str(r.external_id): int(r.player_id) for r in rows.itertuples()}
    external_ids = list(ext_to_pid.keys())

    total_updated = 0
    total_missing = 0
    n_batches = (len(external_ids) + BATCH - 1) // BATCH
    for i in range(0, len(external_ids), BATCH):
        batch = external_ids[i:i + BATCH]
        results = _fetch_batch(batch)

        updates = []
        for ext_id in batch:
            if ext_id not in results:
                total_missing += 1
                continue
            bats, throws, pos = results[ext_id]
            updates.append((bats, throws, pos, ext_to_pid[ext_id]))

        n = _update_rows(updates)
        total_updated += n
        print(f"  batch {i // BATCH + 1}/{n_batches}: fetched {len(results)}/{len(batch)}, updated {n}")
        time.sleep(PAUSE)

    print(f"\nTotal updated: {total_updated:,}")
    print(f"Total missing from API: {total_missing:,}")

    # Verify
    check = query("""
        SELECT
            COUNT(*) AS total,
            COUNT(bats)   AS bats_pop,
            COUNT(throws) AS throws_pop,
            COUNT(position) AS pos_pop
        FROM players WHERE sport_id = 2
    """)
    print("\nFinal coverage:")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
