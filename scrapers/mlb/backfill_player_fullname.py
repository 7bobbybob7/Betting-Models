"""
scrapers/mlb/backfill_player_fullname.py — Populate players.full_name from MLB Stats API.

Our players.name is inconsistent ("Last, F" for ~29%, bare "Last" for the rest), which
breaks name-matching against external prop feeds (BettingPros/Underdog use full names).
This stores the canonical fullName ("Christian Walker") so matching can be exact.

Batched against /api/v1/people?personIds=... (up to 100 IDs per call). Idempotent.

Usage:
    python -m scrapers.mlb.backfill_player_fullname              # only NULL full_name
    python -m scrapers.mlb.backfill_player_fullname --force-all  # refetch everyone
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import requests

from db.db import query, get_conn


API = "https://statsapi.mlb.com/api/v1/people"
BATCH = 100
PAUSE = 0.25


def _select(force_all: bool):
    where = "sport_id = 2 AND external_id IS NOT NULL"
    if not force_all:
        where += " AND full_name IS NULL"
    return query(f"SELECT player_id, external_id FROM players WHERE {where}")


def _fetch_batch(external_ids):
    params = {"personIds": ",".join(external_ids)}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=30)
            r.raise_for_status()
            return {str(p.get("id")): p.get("fullName")
                    for p in r.json().get("people", []) if p.get("fullName")}
        except requests.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"    batch error attempt {attempt + 1}: {e}; sleeping {wait}s")
            time.sleep(wait)
    print(f"    batch FAILED after retries — skipping {len(external_ids)} players")
    return {}


def _update(updates):
    """updates = list of (full_name, player_id)."""
    if not updates:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany("UPDATE players SET full_name = %s WHERE player_id = %s", updates)
            n = cur.rowcount
        conn.commit()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-all", action="store_true")
    args = parser.parse_args()

    rows = _select(args.force_all)
    print(f"Players needing full_name: {len(rows):,}")
    if len(rows) == 0:
        return

    ext_to_pid = {str(r.external_id): int(r.player_id) for r in rows.itertuples()}
    external_ids = list(ext_to_pid.keys())

    total = 0
    n_batches = (len(external_ids) + BATCH - 1) // BATCH
    for i in range(0, len(external_ids), BATCH):
        batch = external_ids[i:i + BATCH]
        results = _fetch_batch(batch)
        updates = [(results[e], ext_to_pid[e]) for e in batch if e in results]
        n = _update(updates)
        total += n
        print(f"  batch {i // BATCH + 1}/{n_batches}: updated {n}")
        time.sleep(PAUSE)

    print(f"\nTotal updated: {total:,}")
    check = query("""
        SELECT COUNT(*) AS total, COUNT(full_name) AS pop
        FROM players WHERE sport_id = 2
    """)
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
