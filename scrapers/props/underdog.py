"""
scrapers/props/underdog.py - Capture Underdog Fantasy MLB player props.

Snapshots all current MLB over/under lines with both sides' odds, payout
multipliers, and player/game context. Writes to the `underdog_props` DB
table (one row per snapshot x line x side).

Usage:
    python -m scrapers.props.underdog                 # write to DB
    python -m scrapers.props.underdog --csv-only      # write to CSV only
    python -m scrapers.props.underdog --csv-also      # write to both
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from db.db import bulk_insert


UNDERDOG_API = "https://api.underdogfantasy.com/beta/v5/over_under_lines"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def fetch_lines():
    r = requests.get(UNDERDOG_API, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_mlb_props(data, snapshot_ts, sport="MLB"):
    games = {g["id"]: g for g in data.get("games", [])}
    players = {p["id"]: p for p in data.get("players", [])}

    mlb_game_ids = {g["id"] for g in data.get("games", []) if g.get("sport_id") == sport}

    mlb_app = {}
    for app in data.get("appearances", []):
        if app.get("match_id") in mlb_game_ids and app.get("match_type") == "Game":
            mlb_app[app["id"]] = {
                "player_id": app.get("player_id"),
                "team_id": app.get("team_id"),
                "game_id": app.get("match_id"),
            }

    rows = []
    for line in data.get("over_under_lines", []):
        ou = line.get("over_under", {})
        appearance_stat = ou.get("appearance_stat", {})
        app_id = appearance_stat.get("appearance_id")

        if app_id not in mlb_app:
            continue

        ctx = mlb_app[app_id]
        player = players.get(ctx["player_id"], {})
        game = games.get(ctx["game_id"], {})

        base = {
            "snapshot_ts": snapshot_ts,
            "line_id": line.get("id"),
            "stable_id": line.get("stable_id"),
            "line_type": line.get("line_type"),
            "line_status": line.get("status"),
            "stat_value": _to_float(line.get("stat_value")),
            "stat_type": appearance_stat.get("display_stat"),
            "stat_internal": appearance_stat.get("stat"),
            "category": ou.get("category"),
            "has_alternates": ou.get("has_alternates"),
            "underdog_player_id": ctx["player_id"],
            "player_first_name": player.get("first_name"),
            "player_last_name": player.get("last_name"),
            "player_position": player.get("position_display_name"),
            "player_team_id": ctx["team_id"],
            "player_jersey": player.get("jersey_number"),
            "underdog_game_id": ctx["game_id"],
            "scheduled_start": game.get("scheduled_at"),
            "game_title": game.get("full_team_names_title"),
            "home_team_id": game.get("home_team_id"),
            "away_team_id": game.get("away_team_id"),
            "match_progress": game.get("match_progress"),
        }

        for opt in line.get("options", []):
            row = dict(base)
            row.update({
                "choice": opt.get("choice"),
                "choice_display": opt.get("choice_display"),
                "american_price": _to_int(opt.get("american_price")),
                "decimal_price": _to_float(opt.get("decimal_price")),
                "payout_multiplier": _to_float(opt.get("payout_multiplier")),
                "option_status": opt.get("status"),
            })
            rows.append(row)

    return rows


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# Column order MUST match the underdog_props table definition
DB_COLUMNS = [
    "snapshot_ts", "line_id", "stable_id", "line_type", "line_status",
    "stat_value", "stat_type", "stat_internal", "category", "has_alternates",
    "underdog_player_id", "player_first_name", "player_last_name",
    "player_position", "player_team_id", "player_jersey",
    "underdog_game_id", "scheduled_start", "game_title", "home_team_id", "away_team_id",
    "match_progress",
    "choice", "choice_display", "american_price", "decimal_price",
    "payout_multiplier", "option_status",
]


def write_to_db(rows):
    """Bulk insert into underdog_props.

    Returns count inserted. Handles duplicate keys via ON CONFLICT.
    """
    if not rows:
        return 0

    # Convert rows (dicts) to tuples in DB_COLUMNS order
    tuples = []
    for row in rows:
        tuples.append(tuple(row.get(col) for col in DB_COLUMNS))

    # bulk_insert may not support ON CONFLICT; use execute many with conflict handling
    # via a custom INSERT
    from db.db import get_conn
    with get_conn() as conn:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(DB_COLUMNS))
            cols = ", ".join(DB_COLUMNS)
            sql = (
                f"INSERT INTO underdog_props ({cols}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (snapshot_ts, line_id, choice) DO NOTHING"
            )
            cur.executemany(sql, tuples)
            inserted = cur.rowcount
        conn.commit()

    return inserted


def write_to_csv(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"underdog_mlb_{ts_str}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", default="MLB")
    parser.add_argument("--csv-only", action="store_true",
                        help="Skip DB write, only output CSV")
    parser.add_argument("--csv-also", action="store_true",
                        help="Write to both DB and CSV")
    parser.add_argument("--output-dir", default="data/underdog_props")
    args = parser.parse_args()

    snapshot_ts = datetime.now(timezone.utc).isoformat()
    print(f"[{snapshot_ts}] Fetching Underdog over/under lines...")

    data = fetch_lines()
    total_lines = len(data.get("over_under_lines", []))
    print(f"  {total_lines} total lines across all sports")

    rows = extract_mlb_props(data, snapshot_ts, sport=args.sport)
    print(f"  {len(rows)} MLB prop rows (lines x sides)")

    if not rows:
        print("  No MLB props found. Skipping write.")
        return

    if not args.csv_only:
        inserted = write_to_db(rows)
        print(f"  Inserted {inserted} new rows into underdog_props "
              f"({len(rows) - inserted} duplicates skipped)")

    if args.csv_only or args.csv_also:
        path = write_to_csv(rows, args.output_dir)
        print(f"  Saved CSV to {path}")

    df = pd.DataFrame(rows)
    print(f"\n  By stat type:")
    print(df.groupby("stat_type").size().sort_values(ascending=False).to_string())
    print(f"\n  Unique players: {df['underdog_player_id'].nunique()}")
    print(f"  Unique games: {df['underdog_game_id'].nunique()}")


if __name__ == "__main__":
    main()
