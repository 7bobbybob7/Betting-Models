"""
scrapers/mlb/statcast.py - Pull Statcast pitch-level data.

Uses pybaseball to fetch pitch-by-pitch data from Baseball Savant.
Populates: mlb_pitches (and ensures players exist)

NOTE: This is the heaviest pull. ~700k pitches per season.
      pybaseball rate-limits automatically but expect 5-15 min per season.

Usage:
    python -m scrapers.mlb.statcast --start 2015 --end 2026
    python -m scrapers.mlb.statcast --start 2025 --end 2025  # just one season
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pandas as pd
import numpy as np
from pybaseball import statcast
from tqdm import tqdm

from db.db import get_conn, query, execute, bulk_insert
from scrapers.mlb.games import ensure_player, get_sport_id


def get_game_lookup(sport_id):
    """Build external_id -> game_id mapping for MLB games."""
    df = query(
        "SELECT game_id, external_id FROM games WHERE sport_id = %s",
        [sport_id]
    )
    return dict(zip(df["external_id"].astype(str), df["game_id"]))


def get_existing_statcast_games():
    """Get set of game_ids that already have pitch data."""
    df = query("SELECT DISTINCT game_id FROM mlb_pitches")
    return set(df["game_id"])


def pull_statcast_season(sport_id, year, game_lookup, existing_games):
    """Pull an entire season of Statcast data and insert into mlb_pitches."""

    # Statcast data is available from ~March to ~November
    start_dt = f"{year}-03-01"
    end_dt = f"{year}-11-30"

    print(f"  Fetching Statcast data for {year} ({start_dt} to {end_dt})...")
    print(f"  This may take several minutes...")

    try:
        data = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception as e:
        print(f"  ERROR fetching Statcast data: {e}")
        return 0

    if data is None or len(data) == 0:
        print(f"  No Statcast data returned for {year}")
        return 0

    print(f"  Raw pitches fetched: {len(data)}")

    # Map game_pk to our game_id
    data["game_pk_str"] = data["game_pk"].astype(str)
    data["db_game_id"] = data["game_pk_str"].map(game_lookup)

    # Filter to games we have in DB
    matched = data[data["db_game_id"].notna()].copy()
    unmatched = len(data) - len(matched)
    if unmatched > 0:
        print(f"  {unmatched} pitches skipped (game not in DB)")

    # Filter out games already loaded
    matched["db_game_id"] = matched["db_game_id"].astype(int)
    matched = matched[~matched["db_game_id"].isin(existing_games)]

    if len(matched) == 0:
        print(f"  All Statcast data already loaded for {year}")
        return 0

    print(f"  Processing {len(matched)} new pitches...")

    # Ensure all pitchers and batters exist
    pitcher_cache = {}
    batter_cache = {}

    unique_pitchers = matched[["pitcher", "player_name"]].drop_duplicates(subset=["pitcher"])
    for _, row in unique_pitchers.iterrows():
        mlb_id = int(row["pitcher"])
        name = row.get("player_name", f"Unknown_{mlb_id}")
        pid = ensure_player(sport_id, mlb_id, str(name), "P")
        pitcher_cache[mlb_id] = pid

    unique_batters = matched[["batter"]].drop_duplicates()
    for _, row in unique_batters.iterrows():
        mlb_id = int(row["batter"])
        if mlb_id not in pitcher_cache:  # might already be cached as pitcher
            pid = ensure_player(sport_id, mlb_id, f"Batter_{mlb_id}")
            batter_cache[mlb_id] = pid
        else:
            batter_cache[mlb_id] = pitcher_cache[mlb_id]

    all_player_cache = {**pitcher_cache, **batter_cache}

    # Build insert rows
    rows = []
    for _, pitch in matched.iterrows():
        pitcher_id = all_player_cache.get(int(pitch["pitcher"]))
        batter_id = all_player_cache.get(int(pitch["batter"]))

        if not pitcher_id or not batter_id:
            continue

        # Determine swing/whiff/in_play
        desc = str(pitch.get("description", "")).lower()
        is_strike = desc in (
            "called_strike", "swinging_strike", "swinging_strike_blocked",
            "foul", "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip"
        )
        is_swing = desc in (
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play", "hit_into_play_no_out",
            "hit_into_play_score", "foul_bunt", "bunt_foul_tip"
        )
        is_whiff = desc in ("swinging_strike", "swinging_strike_blocked")
        is_in_play = desc in (
            "hit_into_play", "hit_into_play_no_out", "hit_into_play_score"
        )

        # Top/bottom of inning
        inning_half = pitch.get("inning_topbot", "")
        top_bottom = None
        if isinstance(inning_half, str):
            if inning_half.lower().startswith("top"):
                top_bottom = "top"
            elif inning_half.lower().startswith("bot"):
                top_bottom = "bot"

        rows.append((
            int(pitch["db_game_id"]),
            pitcher_id,
            batter_id,
            _safe_int(pitch.get("inning")),
            top_bottom,
            _safe_int(pitch.get("at_bat_number")),
            _safe_int(pitch.get("pitch_number")),
            _safe_str(pitch.get("pitch_type")),
            _safe_float(pitch.get("release_speed")),
            _safe_int(pitch.get("release_spin_rate")),
            _safe_float(pitch.get("release_extension")),
            _safe_float(pitch.get("pfx_x")),
            _safe_float(pitch.get("pfx_z")),
            _safe_float(pitch.get("plate_x")),
            _safe_float(pitch.get("plate_z")),
            _safe_int(pitch.get("zone")),
            _safe_str(pitch.get("description")),
            _safe_str(pitch.get("events")),
            is_strike,
            is_swing,
            is_whiff,
            is_in_play,
            _safe_float(pitch.get("launch_speed")),
            _safe_float(pitch.get("launch_angle")),
            _safe_float(pitch.get("hit_distance_sc")),
            _safe_float(pitch.get("estimated_ba_using_speedangle")),
            _safe_float(pitch.get("estimated_slg_using_speedangle")),
            _safe_float(pitch.get("estimated_woba_using_speedangle")),
            _safe_int(pitch.get("balls")),
            _safe_int(pitch.get("strikes")),
        ))

    # Bulk insert in chunks (large dataset)
    columns = [
        "game_id", "pitcher_id", "batter_id",
        "inning", "top_bottom", "at_bat_number", "pitch_number",
        "pitch_type", "release_speed", "release_spin_rate", "release_extension",
        "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
        "description", "result",
        "is_strike", "is_swing", "is_whiff", "is_in_play",
        "launch_speed", "launch_angle", "hit_distance",
        "xba", "xslg", "xwoba",
        "balls", "strikes_count",
    ]

    chunk_size = 10000
    inserted = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        try:
            bulk_insert("mlb_pitches", columns, chunk)
            inserted += len(chunk)
        except Exception as e:
            print(f"  Error inserting chunk at {i}: {e}")
            # Try row by row
            for row in chunk:
                try:
                    cols = ", ".join(columns)
                    placeholders = ", ".join(["%s"] * len(columns))
                    sql = f"INSERT INTO mlb_pitches ({cols}) VALUES ({placeholders})"
                    execute(sql, list(row))
                    inserted += 1
                except Exception:
                    continue

    print(f"  {year}: {inserted} pitches inserted")
    return inserted


def _safe_int(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 3)
    except (ValueError, TypeError):
        return None


def _safe_str(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Pull Statcast pitch data")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2026, help="End year (inclusive)")
    args = parser.parse_args()

    sport_id = get_sport_id()
    game_lookup = get_game_lookup(sport_id)
    existing_games = get_existing_statcast_games()

    print(f"MLB sport_id: {sport_id}")
    print(f"Games in DB: {len(game_lookup)}")
    print(f"Games with Statcast: {len(existing_games)}")
    print(f"Pulling Statcast for {args.start} to {args.end}\n")

    total_pitches = 0
    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        count = pull_statcast_season(sport_id, year, game_lookup, existing_games)
        total_pitches += count

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_pitches} total pitches inserted")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
