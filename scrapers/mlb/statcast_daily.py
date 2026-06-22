"""
scrapers/mlb/statcast_daily.py - Pull yesterday's Statcast data only.

For daily cron use. Pulls a small date window (default 3 days back to today)
and inserts only pitches for games not already in mlb_pitches.

Different from scrapers/mlb/statcast.py which pulls full seasons (slow).

Usage:
    python -m scrapers.mlb.statcast_daily            # last 3 days
    python -m scrapers.mlb.statcast_daily --days 7   # last 7 days
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from datetime import date, timedelta

import numpy as np
from pybaseball import statcast

from db.db import query, bulk_insert
from scrapers.mlb.games import ensure_player, get_sport_id


def get_game_lookup(sport_id):
    df = query(
        "SELECT game_id, external_id FROM games WHERE sport_id = %s",
        [sport_id]
    )
    return dict(zip(df["external_id"].astype(str), df["game_id"]))


def get_existing_statcast_games(start_date):
    """Get game_ids that already have pitch data, within a recent window."""
    df = query(
        "SELECT DISTINCT p.game_id FROM mlb_pitches p "
        "JOIN games g ON p.game_id = g.game_id "
        "WHERE g.game_date >= %s",
        [start_date]
    )
    return set(df["game_id"])


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


def pull_window(start_dt, end_dt, sport_id):
    """Pull a small date window and insert missing pitches."""
    game_lookup = get_game_lookup(sport_id)
    existing_games = get_existing_statcast_games(start_dt)

    print(f"  Fetching Statcast {start_dt} to {end_dt}...")
    try:
        data = statcast(start_dt=str(start_dt), end_dt=str(end_dt))
    except Exception as e:
        print(f"  ERROR fetching Statcast: {e}")
        return 0

    if data is None or len(data) == 0:
        print(f"  No Statcast data returned")
        return 0

    print(f"  Raw pitches fetched: {len(data)}")

    data["game_pk_str"] = data["game_pk"].astype(str)
    data["db_game_id"] = data["game_pk_str"].map(game_lookup)
    matched = data[data["db_game_id"].notna()].copy()
    matched["db_game_id"] = matched["db_game_id"].astype(int)
    matched = matched[~matched["db_game_id"].isin(existing_games)]

    if len(matched) == 0:
        print(f"  All pitches in window already loaded")
        return 0

    print(f"  Processing {len(matched)} new pitches...")

    pitcher_cache = {}
    batter_cache = {}
    for _, row in matched[["pitcher", "player_name"]].drop_duplicates(subset=["pitcher"]).iterrows():
        mlb_id = int(row["pitcher"])
        pitcher_cache[mlb_id] = ensure_player(sport_id, mlb_id, str(row.get("player_name", f"Unknown_{mlb_id}")), "P")
    for _, row in matched[["batter"]].drop_duplicates().iterrows():
        mlb_id = int(row["batter"])
        if mlb_id not in pitcher_cache:
            batter_cache[mlb_id] = ensure_player(sport_id, mlb_id, f"Batter_{mlb_id}")
        else:
            batter_cache[mlb_id] = pitcher_cache[mlb_id]
    all_players = {**pitcher_cache, **batter_cache}

    rows = []
    for _, pitch in matched.iterrows():
        pitcher_id = all_players.get(int(pitch["pitcher"]))
        batter_id = all_players.get(int(pitch["batter"]))
        if not pitcher_id or not batter_id:
            continue

        desc = str(pitch.get("description", "")).lower()
        is_strike = desc in (
            "called_strike", "swinging_strike", "swinging_strike_blocked",
            "foul", "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip",
        )
        is_swing = desc in (
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play", "hit_into_play_no_out",
            "hit_into_play_score", "foul_bunt", "bunt_foul_tip",
        )
        is_whiff = desc in ("swinging_strike", "swinging_strike_blocked")
        is_in_play = desc in ("hit_into_play", "hit_into_play_no_out", "hit_into_play_score")

        inning_half = pitch.get("inning_topbot", "")
        top_bottom = None
        if isinstance(inning_half, str):
            if inning_half.lower().startswith("top"):
                top_bottom = "top"
            elif inning_half.lower().startswith("bot"):
                top_bottom = "bot"

        rows.append((
            int(pitch["db_game_id"]),
            pitcher_id, batter_id,
            _safe_int(pitch.get("inning")), top_bottom,
            _safe_int(pitch.get("at_bat_number")),
            _safe_int(pitch.get("pitch_number")),
            _safe_str(pitch.get("pitch_type")),
            _safe_float(pitch.get("release_speed")),
            _safe_int(pitch.get("release_spin_rate")),
            _safe_float(pitch.get("release_extension")),
            _safe_float(pitch.get("pfx_x")), _safe_float(pitch.get("pfx_z")),
            _safe_float(pitch.get("plate_x")), _safe_float(pitch.get("plate_z")),
            _safe_int(pitch.get("zone")),
            _safe_str(pitch.get("description")),
            _safe_str(pitch.get("events")),
            is_strike, is_swing, is_whiff, is_in_play,
            _safe_float(pitch.get("launch_speed")),
            _safe_float(pitch.get("launch_angle")),
            _safe_float(pitch.get("hit_distance_sc")),
            _safe_float(pitch.get("estimated_ba_using_speedangle")),
            _safe_float(pitch.get("estimated_slg_using_speedangle")),
            _safe_float(pitch.get("estimated_woba_using_speedangle")),
            _safe_int(pitch.get("balls")),
            _safe_int(pitch.get("strikes")),
        ))

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

    inserted = 0
    chunk_size = 5000
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        try:
            bulk_insert("mlb_pitches", columns, chunk)
            inserted += len(chunk)
        except Exception as e:
            print(f"  Insert error: {e}")
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3,
                        help="How many days back to pull (default: 3 — safety overlap)")
    args = parser.parse_args()

    sport_id = get_sport_id()
    today = date.today()
    start_dt = today - timedelta(days=args.days)
    end_dt = today

    print(f"\n=== DAILY STATCAST PULL ===")
    print(f"Window: {start_dt} to {end_dt}")

    count = pull_window(start_dt, end_dt, sport_id)
    print(f"\n  Inserted {count} pitches")


if __name__ == "__main__":
    main()
