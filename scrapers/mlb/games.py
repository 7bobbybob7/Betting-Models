"""
scrapers/mlb/games.py - Pull MLB game results and metadata.

Uses MLB-StatsAPI to fetch:
    - Game schedules and final scores
    - Starting pitchers, venue, weather
    - Populates: teams, seasons, games, mlb_game_info

Usage:
    python -m scrapers.mlb.games --start 2015 --end 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
from datetime import datetime
import statsapi
import pandas as pd
from tqdm import tqdm

from db.db import get_conn, bulk_insert, query, execute


# ---------------------------------------------------------------------------
# In-memory caches — avoid repeated DB lookups for the same entities
# ---------------------------------------------------------------------------
_season_cache = {}   # (sport_id, year) -> season_id
_team_cache = {}     # (sport_id, name) -> team_id
_player_cache = {}   # (sport_id, external_id) -> player_id


def warm_caches(sport_id):
    """Pre-load caches from DB. Call once at startup to cut ~95% of lookups."""
    global _season_cache, _team_cache, _player_cache

    seasons = query("SELECT season_id, sport_id, year FROM seasons WHERE sport_id = %s", [sport_id])
    for _, r in seasons.iterrows():
        _season_cache[(int(r["sport_id"]), int(r["year"]))] = int(r["season_id"])

    teams = query("SELECT team_id, sport_id, name FROM teams WHERE sport_id = %s", [sport_id])
    for _, r in teams.iterrows():
        _team_cache[(int(r["sport_id"]), r["name"])] = int(r["team_id"])

    players = query("SELECT player_id, sport_id, external_id FROM players WHERE sport_id = %s", [sport_id])
    for _, r in players.iterrows():
        _player_cache[(int(r["sport_id"]), str(r["external_id"]))] = int(r["player_id"])

    print(f"  Caches warmed: {len(_season_cache)} seasons, {len(_team_cache)} teams, {len(_player_cache)} players")


def ensure_season(sport_id, year):
    """Insert season if it doesn't exist, return season_id."""
    key = (sport_id, year)
    if key in _season_cache:
        return _season_cache[key]

    execute(
        "INSERT INTO seasons (sport_id, year) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [sport_id, year]
    )
    result = query(
        "SELECT season_id FROM seasons WHERE sport_id = %s AND year = %s",
        [sport_id, year]
    )
    sid = int(result.iloc[0]["season_id"])
    _season_cache[key] = sid
    return sid


def ensure_team(sport_id, name, abbreviation=None):
    """Insert team if it doesn't exist, return team_id."""
    key = (sport_id, name)
    if key in _team_cache:
        return _team_cache[key]

    execute(
        "INSERT INTO teams (sport_id, name, abbreviation) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        [sport_id, name, abbreviation]
    )
    result = query(
        "SELECT team_id FROM teams WHERE sport_id = %s AND name = %s",
        [sport_id, name]
    )
    tid = int(result.iloc[0]["team_id"])
    _team_cache[key] = tid
    return tid


def ensure_player(sport_id, mlb_id, name, position=None):
    """Insert player if it doesn't exist, return player_id."""
    key = (sport_id, str(mlb_id))
    if key in _player_cache:
        return _player_cache[key]

    execute(
        "INSERT INTO players (sport_id, external_id, name, position) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        [sport_id, str(mlb_id), name, position]
    )
    result = query(
        "SELECT player_id FROM players WHERE sport_id = %s AND external_id = %s",
        [sport_id, str(mlb_id)]
    )
    if len(result) > 0:
        pid = int(result.iloc[0]["player_id"])
        _player_cache[key] = pid
        return pid
    return None


def get_sport_id():
    result = query("SELECT sport_id FROM sports WHERE name = 'mlb'")
    return int(result.iloc[0]["sport_id"])


def pull_season_games(sport_id, year):
    """Pull all regular season + postseason games for a given year."""
    season_id = ensure_season(sport_id, year)

    # Check what we already have
    existing = query(
        "SELECT external_id FROM games WHERE sport_id = %s AND season_id = %s",
        [sport_id, season_id]
    )
    existing_ids = set(existing["external_id"].astype(str))

    # MLB regular season typically runs late March to early October
    # Postseason through November
    start_date = f"{year}-02-20"  # spring training starts, regular season late March
    end_date = f"{year}-11-15"

    print(f"  Fetching schedule for {year} ({start_date} to {end_date})...")
    schedule = None
    for attempt in range(3):
        try:
            schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
            break
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    if schedule is None:
        print(f"  SKIPPING {year} — API unavailable after 3 attempts")
        return 0

    games_to_insert = []
    game_info_to_insert = []
    skipped = 0
    new_count = 0

    for game in tqdm(schedule, desc=f"  Processing {year}", leave=False):
        game_pk = str(game["game_id"])

        # Skip if already in DB
        if game_pk in existing_ids:
            skipped += 1
            continue

        # Only final games
        status = game.get("status", "")
        if "Final" not in status and "Completed" not in status:
            continue

        # Skip spring training, all-star, etc.
        game_type = game.get("game_type", "R")
        if game_type not in ("R", "P", "W", "D", "L", "F"):  # Regular, Postseason types
            continue

        home_name = game.get("home_name", "")
        away_name = game.get("away_name", "")
        if not home_name or not away_name:
            continue

        home_team_id = ensure_team(sport_id, home_name)
        away_team_id = ensure_team(sport_id, away_name)

        game_date = game.get("game_date", "")
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        venue = game.get("venue_name", "")
        is_postseason = game_type != "R"

        games_to_insert.append((
            sport_id,
            season_id,
            game_pk,
            game_date,
            None,  # game_time
            home_team_id,
            away_team_id,
            home_score,
            away_score,
            "final",
            venue,
            is_postseason,
            False,  # is_neutral_site
        ))
        new_count += 1

    if games_to_insert:
        columns = [
            "sport_id", "season_id", "external_id", "game_date", "game_time",
            "home_team_id", "away_team_id", "home_score", "away_score",
            "status", "venue", "is_postseason", "is_neutral_site"
        ]
        bulk_insert("games", columns, games_to_insert)

    print(f"  {year}: {new_count} new games inserted, {skipped} already existed")
    return new_count


def pull_game_info(sport_id, year):
    """
    Pull detailed game info (starters, weather) for games that don't have it yet.
    This makes individual API calls per game, so it's slower.
    """
    season_id = ensure_season(sport_id, year)

    # Find games without mlb_game_info
    games_without_info = query("""
        SELECT g.game_id, g.external_id
        FROM games g
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = %s AND g.season_id = %s AND gi.game_id IS NULL
    """, [sport_id, season_id])

    if len(games_without_info) == 0:
        print(f"  {year}: All games already have detailed info")
        return 0

    print(f"  Fetching detailed info for {len(games_without_info)} games in {year}...")
    inserted = 0

    for _, row in tqdm(games_without_info.iterrows(), total=len(games_without_info),
                        desc=f"  Game info {year}", leave=False):
        game_id = int(row["game_id"])
        game_pk = int(row["external_id"])

        boxscore = None
        for attempt in range(3):
            try:
                boxscore = statsapi.boxscore_data(game_pk)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if boxscore is None:
            continue

        # Extract starting pitchers
        home_starter_id = None
        away_starter_id = None

        try:
            home_pitcher_info = boxscore.get("homePitchers", [])
            away_pitcher_info = boxscore.get("awayPitchers", [])

            # First pitcher listed with GS is typically the starter
            for p in home_pitcher_info:
                if p.get("namefield", "").strip() and p.get("ip"):
                    pid = p.get("personId")
                    name = p.get("name", p.get("namefield", "").strip().lstrip("- "))
                    if pid:
                        home_starter_id = ensure_player(sport_id, pid, name, "P")
                    break

            for p in away_pitcher_info:
                if p.get("namefield", "").strip() and p.get("ip"):
                    pid = p.get("personId")
                    name = p.get("name", p.get("namefield", "").strip().lstrip("- "))
                    if pid:
                        away_starter_id = ensure_player(sport_id, pid, name, "P")
                    break
        except Exception:
            pass

        # Weather info (not always available)
        game_info = boxscore.get("gameBoxInfo", [])
        weather_temp = None
        weather_wind = None
        weather_cond = None

        for info in game_info:
            label = info.get("label", "")
            value = info.get("value", "")
            if label == "Weather":
                # Parse "72 degrees, Cloudy" or similar
                try:
                    parts = value.split(",")
                    temp_str = parts[0].strip().split()[0]
                    weather_temp = int(temp_str)
                    if len(parts) > 1:
                        weather_cond = parts[1].strip().lower()
                except Exception:
                    pass
            elif label == "Wind":
                try:
                    wind_str = value.split()[0]
                    weather_wind = int(wind_str)
                except Exception:
                    pass

        # Insert
        try:
            execute("""
                INSERT INTO mlb_game_info (game_id, home_starter_id, away_starter_id,
                    weather_temp, weather_wind, weather_cond)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO NOTHING
            """, [game_id, home_starter_id, away_starter_id,
                  weather_temp, weather_wind, weather_cond])
            inserted += 1
        except Exception as e:
            continue

    print(f"  {year}: {inserted} game info records inserted")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Pull MLB game data")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2026, help="End year (inclusive)")
    parser.add_argument("--info", action="store_true", help="Also pull detailed game info (slower)")
    args = parser.parse_args()

    sport_id = get_sport_id()
    warm_caches(sport_id)
    print(f"\nMLB sport_id: {sport_id}")
    print(f"Pulling seasons {args.start} to {args.end}\n")

    total_games = 0
    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        count = pull_season_games(sport_id, year)
        total_games += count

        if args.info:
            pull_game_info(sport_id, year)

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_games} total new games inserted")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
