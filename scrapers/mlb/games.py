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
    """Pull all regular season + postseason games for a given year.

    Inserts new games and updates existing games' status/scores. For scheduled
    games in the future, also pulls probable pitchers via schedule hydration.
    """
    season_id = ensure_season(sport_id, year)

    # Check what we already have (external_id -> (game_id, status, home_score, away_score))
    existing = query(
        "SELECT game_id, external_id, status, home_score, away_score FROM games WHERE sport_id = %s AND season_id = %s",
        [sport_id, season_id]
    )
    existing_map = {}
    for _, r in existing.iterrows():
        existing_map[str(r["external_id"])] = {
            "game_id": int(r["game_id"]),
            "status": r["status"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
        }

    # MLB regular season typically runs late March to early October
    # Postseason through November
    start_date = f"{year}-02-20"  # spring training starts, regular season late March
    end_date = f"{year}-11-15"

    print(f"  Fetching schedule for {year} ({start_date} to {end_date})...")
    schedule = None
    for attempt in range(3):
        try:
            # Use hydrated schedule to get probable pitcher IDs for upcoming games
            raw = statsapi.get("schedule", {
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "hydrate": "probablePitcher",
            })
            # Flatten dates -> games
            schedule = []
            for d in raw.get("dates", []):
                schedule.extend(d.get("games", []))
            break
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    if schedule is None:
        print(f"  SKIPPING {year} — API unavailable after 3 attempts")
        return 0

    games_to_insert = []
    probables_to_upsert = []  # (game_id, home_starter_id, away_starter_id)
    updates = []  # (status, home_score, away_score, game_id)
    new_count = 0
    updated_count = 0

    for game in tqdm(schedule, desc=f"  Processing {year}", leave=False):
        game_pk = str(game.get("gamePk", ""))
        if not game_pk:
            continue

        # Parse status
        status_obj = game.get("status", {})
        status_str = status_obj.get("abstractGameState", "") + " " + status_obj.get("detailedState", "")
        if "Final" in status_str or "Completed" in status_str:
            game_status = "final"
        elif "Live" in status_str or "In Progress" in status_str:
            game_status = "in_progress"
        elif "Preview" in status_str or "Scheduled" in status_str or "Pre-Game" in status_str or "Warmup" in status_str:
            game_status = "scheduled"
        else:
            continue

        # Skip spring training, all-star, etc.
        game_type = game.get("gameType", "R")
        if game_type not in ("R", "P", "W", "D", "L", "F"):
            continue

        teams = game.get("teams", {})
        home_team_data = teams.get("home", {}).get("team", {})
        away_team_data = teams.get("away", {}).get("team", {})
        home_name = home_team_data.get("name", "")
        away_name = away_team_data.get("name", "")
        if not home_name or not away_name:
            continue

        home_team_id = ensure_team(sport_id, home_name)
        away_team_id = ensure_team(sport_id, away_name)

        # Get game date (officialDate is preferred over gameDate for DST)
        game_date = game.get("officialDate") or game.get("gameDate", "")[:10]

        home_score = teams.get("home", {}).get("score")
        away_score = teams.get("away", {}).get("score")

        venue = game.get("venue", {}).get("name", "")
        is_postseason = game_type != "R"

        # Extract probable pitchers (hydrated)
        home_prob = teams.get("home", {}).get("probablePitcher", {})
        away_prob = teams.get("away", {}).get("probablePitcher", {})
        home_starter_id = None
        away_starter_id = None
        if home_prob and home_prob.get("id"):
            home_starter_id = ensure_player(sport_id, home_prob["id"], home_prob.get("fullName", ""), "P")
        if away_prob and away_prob.get("id"):
            away_starter_id = ensure_player(sport_id, away_prob["id"], away_prob.get("fullName", ""), "P")

        if game_pk in existing_map:
            # Game exists — check if status or scores changed
            ex = existing_map[game_pk]
            new_home_score = home_score if game_status == "final" else None
            new_away_score = away_score if game_status == "final" else None
            if (ex["status"] != game_status or
                ex["home_score"] != new_home_score or
                ex["away_score"] != new_away_score):
                updates.append((game_status, new_home_score, new_away_score, ex["game_id"]))
                updated_count += 1
            # Track probable pitcher info for mlb_game_info upsert
            if home_starter_id or away_starter_id:
                probables_to_upsert.append((ex["game_id"], home_starter_id, away_starter_id))
        else:
            # New game — insert
            games_to_insert.append((
                sport_id, season_id, game_pk, game_date,
                None,  # game_time
                home_team_id, away_team_id,
                home_score if game_status == "final" else None,
                away_score if game_status == "final" else None,
                game_status, venue, is_postseason, False,
            ))
            new_count += 1

    if games_to_insert:
        columns = [
            "sport_id", "season_id", "external_id", "game_date", "game_time",
            "home_team_id", "away_team_id", "home_score", "away_score",
            "status", "venue", "is_postseason", "is_neutral_site"
        ]
        bulk_insert("games", columns, games_to_insert)

    # Apply updates to existing games
    for game_status, h_score, a_score, game_id in updates:
        try:
            execute(
                "UPDATE games SET status = %s, home_score = %s, away_score = %s WHERE game_id = %s",
                [game_status, h_score, a_score, game_id]
            )
        except Exception as e:
            print(f"  Failed to update game {game_id}: {e}")

    # Upsert probable pitchers into mlb_game_info
    for game_id, h_starter, a_starter in probables_to_upsert:
        if h_starter is None and a_starter is None:
            continue
        try:
            execute("""
                INSERT INTO mlb_game_info (game_id, home_starter_id, away_starter_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE
                SET home_starter_id = COALESCE(EXCLUDED.home_starter_id, mlb_game_info.home_starter_id),
                    away_starter_id = COALESCE(EXCLUDED.away_starter_id, mlb_game_info.away_starter_id)
            """, [game_id, h_starter, a_starter])
        except Exception:
            pass

    print(f"  {year}: {new_count} new, {updated_count} updated, {len(probables_to_upsert)} with probable pitchers")
    return new_count


def pull_game_info(sport_id, year):
    """
    Pull detailed game info (starters, weather) for games that don't have it yet.
    This makes individual API calls per game, so it's slower.
    """
    season_id = ensure_season(sport_id, year)

    # Find games without mlb_game_info OR missing wind_dir/umpire
    games_without_info = query("""
        SELECT g.game_id, g.external_id
        FROM games g
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        WHERE g.sport_id = %s AND g.season_id = %s
          AND (gi.game_id IS NULL OR gi.weather_dir IS NULL OR gi.umpire_hp IS NULL)
          AND g.status = 'final'
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

        # Weather + umpire info (not always available)
        game_info = boxscore.get("gameBoxInfo", [])
        weather_temp = None
        weather_wind = None
        weather_dir = None
        weather_cond = None
        umpire_hp = None

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
                # Parse "8 mph, Out To CF" or "12 mph, In From LF"
                try:
                    wind_str = value.split()[0]
                    weather_wind = int(wind_str)
                    # Direction is everything after "mph, "
                    if "," in value:
                        weather_dir = value.split(",", 1)[1].strip().lower()
                except Exception:
                    pass
            elif label == "Umpires":
                # Parse "HP: Clint Vondrak. 1B: Brock Ballou. ..."
                try:
                    if "HP:" in value:
                        hp_part = value.split("HP:")[1].split(".")[0].strip()
                        umpire_hp = hp_part if hp_part else None
                except Exception:
                    pass

        # Insert
        try:
            execute("""
                INSERT INTO mlb_game_info (game_id, home_starter_id, away_starter_id,
                    weather_temp, weather_wind, weather_dir, weather_cond, umpire_hp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    weather_dir = COALESCE(EXCLUDED.weather_dir, mlb_game_info.weather_dir),
                    umpire_hp = COALESCE(EXCLUDED.umpire_hp, mlb_game_info.umpire_hp),
                    home_starter_id = COALESCE(EXCLUDED.home_starter_id, mlb_game_info.home_starter_id),
                    away_starter_id = COALESCE(EXCLUDED.away_starter_id, mlb_game_info.away_starter_id)
            """, [game_id, home_starter_id, away_starter_id,
                  weather_temp, weather_wind, weather_dir, weather_cond, umpire_hp])
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
