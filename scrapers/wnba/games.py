"""
scrapers/wnba/games.py - Pull WNBA game results and box scores from ESPN API.

Populates: teams, seasons, games, wnba_team_game_stats, wnba_player_game, players

Usage:
    python -m scrapers.wnba.games --start 2020 --end 2025
    python -m scrapers.wnba.games --start 2025 --end 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import requests
from datetime import date, timedelta
from tqdm import tqdm

from db.db import query, execute, bulk_insert


SPORT_ID = None
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
})

# Caches
_team_cache = {}
_player_cache = {}
_season_cache = {}


def get_sport_id():
    global SPORT_ID
    if SPORT_ID is None:
        r = query("SELECT sport_id FROM sports WHERE name = 'wnba'")
        SPORT_ID = int(r.iloc[0]["sport_id"])
    return SPORT_ID


def warm_caches():
    global _team_cache, _player_cache, _season_cache
    sid = get_sport_id()

    teams = query("SELECT team_id, name FROM teams WHERE sport_id = %s", [sid])
    for _, r in teams.iterrows():
        _team_cache[r["name"]] = int(r["team_id"])

    players = query("SELECT player_id, external_id FROM players WHERE sport_id = %s", [sid])
    for _, r in players.iterrows():
        _player_cache[str(r["external_id"])] = int(r["player_id"])

    seasons = query("SELECT season_id, year FROM seasons WHERE sport_id = %s", [sid])
    for _, r in seasons.iterrows():
        _season_cache[int(r["year"])] = int(r["season_id"])

    print(f"  Caches: {len(_team_cache)} teams, {len(_player_cache)} players, {len(_season_cache)} seasons")


def ensure_team(name, abbreviation=None):
    if name in _team_cache:
        return _team_cache[name]
    sid = get_sport_id()
    execute("INSERT INTO teams (sport_id, name, abbreviation) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            [sid, name, abbreviation])
    r = query("SELECT team_id FROM teams WHERE sport_id = %s AND name = %s", [sid, name])
    tid = int(r.iloc[0]["team_id"])
    _team_cache[name] = tid
    return tid


def ensure_season(year):
    if year in _season_cache:
        return _season_cache[year]
    sid = get_sport_id()
    execute("INSERT INTO seasons (sport_id, year) VALUES (%s, %s) ON CONFLICT DO NOTHING", [sid, year])
    r = query("SELECT season_id FROM seasons WHERE sport_id = %s AND year = %s", [sid, year])
    season_id = int(r.iloc[0]["season_id"])
    _season_cache[year] = season_id
    return season_id


def ensure_player(espn_id, name, position=None):
    ext = str(espn_id)
    if ext in _player_cache:
        return _player_cache[ext]
    sid = get_sport_id()
    execute("INSERT INTO players (sport_id, external_id, name, position) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [sid, ext, name, position])
    r = query("SELECT player_id FROM players WHERE sport_id = %s AND external_id = %s", [sid, ext])
    if len(r) > 0:
        pid = int(r.iloc[0]["player_id"])
        _player_cache[ext] = pid
        return pid
    return None


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def get_season_games(year):
    """Get all game event IDs for a WNBA season."""
    # WNBA runs May-October
    start = date(year, 5, 1)
    end = date(year, 10, 31)

    event_ids = []
    current = start

    while current <= end:
        dt_str = current.strftime("%Y%m%d")
        data = fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={dt_str}")
        if data:
            for event in data.get("events", []):
                event_ids.append(event["id"])
        current += timedelta(days=1)
        time.sleep(0.3)

    return list(set(event_ids))  # dedupe


def process_game(event_id, year):
    """Pull full box score for a single game and insert into DB."""
    sid = get_sport_id()
    season_id = ensure_season(year)

    # Check if already exists
    existing = query("SELECT game_id FROM games WHERE sport_id = %s AND external_id = %s", [sid, str(event_id)])
    if len(existing) > 0:
        return False  # already loaded

    # Fetch game summary
    data = fetch_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={event_id}")
    if not data:
        return False

    header = data.get("header", {})
    competitions = header.get("competitions", [])
    if not competitions:
        return False

    comp = competitions[0]
    status = comp.get("status", {}).get("type", {}).get("name", "")
    if status != "STATUS_FINAL":
        return False

    game_date_str = comp.get("date", "")[:10]
    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return False

    # Identify home/away
    home_comp = away_comp = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home_comp = c
        else:
            away_comp = c

    if not home_comp or not away_comp:
        return False

    home_name = home_comp.get("team", {}).get("displayName", "")
    away_name = away_comp.get("team", {}).get("displayName", "")
    home_abbr = home_comp.get("team", {}).get("abbreviation", "")
    away_abbr = away_comp.get("team", {}).get("abbreviation", "")
    home_score = int(home_comp.get("score", 0))
    away_score = int(away_comp.get("score", 0))

    home_tid = ensure_team(home_name, home_abbr)
    away_tid = ensure_team(away_name, away_abbr)

    venue = data.get("gameInfo", {}).get("venue", {}).get("fullName", "")

    # Insert game
    execute("""
        INSERT INTO games (sport_id, season_id, external_id, game_date, home_team_id, away_team_id,
            home_score, away_score, status, venue, is_postseason, is_neutral_site)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'final', %s, false, false)
        ON CONFLICT DO NOTHING
    """, [sid, season_id, str(event_id), game_date_str, home_tid, away_tid,
          home_score, away_score, venue])

    # Get game_id
    game_row = query("SELECT game_id FROM games WHERE sport_id = %s AND external_id = %s", [sid, str(event_id)])
    if len(game_row) == 0:
        return False
    game_id = int(game_row.iloc[0]["game_id"])

    # Parse box score
    boxscore = data.get("boxscore", {})

    # Team stats
    teams_box = boxscore.get("teams", [])
    for team_box in teams_box:
        team_info = team_box.get("team", {})
        team_name = team_info.get("displayName", "")
        tid = _team_cache.get(team_name)
        if not tid:
            continue

        is_home = tid == home_tid
        stats = {}
        for stat in team_box.get("statistics", []):
            stats[stat.get("name", "")] = stat.get("displayValue", "")

        # Parse team stats
        def _pi(key):
            v = stats.get(key, "")
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        def _parse_made_att(key):
            v = stats.get(key, "0-0")
            parts = v.split("-")
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass
            return None, None

        fgm, fga = _parse_made_att("fieldGoalsMade-fieldGoalsAttempted")
        fg3m, fg3a = _parse_made_att("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        ftm, fta = _parse_made_att("freeThrowsMade-freeThrowsAttempted")

        orb = _pi("offensiveRebounds")
        drb = _pi("defensiveRebounds")
        ast = _pi("assists")
        stl = _pi("steals")
        blk = _pi("blocks")
        tov = _pi("turnovers")
        fouls = _pi("fouls")
        pts = home_score if is_home else away_score

        # Compute efficiency metrics
        poss = fga - orb + tov + 0.475 * fta if all(v is not None for v in [fga, orb, tov, fta]) else None
        off_eff = round(pts / poss * 100, 2) if poss and poss > 0 else None
        efg = round((fgm + 0.5 * fg3m) / fga, 4) if fga and fga > 0 and fgm is not None and fg3m is not None else None
        tov_pct = round(tov / poss, 4) if poss and poss > 0 and tov is not None else None
        ft_rate = round(fta / fga, 4) if fga and fga > 0 and fta is not None else None

        execute("""
            INSERT INTO wnba_team_game_stats (game_id, team_id, is_home,
                points, fgm, fga, fg3m, fg3a, ftm, fta,
                offensive_rebounds, defensive_rebounds, assists, steals, blocks, turnovers, fouls,
                offensive_efficiency, efg_pct, turnover_pct, ft_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, [game_id, tid, is_home,
              pts, fgm, fga, fg3m, fg3a, ftm, fta,
              orb, drb, ast, stl, blk, tov, fouls,
              off_eff, efg, tov_pct, ft_rate])

    # Player stats
    players_box = boxscore.get("players", [])
    for team_players in players_box:
        team_info = team_players.get("team", {})
        team_name = team_info.get("displayName", "")
        tid = _team_cache.get(team_name)
        if not tid:
            continue

        stat_groups = team_players.get("statistics", [])
        if not stat_groups:
            continue

        labels = stat_groups[0].get("labels", [])
        athletes = stat_groups[0].get("athletes", [])

        for athlete in athletes:
            athlete_info = athlete.get("athlete", {})
            espn_id = athlete_info.get("id")
            name = athlete_info.get("displayName", "")
            position = athlete_info.get("position", {}).get("abbreviation", "")
            starter = athlete.get("starter", False)

            if not espn_id or not name:
                continue

            pid = ensure_player(espn_id, name, position)
            if not pid:
                continue

            stats_vals = athlete.get("stats", [])
            if not stats_vals or stats_vals[0] == "--":
                continue  # DNP

            # Map labels to values
            stat_map = {}
            for i, label in enumerate(labels):
                if i < len(stats_vals):
                    stat_map[label] = stats_vals[i]

            def _pstat(key):
                v = stat_map.get(key, "0")
                if v == "--" or v == "":
                    return None
                try:
                    return int(v)
                except ValueError:
                    try:
                        return int(float(v))
                    except (ValueError, TypeError):
                        return None

            def _pmin(key):
                v = stat_map.get(key, "0")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            def _pmade_att(key):
                v = stat_map.get(key, "0-0")
                parts = v.split("-")
                if len(parts) == 2:
                    try:
                        return int(parts[0]), int(parts[1])
                    except ValueError:
                        pass
                return None, None

            minutes = _pmin("MIN")
            pts = _pstat("PTS")
            fgm, fga = _pmade_att("FG")
            fg3m, fg3a = _pmade_att("3PT")
            ftm, fta = _pmade_att("FT")
            reb = _pstat("REB")
            ast = _pstat("AST")
            tov = _pstat("TO")
            stl = _pstat("STL")
            blk = _pstat("BLK")

            execute("""
                INSERT INTO wnba_player_game (game_id, player_id, team_id, is_starter,
                    minutes, points, fgm, fga, fg3m, fg3a, ftm, fta,
                    drb, assists, steals, blocks, turnovers, fouls)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, [game_id, pid, tid, starter,
                  minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta,
                  reb, ast, stl, blk, tov, None])

    return True


def main():
    parser = argparse.ArgumentParser(description="Pull WNBA data from ESPN")
    parser.add_argument("--start", type=int, default=2020)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  WNBA DATA PULL")
    print(f"  Seasons: {args.start} - {args.end}")
    print(f"{'='*60}")

    get_sport_id()
    warm_caches()

    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")

        # Get all event IDs for the season
        print(f"  Fetching schedule...")
        event_ids = get_season_games(year)
        print(f"  Found {len(event_ids)} games")

        if not event_ids:
            continue

        # Process each game
        success = 0
        skipped = 0
        for eid in tqdm(event_ids, desc=f"  {year}", leave=False):
            result = process_game(eid, year)
            if result:
                success += 1
            else:
                skipped += 1
            time.sleep(0.5)  # be respectful

        print(f"  {year}: {success} new, {skipped} skipped")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    sid = get_sport_id()
    tables = {
        "teams": f"SELECT COUNT(*) as cnt FROM teams WHERE sport_id = {sid}",
        "games": f"SELECT COUNT(*) as cnt FROM games WHERE sport_id = {sid}",
        "team_stats": "SELECT COUNT(*) as cnt FROM wnba_team_game_stats",
        "player_stats": "SELECT COUNT(*) as cnt FROM wnba_player_game",
        "players": f"SELECT COUNT(*) as cnt FROM players WHERE sport_id = {sid}",
    }
    for label, sql in tables.items():
        r = query(sql)
        print(f"  {label}: {int(r.iloc[0]['cnt']):,}")


if __name__ == "__main__":
    main()
