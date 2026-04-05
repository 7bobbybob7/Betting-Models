"""
scrapers/wnba/load_history.py - Load WNBA historical data from nba_api.

Pulls team-level game results and box scores for all WNBA seasons.
Much deeper history than ESPN (2003+ vs 2017+).

Usage:
    python -m scrapers.wnba.load_history --start 2003 --end 2024
    python -m scrapers.wnba.load_history --start 2020 --end 2024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder

from db.db import query, execute, bulk_insert


SPORT_ID = None
_team_cache = {}
_season_cache = {}
_game_cache = set()


def get_sport_id():
    global SPORT_ID
    if SPORT_ID is None:
        r = query("SELECT sport_id FROM sports WHERE name = 'wnba'")
        SPORT_ID = int(r.iloc[0]["sport_id"])
    return SPORT_ID


def warm_caches():
    global _team_cache, _season_cache, _game_cache
    sid = get_sport_id()

    teams = query("SELECT team_id, name FROM teams WHERE sport_id = %s", [sid])
    for _, r in teams.iterrows():
        _team_cache[r["name"]] = int(r["team_id"])

    seasons = query("SELECT season_id, year FROM seasons WHERE sport_id = %s", [sid])
    for _, r in seasons.iterrows():
        _season_cache[int(r["year"])] = int(r["season_id"])

    games = query("SELECT external_id FROM games WHERE sport_id = %s", [sid])
    _game_cache = set(games["external_id"].astype(str))

    print(f"  Caches: {len(_team_cache)} teams, {len(_season_cache)} seasons, {len(_game_cache)} existing games")


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


def load_season(year):
    """Load one WNBA season from nba_api."""
    sid = get_sport_id()
    season_id = ensure_season(year)

    print(f"  Fetching {year} from nba_api...")
    try:
        result = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='10',
            season_nullable=str(year),
            season_type_nullable='Regular Season'
        )
        df = result.get_data_frames()[0]
    except Exception as e:
        print(f"    ERROR: {e}")
        return 0

    if len(df) == 0:
        print(f"    No data for {year}")
        return 0

    print(f"    Raw rows: {len(df)} (each game appears twice)")

    # Each game appears twice (one row per team). Group by GAME_ID.
    games_grouped = df.groupby("GAME_ID")

    new_games = 0
    game_rows = []
    stat_rows = []

    for game_id, game_df in games_grouped:
        ext_id = str(game_id)

        if ext_id in _game_cache:
            continue

        if len(game_df) != 2:
            continue

        # Determine home/away from MATCHUP field
        # "CHI vs. CON" = CHI is home, "CHI @ CON" = CHI is away
        row1 = game_df.iloc[0]
        row2 = game_df.iloc[1]

        if "vs." in str(row1["MATCHUP"]):
            home_row, away_row = row1, row2
        elif "@" in str(row1["MATCHUP"]):
            home_row, away_row = row2, row1
        else:
            continue

        home_name = home_row["TEAM_NAME"]
        away_name = away_row["TEAM_NAME"]
        home_abbr = home_row["TEAM_ABBREVIATION"]
        away_abbr = away_row["TEAM_ABBREVIATION"]

        home_tid = ensure_team(home_name, home_abbr)
        away_tid = ensure_team(away_name, away_abbr)

        game_date = str(home_row["GAME_DATE"])[:10]
        home_score = int(home_row["PTS"])
        away_score = int(away_row["PTS"])

        game_rows.append((
            sid, season_id, ext_id, game_date, None,
            home_tid, away_tid, home_score, away_score,
            "final", None, False, False
        ))
        _game_cache.add(ext_id)

        # Team stats for both sides
        for row, tid, is_home in [(home_row, home_tid, True), (away_row, away_tid, False)]:
            fgm = _si(row.get("FGM"))
            fga = _si(row.get("FGA"))
            fg3m = _si(row.get("FG3M"))
            fg3a = _si(row.get("FG3A"))
            ftm = _si(row.get("FTM"))
            fta = _si(row.get("FTA"))
            orb = _si(row.get("OREB"))
            drb = _si(row.get("DREB"))
            ast = _si(row.get("AST"))
            stl = _si(row.get("STL"))
            blk = _si(row.get("BLK"))
            tov = _si(row.get("TOV"))
            fouls = _si(row.get("PF"))
            pts = int(row["PTS"])

            # Derived metrics
            poss = fga - orb + tov + 0.475 * fta if all(v is not None for v in [fga, orb, tov, fta]) else None
            off_eff = round(pts / poss * 100, 2) if poss and poss > 0 else None
            efg = round((fgm + 0.5 * fg3m) / fga, 4) if fga and fga > 0 and fgm is not None and fg3m is not None else None
            tov_pct = round(tov / poss, 4) if poss and poss > 0 and tov is not None else None
            ft_rate = round(fta / fga, 4) if fga and fga > 0 and fta is not None else None

            stat_rows.append((
                ext_id,  # placeholder — will resolve to game_id after insert
                tid, is_home, pts, fgm, fga, fg3m, fg3a, ftm, fta,
                orb, drb, ast, stl, blk, tov, fouls,
                off_eff, None, None, efg, tov_pct, None, ft_rate
            ))

        new_games += 1

    # Insert games
    if game_rows:
        game_cols = [
            "sport_id", "season_id", "external_id", "game_date", "game_time",
            "home_team_id", "away_team_id", "home_score", "away_score",
            "status", "venue", "is_postseason", "is_neutral_site"
        ]
        bulk_insert("games", game_cols, game_rows)

    # Now resolve external_id -> game_id for stats
    if stat_rows:
        game_id_map = {}
        gdf = query("SELECT game_id, external_id FROM games WHERE sport_id = %s AND season_id = %s",
                     [sid, season_id])
        for _, r in gdf.iterrows():
            game_id_map[str(r["external_id"])] = int(r["game_id"])

        resolved_stats = []
        for row in stat_rows:
            ext_id = row[0]
            game_id = game_id_map.get(ext_id)
            if game_id:
                resolved_stats.append((game_id,) + row[1:])

        if resolved_stats:
            stat_cols = [
                "game_id", "team_id", "is_home",
                "points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                "offensive_rebounds", "defensive_rebounds",
                "assists", "steals", "blocks", "turnovers", "fouls",
                "offensive_efficiency", "defensive_efficiency", "tempo",
                "efg_pct", "turnover_pct", "orb_pct", "ft_rate"
            ]
            bulk_insert("wnba_team_game_stats", stat_cols, resolved_stats)

    print(f"    {new_games} new games inserted")
    return new_games


def _si(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Load WNBA history from nba_api")
    parser.add_argument("--start", type=int, default=2003)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  WNBA HISTORICAL DATA LOAD (nba_api)")
    print(f"  Seasons: {args.start} - {args.end}")
    print(f"{'='*60}")

    get_sport_id()
    warm_caches()

    total = 0
    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        count = load_season(year)
        total += count
        time.sleep(2)  # respect rate limits

    # Summary
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {total} total new games")
    print(f"{'='*60}")

    sid = get_sport_id()
    for label, sql in [
        ("teams", f"SELECT COUNT(*) as cnt FROM teams WHERE sport_id = {sid}"),
        ("seasons", f"SELECT COUNT(*) as cnt FROM seasons WHERE sport_id = {sid}"),
        ("games", f"SELECT COUNT(*) as cnt FROM games WHERE sport_id = {sid}"),
        ("team_stats", "SELECT COUNT(*) as cnt FROM wnba_team_game_stats"),
    ]:
        r = query(sql)
        print(f"  {label}: {int(r.iloc[0]['cnt']):,}")

    # Games by season
    print(f"\n  By season:")
    by_season = query(f"""
        SELECT s.year, COUNT(*) as games
        FROM games g JOIN seasons s ON g.season_id = s.season_id
        WHERE g.sport_id = {sid}
        GROUP BY s.year ORDER BY s.year
    """)
    for _, r in by_season.iterrows():
        print(f"    {int(r['year'])}: {int(r['games'])} games")


if __name__ == "__main__":
    main()
