"""
scripts/daily_refresh.py - Daily data refresh.

Pulls yesterday's game results and box scores from MLB Stats API.
Run this every morning to keep the database current.

Usage:
    python scripts/daily_refresh.py
    python scripts/daily_refresh.py --days 3  # backfill last 3 days
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from datetime import date, timedelta

from scrapers.mlb.games import get_sport_id, warm_caches, pull_season_games, pull_game_info, ensure_season
from scrapers.mlb.boxscores import get_games_needing_boxscores, pull_boxscore
from db.db import query


def main():
    parser = argparse.ArgumentParser(description="Daily data refresh")
    parser.add_argument("--days", type=int, default=1, help="How many days back to refresh")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  DAILY DATA REFRESH")
    print(f"{'='*60}")

    sport_id = get_sport_id()
    warm_caches(sport_id)

    today = date.today()
    current_year = today.year

    # Ensure current season exists
    ensure_season(sport_id, current_year)

    # Step 1: Pull recent games
    print(f"\n  Step 1: Pulling game schedules...")
    total_new = pull_season_games(sport_id, current_year)
    print(f"    {total_new} new games added")

    # Step 2: Pull game info (starters, weather) for recent games
    print(f"\n  Step 2: Pulling game info (starters, weather)...")
    pull_game_info(sport_id, current_year)

    # Step 3: Pull box scores for games that need them
    print(f"\n  Step 3: Pulling box scores...")
    games = get_games_needing_boxscores(sport_id, current_year)
    if len(games) == 0:
        print("    All box scores up to date")
    else:
        print(f"    {len(games)} games need box scores...")
        success = 0
        for _, row in games.iterrows():
            ok = pull_boxscore(
                sport_id,
                int(row["game_id"]),
                row["external_id"],
                int(row["home_team_id"]),
                int(row["away_team_id"]),
            )
            if ok:
                success += 1
        print(f"    {success}/{len(games)} box scores pulled")

    # Summary
    print(f"\n  {'='*60}")
    print("  REFRESH COMPLETE")
    print(f"  {'='*60}")

    counts = {
        "games": f"SELECT COUNT(*) as cnt FROM games WHERE sport_id = 2 AND game_date >= '{today - timedelta(days=7)}'",
        "batting": f"SELECT COUNT(*) as cnt FROM mlb_batting_game bg JOIN games g ON bg.game_id = g.game_id WHERE g.game_date >= '{today - timedelta(days=7)}'",
        "pitching": f"SELECT COUNT(*) as cnt FROM mlb_pitching_game pg JOIN games g ON pg.game_id = g.game_id WHERE g.game_date >= '{today - timedelta(days=7)}'",
    }
    for label, sql in counts.items():
        try:
            r = query(sql)
            print(f"    {label} (last 7 days): {int(r.iloc[0]['cnt'])}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
