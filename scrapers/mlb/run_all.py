"""
scrapers/mlb/run_all.py - Run the full MLB data pull pipeline.

Order:
    1. Games (schedules + scores)
    2. Game info (starters, weather)
    3. Box scores (batting + pitching per game)
    4. Statcast (pitch-level data)

Usage:
    python -m scrapers.mlb.run_all --start 2015 --end 2026
    python -m scrapers.mlb.run_all --start 2025 --end 2026 --skip-statcast  # faster
    python -m scrapers.mlb.run_all --start 2026 --end 2026  # current season only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from db.db import query
from scrapers.mlb.games import get_sport_id, pull_season_games, pull_game_info, warm_caches
from scrapers.mlb.boxscores import get_games_needing_boxscores, pull_boxscore
from scrapers.mlb.statcast import pull_statcast_season, get_game_lookup, get_existing_statcast_games
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Full MLB data pull")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--skip-statcast", action="store_true",
                        help="Skip pitch-level Statcast (much faster)")
    parser.add_argument("--skip-boxscores", action="store_true",
                        help="Skip individual box scores")
    parser.add_argument("--skip-info", action="store_true",
                        help="Skip detailed game info (starters, weather)")
    args = parser.parse_args()

    sport_id = get_sport_id()
    warm_caches(sport_id)

    print("=" * 60)
    print("  MLB FULL DATA PULL")
    print(f"  Seasons: {args.start} - {args.end}")
    print("=" * 60)

    # --- Step 1: Games ---
    print("\n\n" + "=" * 60)
    print("STEP 1: GAME SCHEDULES & SCORES")
    print("=" * 60)
    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        pull_season_games(sport_id, year)

    # --- Step 2: Game Info ---
    if not args.skip_info:
        print(f"\n\n{'='*60}")
        print("STEP 2: GAME INFO (starters, weather)")
        print("=" * 60)
        for year in range(args.start, args.end + 1):
            print(f"\n--- {year} ---")
            pull_game_info(sport_id, year)

    # --- Step 3: Box Scores ---
    if not args.skip_boxscores:
        print(f"\n\n{'='*60}")
        print("STEP 3: BOX SCORES (batting + pitching)")
        print("=" * 60)
        for year in range(args.start, args.end + 1):
            print(f"\n--- {year} ---")
            games = get_games_needing_boxscores(sport_id, year)
            if len(games) == 0:
                print("  All box scores up to date")
                continue

            print(f"  Pulling {len(games)} box scores...")
            success = 0
            for _, row in tqdm(games.iterrows(), total=len(games),
                               desc=f"  {year}", leave=False):
                ok = pull_boxscore(
                    sport_id,
                    int(row["game_id"]),
                    row["external_id"],
                    int(row["home_team_id"]),
                    int(row["away_team_id"]),
                )
                if ok:
                    success += 1
            print(f"  {year}: {success}/{len(games)} box scores pulled")

    # --- Step 4: Statcast ---
    if not args.skip_statcast:
        print(f"\n\n{'='*60}")
        print("STEP 4: STATCAST PITCH DATA")
        print("=" * 60)
        game_lookup = get_game_lookup(sport_id)
        existing_games = get_existing_statcast_games()

        for year in range(args.start, args.end + 1):
            print(f"\n--- {year} ---")
            pull_statcast_season(sport_id, year, game_lookup, existing_games)

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print("DATA PULL COMPLETE - SUMMARY")
    print("=" * 60)

    tables = {
        "games (MLB)": "SELECT COUNT(*) as cnt FROM games WHERE sport_id = {}".format(sport_id),
        "mlb_game_info": "SELECT COUNT(*) as cnt FROM mlb_game_info",
        "players (MLB)": "SELECT COUNT(*) as cnt FROM players WHERE sport_id = {}".format(sport_id),
        "mlb_batting_game": "SELECT COUNT(*) as cnt FROM mlb_batting_game",
        "mlb_pitching_game": "SELECT COUNT(*) as cnt FROM mlb_pitching_game",
        "mlb_pitches": "SELECT COUNT(*) as cnt FROM mlb_pitches",
    }

    for label, sql in tables.items():
        try:
            result = query(sql)
            count = int(result.iloc[0]["cnt"])
            print(f"  {label}: {count:,} rows")
        except Exception:
            print(f"  {label}: error querying")


if __name__ == "__main__":
    main()
