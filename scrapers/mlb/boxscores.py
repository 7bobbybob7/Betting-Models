"""
scrapers/mlb/boxscores.py - Pull MLB batting and pitching box scores.

Uses MLB-StatsAPI to fetch per-game player stats.
Populates: players, mlb_batting_game, mlb_pitching_game

Usage:
    python -m scrapers.mlb.boxscores --start 2015 --end 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import statsapi
from tqdm import tqdm

from db.db import get_conn, query, execute
from scrapers.mlb.games import ensure_player, get_sport_id


def get_games_needing_boxscores(sport_id, year):
    """Find games that have scores but no batting/pitching records yet."""
    result = query("""
        SELECT g.game_id, g.external_id, g.home_team_id, g.away_team_id
        FROM games g
        JOIN seasons s ON g.season_id = s.season_id
        LEFT JOIN mlb_batting_game bg ON g.game_id = bg.game_id
        WHERE g.sport_id = %s
          AND s.year = %s
          AND g.status = 'final'
          AND bg.stat_id IS NULL
        ORDER BY g.game_date
    """, [sport_id, year])
    return result


def pull_boxscore(sport_id, game_id, game_pk, home_team_id, away_team_id):
    """Pull and insert batting + pitching stats for a single game."""
    box = None
    for attempt in range(3):
        try:
            box = statsapi.boxscore_data(int(game_pk))
            break
        except Exception:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if box is None:
        return False

    batting_data = []
    pitching_data = []

    # Process both sides
    for side, team_id in [("home", home_team_id), ("away", away_team_id)]:
        batters_key = f"{side}Batters"
        pitchers_key = f"{side}Pitchers"

        # --- Batting ---
        batters = box.get(batters_key, [])
        batting_order = 0

        for batter in batters:
            pid = batter.get("personId")
            name = batter.get("name", "")
            if not pid or not name:
                continue
            # Skip header/summary rows
            if batter.get("namefield", "").strip() in ("", "Totals"):
                continue
            if batter.get("ab") is None:
                continue

            player_id = ensure_player(sport_id, pid, name)
            if not player_id:
                continue

            # Determine batting order (1-9 for starters, None for subs)
            batting_order_val = None
            namefield = batter.get("namefield", "")
            if namefield and not namefield.startswith("-"):
                batting_order += 1
                batting_order_val = batting_order if batting_order <= 9 else None

            batting_data.append((
                game_id,
                player_id,
                team_id,
                batting_order_val,
                _safe_int(batter.get("ab", 0)) + _safe_int(batter.get("bb", 0)) +
                    _safe_int(batter.get("hbp", 0)) + _safe_int(batter.get("sf", 0)) +
                    _safe_int(batter.get("sac", 0)),  # PA estimate
                _safe_int(batter.get("ab")),
                _safe_int(batter.get("h")),
                _safe_int(batter.get("doubles")),
                _safe_int(batter.get("triples")),
                _safe_int(batter.get("hr")),
                _safe_int(batter.get("rbi")),
                _safe_int(batter.get("bb")),
                _safe_int(batter.get("k")),
                _safe_int(batter.get("hbp")),
                _safe_int(batter.get("sb")),
                _safe_int(batter.get("cs")),
            ))

        # --- Pitching ---
        pitchers = box.get(pitchers_key, [])
        is_first = True

        for pitcher in pitchers:
            pid = pitcher.get("personId")
            name = pitcher.get("name", "")
            if not pid or not name:
                continue
            if pitcher.get("namefield", "").strip() in ("", "Totals"):
                continue
            if pitcher.get("ip") is None:
                continue

            player_id = ensure_player(sport_id, pid, name, "P")
            if not player_id:
                continue

            # Parse IP (e.g., "6.2" means 6 and 2/3 innings)
            ip_raw = pitcher.get("ip", "0")
            try:
                ip = float(ip_raw)
            except (ValueError, TypeError):
                ip = 0.0

            # Parse decision from namefield (e.g., "Smith (W, 5-2)")
            decision = None
            nf = pitcher.get("namefield", "")
            if "(W" in nf:
                decision = "W"
            elif "(L" in nf:
                decision = "L"
            elif "(S" in nf:
                decision = "S"
            elif "(H" in nf:
                decision = "H"

            pitching_data.append((
                game_id,
                player_id,
                team_id,
                is_first,  # is_starter
                ip,
                _safe_int(pitcher.get("h")),
                _safe_int(pitcher.get("r")),
                _safe_int(pitcher.get("er")),
                _safe_int(pitcher.get("bb")),
                _safe_int(pitcher.get("k")),
                _safe_int(pitcher.get("hr")),
                _safe_int(pitcher.get("p")),       # pitches
                _safe_int(pitcher.get("s")),       # strikes
                decision,
            ))
            is_first = False

    # Insert batting
    if batting_data:
        batting_cols = [
            "game_id", "player_id", "team_id", "batting_order",
            "pa", "ab", "hits", "doubles", "triples", "hr", "rbi",
            "bb", "so", "hbp", "sb", "cs"
        ]
        _bulk_insert_safe("mlb_batting_game", batting_cols, batting_data)

    # Insert pitching
    if pitching_data:
        pitching_cols = [
            "game_id", "player_id", "team_id", "is_starter",
            "ip", "hits_allowed", "runs", "earned_runs", "bb", "so",
            "hr_allowed", "pitches", "strikes", "decision"
        ]
        _bulk_insert_safe("mlb_pitching_game", pitching_cols, pitching_data)

    return True


def _safe_int(val):
    """Convert to int, handling None/NaN/empty."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _bulk_insert_safe(table, columns, data):
    """Insert with ON CONFLICT DO NOTHING, row by row if bulk fails."""
    from db.db import bulk_insert
    try:
        bulk_insert(table, columns, data)
    except Exception:
        # Fall back to row-by-row for duplicate handling
        cols = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        with get_conn() as conn:
            with conn.cursor() as cur:
                for row in data:
                    try:
                        cur.execute(sql, row)
                    except Exception:
                        conn.rollback()
                        continue
            conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Pull MLB box scores")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2026, help="End year (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Max games per year (for testing)")
    args = parser.parse_args()

    sport_id = get_sport_id()
    print(f"\nMLB sport_id: {sport_id}")
    print(f"Pulling box scores for {args.start} to {args.end}\n")

    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        games = get_games_needing_boxscores(sport_id, year)

        if len(games) == 0:
            print(f"  All games already have box scores")
            continue

        if args.limit:
            games = games.head(args.limit)

        print(f"  Pulling box scores for {len(games)} games...")
        success = 0
        failed = 0

        for _, row in tqdm(games.iterrows(), total=len(games),
                           desc=f"  Boxscores {year}", leave=False):
            ok = pull_boxscore(
                sport_id,
                int(row["game_id"]),
                row["external_id"],
                int(row["home_team_id"]),
                int(row["away_team_id"]),
            )
            if ok:
                success += 1
            else:
                failed += 1

        print(f"  {year}: {success} games processed, {failed} failed")

    print(f"\n{'='*60}")
    print("BOX SCORE PULL COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
