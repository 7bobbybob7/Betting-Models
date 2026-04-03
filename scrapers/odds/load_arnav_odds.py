"""
scrapers/odds/load_arnav_odds.py - Load odds from ArnavSaraogi/mlb-odds-scraper dataset.

Loads the 76MB JSON file with multi-book odds (BetMGM, FanDuel, Caesars,
Bet365, DraftKings, BetRivers) for 2021-2025 into the odds table.

Usage:
    python -m scrapers.odds.load_arnav_odds
    python -m scrapers.odds.load_arnav_odds --start 2024-05-01  # only load from this date
    python -m scrapers.odds.load_arnav_odds --compare            # compare with ESPN odds
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
from tqdm import tqdm

from db.db import query, bulk_insert


DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/mlb_odds_dataset.json")

# Map dataset sportsbook names to our DB names
BOOK_MAP = {
    "betmgm": "betmgm",
    "fanduel": "fanduel",
    "caesars": "caesars",
    "bet365": "bet365",
    "draftkings": "draftkings",
    "bet_rivers_ny": "betrivers",
}


def clamp_odds(val):
    """Clamp odds to fit DECIMAL(8,3) — abs value < 100000."""
    if val is None:
        return None
    if abs(val) > 99999:
        return -99999.0 if val < 0 else 99999.0
    return float(val)


def devig_american(home_ml, away_ml):
    """Convert American odds to de-vigged implied probabilities."""
    if home_ml is None or away_ml is None:
        return None, None

    def to_implied(odds):
        if odds >= 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    h = to_implied(home_ml)
    a = to_implied(away_ml)
    total = h + a
    if total == 0:
        return None, None
    return round(h / total, 4), round(a / total, 4)


def build_game_lookup():
    """Build (date, home_fullName, away_fullName) -> game_id lookup."""
    sport_id = query("SELECT sport_id FROM sports WHERE name = 'mlb'").iloc[0]["sport_id"]
    games = query("""
        SELECT g.game_id, g.game_date, ht.name as home_team, at.name as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.sport_id = %s AND g.status = 'final'
    """, [int(sport_id)])

    lookup = {}
    for _, row in games.iterrows():
        key = (str(row["game_date"]), row["home_team"], row["away_team"])
        lookup[key] = int(row["game_id"])
    return lookup


def get_existing_odds():
    """Get set of (game_id, sportsbook, market) already in odds table."""
    df = query("SELECT game_id, sportsbook, market FROM odds")
    return set(zip(df["game_id"], df["sportsbook"], df["market"]))


def load_odds(start_date=None):
    """Load the Arnav dataset into the odds table."""
    print("Loading dataset...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Dataset dates: {len(data)} days")

    print("Building game lookup...")
    game_lookup = build_game_lookup()
    print(f"  {len(game_lookup)} games in DB")

    existing = get_existing_odds()
    print(f"  {len(existing)} existing odds records")

    all_rows = []
    matched = 0
    unmatched = 0

    sorted_dates = sorted(data.keys())
    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
        print(f"  Filtering to dates >= {start_date}: {len(sorted_dates)} days")

    for date_str in tqdm(sorted_dates, desc="Processing"):
        games = data[date_str]
        game_date = date_str  # already YYYY-MM-DD

        for game in games:
            gv = game.get("gameView", {})

            # Skip non-final games
            if gv.get("gameStatusText", "") != "Final":
                continue
            # Skip spring training / all-star
            if gv.get("gameType", "R") not in ("R", "P", "W", "D", "L", "F"):
                continue

            home_name = gv.get("homeTeam", {}).get("fullName", "")
            away_name = gv.get("awayTeam", {}).get("fullName", "")
            if not home_name or not away_name:
                continue

            game_id = game_lookup.get((game_date, home_name, away_name))
            if not game_id:
                unmatched += 1
                continue
            matched += 1

            odds = game.get("odds", {})

            # --- Moneyline ---
            for entry in odds.get("moneyline", []):
                book_raw = entry.get("sportsbook", "")
                book = BOOK_MAP.get(book_raw, book_raw)
                line = entry.get("currentLine", {})
                home_ml = line.get("homeOdds")
                away_ml = line.get("awayOdds")

                if home_ml is None or away_ml is None:
                    continue
                if (game_id, book, "moneyline") in existing:
                    continue

                home_imp, away_imp = devig_american(home_ml, away_ml)
                all_rows.append((
                    game_id, book, "moneyline",
                    clamp_odds(home_ml), clamp_odds(away_ml), None,
                    None, None,
                    home_imp, away_imp, True
                ))

            # --- Spread (run line) ---
            for entry in odds.get("pointspread", []):
                book_raw = entry.get("sportsbook", "")
                book = BOOK_MAP.get(book_raw, book_raw)
                line = entry.get("currentLine", {})
                home_spread = line.get("homeSpread")
                home_odds = line.get("homeOdds")
                away_odds = line.get("awayOdds")

                if home_spread is None:
                    continue
                if (game_id, book, "spread") in existing:
                    continue

                all_rows.append((
                    game_id, book, "spread",
                    float(home_spread), float(-home_spread), None,
                    clamp_odds(home_odds), clamp_odds(away_odds),
                    None, None, True
                ))

            # --- Totals ---
            for entry in odds.get("totals", []):
                book_raw = entry.get("sportsbook", "")
                book = BOOK_MAP.get(book_raw, book_raw)
                line = entry.get("currentLine", {})
                total = line.get("total")
                over_odds = line.get("overOdds")
                under_odds = line.get("underOdds")

                if total is None:
                    continue
                if (game_id, book, "total") in existing:
                    continue

                all_rows.append((
                    game_id, book, "total",
                    None, None, float(total),
                    clamp_odds(over_odds), clamp_odds(under_odds),
                    None, None, True
                ))

    print(f"\nMatched: {matched} games, Unmatched: {unmatched}")
    print(f"New odds rows to insert: {len(all_rows)}")

    if all_rows:
        # Insert in chunks to avoid memory issues
        chunk_size = 50000
        for i in range(0, len(all_rows), chunk_size):
            chunk = all_rows[i:i + chunk_size]
            cols = [
                "game_id", "sportsbook", "market",
                "home_line", "away_line", "total_line",
                "over_odds", "under_odds",
                "home_implied", "away_implied", "is_closing"
            ]
            bulk_insert("odds", cols, chunk)

    return matched, len(all_rows)


def compare_with_espn():
    """Compare Arnav dataset odds vs ESPN odds for overlapping games."""
    print("\n" + "=" * 60)
    print("COMPARING ARNAV vs ESPN ODDS")
    print("=" * 60)

    # Get ESPN odds (sportsbooks from ESPN)
    espn_books = query("""
        SELECT DISTINCT sportsbook FROM odds
        WHERE sportsbook LIKE 'espn%%'
           OR sportsbook LIKE 'draftkings%%'
           OR sportsbook LIKE 'bet365%%'
           OR sportsbook LIKE 'caesars%%'
           OR sportsbook LIKE 'mgm%%'
    """)
    print(f"\nESPN-sourced sportsbooks in DB:")
    for _, r in espn_books.iterrows():
        print(f"  {r['sportsbook']}")

    # Find games with odds from BOTH sources
    # Arnav uses: betmgm, fanduel, caesars, bet365, draftkings, betrivers
    # ESPN uses: espn_bet, draftkings_old, bet365, caesars_*, mgm, etc.

    # Compare DraftKings moneylines where both exist
    comparison = query("""
        SELECT
            o1.game_id,
            g.game_date,
            ht.name as home_team,
            o1.home_line as arnav_home_ml,
            o1.away_line as arnav_away_ml,
            o2.home_line as espn_home_ml,
            o2.away_line as espn_away_ml,
            o1.home_implied as arnav_home_imp,
            o2.home_implied as espn_home_imp
        FROM odds o1
        JOIN odds o2 ON o1.game_id = o2.game_id
            AND o2.market = 'moneyline'
        JOIN games g ON o1.game_id = g.game_id
        JOIN teams ht ON g.home_team_id = ht.team_id
        WHERE o1.sportsbook = 'draftkings'
          AND o1.market = 'moneyline'
          AND o2.sportsbook LIKE 'draftkings%%'
          AND o2.sportsbook != 'draftkings'
        ORDER BY g.game_date
        LIMIT 20
    """)

    if len(comparison) == 0:
        # Try a broader comparison
        comparison = query("""
            WITH arnav AS (
                SELECT game_id, home_line, away_line, home_implied
                FROM odds
                WHERE sportsbook = 'fanduel' AND market = 'moneyline'
            ),
            espn AS (
                SELECT game_id, home_line, away_line, home_implied
                FROM odds
                WHERE sportsbook = 'espn_bet' AND market = 'moneyline'
            )
            SELECT
                a.game_id,
                g.game_date,
                ht.name as home_team,
                a.home_line as arnav_fanduel_ml,
                e.home_line as espn_bet_ml,
                a.home_implied as arnav_implied,
                e.home_implied as espn_implied,
                ABS(a.home_implied - e.home_implied) as imp_diff
            FROM arnav a
            JOIN espn e ON a.game_id = e.game_id
            JOIN games g ON a.game_id = g.game_id
            JOIN teams ht ON g.home_team_id = ht.team_id
            ORDER BY g.game_date
            LIMIT 20
        """)

    if len(comparison) > 0:
        print(f"\nSample comparison (first 20 games with both sources):")
        print(comparison.to_string())

        # Summary stats
        if "imp_diff" in comparison.columns:
            print(f"\nImplied probability difference (FanDuel vs ESPN BET):")
            print(f"  Mean:   {comparison['imp_diff'].mean():.4f}")
            print(f"  Median: {comparison['imp_diff'].median():.4f}")
            print(f"  Max:    {comparison['imp_diff'].max():.4f}")
    else:
        print("\nNo overlapping games found between sources yet.")
        print("This may be because ESPN odds are still loading.")

    # Show coverage by source
    coverage = query("""
        SELECT
            sportsbook,
            market,
            COUNT(*) as cnt,
            MIN(g.game_date) as earliest,
            MAX(g.game_date) as latest
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        WHERE g.sport_id = 2
        GROUP BY sportsbook, market
        ORDER BY sportsbook, market
    """)
    print(f"\nOdds coverage by sportsbook + market:")
    print(coverage.to_string())


def main():
    parser = argparse.ArgumentParser(description="Load Arnav MLB odds dataset")
    parser.add_argument("--start", type=str, default=None,
                        help="Only load dates >= this (YYYY-MM-DD)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with ESPN odds after loading")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Download from: https://github.com/ArnavSaraogi/mlb-odds-scraper/releases")
        sys.exit(1)

    matched, inserted = load_odds(start_date=args.start)

    # Summary
    print(f"\n{'='*60}")
    print("LOAD COMPLETE")
    print(f"{'='*60}")
    total = query("SELECT COUNT(*) as cnt FROM odds")
    print(f"Total odds in DB: {int(total.iloc[0]['cnt']):,}")

    by_book = query("""
        SELECT sportsbook, COUNT(*) as cnt
        FROM odds
        GROUP BY sportsbook
        ORDER BY cnt DESC
    """)
    print(f"\nBy sportsbook:")
    print(by_book.to_string())

    if args.compare:
        compare_with_espn()


if __name__ == "__main__":
    main()
