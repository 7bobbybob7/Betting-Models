"""
scrapers/odds/sbr_scraper.py - Daily SportsBookReview odds scraper.

Scrapes today's (or specified date's) odds from SportsBookReview.
Captures moneyline, spread, and totals from 6 sportsbooks.

Usage:
    python -m scrapers.odds.sbr_scraper                    # today
    python -m scrapers.odds.sbr_scraper --date 2026-04-04  # specific date
    python -m scrapers.odds.sbr_scraper --sport wnba       # other sports
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import re
import time
import requests
from datetime import date, timedelta

from db.db import query, bulk_insert


SPORT_CONFIG = {
    "mlb": {
        "url": "https://www.sportsbookreview.com/betting-odds/mlb-baseball",
        "db_sport": "mlb",
    },
    "wnba": {
        "url": "https://www.sportsbookreview.com/betting-odds/wnba-basketball",
        "db_sport": "wnba",
    },
}

MARKET_URLS = {
    "moneyline": "",            # default page
    "pointspread": "/pointspread",
    "totals": "/totals",
}

NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
})


def devig_american(home_ml, away_ml):
    if home_ml is None or away_ml is None:
        return None, None
    def to_imp(odds):
        return 100 / (odds + 100) if odds >= 0 else abs(odds) / (abs(odds) + 100)
    h, a = to_imp(home_ml), to_imp(away_ml)
    total = h + a
    if total == 0:
        return None, None
    return round(h / total, 4), round(a / total, 4)


def fetch_sbr_page(url, dt):
    """Fetch SBR page and extract __NEXT_DATA__ JSON."""
    full_url = f"{url}/?date={dt}"
    try:
        resp = SESSION.get(full_url, timeout=15)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {full_url}")
            return None
        match = NEXT_DATA_PATTERN.search(resp.text)
        if not match:
            return None
        return json.loads(match.group(1))
    except Exception as e:
        print(f"    Error fetching {full_url}: {e}")
        return None


def parse_games(data):
    """Extract games and odds from SBR __NEXT_DATA__."""
    tables = data.get("props", {}).get("pageProps", {}).get("oddsTables", [])
    if not tables:
        return []

    games = []
    for table in tables:
        rows = table.get("oddsTableModel", {}).get("gameRows", [])
        for row in rows:
            gv = row.get("gameView", {})
            games.append({
                "home_team": gv.get("homeTeam", {}).get("fullName", ""),
                "away_team": gv.get("awayTeam", {}).get("fullName", ""),
                "status": gv.get("gameStatusText", ""),
                "odds": row.get("oddsViews", row.get("odds", {})),
                "game_view": gv,
            })
    return games


def extract_moneyline_odds(game_data):
    """Extract moneyline odds from SBR game data."""
    odds = game_data.get("odds", {})
    if not odds:
        return []

    # The odds structure varies — handle both formats
    rows = []

    # Format 1: odds is a dict of sportsbook entries
    if isinstance(odds, dict):
        for book_key, book_data in odds.items():
            if not isinstance(book_data, dict):
                continue
            book_name = book_data.get("sportsbook", book_key)
            if isinstance(book_name, dict):
                book_name = book_name.get("name", book_key)

            current = book_data.get("currentLine", book_data)
            home_ml = current.get("homeOdds")
            away_ml = current.get("awayOdds")

            if home_ml is not None and away_ml is not None:
                rows.append({
                    "sportsbook": str(book_name).lower().replace(" ", "_")[:30],
                    "home_ml": float(home_ml),
                    "away_ml": float(away_ml),
                })

    # Format 2: odds is from the moneyline array in the Arnav-style format
    elif isinstance(odds, list):
        for entry in odds:
            if entry is None:
                continue
            book = entry.get("sportsbook", "unknown")
            current = entry.get("currentLine", {})
            if current is None:
                continue
            home_ml = current.get("homeOdds")
            away_ml = current.get("awayOdds")
            if home_ml is not None and away_ml is not None:
                home_ml_f = max(min(float(home_ml), 99999), -99999)
                away_ml_f = max(min(float(away_ml), 99999), -99999)
                rows.append({
                    "sportsbook": str(book).lower().replace(" ", "_")[:30],
                    "home_ml": home_ml_f,
                    "away_ml": away_ml_f,
                })

    return rows


def scrape_date(sport_key, dt):
    """Scrape all markets for a given date."""
    cfg = SPORT_CONFIG[sport_key]
    sport_id_result = query("SELECT sport_id FROM sports WHERE name = %s", [cfg["db_sport"]])
    if len(sport_id_result) == 0:
        print(f"  Sport {cfg['db_sport']} not found in DB")
        return 0

    sport_id = int(sport_id_result.iloc[0]["sport_id"])
    dt_str = dt.strftime("%Y-%m-%d")

    # Build game lookup for this date
    games_db = query("""
        SELECT g.game_id, g.game_date, ht.name as home_team, at.name as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.sport_id = %s AND g.game_date = %s
    """, [sport_id, dt_str])

    game_lookup = {}
    for _, r in games_db.iterrows():
        game_lookup[(r["home_team"], r["away_team"])] = int(r["game_id"])

    if not game_lookup:
        print(f"  No games in DB for {dt_str}")
        return 0

    print(f"  {dt_str}: {len(game_lookup)} games in DB")

    # Check what odds we already have (by market, so moneyline doesn't block totals)
    existing_ml = query("""
        SELECT DISTINCT game_id FROM odds
        WHERE market = 'moneyline' AND game_id IN (SELECT game_id FROM games WHERE sport_id = %s AND game_date = %s)
    """, [sport_id, dt_str])
    existing_ml_ids = set(existing_ml["game_id"]) if len(existing_ml) > 0 else set()

    existing_tot = query("""
        SELECT DISTINCT game_id FROM odds
        WHERE market = 'total' AND game_id IN (SELECT game_id FROM games WHERE sport_id = %s AND game_date = %s)
    """, [sport_id, dt_str])
    existing_tot_ids = set(existing_tot["game_id"]) if len(existing_tot) > 0 else set()

    # Fetch moneyline page
    data = fetch_sbr_page(cfg["url"], dt_str)
    if not data:
        print(f"  No data returned for {dt_str}")
        return 0

    games = parse_games(data)
    print(f"  SBR returned {len(games)} games")

    all_rows = []
    matched = 0

    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        # Match to our DB
        game_id = game_lookup.get((home, away))
        if not game_id:
            # Try fuzzy match
            for (h, a), gid in game_lookup.items():
                if home.lower() in h.lower() or h.lower() in home.lower():
                    if away.lower() in a.lower() or a.lower() in away.lower():
                        game_id = gid
                        break

        if not game_id:
            continue

        if game_id in existing_ml_ids:
            continue

        matched += 1

        # Extract moneyline odds
        ml_odds = extract_moneyline_odds(game)
        for ml in ml_odds:
            home_imp, away_imp = devig_american(ml["home_ml"], ml["away_ml"])
            all_rows.append((
                game_id, ml["sportsbook"], "moneyline",
                ml["home_ml"], ml["away_ml"], None,
                None, None,
                home_imp, away_imp, True,
            ))

    # Fetch totals page
    totals_data = fetch_sbr_page(cfg["url"] + "/totals", dt_str)
    if totals_data:
        totals_games = parse_games(totals_data)
        for game in totals_games:
            home = game["home_team"]
            away = game["away_team"]
            game_id = game_lookup.get((home, away))
            if not game_id:
                for (h, a), gid in game_lookup.items():
                    if home.lower() in h.lower() or h.lower() in home.lower():
                        if away.lower() in a.lower() or a.lower() in away.lower():
                            game_id = gid
                            break
            if not game_id or game_id in existing_tot_ids:
                continue

            odds = game.get("odds", {})

            # Handle list format (oddsViews from SBR)
            if isinstance(odds, list):
                for entry in odds:
                    if entry is None:
                        continue
                    book = entry.get("sportsbook", "unknown")
                    current = entry.get("currentLine", {})
                    if current is None:
                        continue
                    total = current.get("total")
                    over_odds_val = current.get("overOdds")
                    under_odds_val = current.get("underOdds")
                    if total is not None:
                        ov = max(min(float(over_odds_val), 99999), -99999) if over_odds_val else None
                        un = max(min(float(under_odds_val), 99999), -99999) if under_odds_val else None
                        all_rows.append((
                            game_id, str(book).lower().replace(" ", "_")[:30], "total",
                            None, None, float(total),
                            ov, un,
                            None, None, True,
                        ))

            # Handle dict format (older data)
            elif isinstance(odds, dict):
                for book_key, book_data in odds.items():
                    if not isinstance(book_data, dict):
                        continue
                    current = book_data.get("currentLine", book_data)
                    total = current.get("total")
                    over_odds_val = current.get("overOdds")
                    under_odds_val = current.get("underOdds")
                    if total is not None:
                        book_name = book_data.get("sportsbook", book_key)
                        if isinstance(book_name, dict):
                            book_name = book_name.get("name", book_key)
                        ov = max(min(float(over_odds_val), 99999), -99999) if over_odds_val else None
                        un = max(min(float(under_odds_val), 99999), -99999) if under_odds_val else None
                        all_rows.append((
                            game_id, str(book_name).lower().replace(" ", "_")[:30], "total",
                            None, None, float(total),
                            ov, un,
                            None, None, True,
                        ))
    else:
        print(f"    No totals data for {dt_str}")

    if all_rows:
        cols = [
            "game_id", "sportsbook", "market",
            "home_line", "away_line", "total_line",
            "over_odds", "under_odds",
            "home_implied", "away_implied", "is_closing"
        ]
        bulk_insert("odds", cols, all_rows)

    print(f"  Matched {matched} games, inserted {len(all_rows)} odds rows")
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description="SBR daily odds scraper")
    parser.add_argument("--sport", type=str, default="mlb", choices=list(SPORT_CONFIG.keys()))
    parser.add_argument("--date", type=str, default=None, help="Date (YYYY-MM-DD), default today")
    parser.add_argument("--days-back", type=int, default=0, help="Also scrape N days back")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SBR DAILY ODDS SCRAPER — {args.sport.upper()}")
    print(f"{'='*60}")

    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = date.today()

    total = 0
    for i in range(args.days_back, -1, -1):
        dt = target - timedelta(days=i)
        count = scrape_date(args.sport, dt)
        total += count
        if i > 0:
            time.sleep(2)  # be respectful

    print(f"\n  Total odds rows inserted: {total}")


if __name__ == "__main__":
    main()
