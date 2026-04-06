"""
scrapers/odds/espn_odds.py - Pull historical odds from ESPN's public API.

Pulls moneyline, spread, and totals with open/close lines from multiple
sportsbooks. ESPN has multi-book data for 2015-early 2024, then ESPN BET
only for late 2024+.

Supports: MLB, CBB, WNBA (any sport ESPN covers).

Usage:
    python -m scrapers.odds.espn_odds --sport mlb --start 2015 --end 2024
    python -m scrapers.odds.espn_odds --sport mlb --start 2024 --end 2024 --month 6
    python -m scrapers.odds.espn_odds --sport cbb --start 2023 --end 2024
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import time
import requests
from datetime import date, timedelta
from tqdm import tqdm

from db.db import query, bulk_insert


# ---------------------------------------------------------------------------
# ESPN API config
# ---------------------------------------------------------------------------
SPORT_CONFIG = {
    "mlb": {
        "espn_sport": "baseball",
        "espn_league": "mlb",
        "season_start": (3, 20),   # March 20
        "season_end": (11, 15),    # Nov 15
    },
    "cbb": {
        "espn_sport": "basketball",
        "espn_league": "mens-college-basketball",
        "season_start": (11, 1),   # Nov 1
        "season_end": (4, 15),     # Apr 15 (next year)
    },
    "wnba": {
        "espn_sport": "basketball",
        "espn_league": "wnba",
        "season_start": (5, 1),    # May 1
        "season_end": (10, 31),    # Oct 31
    },
}

# Sportsbooks we care about (skip consensus/prediction providers)
SKIP_PROVIDERS = {"Consensus", "TeamRankings", "NumberFire"}

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
ODDS_URL = "https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/events/{event_id}/competitions/{event_id}/odds"

SESSION = requests.Session()


# ---------------------------------------------------------------------------
# De-vig helper
# ---------------------------------------------------------------------------
def devig_american(home_ml, away_ml):
    """Convert American odds to de-vigged implied probabilities (power method)."""
    if home_ml is None or away_ml is None:
        return None, None

    def american_to_implied(odds):
        if odds >= 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    h_imp = american_to_implied(home_ml)
    a_imp = american_to_implied(away_ml)
    total = h_imp + a_imp

    if total == 0:
        return None, None

    return round(h_imp / total, 4), round(a_imp / total, 4)


# ---------------------------------------------------------------------------
# ESPN API helpers
# ---------------------------------------------------------------------------
def fetch_json(url, params=None, retries=3):
    """Fetch JSON with retry logic."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None


def get_espn_events_for_date(sport_cfg, dt):
    """Get all ESPN events for a single date. Returns list of (event_id, home_team, away_team)."""
    url = SCOREBOARD_URL.format(sport=sport_cfg["espn_sport"], league=sport_cfg["espn_league"])
    data = fetch_json(url, params={"dates": dt.strftime("%Y%m%d"), "limit": 100})
    if not data:
        return []

    events = []
    for event in data.get("events", []):
        event_id = event.get("id")
        if not event_id:
            continue

        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_name = None
        away_name = None
        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            if c.get("homeAway") == "home":
                home_name = team_name
            else:
                away_name = team_name

        if home_name and away_name:
            events.append((event_id, home_name, away_name))

    return events


def get_odds_for_event(sport_cfg, event_id):
    """Fetch all odds providers for an ESPN event."""
    url = ODDS_URL.format(
        sport=sport_cfg["espn_sport"],
        league=sport_cfg["espn_league"],
        event_id=event_id
    )
    data = fetch_json(url)
    if not data:
        return []

    return data.get("items", [])


def parse_odds_item(item):
    """Extract structured odds from a single ESPN odds item."""
    provider = item.get("provider", {})
    provider_name = provider.get("name", "unknown")
    provider_id = provider.get("id", "")

    if provider_name in SKIP_PROVIDERS:
        return []

    # Skip live odds providers
    if "Live" in provider_name:
        return []

    sportsbook = provider_name.lower().replace(" ", "_").replace("(", "").replace(")", "")[:30]

    rows = []

    # --- Moneyline ---
    home_odds = item.get("homeTeamOdds", {})
    away_odds = item.get("awayTeamOdds", {})
    home_ml = home_odds.get("moneyLine")
    away_ml = away_odds.get("moneyLine")

    # Prefer close lines, fall back to current, then top-level
    home_close = home_odds.get("close", {}).get("moneyLine", {})
    away_close = away_odds.get("close", {}).get("moneyLine", {})

    # Use close ML if available (american format)
    close_home_ml = None
    close_away_ml = None
    if home_close:
        alt = home_close.get("alternateDisplayValue", "")
        if alt:
            try:
                close_home_ml = int(alt)
            except ValueError:
                pass
    if away_close:
        alt = away_close.get("alternateDisplayValue", "")
        if alt:
            try:
                close_away_ml = int(alt)
            except ValueError:
                pass

    # Use close if available, otherwise top-level
    final_home_ml = close_home_ml if close_home_ml is not None else home_ml
    final_away_ml = close_away_ml if close_away_ml is not None else away_ml

    if final_home_ml is not None and final_away_ml is not None:
        home_imp, away_imp = devig_american(final_home_ml, final_away_ml)
        rows.append({
            "sportsbook": sportsbook,
            "market": "moneyline",
            "home_line": float(final_home_ml),
            "away_line": float(final_away_ml),
            "total_line": None,
            "over_odds": None,
            "under_odds": None,
            "home_implied": home_imp,
            "away_implied": away_imp,
            "is_closing": True,
        })

    # --- Spread ---
    # Get the ACTUAL home spread from pointSpread (not the top-level spread which is always positive)
    home_point_spread = home_odds.get("close", {}).get("pointSpread", {}) or home_odds.get("current", {}).get("pointSpread", {})
    home_spread_odds = home_odds.get("close", {}).get("spread", {}) or home_odds.get("current", {}).get("spread", {})
    away_spread_odds = away_odds.get("close", {}).get("spread", {}) or away_odds.get("current", {}).get("spread", {})

    # Extract actual home spread value (e.g., "+1.5" or "-1.5")
    home_spread_val = None
    if home_point_spread:
        ps_str = home_point_spread.get("alternateDisplayValue", "") or home_point_spread.get("american", "")
        try:
            home_spread_val = float(ps_str)
        except (ValueError, TypeError):
            pass

    if home_spread_val is not None:
        home_spread_american = None
        away_spread_american = None
        if home_spread_odds:
            alt = home_spread_odds.get("alternateDisplayValue", "")
            try:
                home_spread_american = int(alt)
            except (ValueError, TypeError):
                pass
        if away_spread_odds:
            alt = away_spread_odds.get("alternateDisplayValue", "")
            try:
                away_spread_american = int(alt)
            except (ValueError, TypeError):
                pass

        rows.append({
            "sportsbook": sportsbook,
            "market": "spread",
            "home_line": float(home_spread_val),
            "away_line": float(-home_spread_val),
            "total_line": None,
            "over_odds": float(home_spread_american) if home_spread_american else None,
            "under_odds": float(away_spread_american) if away_spread_american else None,
            "home_implied": None,
            "away_implied": None,
            "is_closing": True,
        })

    # --- Totals ---
    over_under = item.get("overUnder")
    if over_under is not None:
        # Prefer close totals, fall back to top-level
        close_totals = item.get("close", {})
        over_american = item.get("overOdds")
        under_american = item.get("underOdds")

        if close_totals:
            over_info = close_totals.get("over", {})
            under_info = close_totals.get("under", {})
            if over_info.get("alternateDisplayValue"):
                try:
                    over_american = int(over_info["alternateDisplayValue"])
                except (ValueError, TypeError):
                    pass
            if under_info.get("alternateDisplayValue"):
                try:
                    under_american = int(under_info["alternateDisplayValue"])
                except (ValueError, TypeError):
                    pass

        rows.append({
            "sportsbook": sportsbook,
            "market": "total",
            "home_line": None,
            "away_line": None,
            "total_line": float(over_under),
            "over_odds": float(over_american) if over_american else None,
            "under_odds": float(under_american) if under_american else None,
            "home_implied": None,
            "away_implied": None,
            "is_closing": True,
        })

    return rows


# ---------------------------------------------------------------------------
# Main scraper logic
# ---------------------------------------------------------------------------
def build_game_lookup(sport_id):
    """Build (game_date, home_team_name, away_team_name) -> game_id lookup."""
    games = query("""
        SELECT g.game_id, g.game_date, ht.name as home_team, at.name as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.sport_id = %s AND g.status = 'final'
    """, [sport_id])

    lookup = {}
    for _, row in games.iterrows():
        gd = str(row["game_date"])
        key = (gd, row["home_team"], row["away_team"])
        lookup[key] = int(row["game_id"])
    return lookup


def get_existing_odds_game_ids():
    """Get set of game_ids that already have odds from ESPN."""
    df = query("SELECT DISTINCT game_id FROM odds")
    return set(df["game_id"])


def fuzzy_match_team(espn_name, our_teams):
    """Try to match ESPN team name to our DB name. Returns our name or None."""
    # Exact match
    if espn_name in our_teams:
        return espn_name

    # ESPN sometimes uses slightly different names
    espn_lower = espn_name.lower()
    for name in our_teams:
        if name.lower() == espn_lower:
            return name
        # Partial match: "New York Yankees" matches "New York Yankees"
        # Handle cases like "LA Dodgers" vs "Los Angeles Dodgers"
        if espn_lower in name.lower() or name.lower() in espn_lower:
            return name

    return None


def scrape_date(sport_cfg, dt, game_lookup, our_teams, existing_odds_ids):
    """Scrape odds for all games on a given date. Returns count of new odds rows inserted."""
    events = get_espn_events_for_date(sport_cfg, dt)
    if not events:
        return 0

    all_rows = []

    for event_id, espn_home, espn_away in events:
        # Match ESPN teams to our DB teams
        our_home = fuzzy_match_team(espn_home, our_teams)
        our_away = fuzzy_match_team(espn_away, our_teams)
        if not our_home or not our_away:
            continue

        # Find game_id
        game_key = (str(dt), our_home, our_away)
        game_id = game_lookup.get(game_key)
        if not game_id:
            continue

        # Skip if already have odds
        if game_id in existing_odds_ids:
            continue

        # Fetch odds
        odds_items = get_odds_for_event(sport_cfg, event_id)
        for item in odds_items:
            parsed = parse_odds_item(item)
            for row in parsed:
                all_rows.append((
                    game_id,
                    row["sportsbook"],
                    row["market"],
                    row["home_line"],
                    row["away_line"],
                    row["total_line"],
                    row["over_odds"],
                    row["under_odds"],
                    row["home_implied"],
                    row["away_implied"],
                    row["is_closing"],
                ))

    if all_rows:
        cols = [
            "game_id", "sportsbook", "market",
            "home_line", "away_line", "total_line",
            "over_odds", "under_odds",
            "home_implied", "away_implied", "is_closing"
        ]
        bulk_insert("odds", cols, all_rows)
        for gid in set(r[0] for r in all_rows):
            existing_odds_ids.add(gid)

    return len(all_rows)


def get_season_dates(sport_key, year):
    """Generate list of dates for a season."""
    cfg = SPORT_CONFIG[sport_key]
    start_month, start_day = cfg["season_start"]
    end_month, end_day = cfg["season_end"]

    if sport_key == "cbb":
        # CBB season crosses year boundary: Nov {year-1} to Apr {year}
        start = date(year - 1, start_month, start_day)
        end = date(year, end_month, end_day)
    else:
        start = date(year, start_month, start_day)
        end = date(year, end_month, end_day)

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="Pull historical odds from ESPN")
    parser.add_argument("--sport", type=str, required=True, choices=["mlb", "cbb", "wnba"])
    parser.add_argument("--start", type=int, required=True, help="Start season year")
    parser.add_argument("--end", type=int, required=True, help="End season year (inclusive)")
    parser.add_argument("--month", type=int, default=None, help="Only scrape a specific month (1-12)")
    args = parser.parse_args()

    sport_cfg = SPORT_CONFIG[args.sport]

    # Get sport_id
    sport_result = query("SELECT sport_id FROM sports WHERE name = %s", [args.sport])
    sport_id = int(sport_result.iloc[0]["sport_id"])

    print(f"\n{'='*60}")
    print(f"  ESPN ODDS SCRAPER — {args.sport.upper()}")
    print(f"  Seasons: {args.start} - {args.end}")
    print(f"{'='*60}")

    # Build lookups
    print("\nBuilding game lookup...")
    game_lookup = build_game_lookup(sport_id)
    our_teams = set(name for (_, name, _) in game_lookup.keys())
    print(f"  {len(game_lookup)} games, {len(our_teams)} teams")

    existing_odds_ids = get_existing_odds_game_ids()
    print(f"  {len(existing_odds_ids)} games already have odds")

    total_rows = 0

    for year in range(args.start, args.end + 1):
        print(f"\n--- {year} ---")
        dates = get_season_dates(args.sport, year)

        if args.month:
            dates = [d for d in dates if d.month == args.month]

        inserted_season = 0
        for dt in tqdm(dates, desc=f"  {year}", leave=False):
            count = scrape_date(sport_cfg, dt, game_lookup, our_teams, existing_odds_ids)
            inserted_season += count
            # Be respectful to ESPN's API
            time.sleep(0.5)

        print(f"  {year}: {inserted_season} odds rows inserted")
        total_rows += inserted_season

    # Summary
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_rows} total odds rows inserted")
    print(f"{'='*60}")

    total_odds = query("SELECT COUNT(*) as cnt FROM odds")
    print(f"Total odds in DB: {int(total_odds.iloc[0]['cnt']):,}")


if __name__ == "__main__":
    main()
