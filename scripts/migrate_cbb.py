"""
scripts/migrate_cbb.py - Migrate CBB data from the old cbb-betting-model repo.

Reads 5 CSVs from the old project and loads them into the Postgres schema:
    1. all_games.csv       -> teams, seasons, games
    2. boxscores_flat.csv  -> cbb_team_game_stats
    3. elo_game_log.csv    -> elo_ratings
    4. odds_devigged.csv   -> odds
    5. bets_log.csv        -> predictions

Usage:
    python scripts/migrate_cbb.py
    python scripts/migrate_cbb.py --data-dir /path/to/csvs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from db.db import query, execute, bulk_insert


# ---------------------------------------------------------------------------
# Paths to source CSVs (default: old repo location)
# ---------------------------------------------------------------------------
OLD_REPO = "/Users/cody/Projects/cbb-betting-model"
DEFAULT_PATHS = {
    "all_games": f"{OLD_REPO}/data/processed/all_games.csv",
    "boxscores": f"{OLD_REPO}/data/processed/boxscores_flat.csv",
    "elo":       f"{OLD_REPO}/data/processed/elo_game_log.csv",
    "odds":      f"{OLD_REPO}/data/odds/odds_devigged.csv",
    "bets":      f"{OLD_REPO}/data/bets_log.csv",
}


def get_sport_id():
    result = query("SELECT sport_id FROM sports WHERE name = 'cbb'")
    return int(result.iloc[0]["sport_id"])


def ensure_season(sport_id, year):
    execute(
        "INSERT INTO seasons (sport_id, year) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [sport_id, year]
    )
    result = query(
        "SELECT season_id FROM seasons WHERE sport_id = %s AND year = %s",
        [sport_id, year]
    )
    return int(result.iloc[0]["season_id"])


def migrate_games(sport_id, paths):
    """Load all_games.csv -> teams, seasons, games."""
    print("\n=== STEP 1: Games ===")
    df = pd.read_csv(paths["all_games"])
    print(f"  Source rows: {len(df)}")

    # --- Teams ---
    home_teams = df[["home_team", "home_id"]].rename(columns={"home_team": "name", "home_id": "ext_id"})
    away_teams = df[["away_team", "away_id"]].rename(columns={"away_team": "name", "away_id": "ext_id"})
    all_teams = pd.concat([home_teams, away_teams]).drop_duplicates(subset=["name"])

    team_rows = [(sport_id, row["name"], None) for _, row in all_teams.iterrows()]
    bulk_insert("teams", ["sport_id", "name", "abbreviation"], team_rows)

    # Build name -> team_id map
    team_map = {}
    teams_db = query("SELECT team_id, name FROM teams WHERE sport_id = %s", [sport_id])
    for _, r in teams_db.iterrows():
        team_map[r["name"]] = int(r["team_id"])

    # --- Seasons ---
    season_map = {}
    for year in sorted(df["season"].unique()):
        season_map[int(year)] = ensure_season(sport_id, int(year))

    # --- Games ---
    game_rows = []
    for _, row in df.iterrows():
        home_tid = team_map.get(row["home_team"])
        away_tid = team_map.get(row["away_team"])
        if not home_tid or not away_tid:
            continue

        season_id = season_map[int(row["season"])]
        game_date = str(row["date"])[:10]  # extract date from ISO timestamp
        neutral = bool(row.get("neutral_site", False))

        game_rows.append((
            sport_id, season_id, str(row["game_id"]),
            game_date, None,
            home_tid, away_tid,
            int(row["home_score"]) if pd.notna(row["home_score"]) else None,
            int(row["away_score"]) if pd.notna(row["away_score"]) else None,
            "final", None, False, neutral
        ))

    cols = [
        "sport_id", "season_id", "external_id", "game_date", "game_time",
        "home_team_id", "away_team_id", "home_score", "away_score",
        "status", "venue", "is_postseason", "is_neutral_site"
    ]
    bulk_insert("games", cols, game_rows)

    print(f"  Teams: {len(all_teams)}, Seasons: {len(season_map)}, Games: {len(game_rows)}")
    return team_map, season_map


def _build_game_map(sport_id):
    """Build external_id -> game_id map for CBB."""
    gdf = query(
        "SELECT game_id, external_id FROM games WHERE sport_id = %s",
        [sport_id]
    )
    return dict(zip(gdf["external_id"].astype(str), gdf["game_id"].astype(int)))


def migrate_boxscores(sport_id, paths, team_map):
    """Load boxscores_flat.csv -> cbb_team_game_stats."""
    print("\n=== STEP 2: Box Scores ===")
    df = pd.read_csv(paths["boxscores"])
    print(f"  Source rows: {len(df)}")

    game_map = _build_game_map(sport_id)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Boxscores", leave=False):
        gid_str = str(int(row["game_id"])) if pd.notna(row["game_id"]) else None
        game_id = game_map.get(gid_str)
        if not game_id:
            continue

        # Process home side
        home_tid = team_map.get(row.get("home_team_name"))
        if home_tid:
            fgm = _si(row.get("home_fieldGoalsMade"))
            fga = _si(row.get("home_fieldGoalsAttempted"))
            fg3m = _si(row.get("home_threePointFieldGoalsMade"))
            fg3a = _si(row.get("home_threePointFieldGoalsAttempted"))
            ftm = _si(row.get("home_freeThrowsMade"))
            fta = _si(row.get("home_freeThrowsAttempted"))
            orb = _si(row.get("home_offensiveRebounds"))
            drb = _si(row.get("home_defensiveRebounds"))
            ast = _si(row.get("home_assists"))
            stl = _si(row.get("home_steals"))
            blk = _si(row.get("home_blocks"))
            tov = _si(row.get("home_turnovers"))
            fouls = _si(row.get("home_fouls"))
            pts = (fgm - fg3m) * 2 + fg3m * 3 + ftm if all(v is not None for v in [fgm, fg3m, ftm]) else None

            # Derived metrics
            off_eff, def_eff, tempo, efg, tov_pct, orb_pct, ft_rate = _compute_efficiency(
                fgm, fga, fg3m, ftm, fta, orb, tov, pts,
                _si(row.get("away_fieldGoalsAttempted")),
                _si(row.get("away_offensiveRebounds")),
                _si(row.get("away_turnovers")),
                _si(row.get("away_freeThrowsAttempted")),
            )

            rows.append((
                game_id, home_tid, True,
                pts, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb,
                ast, stl, blk, tov, fouls,
                off_eff, def_eff, tempo, efg, tov_pct, orb_pct, ft_rate
            ))

        # Process away side
        away_tid = team_map.get(row.get("away_team_name"))
        if away_tid:
            fgm = _si(row.get("away_fieldGoalsMade"))
            fga = _si(row.get("away_fieldGoalsAttempted"))
            fg3m = _si(row.get("away_threePointFieldGoalsMade"))
            fg3a = _si(row.get("away_threePointFieldGoalsAttempted"))
            ftm = _si(row.get("away_freeThrowsMade"))
            fta = _si(row.get("away_freeThrowsAttempted"))
            orb = _si(row.get("away_offensiveRebounds"))
            drb = _si(row.get("away_defensiveRebounds"))
            ast = _si(row.get("away_assists"))
            stl = _si(row.get("away_steals"))
            blk = _si(row.get("away_blocks"))
            tov = _si(row.get("away_turnovers"))
            fouls = _si(row.get("away_fouls"))
            pts = (fgm - fg3m) * 2 + fg3m * 3 + ftm if all(v is not None for v in [fgm, fg3m, ftm]) else None

            off_eff, def_eff, tempo, efg, tov_pct, orb_pct, ft_rate = _compute_efficiency(
                fgm, fga, fg3m, ftm, fta, orb, tov, pts,
                _si(row.get("home_fieldGoalsAttempted")),
                _si(row.get("home_offensiveRebounds")),
                _si(row.get("home_turnovers")),
                _si(row.get("home_freeThrowsAttempted")),
            )

            rows.append((
                game_id, away_tid, False,
                pts, fgm, fga, fg3m, fg3a, ftm, fta, orb, drb,
                ast, stl, blk, tov, fouls,
                off_eff, def_eff, tempo, efg, tov_pct, orb_pct, ft_rate
            ))

    cols = [
        "game_id", "team_id", "is_home",
        "points", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
        "offensive_rebounds", "defensive_rebounds",
        "assists", "steals", "blocks", "turnovers", "fouls",
        "offensive_efficiency", "defensive_efficiency", "tempo",
        "efg_pct", "turnover_pct", "orb_pct", "ft_rate"
    ]
    bulk_insert("cbb_team_game_stats", cols, rows)
    print(f"  Inserted: {len(rows)} box score rows")


def _compute_efficiency(fgm, fga, fg3m, ftm, fta, orb, tov, pts,
                         opp_fga, opp_orb, opp_tov, opp_fta):
    """Compute four factors + efficiency metrics."""
    try:
        # Possessions estimate (team)
        poss = fga - orb + tov + 0.475 * fta if all(v is not None for v in [fga, orb, tov, fta]) else None
        # Opponent possessions
        opp_poss = opp_fga - opp_orb + opp_tov + 0.475 * opp_fta if all(v is not None for v in [opp_fga, opp_orb, opp_tov, opp_fta]) else None

        off_eff = round(pts / poss * 100, 2) if poss and pts is not None else None
        # We don't have opponent points in this row, so def_eff is left None here
        def_eff = None
        tempo = round((poss + opp_poss) / 2, 2) if poss and opp_poss else None
        efg = round((fgm + 0.5 * fg3m) / fga, 4) if fga else None
        tov_pct = round(tov / poss, 4) if poss and tov is not None else None
        orb_pct_val = round(orb / (orb + (opp_fga - (opp_fga - fga + orb))), 4) if orb is not None and fga and opp_fga else None
        # Simplified ORB%: orb / (orb + opp_drb). We don't have opp_drb directly, skip complex calc
        orb_pct_val = None  # Too complex without opponent DRB, leave null
        ft_rate = round(fta / fga, 4) if fga else None

        return off_eff, def_eff, tempo, efg, tov_pct, orb_pct_val, ft_rate
    except Exception:
        return None, None, None, None, None, None, None


def migrate_elo(sport_id, paths, team_map):
    """Load elo_game_log.csv -> elo_ratings."""
    print("\n=== STEP 3: ELO Ratings ===")
    df = pd.read_csv(paths["elo"])
    print(f"  Source rows: {len(df)}")

    game_map = _build_game_map(sport_id)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  ELO", leave=False):
        gid_str = str(int(row["game_id"])) if pd.notna(row["game_id"]) else None
        game_id = game_map.get(gid_str)
        if not game_id:
            continue

        game_date = str(row["date"])[:10]

        # Home team post-ELO
        home_tid = team_map.get(row["home_team"])
        if home_tid and pd.notna(row.get("home_elo_post")):
            rows.append((home_tid, game_id, round(float(row["home_elo_post"]), 2), game_date))

        # Away team post-ELO
        away_tid = team_map.get(row["away_team"])
        if away_tid and pd.notna(row.get("away_elo_post")):
            rows.append((away_tid, game_id, round(float(row["away_elo_post"]), 2), game_date))

    cols = ["team_id", "game_id", "rating", "rating_date"]
    bulk_insert("elo_ratings", cols, rows)
    print(f"  Inserted: {len(rows)} ELO records")


def migrate_odds(sport_id, paths):
    """Load odds_devigged.csv -> odds."""
    print("\n=== STEP 4: Odds ===")
    df = pd.read_csv(paths["odds"])
    print(f"  Source rows: {len(df)}")

    game_map = _build_game_map(sport_id)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Odds", leave=False):
        gid_str = str(int(row["game_id"])) if pd.notna(row["game_id"]) else None
        game_id = game_map.get(gid_str)
        if not game_id:
            continue

        sportsbook = str(row.get("provider_name", "unknown")).lower().replace(" ", "_")
        home_ml = _sf(row.get("home_ml"))
        away_ml = _sf(row.get("away_ml"))
        # Clamp extreme ML odds to fit DECIMAL(8,3)
        if home_ml is not None and abs(home_ml) > 99999:
            home_ml = -99999.0 if home_ml < 0 else 99999.0
        if away_ml is not None and abs(away_ml) > 99999:
            away_ml = -99999.0 if away_ml < 0 else 99999.0
        spread = _sf(row.get("spread"))
        total = _sf(row.get("over_under"))
        home_implied = _sf(row.get("home_fair"))
        away_implied = _sf(row.get("away_fair"))

        # Moneyline row
        rows.append((
            game_id, sportsbook, "moneyline",
            home_ml, away_ml, total, None, None,
            home_implied, away_implied, True
        ))

        # Spread row (if available)
        if spread is not None:
            rows.append((
                game_id, sportsbook, "spread",
                spread, -spread if spread else None, total, None, None,
                home_implied, away_implied, True
            ))

        # Total row (if available)
        if total is not None:
            rows.append((
                game_id, sportsbook, "total",
                None, None, total, None, None,
                None, None, True
            ))

    cols = [
        "game_id", "sportsbook", "market",
        "home_line", "away_line", "total_line", "over_odds", "under_odds",
        "home_implied", "away_implied", "is_closing"
    ]
    bulk_insert("odds", cols, rows)
    print(f"  Inserted: {len(rows)} odds records")


def migrate_bets(sport_id, paths, team_map):
    """Load bets_log.csv -> predictions."""
    print("\n=== STEP 5: Predictions/Bets ===")
    df = pd.read_csv(paths["bets"])
    print(f"  Source rows: {len(df)}")

    game_map = _build_game_map(sport_id)
    rows = []

    for _, row in df.iterrows():
        gid_str = str(int(row["game_id"])) if pd.notna(row["game_id"]) else None
        game_id = game_map.get(gid_str)
        if not game_id:
            continue

        result_str = str(row.get("result", "")).upper()
        outcome = "win" if result_str == "W" else "loss" if result_str == "L" else None
        pnl = _sf(row.get("profit"))
        bet_amount = _sf(row.get("bet_amount"))
        model_prob = _sf(row.get("model_prob"))
        market_prob = _sf(row.get("market_prob"))
        decimal_odds = _sf(row.get("decimal_odds"))
        edge = round(model_prob - market_prob, 4) if model_prob and market_prob else None

        rows.append((
            game_id, "cbb_elo_v1", "moneyline",
            model_prob, None, edge,
            True, bet_amount, decimal_odds,
            outcome, pnl
        ))

    cols = [
        "game_id", "model_name", "market",
        "predicted_prob", "predicted_value", "edge",
        "bet_placed", "bet_amount", "bet_odds",
        "outcome", "pnl"
    ]
    bulk_insert("predictions", cols, rows)
    print(f"  Inserted: {len(rows)} prediction records")


def _si(val):
    """Safe int."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _sf(val):
    """Safe float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Migrate CBB data from old repo")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override base directory for CSVs")
    args = parser.parse_args()

    # Resolve paths
    paths = DEFAULT_PATHS.copy()
    if args.data_dir:
        paths = {
            "all_games": f"{args.data_dir}/all_games.csv",
            "boxscores": f"{args.data_dir}/boxscores_flat.csv",
            "elo":       f"{args.data_dir}/elo_game_log.csv",
            "odds":      f"{args.data_dir}/odds_devigged.csv",
            "bets":      f"{args.data_dir}/bets_log.csv",
        }

    # Verify files exist
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            print("Use --data-dir to specify the CSV directory")
            sys.exit(1)

    sport_id = get_sport_id()
    print(f"CBB sport_id: {sport_id}")

    # Run all migrations
    team_map, season_map = migrate_games(sport_id, paths)
    migrate_boxscores(sport_id, paths, team_map)
    migrate_elo(sport_id, paths, team_map)
    migrate_odds(sport_id, paths)
    migrate_bets(sport_id, paths, team_map)

    # Summary
    print(f"\n{'='*60}")
    print("CBB MIGRATION COMPLETE")
    print(f"{'='*60}")

    tables = {
        "teams (CBB)": f"SELECT COUNT(*) as cnt FROM teams WHERE sport_id = {sport_id}",
        "seasons (CBB)": f"SELECT COUNT(*) as cnt FROM seasons WHERE sport_id = {sport_id}",
        "games (CBB)": f"SELECT COUNT(*) as cnt FROM games WHERE sport_id = {sport_id}",
        "cbb_team_game_stats": "SELECT COUNT(*) as cnt FROM cbb_team_game_stats",
        "elo_ratings": "SELECT COUNT(*) as cnt FROM elo_ratings",
        "odds": f"SELECT COUNT(*) as cnt FROM odds WHERE game_id IN (SELECT game_id FROM games WHERE sport_id = {sport_id})",
        "predictions": "SELECT COUNT(*) as cnt FROM predictions",
    }
    for label, sql in tables.items():
        try:
            result = query(sql)
            count = int(result.iloc[0]["cnt"])
            print(f"  {label}: {count:,} rows")
        except Exception as e:
            print(f"  {label}: error - {e}")


if __name__ == "__main__":
    main()
