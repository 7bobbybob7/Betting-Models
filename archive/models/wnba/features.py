"""
models/wnba/features.py - WNBA feature engineering pipeline.

Builds game-level feature matrix with four factors, ELO, and contextual features.
Excludes 2020 bubble from model training (kept in ELO chain).

Usage:
    python -m models.wnba.features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from tqdm import tqdm

from db.db import query
from models.wnba.elo import WNBAElo


BUBBLE_YEAR = 2020


def load_games():
    """Load all WNBA games with team names."""
    df = query("""
        SELECT g.game_id, g.game_date, g.season_id,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               s.year as season,
               ht.name as home_team, at.name as away_team
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE g.sport_id = 3 AND g.status = 'final'
        ORDER BY g.game_date, g.game_id
    """)
    # Deduplicate
    df = df.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"], keep="first")
    return df


def load_team_stats():
    """Load team game stats."""
    df = query("""
        SELECT ts.game_id, ts.team_id, ts.is_home,
               ts.points, ts.fgm, ts.fga, ts.fg3m, ts.fg3a,
               ts.ftm, ts.fta, ts.offensive_rebounds, ts.defensive_rebounds,
               ts.assists, ts.steals, ts.blocks, ts.turnovers, ts.fouls,
               ts.offensive_efficiency, ts.efg_pct, ts.turnover_pct, ts.ft_rate,
               g.game_date, g.home_team_id, g.away_team_id
        FROM wnba_team_game_stats ts
        JOIN games g ON ts.game_id = g.game_id
        ORDER BY g.game_date, ts.game_id
    """)
    return df


def build_rolling_features(team_stats):
    """Build rolling four factors per team."""
    print("  Building rolling team features...")
    df = team_stats.sort_values("game_date").copy()

    # Compute per-game metrics
    df["poss"] = df["fga"] - df["offensive_rebounds"] + df["turnovers"] + 0.475 * df["fta"]
    df["opp_points"] = None  # will fill below

    # For defensive efficiency, we need opponent's points
    # Match each team-game row with the opponent's row
    game_pairs = df.groupby("game_id")
    opp_points = {}
    opp_poss = {}
    for gid, grp in game_pairs:
        if len(grp) == 2:
            r1, r2 = grp.iloc[0], grp.iloc[1]
            opp_points[(gid, r1["team_id"])] = r2["points"]
            opp_points[(gid, r2["team_id"])] = r1["points"]
            opp_poss[(gid, r1["team_id"])] = r2["poss"] if pd.notna(r2["poss"]) else None
            opp_poss[(gid, r2["team_id"])] = r1["poss"] if pd.notna(r1["poss"]) else None

    df["opp_points"] = df.apply(lambda r: opp_points.get((r["game_id"], r["team_id"])), axis=1)
    df["def_eff"] = np.where(
        df["poss"] > 0,
        df["opp_points"] / df["poss"] * 100,
        None
    )
    df["off_eff"] = np.where(
        df["poss"] > 0,
        df["points"] / df["poss"] * 100,
        None
    )

    features = {}

    for tid, grp in tqdm(df.groupby("team_id"), desc="  Teams", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            gid = int(row["game_id"])
            prior = grp.iloc[:i]

            if len(prior) < 3:
                features[(gid, int(tid))] = _empty_features()
                continue

            feat = {}
            for window, suffix in [(5, "5"), (10, "10")]:
                w = prior.tail(window)

                fga = w["fga"].sum()
                fgm = w["fgm"].sum()
                fg3m = w["fg3m"].sum()
                fta = w["fta"].sum()
                ftm = w["ftm"].sum()
                orb = w["offensive_rebounds"].sum()
                tov = w["turnovers"].sum()
                pts = w["points"].sum()

                feat[f"off_eff_{suffix}"] = round(w["off_eff"].dropna().mean(), 2) if w["off_eff"].notna().any() else None
                feat[f"def_eff_{suffix}"] = round(w["def_eff"].dropna().mean(), 2) if w["def_eff"].notna().any() else None
                feat[f"pace_{suffix}"] = round(w["poss"].dropna().mean(), 2) if w["poss"].notna().any() else None
                feat[f"efg_{suffix}"] = round((fgm + 0.5 * fg3m) / fga, 4) if fga > 0 else None
                feat[f"tov_pct_{suffix}"] = round(tov / (fga + 0.44 * fta + tov), 4) if (fga + 0.44 * fta + tov) > 0 else None
                feat[f"ft_rate_{suffix}"] = round(fta / fga, 4) if fga > 0 else None
                feat[f"ppg_{suffix}"] = round(pts / len(w), 2) if len(w) > 0 else None

            features[(gid, int(tid))] = feat

    return features


def _empty_features():
    feat = {}
    for suffix in ["5", "10"]:
        for k in ["off_eff", "def_eff", "pace", "efg", "tov_pct", "ft_rate", "ppg"]:
            feat[f"{k}_{suffix}"] = None
    return feat


def build_feature_matrix():
    """Build the full WNBA feature matrix."""
    print("\nBuilding WNBA feature matrix...")

    games = load_games()
    team_stats = load_team_stats()
    print(f"  Games: {len(games)}, Team stats: {len(team_stats)}")

    # Build rolling features
    rolling = build_rolling_features(team_stats)

    # Build ELO
    elo = WNBAElo()
    elo.run(start_year=2015, end_year=2024)
    elo_data = elo.get_game_elos()

    # Assemble
    print("  Assembling feature matrix...")
    rows = []

    for _, game in tqdm(games.iterrows(), total=len(games), desc="  Games", leave=False):
        gid = int(game["game_id"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])
        season = int(game["season"])

        row = {
            "game_id": gid,
            "game_date": game["game_date"],
            "season": season,
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "total_points": game["home_score"] + game["away_score"],
            "home_win": 1 if game["home_score"] > game["away_score"] else 0,
            "is_bubble": season == BUBBLE_YEAR,
        }

        # ELO
        ge = elo_data.get(gid, {})
        row["home_elo"] = ge.get("home_elo")
        row["away_elo"] = ge.get("away_elo")
        row["elo_diff"] = ge.get("elo_diff")
        row["elo_win_prob"] = ge.get("home_win_prob")

        # Home team rolling features
        h_feat = rolling.get((gid, home_tid), _empty_features())
        for k, v in h_feat.items():
            row[f"home_{k}"] = v

        # Away team rolling features
        a_feat = rolling.get((gid, away_tid), _empty_features())
        for k, v in a_feat.items():
            row[f"away_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add differentials
    for col in ["off_eff_5", "def_eff_5", "pace_5", "efg_5", "tov_pct_5", "ppg_5"]:
        h, a = f"home_{col}", f"away_{col}"
        if h in df.columns and a in df.columns:
            invert = col in ["def_eff_5", "tov_pct_5"]  # lower is better
            if invert:
                df[f"diff_{col}"] = df[a] - df[h]
            else:
                df[f"diff_{col}"] = df[h] - df[a]

    print(f"\n  Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Bubble games: {df['is_bubble'].sum()}")
    print(f"  Home win rate (excl bubble): {df[~df['is_bubble']]['home_win'].mean():.3f}")

    return df


def main():
    df = build_feature_matrix()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/wnba_features.csv", index=False)
    print(f"\nSaved to data/wnba_features.csv")

    print(f"\n{'='*60}")
    print("  WNBA FEATURE SUMMARY")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Games per season:")
    for yr in sorted(df["season"].unique()):
        n = len(df[df["season"] == yr])
        bubble = " (bubble)" if yr == BUBBLE_YEAR else ""
        print(f"    {yr}: {n}{bubble}")


if __name__ == "__main__":
    main()
