"""
models/mlb/elo.py - MLB ELO rating system with starter adjustments.

Features:
    - Team ELO updated after every game
    - Starter adjustment: pitcher quality shifts team's effective ELO per game
    - Margin-of-victory multiplier with diminishing returns
    - Home advantage parameter
    - Season-to-season regression toward mean
    - Outputs: pre-game ELO for both teams, starter-adjusted ELO, win probability

Usage:
    from models.mlb.elo import MLBElo
    elo = MLBElo()
    elo.run(start_year=2015, end_year=2025)
    game_elos = elo.get_game_elos()  # dict: game_id -> {home_elo, away_elo, ...}
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from db.db import query


# ---------------------------------------------------------------------------
# ELO Parameters (tuned for MLB)
# ---------------------------------------------------------------------------
BASE_ELO = 1500
K_FACTOR = 6            # Lower than CBB (~20) because MLB has 162 games per season
HOME_ADVANTAGE = 24     # ~54% implied home win rate
SEASON_REGRESSION = 0.6  # Regress 40% toward mean between seasons (MLB rosters more stable)
MOV_MULTIPLIER = 1.0    # Margin-of-victory scaling factor
STARTER_WEIGHT = 40     # Max ELO points a starter can shift (ace vs replacement)


def expected_score(elo_a, elo_b):
    """Expected win probability for team A given ELO ratings."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(margin):
    """Margin-of-victory multiplier with diminishing returns (log-based)."""
    if margin == 0:
        return 1.0
    return np.log(abs(margin) + 1) * MOV_MULTIPLIER


class MLBElo:
    def __init__(self):
        self.team_elos = defaultdict(lambda: BASE_ELO)
        self.pitcher_quality = {}  # player_id -> quality score (lower = better)
        self.game_elos = {}        # game_id -> dict of elo values
        self._games_df = None

    def _load_data(self):
        """Load games and pitcher stats."""
        print("  Loading games...")
        self._games_df = query("""
            SELECT
                g.game_id, g.game_date, g.home_team_id, g.away_team_id,
                g.home_score, g.away_score, s.year as season,
                hp.player_id as home_starter_id,
                ap.player_id as away_starter_id
            FROM games g
            JOIN seasons s ON g.season_id = s.season_id
            LEFT JOIN mlb_pitching_game hp ON g.game_id = hp.game_id
                AND hp.team_id = g.home_team_id AND hp.is_starter = true
            LEFT JOIN mlb_pitching_game ap ON g.game_id = ap.game_id
                AND ap.team_id = g.away_team_id AND ap.is_starter = true
            WHERE g.sport_id = 2 AND g.status = 'final'
            ORDER BY g.game_date, g.game_id
        """)
        print(f"    {len(self._games_df)} games loaded")

    def _build_pitcher_quality(self):
        """
        Compute pitcher quality scores from career FIP.
        Quality = how much better/worse than league average.
        Positive = better than average (shifts team ELO up).
        """
        print("  Computing pitcher quality scores...")
        pitching = query("""
            SELECT
                player_id, team_id,
                SUM(so) as so, SUM(bb) as bb, SUM(hr_allowed) as hr,
                SUM(CASE WHEN ip > 0 THEN
                    CAST(ip AS INTEGER) + (ip - CAST(ip AS INTEGER)) * 10 / 3.0
                    ELSE 0 END) as ip_true,
                COUNT(*) as starts
            FROM mlb_pitching_game
            WHERE is_starter = true AND ip > 0
            GROUP BY player_id, team_id
        """)

        # Compute FIP per pitcher
        pitching["fip"] = np.where(
            pitching["ip_true"] > 0,
            (13 * pitching["hr"] + 3 * pitching["bb"] - 2 * pitching["so"]) / pitching["ip_true"] + 3.10,
            None
        )

        # League average FIP
        total_ip = pitching["ip_true"].sum()
        league_fip = (13 * pitching["hr"].sum() + 3 * pitching["bb"].sum() -
                      2 * pitching["so"].sum()) / total_ip + 3.10

        # Quality score: how many ELO points this pitcher is worth vs average
        # Lower FIP = better pitcher = positive quality
        # Scale: 1 FIP point difference ≈ STARTER_WEIGHT ELO points
        for _, row in pitching.iterrows():
            if row["starts"] >= 3 and pd.notna(row["fip"]):
                # Regress toward league average based on sample size
                # More starts = trust the observed FIP more
                regress_factor = min(row["starts"] / 30, 1.0)  # Full weight at 30+ starts
                regressed_fip = row["fip"] * regress_factor + league_fip * (1 - regress_factor)
                quality = (league_fip - regressed_fip) * (STARTER_WEIGHT / 1.5)
                self.pitcher_quality[int(row["player_id"])] = round(quality, 1)

        print(f"    {len(self.pitcher_quality)} pitchers rated")
        print(f"    League avg FIP: {league_fip:.3f}")

        # Show top/bottom pitchers
        sorted_pitchers = sorted(self.pitcher_quality.items(), key=lambda x: -x[1])
        if sorted_pitchers:
            print(f"    Best: player {sorted_pitchers[0][0]} (+{sorted_pitchers[0][1]:.1f} ELO)")
            print(f"    Worst: player {sorted_pitchers[-1][0]} ({sorted_pitchers[-1][1]:.1f} ELO)")

    def _get_starter_adjustment(self, player_id):
        """Get ELO adjustment for a starter. 0 if unknown."""
        if player_id is None or pd.isna(player_id):
            return 0.0
        return self.pitcher_quality.get(int(player_id), 0.0)

    def _regress_season(self, new_season):
        """Regress all team ELOs toward the mean between seasons."""
        for team_id in list(self.team_elos.keys()):
            self.team_elos[team_id] = (
                BASE_ELO * (1 - SEASON_REGRESSION) +
                self.team_elos[team_id] * SEASON_REGRESSION
            )

    def run(self, start_year=2015, end_year=2025):
        """Run the full ELO simulation."""
        print("\n=== MLB ELO SYSTEM ===")
        self._load_data()
        self._build_pitcher_quality()

        df = self._games_df
        df = df[(df["season"] >= start_year) & (df["season"] <= end_year)]

        current_season = None
        correct = 0
        total = 0

        print(f"\n  Running ELO simulation ({start_year}-{end_year})...")
        for _, game in tqdm(df.iterrows(), total=len(df), desc="  Games", leave=False):
            season = int(game["season"])
            game_id = int(game["game_id"])
            home_tid = int(game["home_team_id"])
            away_tid = int(game["away_team_id"])
            home_score = game["home_score"]
            away_score = game["away_score"]

            # Season transition
            if current_season is not None and season != current_season:
                self._regress_season(season)
            current_season = season

            # Pre-game ELOs
            home_elo = self.team_elos[home_tid]
            away_elo = self.team_elos[away_tid]

            # Starter adjustments
            home_starter_adj = self._get_starter_adjustment(game.get("home_starter_id"))
            away_starter_adj = self._get_starter_adjustment(game.get("away_starter_id"))

            # Effective ELOs (team + starter + home advantage)
            home_elo_eff = home_elo + home_starter_adj + HOME_ADVANTAGE
            away_elo_eff = away_elo + away_starter_adj

            # Win probability
            home_win_prob = expected_score(home_elo_eff, away_elo_eff)

            # Actual result
            if home_score > away_score:
                actual = 1.0
            elif away_score > home_score:
                actual = 0.0
            else:
                actual = 0.5  # ties (rare in MLB)

            # Track accuracy
            if actual == 1.0 and home_win_prob > 0.5:
                correct += 1
            elif actual == 0.0 and home_win_prob < 0.5:
                correct += 1
            total += 1

            # Margin of victory
            margin = abs(home_score - away_score)
            mov_mult = mov_multiplier(margin)

            # ELO update
            delta = K_FACTOR * mov_mult * (actual - home_win_prob)
            self.team_elos[home_tid] += delta
            self.team_elos[away_tid] -= delta

            # Store pre-game values (what would be known before the game)
            self.game_elos[game_id] = {
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "home_starter_adj": round(home_starter_adj, 1),
                "away_starter_adj": round(away_starter_adj, 1),
                "home_elo_eff": round(home_elo_eff, 1),
                "away_elo_eff": round(away_elo_eff, 1),
                "elo_diff": round(home_elo_eff - away_elo_eff, 1),
                "home_win_prob": round(home_win_prob, 4),
            }

        accuracy = correct / total if total > 0 else 0
        print(f"\n  Results:")
        print(f"    Games: {total}")
        print(f"    Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"    Home win rate (actual): {df[df['home_score'] > df['away_score']].shape[0] / len(df):.3f}")

        # Show final team ELOs
        sorted_teams = sorted(self.team_elos.items(), key=lambda x: -x[1])
        teams = query("SELECT team_id, name FROM teams WHERE sport_id = 2")
        team_names = dict(zip(teams["team_id"], teams["name"]))

        print(f"\n  Final ELO rankings (top 10):")
        for tid, elo in sorted_teams[:10]:
            name = team_names.get(tid, f"Team {tid}")
            print(f"    {name:30s} {elo:.0f}")

        return self.game_elos

    def get_game_elos(self):
        """Return dict: game_id -> elo features."""
        return self.game_elos

    def to_dataframe(self):
        """Convert game ELOs to DataFrame for merging with feature matrix."""
        rows = []
        for game_id, elos in self.game_elos.items():
            rows.append({"game_id": game_id, **elos})
        return pd.DataFrame(rows)


def main():
    """Run ELO and print results."""
    elo = MLBElo()
    elo.run(start_year=2015, end_year=2025)


if __name__ == "__main__":
    main()
