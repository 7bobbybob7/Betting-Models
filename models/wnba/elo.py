"""
models/wnba/elo.py - WNBA ELO rating system.

Tuned for WNBA:
    - Higher K-factor than MLB (40 games/season vs 162)
    - Home advantage calibrated from data (excluding 2020 bubble)
    - Stronger season regression (WNBA rosters change more)
    - 2020 bubble included in ELO chain with home advantage = 0

Usage:
    from models.wnba.elo import WNBAElo
    elo = WNBAElo()
    elo.run()
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from db.db import query


BASE_ELO = 1500
K_FACTOR = 22
HOME_ADVANTAGE = 55  # ~3.5 points -> ~55 ELO points at WNBA scoring levels
SEASON_REGRESSION = 0.5  # 50% carryover between seasons
BUBBLE_YEAR = 2020


def expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(margin):
    if margin == 0:
        return 1.0
    return np.log(abs(margin) + 1)


class WNBAElo:
    def __init__(self):
        self.team_elos = defaultdict(lambda: BASE_ELO)
        self.game_elos = {}

    def _load_games(self):
        print("  Loading WNBA games...")
        df = query("""
            SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score, s.year as season,
                   g.external_id
            FROM games g
            JOIN seasons s ON g.season_id = s.season_id
            WHERE g.sport_id = 3 AND g.status = 'final'
            ORDER BY g.game_date, g.game_id
        """)

        # Deduplicate — keep first occurrence per (game_date, home_team_id, away_team_id)
        df = df.drop_duplicates(subset=["game_date", "home_team_id", "away_team_id"], keep="first")

        print(f"    {len(df)} games (after dedup)")
        return df

    def run(self, start_year=2015, end_year=2024):
        print("\n=== WNBA ELO SYSTEM ===")
        df = self._load_games()
        df = df[(df["season"] >= start_year) & (df["season"] <= end_year)]

        current_season = None
        correct = 0
        total = 0

        print(f"  Running ELO simulation ({start_year}-{end_year})...")
        for _, game in tqdm(df.iterrows(), total=len(df), desc="  Games", leave=False):
            season = int(game["season"])
            game_id = int(game["game_id"])
            home_tid = int(game["home_team_id"])
            away_tid = int(game["away_team_id"])
            home_score = game["home_score"]
            away_score = game["away_score"]

            # Season transition
            if current_season is not None and season != current_season:
                for tid in list(self.team_elos.keys()):
                    self.team_elos[tid] = BASE_ELO * (1 - SEASON_REGRESSION) + self.team_elos[tid] * SEASON_REGRESSION
            current_season = season

            home_elo = self.team_elos[home_tid]
            away_elo = self.team_elos[away_tid]

            # No home advantage in 2020 bubble
            ha = 0 if season == BUBBLE_YEAR else HOME_ADVANTAGE
            home_elo_eff = home_elo + ha
            away_elo_eff = away_elo

            home_win_prob = expected_score(home_elo_eff, away_elo_eff)

            if home_score > away_score:
                actual = 1.0
            elif away_score > home_score:
                actual = 0.0
            else:
                actual = 0.5

            if actual == 1.0 and home_win_prob > 0.5:
                correct += 1
            elif actual == 0.0 and home_win_prob < 0.5:
                correct += 1
            total += 1

            margin = abs(home_score - away_score)
            mov = mov_multiplier(margin)
            delta = K_FACTOR * mov * (actual - home_win_prob)

            self.team_elos[home_tid] += delta
            self.team_elos[away_tid] -= delta

            self.game_elos[game_id] = {
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "home_elo_eff": round(home_elo_eff, 1),
                "away_elo_eff": round(away_elo_eff, 1),
                "elo_diff": round(home_elo_eff - away_elo_eff, 1),
                "home_win_prob": round(home_win_prob, 4),
            }

        accuracy = correct / total if total > 0 else 0
        print(f"\n  Results:")
        print(f"    Games: {total}")
        print(f"    Accuracy: {accuracy:.3f}")

        # Home win rate by year
        home_wins = df[df["home_score"] > df["away_score"]]
        non_bubble = df[df["season"] != BUBBLE_YEAR]
        print(f"    Home win rate (all): {len(home_wins)/len(df):.3f}")
        print(f"    Home win rate (excl 2020): {len(home_wins[home_wins['season']!=BUBBLE_YEAR])/len(non_bubble):.3f}")

        # Top teams
        teams = query("SELECT team_id, name FROM teams WHERE sport_id = 3")
        team_names = dict(zip(teams["team_id"], teams["name"]))
        sorted_teams = sorted(self.team_elos.items(), key=lambda x: -x[1])

        print(f"\n  Final ELO rankings:")
        for tid, elo in sorted_teams[:10]:
            name = team_names.get(tid, f"Team {tid}")
            print(f"    {name:30s} {elo:.0f}")

        return self.game_elos

    def get_game_elos(self):
        return self.game_elos

    def to_dataframe(self):
        rows = [{"game_id": gid, **elos} for gid, elos in self.game_elos.items()]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    elo = WNBAElo()
    elo.run()
