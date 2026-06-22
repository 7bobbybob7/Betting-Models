"""
models/mlb/features.py - MLB feature engineering pipeline.

Builds a game-level feature matrix for model training/prediction.
Each row = one game with home/away pitcher + team + contextual features.

Features:
    Pitcher (per starter):
        - Rolling FIP, K%, BB%, K-BB%, HR/9, WHIP (5-start and season)
        - Rest days since last start
        - Pitch count fatigue (pitches in last 2 starts)
        - Season IP (workload)
        - Home/away splits

    Team batting (per team):
        - Rolling wOBA, OPS, ISO, AVG, K%, BB% (last 15 and 30 games)
        - Runs per game

    Contextual:
        - Park factor (runs scored at venue vs league average)
        - Weather (temperature, wind)
        - Home/away indicator

Usage:
    python -m models.mlb.features --start 2016 --end 2025
    python -m models.mlb.features --start 2024 --end 2024 --out features_2024.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from db.db import query


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pitching():
    """Load all starter pitching game logs."""
    return query("""
        SELECT
            pg.game_id, pg.player_id, pg.team_id, pg.is_starter,
            pg.ip, pg.hits_allowed, pg.runs, pg.earned_runs,
            pg.bb, pg.so, pg.hr_allowed, pg.pitches, pg.strikes,
            g.game_date, g.home_team_id, g.away_team_id, g.season_id
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE pg.is_starter = true
        ORDER BY g.game_date, pg.game_id
    """)


def load_batting():
    """Load all batting game logs (aggregated per team per game)."""
    return query("""
        SELECT
            bg.game_id, bg.team_id,
            SUM(bg.pa) as pa, SUM(bg.ab) as ab, SUM(bg.hits) as hits,
            SUM(bg.doubles) as doubles, SUM(bg.triples) as triples,
            SUM(bg.hr) as hr, SUM(bg.rbi) as rbi, SUM(bg.bb) as bb,
            SUM(bg.so) as so, SUM(bg.hbp) as hbp,
            SUM(bg.sb) as sb, SUM(bg.cs) as cs,
            g.game_date, g.home_team_id, g.away_team_id, g.home_score, g.away_score
        FROM mlb_batting_game bg
        JOIN games g ON bg.game_id = g.game_id
        GROUP BY bg.game_id, bg.team_id,
                 g.game_date, g.home_team_id, g.away_team_id, g.home_score, g.away_score
        ORDER BY g.game_date, bg.game_id
    """)


def load_games(include_scheduled=False):
    """Load all MLB games with team names, starters, and game info.

    For final games, starters come from mlb_pitching_game (actual starter).
    For scheduled games, starters come from mlb_game_info (probable pitcher).
    """
    status_filter = "g.status = 'final'" if not include_scheduled else "g.status IN ('final', 'scheduled', 'pre_game', 'in_progress')"
    return query(f"""
        SELECT
            g.game_id, g.game_date, g.season_id, g.status,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            g.venue, g.is_postseason,
            ht.name as home_team, at.name as away_team,
            COALESCE(hp.player_id, gi.home_starter_id) as home_starter_id,
            COALESCE(ap.player_id, gi.away_starter_id) as away_starter_id,
            gi.weather_temp, gi.weather_wind, gi.weather_dir, gi.weather_cond,
            gi.umpire_hp,
            s.year as season_year
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.team_id
        JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN mlb_pitching_game hp ON g.game_id = hp.game_id
            AND hp.team_id = g.home_team_id AND hp.is_starter = true
        LEFT JOIN mlb_pitching_game ap ON g.game_id = ap.game_id
            AND ap.team_id = g.away_team_id AND ap.is_starter = true
        LEFT JOIN mlb_game_info gi ON g.game_id = gi.game_id
        JOIN seasons s ON g.season_id = s.season_id
        WHERE g.sport_id = 2 AND {status_filter}
        ORDER BY g.game_date, g.game_id
    """)


# ---------------------------------------------------------------------------
# Pitcher features
# ---------------------------------------------------------------------------
def compute_fip(er, hr, bb, hbp, so, ip):
    """Compute FIP (Fielding Independent Pitching). Uses 3.10 as FIP constant."""
    if ip is None or ip == 0:
        return None
    hbp = hbp if hbp else 0
    return ((13 * hr + 3 * (bb + hbp) - 2 * so) / ip) + 3.10


def build_pitcher_features(pitching_df):
    """
    Build rolling pitcher features. For each starter appearance,
    compute features based on their PRIOR starts (no leakage).

    Returns dict: (game_id, player_id) -> feature dict
    """
    print("  Building pitcher features...")
    pitching_df = pitching_df.sort_values("game_date").copy()

    # Convert IP to true innings (6.2 IP = 6.667 innings)
    pitching_df["ip_true"] = pitching_df["ip"].apply(
        lambda x: int(x) + (x % 1) * 10 / 3 if pd.notna(x) else 0
    )

    features = {}
    # Group by pitcher, iterate chronologically
    for pid, grp in tqdm(pitching_df.groupby("player_id"), desc="  Pitchers", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            game_id = int(row["game_id"])

            # Prior starts only (no leakage)
            prior = grp.iloc[:i]

            feat = {}

            if len(prior) == 0:
                # First start of career/dataset — no features
                feat = _empty_pitcher_features()
            else:
                # Rolling 5-start features
                last5 = prior.tail(5)
                feat["p_starts_5"] = len(last5)
                feat["p_ip_5"] = last5["ip_true"].sum()
                feat["p_k9_5"] = _rate(last5["so"].sum(), last5["ip_true"].sum(), 9)
                feat["p_bb9_5"] = _rate(last5["bb"].sum(), last5["ip_true"].sum(), 9)
                feat["p_hr9_5"] = _rate(last5["hr_allowed"].sum(), last5["ip_true"].sum(), 9)
                feat["p_whip_5"] = _rate(
                    last5["hits_allowed"].sum() + last5["bb"].sum(),
                    last5["ip_true"].sum(), 1
                )
                feat["p_kpct_5"] = _safe_div(last5["so"].sum(), _batters_faced(last5))
                feat["p_bbpct_5"] = _safe_div(last5["bb"].sum(), _batters_faced(last5))
                feat["p_kbb_5"] = (feat["p_kpct_5"] or 0) - (feat["p_bbpct_5"] or 0)
                feat["p_fip_5"] = _rolling_fip(last5)

                # Season features
                season_id = row["season_id"]
                season = prior[prior["season_id"] == season_id]
                if len(season) > 0:
                    feat["p_starts_szn"] = len(season)
                    feat["p_ip_szn"] = season["ip_true"].sum()
                    feat["p_k9_szn"] = _rate(season["so"].sum(), season["ip_true"].sum(), 9)
                    feat["p_bb9_szn"] = _rate(season["bb"].sum(), season["ip_true"].sum(), 9)
                    feat["p_fip_szn"] = _rolling_fip(season)
                    feat["p_era_szn"] = _rate(season["earned_runs"].sum(), season["ip_true"].sum(), 9)
                    feat["p_whip_szn"] = _rate(
                        season["hits_allowed"].sum() + season["bb"].sum(),
                        season["ip_true"].sum(), 1
                    )
                else:
                    for k in ["p_starts_szn", "p_ip_szn", "p_k9_szn", "p_bb9_szn",
                              "p_fip_szn", "p_era_szn", "p_whip_szn"]:
                        feat[k] = None

                # Rest days
                prev_date = prior.iloc[-1]["game_date"]
                feat["p_rest_days"] = (row["game_date"] - prev_date).days

                # Pitch count fatigue (last 2 starts)
                last2 = prior.tail(2)
                feat["p_pitches_last2"] = last2["pitches"].sum() if last2["pitches"].notna().all() else None

            features[(game_id, int(pid))] = feat

    return features


def _empty_pitcher_features():
    return {k: None for k in [
        "p_starts_5", "p_ip_5", "p_k9_5", "p_bb9_5", "p_hr9_5",
        "p_whip_5", "p_kpct_5", "p_bbpct_5", "p_kbb_5", "p_fip_5",
        "p_starts_szn", "p_ip_szn", "p_k9_szn", "p_bb9_szn",
        "p_fip_szn", "p_era_szn", "p_whip_szn",
        "p_rest_days", "p_pitches_last2",
    ]}


def _batters_faced(df):
    """Estimate batters faced from available stats."""
    return (df["ip_true"] * 3 + df["hits_allowed"] + df["bb"]).sum()


def _rolling_fip(df):
    ip = df["ip_true"].sum()
    if ip == 0:
        return None
    er = df["earned_runs"].sum()
    hr = df["hr_allowed"].sum()
    bb = df["bb"].sum()
    so = df["so"].sum()
    return round((13 * hr + 3 * bb - 2 * so) / ip + 3.10, 3)


def _rate(num, denom, multiplier):
    if denom is None or denom == 0:
        return None
    return round(num / denom * multiplier, 3)


def _safe_div(num, denom):
    if denom is None or denom == 0:
        return None
    return round(num / denom, 4)


# ---------------------------------------------------------------------------
# Team batting features
# ---------------------------------------------------------------------------
def build_batting_features(batting_df):
    """
    Build rolling team batting features.
    Returns dict: (game_id, team_id) -> feature dict
    """
    print("  Building batting features...")
    batting_df = batting_df.sort_values("game_date").copy()

    # Compute per-game stats
    batting_df["singles"] = batting_df["hits"] - batting_df["doubles"] - batting_df["triples"] - batting_df["hr"]
    batting_df["tb"] = (batting_df["singles"] + 2 * batting_df["doubles"] +
                        3 * batting_df["triples"] + 4 * batting_df["hr"])
    batting_df["obp_num"] = batting_df["hits"] + batting_df["bb"] + batting_df["hbp"].fillna(0)
    batting_df["obp_den"] = batting_df["ab"] + batting_df["bb"] + batting_df["hbp"].fillna(0)
    batting_df["slg"] = batting_df["tb"] / batting_df["ab"].replace(0, np.nan)
    batting_df["iso"] = (batting_df["tb"] - batting_df["hits"]) / batting_df["ab"].replace(0, np.nan)

    # Determine runs scored for this team
    batting_df["runs"] = np.where(
        batting_df["team_id"] == batting_df["home_team_id"],
        batting_df["home_score"],
        batting_df["away_score"]
    )

    # wOBA weights (approximate, standard linear weights)
    W_BB = 0.69
    W_HBP = 0.72
    W_1B = 0.88
    W_2B = 1.27
    W_3B = 1.62
    W_HR = 2.10

    batting_df["woba_num"] = (
        W_BB * batting_df["bb"] +
        W_HBP * batting_df["hbp"].fillna(0) +
        W_1B * batting_df["singles"] +
        W_2B * batting_df["doubles"] +
        W_3B * batting_df["triples"] +
        W_HR * batting_df["hr"]
    )
    batting_df["woba_den"] = (
        batting_df["ab"] + batting_df["bb"] +
        batting_df["hbp"].fillna(0)
    )

    features = {}

    for tid, grp in tqdm(batting_df.groupby("team_id"), desc="  Teams", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            game_id = int(row["game_id"])

            prior = grp.iloc[:i]
            feat = {}

            if len(prior) < 5:
                feat = _empty_batting_features()
            else:
                for window, suffix in [(15, "15"), (30, "30")]:
                    w = prior.tail(window)
                    pa = w["pa"].sum()
                    ab = w["ab"].sum()

                    feat[f"b_avg_{suffix}"] = _safe_div(w["hits"].sum(), ab)
                    feat[f"b_obp_{suffix}"] = _safe_div(w["obp_num"].sum(), w["obp_den"].sum())
                    feat[f"b_slg_{suffix}"] = _safe_div(w["tb"].sum(), ab)
                    feat[f"b_ops_{suffix}"] = (feat[f"b_obp_{suffix}"] or 0) + (feat[f"b_slg_{suffix}"] or 0)
                    feat[f"b_iso_{suffix}"] = _safe_div((w["tb"].sum() - w["hits"].sum()), ab)
                    feat[f"b_woba_{suffix}"] = _safe_div(w["woba_num"].sum(), w["woba_den"].sum())
                    feat[f"b_kpct_{suffix}"] = _safe_div(w["so"].sum(), pa)
                    feat[f"b_bbpct_{suffix}"] = _safe_div(w["bb"].sum(), pa)
                    feat[f"b_rpg_{suffix}"] = round(w["runs"].sum() / len(w), 3) if len(w) > 0 else None

            features[(game_id, int(tid))] = feat

    return features


def _empty_batting_features():
    feat = {}
    for suffix in ["15", "30"]:
        for k in ["b_avg", "b_obp", "b_slg", "b_ops", "b_iso",
                   "b_woba", "b_kpct", "b_bbpct", "b_rpg"]:
            feat[f"{k}_{suffix}"] = None
    return feat


# ---------------------------------------------------------------------------
# Bullpen features
# ---------------------------------------------------------------------------
def load_bullpen():
    """Load all reliever pitching game logs."""
    return query("""
        SELECT
            pg.game_id, pg.player_id, pg.team_id,
            pg.ip, pg.hits_allowed, pg.earned_runs, pg.bb, pg.so, pg.hr_allowed,
            g.game_date, g.home_team_id, g.away_team_id
        FROM mlb_pitching_game pg
        JOIN games g ON pg.game_id = g.game_id
        WHERE pg.is_starter = false AND pg.ip > 0
        ORDER BY g.game_date, pg.game_id
    """)


def build_bullpen_features(bullpen_df):
    """
    Build rolling bullpen features per team.
    Returns dict: (game_id, team_id) -> feature dict
    """
    print("  Building bullpen features...")
    bullpen_df = bullpen_df.sort_values("game_date").copy()

    # Convert IP to true innings
    bullpen_df["ip_true"] = bullpen_df["ip"].apply(
        lambda x: int(x) + (x % 1) * 10 / 3 if pd.notna(x) else 0
    )

    # Aggregate per team per game
    team_game = bullpen_df.groupby(["game_id", "team_id", "game_date"]).agg(
        bp_ip=("ip_true", "sum"),
        bp_er=("earned_runs", "sum"),
        bp_hits=("hits_allowed", "sum"),
        bp_bb=("bb", "sum"),
        bp_so=("so", "sum"),
        bp_hr=("hr_allowed", "sum"),
        bp_arms=("player_id", "nunique"),
    ).reset_index()

    features = {}

    for tid, grp in tqdm(team_game.groupby("team_id"), desc="  Bullpens", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            game_id = int(row["game_id"])
            game_date = row["game_date"]

            # Rolling 7-day window (by date, not by games)
            mask = (grp["game_date"] < game_date) & (grp["game_date"] >= game_date - pd.Timedelta(days=7))
            window = grp[mask]

            # Rolling 3-day fatigue
            mask_3d = (grp["game_date"] < game_date) & (grp["game_date"] >= game_date - pd.Timedelta(days=3))
            window_3d = grp[mask_3d]

            feat = {}
            if len(window) < 2:
                feat = _empty_bullpen_features()
            else:
                ip = window["bp_ip"].sum()
                feat["bp_era_7d"] = _rate(window["bp_er"].sum(), ip, 9)
                feat["bp_whip_7d"] = _rate(window["bp_hits"].sum() + window["bp_bb"].sum(), ip, 1)
                feat["bp_k9_7d"] = _rate(window["bp_so"].sum(), ip, 9)
                feat["bp_ip_7d"] = round(ip, 1)
                feat["bp_ip_3d"] = round(window_3d["bp_ip"].sum(), 1) if len(window_3d) > 0 else 0.0

            features[(game_id, int(tid))] = feat

    return features


def _empty_bullpen_features():
    return {k: None for k in [
        "bp_era_7d", "bp_whip_7d", "bp_k9_7d", "bp_ip_7d", "bp_ip_3d",
        "bp_high_lev_avail",
    ]}


def build_bullpen_availability(bullpen_raw):
    """
    Build high-leverage bullpen availability feature per team per game.

    For each team/game, identifies the top relievers (by save+hold rate in
    trailing 30 days) and checks how many are available (haven't pitched in
    the last 2 days). Returns a fraction: 1.0 = all top relievers rested,
    0.0 = all recently used.

    Returns dict: (game_id, team_id) -> availability fraction
    """
    print("  Building bullpen availability...")
    bp = bullpen_raw.copy()
    bp = bp.sort_values("game_date")

    # Load decision data for save/hold classification
    from db.db import query as db_query
    decisions = db_query("""
        SELECT pg.game_id, pg.player_id, pg.decision
        FROM mlb_pitching_game pg
        WHERE pg.is_starter = false AND pg.decision IN ('S', 'H')
    """)
    # Map (game_id, player_id) -> has_save_or_hold
    sh_set = set()
    for _, r in decisions.iterrows():
        sh_set.add((int(r["game_id"]), int(r["player_id"])))

    # Mark high-leverage appearances in bullpen data
    bp["is_high_lev"] = bp.apply(
        lambda r: (int(r["game_id"]), int(r["player_id"])) in sh_set, axis=1
    )

    availability = {}

    for tid, grp in tqdm(bp.groupby("team_id"), desc="  BP Avail", leave=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        # Get unique game dates for this team
        game_dates = grp.groupby("game_id")["game_date"].first().reset_index()
        game_dates = game_dates.sort_values("game_date")

        for _, gd_row in game_dates.iterrows():
            game_id = int(gd_row["game_id"])
            game_date = gd_row["game_date"]

            # Trailing 30-day window to identify top relievers
            mask_30d = (grp["game_date"] < game_date) & \
                       (grp["game_date"] >= game_date - pd.Timedelta(days=30))
            recent = grp[mask_30d]

            if len(recent) < 5:
                availability[(game_id, tid)] = None
                continue

            # Rank relievers by save+hold appearances in the window
            reliever_scores = recent.groupby("player_id")["is_high_lev"].sum()
            # Top 3 relievers by high-leverage appearances
            top_relievers = reliever_scores.nlargest(3).index.tolist()

            if len(top_relievers) == 0:
                availability[(game_id, tid)] = None
                continue

            # Check which of the top relievers pitched in the last 2 days
            mask_2d = (grp["game_date"] < game_date) & \
                      (grp["game_date"] >= game_date - pd.Timedelta(days=2))
            recent_2d = grp[mask_2d]
            recently_used = set(recent_2d["player_id"].unique())

            available_count = sum(1 for p in top_relievers if p not in recently_used)
            availability[(game_id, tid)] = round(available_count / len(top_relievers), 2)

    return availability


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------
def compute_park_factors(games_df):
    """
    Compute park factors as runs scored at venue vs league average.
    Returns dict: venue -> park_factor
    """
    print("  Computing park factors...")
    games_df = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()].copy()
    games_df["total_runs"] = games_df["home_score"] + games_df["away_score"]

    league_avg = games_df["total_runs"].mean()

    park_factors = {}
    for venue, grp in games_df.groupby("venue"):
        if len(grp) < 20:  # need enough games for stable estimate
            park_factors[venue] = 1.0
        else:
            park_factors[venue] = round(grp["total_runs"].mean() / league_avg, 3)

    return park_factors


# ---------------------------------------------------------------------------
# Assemble feature matrix
# ---------------------------------------------------------------------------
def _inject_scheduled_games(games, pitching, batting, bullpen):
    """Add synthetic rows to pitching/batting/bullpen for scheduled games.

    For the rolling feature computation to work on scheduled games, each needs
    a row in the pitching/batting data. The synthetic row has zero stats — only
    the game_id, player_id/team_id, and game_date are needed so the function
    computes rolling features from PRIOR real entries.
    """
    scheduled = games[games["status"] != "final"]
    if len(scheduled) == 0:
        return pitching, batting, bullpen

    # Synthetic pitching rows for probable starters
    synth_pitch = []
    for _, g in scheduled.iterrows():
        for starter_id, team_id in [
            (g.get("home_starter_id"), g["home_team_id"]),
            (g.get("away_starter_id"), g["away_team_id"]),
        ]:
            if pd.isna(starter_id) if isinstance(starter_id, float) else starter_id is None:
                continue
            synth_pitch.append({
                "game_id": int(g["game_id"]),
                "player_id": int(starter_id),
                "team_id": int(team_id),
                "is_starter": True,
                "ip": 0, "hits_allowed": 0, "runs": 0, "earned_runs": 0,
                "bb": 0, "so": 0, "hr_allowed": 0, "pitches": 0, "strikes": 0,
                "game_date": g["game_date"],
                "home_team_id": int(g["home_team_id"]),
                "away_team_id": int(g["away_team_id"]),
                "season_id": g["season_id"],
            })

    if synth_pitch:
        pitching = pd.concat([pitching, pd.DataFrame(synth_pitch)], ignore_index=True)
        pitching = pitching.sort_values("game_date").reset_index(drop=True)

    # Synthetic batting rows for each team in scheduled games
    synth_bat = []
    for _, g in scheduled.iterrows():
        for team_id in [g["home_team_id"], g["away_team_id"]]:
            synth_bat.append({
                "game_id": int(g["game_id"]),
                "team_id": int(team_id),
                "pa": 0, "ab": 0, "hits": 0, "doubles": 0, "triples": 0,
                "hr": 0, "rbi": 0, "bb": 0, "so": 0, "hbp": 0,
                "sb": 0, "cs": 0,
                "game_date": g["game_date"],
                "home_team_id": int(g["home_team_id"]),
                "away_team_id": int(g["away_team_id"]),
                "home_score": 0, "away_score": 0,
            })

    if synth_bat:
        batting = pd.concat([batting, pd.DataFrame(synth_bat)], ignore_index=True)
        batting = batting.sort_values("game_date").reset_index(drop=True)

    # Synthetic bullpen rows (per team, per scheduled game)
    synth_bp = []
    for _, g in scheduled.iterrows():
        for team_id in [g["home_team_id"], g["away_team_id"]]:
            synth_bp.append({
                "game_id": int(g["game_id"]),
                "player_id": 0,  # dummy
                "team_id": int(team_id),
                "ip": 0, "hits_allowed": 0, "earned_runs": 0,
                "bb": 0, "so": 0, "hr_allowed": 0,
                "game_date": g["game_date"],
                "home_team_id": int(g["home_team_id"]),
                "away_team_id": int(g["away_team_id"]),
            })

    if synth_bp:
        bullpen = pd.concat([bullpen, pd.DataFrame(synth_bp)], ignore_index=True)
        bullpen = bullpen.sort_values("game_date").reset_index(drop=True)

    return pitching, batting, bullpen


def _build_umpire_zones():
    """Compute per-umpire called strike rate from Statcast data.

    Uses only borderline pitches (zones 11-14 = shadow zone) to measure
    how wide each umpire's zone is. A higher rate = wider zone = more K's = fewer runs.

    Returns dict: umpire_name -> {"zone_rate": float, "games": int}
    """
    ump_data = query("""
        SELECT gi.umpire_hp,
               COUNT(*) as borderline_pitches,
               SUM(CASE WHEN p.is_strike AND p.description = 'called_strike' THEN 1 ELSE 0 END) as called_strikes
        FROM mlb_pitches p
        JOIN mlb_game_info gi ON p.game_id = gi.game_id
        WHERE gi.umpire_hp IS NOT NULL
          AND p.zone BETWEEN 11 AND 14
        GROUP BY gi.umpire_hp
        HAVING COUNT(*) >= 200
    """)

    zones = {}
    if len(ump_data) > 0:
        league_avg = ump_data["called_strikes"].sum() / ump_data["borderline_pitches"].sum()
        for _, r in ump_data.iterrows():
            name = r["umpire_hp"]
            rate = r["called_strikes"] / r["borderline_pitches"]
            zones[name] = {
                "zone_rate": round(rate, 4),
                "zone_vs_avg": round(rate - league_avg, 4),  # positive = wider zone = fewer runs
                "pitches": int(r["borderline_pitches"]),
            }

    return zones


def build_feature_matrix(start_year, end_year, include_scheduled=False):
    """Build the full game-level feature matrix.

    When include_scheduled=True, also builds feature rows for scheduled/upcoming
    games using probable pitcher info and rolling team stats from prior games.
    """
    print(f"\nLoading data...")
    games = load_games(include_scheduled=include_scheduled)
    pitching = load_pitching()
    batting = load_batting()
    bullpen_raw = load_bullpen()

    print(f"  Games: {len(games)}, Pitching: {len(pitching)}, Batting: {len(batting)}, Bullpen: {len(bullpen_raw)}")

    # Filter to requested years
    games = games[(games["season_year"] >= start_year) & (games["season_year"] <= end_year)]
    print(f"  Filtered to {start_year}-{end_year}: {len(games)} games")

    # Inject synthetic rows for scheduled games so rolling features get computed
    if include_scheduled:
        pitching, batting, bullpen_raw = _inject_scheduled_games(games, pitching, batting, bullpen_raw)

    # Build umpire zone metrics from Statcast (before main feature build)
    print("  Building umpire zone metrics...")
    umpire_zones = _build_umpire_zones()

    # Build features
    pitcher_feats = build_pitcher_features(pitching)
    batting_feats = build_batting_features(batting)
    bullpen_feats = build_bullpen_features(bullpen_raw)
    bp_availability = build_bullpen_availability(bullpen_raw)
    park_factors = compute_park_factors(games)

    # Build ELO
    from models.mlb.elo import MLBElo
    elo = MLBElo()
    elo.run(start_year=2015, end_year=end_year)
    elo_data = elo.get_game_elos()

    # Assemble rows
    print("  Assembling feature matrix...")
    rows = []

    for _, game in tqdm(games.iterrows(), total=len(games), desc="  Games", leave=False):
        gid = int(game["game_id"])
        home_tid = int(game["home_team_id"])
        away_tid = int(game["away_team_id"])

        home_score = game["home_score"]
        away_score = game["away_score"]
        has_scores = pd.notna(home_score) and pd.notna(away_score)

        row = {
            "game_id": gid,
            "game_date": game["game_date"],
            "season": int(game["season_year"]),
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_score": home_score if has_scores else None,
            "away_score": away_score if has_scores else None,
            "home_win": (1 if home_score > away_score else 0) if has_scores else None,
            "is_postseason": game["is_postseason"],
        }

        # Home starter features
        home_starter = game.get("home_starter_id")
        if pd.notna(home_starter):
            h_pitch = pitcher_feats.get((gid, int(home_starter)), _empty_pitcher_features())
            for k, v in h_pitch.items():
                row[f"home_{k}"] = v
        else:
            for k, v in _empty_pitcher_features().items():
                row[f"home_{k}"] = v

        # Away starter features
        away_starter = game.get("away_starter_id")
        if pd.notna(away_starter):
            a_pitch = pitcher_feats.get((gid, int(away_starter)), _empty_pitcher_features())
            for k, v in a_pitch.items():
                row[f"away_{k}"] = v
        else:
            for k, v in _empty_pitcher_features().items():
                row[f"away_{k}"] = v

        # Home batting features
        h_bat = batting_feats.get((gid, home_tid), _empty_batting_features())
        for k, v in h_bat.items():
            row[f"home_{k}"] = v

        # Away batting features
        a_bat = batting_feats.get((gid, away_tid), _empty_batting_features())
        for k, v in a_bat.items():
            row[f"away_{k}"] = v

        # Bullpen features
        h_bp = bullpen_feats.get((gid, home_tid), _empty_bullpen_features())
        for k, v in h_bp.items():
            row[f"home_{k}"] = v

        a_bp = bullpen_feats.get((gid, away_tid), _empty_bullpen_features())
        for k, v in a_bp.items():
            row[f"away_{k}"] = v

        # Bullpen availability (high-leverage reliever freshness)
        row["home_bp_high_lev_avail"] = bp_availability.get((gid, home_tid))
        row["away_bp_high_lev_avail"] = bp_availability.get((gid, away_tid))

        # ELO features
        game_elo = elo_data.get(gid, {})
        row["home_elo"] = game_elo.get("home_elo")
        row["away_elo"] = game_elo.get("away_elo")
        row["home_elo_eff"] = game_elo.get("home_elo_eff")
        row["away_elo_eff"] = game_elo.get("away_elo_eff")
        row["elo_diff"] = game_elo.get("elo_diff")
        row["elo_win_prob"] = game_elo.get("home_win_prob")

        # Contextual
        row["park_factor"] = park_factors.get(game["venue"], 1.0)
        row["weather_temp"] = game.get("weather_temp")
        row["weather_wind"] = game.get("weather_wind")

        # Wind direction: encode as run-impact categories
        wind_dir = game.get("weather_dir")
        if wind_dir and isinstance(wind_dir, str):
            wind_dir = wind_dir.strip().rstrip(".")
            if "out to" in wind_dir:
                row["wind_out"] = 1  # run-boosting
                row["wind_in"] = 0
            elif "in from" in wind_dir:
                row["wind_out"] = 0
                row["wind_in"] = 1  # run-suppressing
            else:
                row["wind_out"] = 0
                row["wind_in"] = 0
        else:
            row["wind_out"] = None
            row["wind_in"] = None

        # Dome indicator (from weather_cond)
        weather_cond = game.get("weather_cond")
        if weather_cond and isinstance(weather_cond, str):
            row["is_dome"] = 1 if weather_cond.strip().rstrip(".") in ("dome", "roof closed") else 0
        else:
            row["is_dome"] = None

        # Umpire zone (wider zone = more K's = fewer runs)
        ump_name = game.get("umpire_hp")
        if ump_name and ump_name in umpire_zones:
            row["ump_zone_vs_avg"] = umpire_zones[ump_name]["zone_vs_avg"]
        else:
            row["ump_zone_vs_avg"] = None

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute differentials (home - away)
    diff_cols = [
        ("p_fip_5", True),       # lower FIP is better, so invert
        ("p_kpct_5", False),
        ("p_bbpct_5", True),
        ("p_era_szn", True),
        ("b_woba_15", False),
        ("b_ops_15", False),
        ("b_rpg_15", False),
        ("b_kpct_15", True),     # lower K% is better for batters
        ("bp_era_7d", True),     # lower bullpen ERA is better
        ("bp_whip_7d", True),
        ("bp_ip_3d", True),      # more recent bullpen usage = worse (fatigued)
    ]
    for col, invert in diff_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in df.columns and a_col in df.columns:
            if invert:
                df[f"diff_{col}"] = df[a_col] - df[h_col]  # reversed so positive = home advantage
            else:
                df[f"diff_{col}"] = df[h_col] - df[a_col]

    print(f"\n  Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build MLB feature matrix")
    parser.add_argument("--start", type=int, default=2016)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: data/mlb_features.csv)")
    args = parser.parse_args()

    df = build_feature_matrix(args.start, args.end)

    out_path = args.out or "data/mlb_features.csv"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("FEATURE MATRIX SUMMARY")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"  Home win rate: {df['home_win'].mean():.3f}")
    print(f"  Non-null pitcher features: {df['home_p_fip_5'].notna().sum()} / {len(df)}")
    print(f"  Non-null batting features: {df['home_b_woba_15'].notna().sum()} / {len(df)}")
    print(f"  Non-null bullpen features: {df['home_bp_era_7d'].notna().sum()} / {len(df)}")
    print(f"  Non-null ELO features: {df['elo_diff'].notna().sum()} / {len(df)}")
    print(f"\nSample feature stats:")
    feat_cols = [c for c in df.columns if c.startswith("diff_")]
    if feat_cols:
        print(df[feat_cols].describe().round(4).to_string())


if __name__ == "__main__":
    main()
