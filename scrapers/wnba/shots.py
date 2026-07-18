"""
scrapers/wnba/shots.py — WNBA shot-chart backfill via stats.wnba.com (nba_api).

Per-shot rows (location zone, distance, made/missed) -> wnba_shots. Feeds batch-3
shot-profile features (threes market) + derived rebound-environment features.
Usage: python scrapers/wnba/shots.py --seasons 2024,2025,2026
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import pandas as pd
from db.db import query, execute, get_conn
from psycopg2.extras import execute_values
from nba_api.stats.endpoints import shotchartdetail, commonallplayers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025,2026")
    seasons = [s.strip() for s in ap.parse_args().seasons.split(",")]
    execute("""CREATE TABLE IF NOT EXISTS wnba_shots (
        wnba_player_id INT NOT NULL, season INT NOT NULL, game_id_ext VARCHAR(20),
        game_date DATE, period INT, shot_type VARCHAR(20), zone VARCHAR(40),
        area VARCHAR(30), dist INT, made BOOLEAN, x INT, y INT,
        PRIMARY KEY (wnba_player_id, game_id_ext, period, x, y, dist))""")
    for season in seasons:
        ps = commonallplayers.CommonAllPlayers(league_id='10', season=season).get_data_frames()[0]
        active = ps[(ps['TO_YEAR'].astype(int) >= int(season)) &
                    (ps['FROM_YEAR'].astype(int) <= int(season)) &
                    (ps['ROSTERSTATUS'] == 1)]
        print(f"{season}: {len(active)} rostered players")
        n_tot = 0
        for i, r in enumerate(active.itertuples()):
            for attempt in range(3):
                try:
                    df = shotchartdetail.ShotChartDetail(
                        league_id='10', team_id=0, player_id=int(r.PERSON_ID),
                        season_nullable=season, context_measure_simple='FGA',
                        timeout=30).get_data_frames()[0]
                    break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"  skip {r.DISPLAY_FIRST_LAST}"); continue
            if df.empty:
                time.sleep(0.7); continue
            rows = [(int(r.PERSON_ID), int(season), str(g.GAME_ID), pd.to_datetime(g.GAME_DATE).date(),
                     int(g.PERIOD), str(g.ACTION_TYPE)[:20], str(g.SHOT_ZONE_BASIC)[:40],
                     str(g.SHOT_ZONE_AREA)[:30], int(g.SHOT_DISTANCE),
                     bool(g.SHOT_MADE_FLAG), int(g.LOC_X), int(g.LOC_Y)) for g in df.itertuples()]
            with get_conn() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, """INSERT INTO wnba_shots VALUES %s
                        ON CONFLICT DO NOTHING""", rows, page_size=500)
                conn.commit()
            n_tot += len(rows)
            if i % 25 == 0:
                print(f"  [{i}/{len(active)}] {n_tot:,} shots", flush=True)
            time.sleep(0.7)
        print(f"{season}: {n_tot:,} shots loaded")

if __name__ == "__main__":
    main()
