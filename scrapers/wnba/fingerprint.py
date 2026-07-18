"""
scrapers/wnba/fingerprint.py — ESPN Analytics WNBA fingerprint loader.

Per-season player files (unsigned S3, discovered via user's browser cURL):
    .../net-pts/fingerprint-files/wnbafingerprint_{year}.json
Rich profile: height/DOB/avg-position/usage + net-points by 26 play types (o/d/t).
LEAK RULE: season files are season-AGGREGATES -> use PRIOR-season rows as features.
Usage: python scrapers/wnba/fingerprint.py --years 2024,2025,2026
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import math
import requests
from db.db import execute, get_conn
from psycopg2.extras import execute_values

BASE = "https://nfl-player-metrics.s3.us-east-1.amazonaws.com/net-pts/fingerprint-files"
HDRS = {"Referer": "https://espnanalytics.com/", "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"}

def hin(h):
    try:
        f, i = str(h).split('-'); return int(f) * 12 + int(i)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--years", default="2024,2025,2026")
    years = [y.strip() for y in ap.parse_args().years.split(",")]
    execute("""CREATE TABLE IF NOT EXISTS wnba_fingerprint (
        wnba_player_id INT NOT NULL, season INT NOT NULL,
        display_name VARCHAR(60), team VARCHAR(8), height_in INT, dob VARCHAR(12),
        avg_position DECIMAL(6,3), offensive_usage DECIMAL(8,4), assisted_rate DECIMAL(8,4),
        minutes_played DECIMAL(8,1), games_played INT, tposs DECIMAL(10,2),
        data JSONB,
        PRIMARY KEY (wnba_player_id, season))""")
    for y in years:
        r = requests.get(f"{BASE}/wnbafingerprint_{y}.json", headers=HDRS, timeout=60)
        if r.status_code != 200:
            print(f"{y}: HTTP {r.status_code} — skipped"); continue
        d = r.json()
        rows = []
        for pid, v in d.items():
            rows.append((int(pid), int(y), v.get('displayName'), v.get('deanAbbrev'),
                         hin(v.get('height')), v.get('dob'),
                         v.get('average_position'), v.get('offensive_usage'),
                         v.get('assisted_rate'), v.get('minutes_played'),
                         int(v.get('games_played') or 0), v.get('tPoss'),
                         json.dumps({k: (None if isinstance(x, float) and math.isnan(x) else x)
                                     for k, x in v.items()})))
        with get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(cur, """INSERT INTO wnba_fingerprint VALUES %s
                    ON CONFLICT (wnba_player_id, season) DO UPDATE SET
                    data=EXCLUDED.data, minutes_played=EXCLUDED.minutes_played,
                    games_played=EXCLUDED.games_played, tposs=EXCLUDED.tposs,
                    avg_position=EXCLUDED.avg_position,
                    offensive_usage=EXCLUDED.offensive_usage,
                    assisted_rate=EXCLUDED.assisted_rate""", rows, page_size=200)
            conn.commit()
        print(f"{y}: {len(rows)} players loaded")

if __name__ == "__main__":
    main()
