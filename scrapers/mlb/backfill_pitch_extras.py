"""
scrapers/mlb/backfill_pitch_extras.py — backfill Statcast columns we never stored.

Statcast is a full historical archive: these columns exist retroactively for every pitch,
we just didn't keep them. One pull unlocks four datasets (see docs/LEG1_MODEL_V2_PRD.md):

    catcher_mlbam (fielder_2)      -> catcher framing features        (exists 2015+)
    hc_x, hc_y                     -> true spray / pull%              (exists 2015+)
    bat_speed, swing_length        -> bat tracking                    (exists May 2024+)
    arm_angle                      -> pitcher release geometry        (exists 2023-24+)
    attack_angle / attack_direction / swing_path_tilt                 (exists ~2025+)

Writes to mlb_pitch_extras keyed (game_id, at_bat_number, pitch_number) — insert-only,
idempotent, joins 1:1 to mlb_pitches. Default window 2024-03-20+ (bat-tracking era);
extend --start earlier later if spray/framing features prove out and need training depth.

Usage:
    python -m scrapers.mlb.backfill_pitch_extras --start 2024-03-20 --end 2026-07-02
    python -m scrapers.mlb.backfill_pitch_extras --start 2026-06-25   # top-up
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse, os as _os, subprocess, sys as _sys, tempfile, time, warnings
from datetime import date, timedelta, datetime
warnings.filterwarnings("ignore")

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from db.db import query

CHUNK_DAYS = 6
PULL_TIMEOUT_S = 300   # hard cap per chunk pull, enforced at the PROCESS level

# The pull runs in a subprocess killed by the OS on timeout. In-process SIGALRM proved
# insufficient: pybaseball's threaded internals swallowed/evaded it (observed: two
# multi-hour hangs on a stale connection with alarm armed).
_CHILD = r'''
import sys, warnings; warnings.filterwarnings("ignore")
from pybaseball import statcast
start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
df = statcast(start_dt=start, end_dt=end, verbose=False)
keep = [c for c in ["game_pk","at_bat_number","pitch_number","batter","fielder_2",
        "hc_x","hc_y","bat_speed","swing_length","arm_angle",
        "attack_angle","attack_direction","swing_path_tilt"] if c in df.columns]
if df is not None and len(df):
    df[keep].to_parquet(out, index=False)
else:
    open(out + ".empty", "w").write("")
'''


def statcast_with_timeout(start_dt, end_dt, timeout=PULL_TIMEOUT_S):
    """Pull one chunk in a subprocess with an OS-enforced timeout. Returns a DataFrame
    (possibly empty) or raises on timeout/failure."""
    with tempfile.TemporaryDirectory() as td:
        out = _os.path.join(td, "chunk.parquet")
        subprocess.run([_sys.executable, "-c", _CHILD, start_dt, end_dt, out],
                       timeout=timeout, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if _os.path.exists(out + ".empty"):
            return pd.DataFrame()
        return pd.read_parquet(out)
WANT = ['fielder_2', 'hc_x', 'hc_y', 'bat_speed', 'swing_length',
        'arm_angle', 'attack_angle', 'attack_direction', 'swing_path_tilt']
COLS = ['game_id', 'at_bat_number', 'pitch_number', 'batter_id', 'catcher_mlbam',
        'hc_x', 'hc_y', 'bat_speed', 'swing_length', 'arm_angle',
        'attack_angle', 'attack_direction', 'swing_path_tilt']


def game_lookup():
    df = query("SELECT game_id, external_id FROM games WHERE sport_id = 2")
    return dict(zip(df['external_id'].astype(str), df['game_id']))


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _i(v):
    f = _f(v)
    return int(f) if f is not None else None


def _fresh_conn():
    """Dedicated insert connection: TCP keepalives + statement timeout so a silently-dead
    Supabase socket errors in seconds instead of blocking forever (observed hang mode)."""
    return psycopg2.connect(
        _os.environ["DATABASE_URL"], connect_timeout=15,
        keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=3,
        options="-c statement_timeout=120000")


def insert_rows(rows, max_retries=3):
    """Batched insert via execute_values (ONE round trip per ~1000 rows).
    executemany was the original sin here: one round trip PER ROW -> ~15 rows/s to
    Supabase, so a 16K-row chunk took 23 minutes and looked like a network hang."""
    if not rows:
        return 0
    sql = (f"INSERT INTO mlb_pitch_extras ({', '.join(COLS)}) VALUES %s "
           f"ON CONFLICT (game_id, at_bat_number, pitch_number) DO NOTHING")
    for attempt in range(max_retries):
        conn = None
        try:
            conn = _fresh_conn()
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, page_size=1000)
                n = cur.rowcount
            conn.commit()
            conn.close()
            return n
        except Exception as e:
            if conn is not None:
                try: conn.close()
                except Exception: pass
            wait = 2 ** (attempt + 1)
            print(f"    DB error {attempt+1}/{max_retries}: {type(e).__name__}: {e}; "
                  f"sleep {wait}s", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-20")
    ap.add_argument("--end", default=str(date.today()))
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    lut = game_lookup()
    print(f"backfilling pitch extras {start} -> {end} ({len(lut):,} games in lookup)")

    total, cur, failed_chunks = 0, start, []
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        t0 = time.time()
        df = None
        for attempt in range(3):
            try:
                df = statcast_with_timeout(str(cur), str(chunk_end))
                break
            except Exception as e:   # TimeoutExpired, CalledProcessError, parquet errors
                wait = 20 * (attempt + 1)
                print(f"  {cur}..{chunk_end}: pull failed attempt {attempt+1}/3 "
                      f"({type(e).__name__}); sleeping {wait}s", flush=True)
                time.sleep(wait)
        pull_s = time.time() - t0
        if df is None:
            print(f"  {cur}..{chunk_end}: FAILED after 3 attempts; skipping chunk", flush=True)
            cur = chunk_end + timedelta(days=1)
            continue
        if df is None or len(df) == 0:
            print(f"  {cur}..{chunk_end}: 0 pitches")
            cur = chunk_end + timedelta(days=1)
            continue
        for c in WANT:
            if c not in df.columns:
                df[c] = None
        df['db_game_id'] = df['game_pk'].astype(str).map(lut)
        df = df[df['db_game_id'].notna()]
        rows = [(int(r.db_game_id), _i(r.at_bat_number), _i(r.pitch_number), _i(r.batter),
                 _i(r.fielder_2), _f(r.hc_x), _f(r.hc_y), _f(r.bat_speed), _f(r.swing_length),
                 _f(r.arm_angle), _f(r.attack_angle), _f(r.attack_direction), _f(r.swing_path_tilt))
                for r in df.itertuples()
                if r.at_bat_number == r.at_bat_number and r.pitch_number == r.pitch_number]
        t1 = time.time()
        try:
            n = insert_rows(rows)
        except Exception as e:
            print(f"  {cur}..{chunk_end}: INSERT FAILED after retries "
                  f"({type(e).__name__}) — chunk skipped, rerun later", flush=True)
            failed_chunks.append((str(cur), str(chunk_end)))
            cur = chunk_end + timedelta(days=1)
            continue
        total += n
        bt = sum(1 for r in rows if r[7] is not None)
        print(f"  {cur}..{chunk_end}: {len(rows):,} pitches, {n:,} inserted "
              f"(bat-tracking on {bt:,}) [pull {pull_s:.0f}s, insert {time.time()-t1:.0f}s]",
              flush=True)
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)

    print(f"\nTOTAL inserted: {total:,}")
    if failed_chunks:
        print(f"FAILED chunks ({len(failed_chunks)}) — rerun with these ranges:")
        for s, e in failed_chunks:
            print(f"  --start {s} --end {e}")
    chk = query("""SELECT COUNT(*) n, COUNT(bat_speed) bs, COUNT(hc_x) hc, COUNT(catcher_mlbam) cat,
                   COUNT(attack_angle) aa FROM mlb_pitch_extras""")
    print(chk.to_string(index=False))


if __name__ == "__main__":
    main()
