# MLB Hitter Prop Modeling Platform

A data + modeling pipeline for **MLB player prop betting** (Hits + Runs + RBIs,
Total Bases, Hits, etc.) targeting Underdog Fantasy's adjusted-odds market.

Pulls Statcast pitch data, MLB box scores, and Underdog prop lines into
Postgres on a daily schedule. Models are built per-player-game using
matchup-aware features (batter vs pitch type, pitcher arsenal, lineup
context, swing tendencies).

## Why hitter props

This project originally targeted MLB game totals. After 60 days of live
deployment (April-June 2026), all three totals models were unprofitable
(see [`archive/TOTALS_LIVE_TEST_FINDINGS.md`](archive/TOTALS_LIVE_TEST_FINDINGS.md)).
The 2026 environment is structurally UNDER-favored, and our model echoed
market consensus on direction selection — meaning we were betting the same
side as the public and losing.

Hitter props on Underdog have three advantages over game totals:
1. **Less efficient markets.** Books can't price ~250 hitters × 13 stat
   types × 15 games per day with the same rigor as 15 game totals.
2. **Explicit odds and payout multipliers in the API.** Underdog's API
   returns `american_price`, `decimal_price`, and `payout_multiplier`
   per side — no hidden combinatorial pricing like PrizePicks.
3. **Matchup-driven outcomes.** Hitter performance against specific
   pitcher arsenals is more model-tractable than game totals which are
   already a sum of many noisy components.

## Project status

| Component | Status |
|-----------|--------|
| MLB game/box score scraper | ✅ Running daily |
| Statcast pitch data scraper | ✅ Running daily (caught up through current) |
| Underdog props capture | ✅ Running 3x weekday / 4x weekend |
| Hitter prop model | 🚧 Not yet built — next phase |
| Live prop betting | ⏸️ Pending model |

## Architecture

```
                       ┌──────────────────┐
   MLB Stats API ───►  │ scrapers/mlb/    │ ──► games, box scores, weather, umpire
                       └──────────────────┘
                                │
  Baseball Savant       ┌──────────────────┐
   (pybaseball)   ───►  │ scrapers/mlb/    │ ──► mlb_pitches (pitch-by-pitch)
                       │   statcast_daily │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Postgres        │ ◄────── scrapers/props/underdog.py
                       │  (Supabase)      │            (Underdog prop snapshots)
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ models/mlb/      │ (planned)
                       │  hitter prop     │ ──► predicted P(over) per prop line
                       │  classifier      │ ──► EV vs Underdog odds
                       └──────────────────┘
```

## Repo layout

```
.
├── README.md                 — This file
├── archive/                  — Legacy totals/moneyline/WNBA work
│   ├── LEGACY_PATHS.md       — Import-dependency map for reviving any archived file
│   ├── TOTALS_LIVE_TEST_FINDINGS.md  — Why the totals models were retired
│   └── ...                   — All legacy code preserved
│
├── docs/
│   └── CLAUDE.md             — Project notes for AI assistants
│
├── db/
│   ├── db.py                 — Postgres connection pool + helpers
│   ├── schema.sql            — Full schema definition
│   └── migrate_*.sql         — Schema additions over time
│
├── scrapers/
│   ├── mlb/
│   │   ├── games.py          — MLB Stats API: schedules + scores + game info
│   │   ├── boxscores.py      — Per-batter and per-pitcher game stats
│   │   ├── statcast.py       — Full-season Statcast pull (manual catchup)
│   │   ├── statcast_daily.py — Yesterday's pitches only (daily cron)
│   │   └── run_all.py        — Bootstrap helper for new clones
│   └── props/
│       └── underdog.py       — Underdog Fantasy MLB props snapshot capture
│
├── models/
│   └── mlb/
│       ├── statcast_features.py  — Pitcher Statcast features (whiff/chase/velo)
│       ├── lineup_features.py    — Per-player lineup feature aggregation
│       ├── k_model.py            — Pitcher strikeout Poisson model
│       └── hitter_model.py       — Earlier hitter prop attempt (starting point)
│
├── scripts/
│   └── daily_refresh.py      — Morning cron entry point
│
├── .github/workflows/
│   ├── daily_pipeline.yml    — Morning: pulls games + box scores + Statcast
│   └── underdog_capture.yml  — 3-4x daily: snapshots Underdog prop lines
│
├── config.py
├── requirements.txt
└── .env.example              — Set DATABASE_URL here
```

## Data sources

| Source | What it provides | Cost | Where it lands |
|--------|-----------------|------|----------------|
| MLB Stats API | Game schedules, scores, box scores, probable pitchers, weather, umpires | Free | `games`, `mlb_batting_game`, `mlb_pitching_game`, `mlb_game_info` |
| Baseball Savant (via pybaseball) | Pitch-by-pitch Statcast: velocity, spin, location, xwOBA, launch angle | Free | `mlb_pitches` |
| Underdog Fantasy API | Player prop lines: both sides' odds + multipliers | Free (no auth) | `underdog_props` |

## Database

- **Provider:** Supabase Postgres
- **Plan:** Pro ($25/mo, planned downgrade to free after pre-aggregation phase)
- **Current size:** ~2 GB (well under 8 GB Pro limit)
- **Connection:** via `DATABASE_URL` env var, `psycopg2` connection pool in `db/db.py`

Key tables:
- `games` — every MLB game 2015–present (~26K rows)
- `mlb_batting_game` — every batter × game outcome (H, R, RBI, etc.) (~600K rows)
- `mlb_pitching_game` — every pitcher × game outcome (~220K rows)
- `mlb_pitches` — pitch-by-pitch Statcast data (~7.6M rows, biggest table)
- `mlb_game_info` — weather, umpire, probable pitchers
- `underdog_props` — Underdog prop snapshots, multiple per day per line
- `predictions` — model predictions log (currently inactive; was used by retired totals models)

## Daily cron schedule

All times in UTC. ET = UTC-4 during summer.

| Cron | Workflow | What it does |
|------|----------|--------------|
| `0 10 * * *` (6 AM ET) | `daily_pipeline.yml` | Pulls yesterday's box scores + last 3 days of Statcast |
| `0 15 * * 1-5` (11 AM ET, weekday) | `underdog_capture.yml` | Underdog opening-ish lines |
| `0 19 * * 1-5` (3 PM ET, weekday) | `underdog_capture.yml` | Mid-day line movement |
| `0 22 * * 1-5` (6 PM ET, weekday) | `underdog_capture.yml` | Pre-night-games |
| `0 14 * * 0,6` (10 AM ET, weekend) | `underdog_capture.yml` | Opening lines |
| `30 16 * * 0,6` (12:30 PM ET, weekend) | `underdog_capture.yml` | Pre-day-games |
| `0 20 * * 0,6` (4 PM ET, weekend) | `underdog_capture.yml` | Mid-afternoon |
| `0 22 * * 0,6` (6 PM ET, weekend) | `underdog_capture.yml` | Pre-night-games |

## Setup (for a fresh clone)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Env
cp .env.example .env
# Set DATABASE_URL to your Postgres connection string

# 3. Schema
psql $DATABASE_URL -f db/schema.sql
psql $DATABASE_URL -f db/migrate_underdog_props.sql

# 4. Bootstrap historical data (slow: hours)
python -m scrapers.mlb.run_all --start 2015 --end 2026

# 5. Pull Statcast (very slow: ~1 day, rate-limited)
python -m scrapers.mlb.statcast --start 2015 --end 2026

# 6. Verify
python scripts/daily_refresh.py
python -m scrapers.props.underdog
```

## Running a one-off capture

```bash
# Yesterday's MLB data only
python scripts/daily_refresh.py

# Last 3 days of Statcast (used by morning cron)
python -m scrapers.mlb.statcast_daily --days 3

# Snapshot current Underdog MLB props (writes to DB + optionally CSV)
python -m scrapers.props.underdog
python -m scrapers.props.underdog --csv-also  # also write CSV
```

## What's next (planned work)

1. **Build batter arsenal profile features** from `mlb_pitches`:
   - Pull rate, launch angle distribution per batter
   - vs-pitch-type xwOBA (e.g. "Walker vs sinkers" .380, "Walker vs high fastballs" .290)
   - Zone preferences and chase tendencies
   - Stored as pre-computed lookup tables joined at predict time

2. **Build pitcher arsenal profile features**:
   - Pitch mix percentages (% sinker, % slider, etc.)
   - Velocity / spin rate per pitch type
   - Recent velo trend (last 3 starts vs season)

3. **Train hitter prop model** targeting Hits + Runs + RBIs (HRR) over 1.5:
   - Binary classifier with batter + pitcher + matchup + context features
   - L1-regularized LogReg as baseline
   - Expanding-window cross-validation across 2019–2025
   - Direction-level metrics from day 1 (lesson learned from totals OVER bias)

4. **Validate against actual outcomes** using `mlb_batting_game` (H/R/RBI per game).

5. **Once Underdog snapshot dataset is large enough (~30 days):**
   compute predicted P(over) vs actual closing line implied probability per prop,
   measure CLV going forward.

## Lessons from the totals work (preserved in `archive/`)

- Backtest profitability does not guarantee live profitability
- Aggregate calibration can hide systematic bet-selection bias
- Decomposing performance by direction surfaces failure modes aggregate metrics miss
- The "model echoes market consensus" failure mode is real and costs money
- Live test data is the only valid validation
