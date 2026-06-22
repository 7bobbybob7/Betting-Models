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
| MLB game/box score scraper | ✅ Running daily (now captures runs scored) |
| Statcast pitch data scraper | ✅ Running daily (caught up through current) |
| Player handedness (`bats`/`throws`) | ✅ Backfilled (4,160 players); cron refresh for rookies |
| Underdog props capture | ✅ Running 3x weekday / 4x weekend |
| BettingPros historical props (Underdog book_id=36) | ✅ 2026 done, 2024/2025 backfilling |
| `mlb_batting_game.runs` column | ✅ Schema added; historical backfill in progress (~20%) |
| Batter arsenal features | ✅ `models/mlb/batter_arsenal_features.py` (36 features) |
| Pitcher arsenal features (handedness-conditioned mix) | ✅ `models/mlb/pitcher_arsenal_features.py` (45) |
| Context features (rolling park factors, lineup support) | ✅ `models/mlb/context_features.py` (11) |
| Matchup features (weighted xwOBA / whiff / platoon / PA) | ✅ `models/mlb/matchup_features.py` (6) |
| Dataset assembler | ✅ `models/mlb/hitter_prop_dataset.py` (112 features + 3 labels) |
| Hitter prop training script | 🚧 Next step (LR-L1 + LightGBM, expanding-season CV) |
| Backtest script (EV threshold sweep) | 🚧 Pending |
| Live prop betting | ⏸️ Pending model |

## Architecture

```
   MLB Stats API ───►  scrapers/mlb/games.py + boxscores.py  ──► games, mlb_batting_game,
                                                                  mlb_pitching_game, players
                                                                          │
   Baseball Savant ──► scrapers/mlb/statcast.py + statcast_daily.py ──► mlb_pitches
                                                                          │
   Underdog API   ───► scrapers/props/underdog.py             ──► underdog_props (forward only)
                                                                          │
   BettingPros API ──► scrapers/props/bettingpros.py          ──► bettingpros_props
                                                                  (backfill + daily;
                                                                   includes Underdog odds via book_id=36)
                                                                          │
                                                                          ▼
                              Postgres on Supabase
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
   models/mlb/                    models/mlb/                    models/mlb/
   batter_arsenal_features.py     pitcher_arsenal_features.py   context_features.py
   (36 features: per-pitch xwOBA, (45: handedness-conditioned   (11: rolling 365D park
    whiff, plate discipline,       pitch mix, velo, allowed     factors, weather, lineup
    handedness, recent form)       rates, IP/start)              OBP-in-front, SLG-behind)
        │                              │                              │
        └──────────┬───────────────────┴────────────┬─────────────────┘
                   ▼                                ▼
            models/mlb/matchup_features.py    models/mlb/hitter_prop_dataset.py
            (6: weighted xwOBA, weighted       (orchestrates all 4 → 112 features
             whiff, platoon, PA estimates)      + HRR/TB/RBI binary labels)
                                                       │
                                                       ▼
                                  models/mlb/hitter_prop_model.py  (planned)
                                  models/mlb/backtest.py           (planned)
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
│   │   ├── games.py                       — MLB Stats API: schedules + scores
│   │   ├── boxscores.py                   — Per-batter / per-pitcher game stats (now captures runs)
│   │   ├── statcast.py                    — Full-season Statcast pull (manual catchup)
│   │   ├── statcast_daily.py              — Last 3 days of pitches (daily cron)
│   │   ├── backfill_player_handedness.py  — Fetch bats/throws via /api/v1/people
│   │   ├── backfill_batter_runs.py        — Backfill runs scored (re-fetches box scores)
│   │   └── run_all.py                     — Bootstrap helper for new clones
│   └── props/
│       ├── underdog.py       — Underdog Fantasy MLB props snapshot capture (incl. batter walks)
│       └── bettingpros.py    — BettingPros historical/daily props (Underdog odds via book_id=36)
│
├── models/
│   └── mlb/
│       ├── batter_arsenal_features.py    — 36 batter features (per pitch type + buckets)
│       ├── pitcher_arsenal_features.py   — 45 pitcher features (mix conditioned on batter hand)
│       ├── context_features.py           — 11 context features (rolling 365D park factors,
│       │                                   weather, lineup OBP-in-front, SLG-behind)
│       ├── matchup_features.py           — 6 cross-features (handedness-aware weighted xwOBA)
│       └── hitter_prop_dataset.py        — Assembler: 112 features + HRR/TB/RBI labels
│                                            (totals-era models live in archive/models/mlb/)
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
| MLB Stats API | Schedules, scores, box scores, probable pitchers, weather, umpires, player handedness | Free | `games`, `mlb_batting_game` (incl. runs), `mlb_pitching_game`, `mlb_game_info`, `players` |
| Baseball Savant (via pybaseball) | Pitch-by-pitch Statcast: velocity, spin, location, xwOBA, launch angle | Free | `mlb_pitches` |
| Underdog Fantasy API | Live prop lines: both sides' odds + payout multipliers (incl. batter walks) | Free (no auth) | `underdog_props` |
| BettingPros API (`/v3/props`) | Historical prop lines + outcomes; per-book filter so we get the actual Underdog book back to 2024 | Free (no auth) | `bettingpros_props` (book_id=36 for Underdog) |

## Database

- **Provider:** Supabase Postgres
- **Plan:** Pro ($25/mo, planned downgrade to free after pre-aggregation phase)
- **Current size:** ~2 GB (well under 8 GB Pro limit)
- **Connection:** via `DATABASE_URL` env var, `psycopg2` connection pool in `db/db.py`

Key tables:
- `games` — every MLB game 2015–present (~26K rows)
- `mlb_batting_game` — every batter × game outcome (H, R, RBI, runs, etc.) (~620K rows)
- `mlb_pitching_game` — every pitcher × game outcome (~220K rows)
- `mlb_pitches` — pitch-by-pitch Statcast data (~7.9M rows, biggest table)
- `mlb_game_info` — weather, umpire, probable pitchers
- `players` — 4,160 MLB players with `bats` and `throws` populated
- `underdog_props` — Underdog prop snapshots, multiple per day per line (forward capture)
- `bettingpros_props` — historical props for Underdog (book_id=36) + Consensus + Novig, back to Apr 2024
- `predictions` — model predictions log (currently inactive; was used by retired totals models)

## Daily cron schedule

All times in UTC. ET = UTC-4 during summer.

| Cron | Workflow | What it does |
|------|----------|--------------|
| `0 10 * * *` (6 AM ET) | `daily_pipeline.yml` | Box scores + Statcast (last 3 days) + handedness backfill for new players |
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
for f in db/migrate_*.sql; do psql $DATABASE_URL -f "$f"; done

# 4. Bootstrap historical data (slow: hours)
python -m scrapers.mlb.run_all --start 2015 --end 2026

# 5. Pull Statcast (very slow: ~1 day, rate-limited)
python -m scrapers.mlb.statcast --start 2015 --end 2026

# 6. Backfill player handedness (2 min)
python -m scrapers.mlb.backfill_player_handedness

# 7. Backfill historical runs scored (~5-7 hours; re-fetches each box score)
python -m scrapers.mlb.backfill_batter_runs

# 8. Backfill historical prop odds (BettingPros, ~2-3 hours per season)
python -m scrapers.props.bettingpros --start 2024-04-01 --end 2026-12-31

# 9. Verify
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
python -m scrapers.props.underdog --csv-also

# BettingPros: pull today, a single date, or a backfill range
python -m scrapers.props.bettingpros                       # today, all default books
python -m scrapers.props.bettingpros --date 2026-06-21
python -m scrapers.props.bettingpros --start 2024-04-01 --end 2024-10-31 --books 36
```

## Inspecting feature modules

Each module has a built-in smoke test that prints coverage stats:

```bash
python -m models.mlb.batter_arsenal_features  --start 2025-06-15 --end 2025-06-17
python -m models.mlb.pitcher_arsenal_features --start 2025-06-15 --end 2025-06-17
python -m models.mlb.context_features         --start 2025-06-15 --end 2025-06-17
python -m models.mlb.matchup_features                                          # synthetic test row
python -m models.mlb.hitter_prop_dataset      --start 2025-06-15 --end 2025-06-17
```

## v1 design decisions (locked)

**Targets** — three binary classifiers sharing the same feature stack:
- **Primary: HRR > 1.5** — most common Underdog hitter prop, near-even pricing (~50% base rate)
- **Companion: TB > 1.5** — broader market, balanced pricing, biggest natural partner since
  XBH events drive both HRR and TB
- **Companion: RBI > 0.5** — +197 avg over odds gives ROI headroom; directly stresses our
  lineup-OBP-in-front feature

Companions exist because HRR-with-this-line-only didn't launch on BettingPros until 2026 →
HRR ROI backtest is restricted to ~70 days. TB and RBI have full 2024-2025 Underdog odds, so
they validate the same feature stack against more sample.

**Features** — 4 modules, 112 features total. No leakage: every rolling window is half-open
`[as_of − window, as_of)` via pandas `closed='left'`, and every SQL filter uses strict
`game_date < as_of_date`.

| Module | Count | Highlights |
|---|---|---|
| `batter_arsenal_features` | 36 | Per-pitch-type xwOBA + whiff vs 9 individual types (FF, SI, FC, SL, ST, SV, CU, KC, CH, FS, FO) with bucket fallbacks (FB / BR / OS); plate discipline; vs RHP / vs LHP; recent box-score form |
| `pitcher_arsenal_features` | 45 | **Pitch mix conditioned on batter handedness** (key insight: RHP throws 25% sliders to RHB but 5% to LHB — substitutes changeups); per-pitch results allowed; fastball velo + 30D vs 180D trend; season K / BB / HR-per-9 rates; IP per start (workload) |
| `context_features` | 11 | **Rolling 365-day park factors** (Camden 2022/2025 wall change visible in the numbers); weather; lineup OBP-in-front (RBI opportunity); lineup SLG-behind (run-scoring opportunity); home/away |
| `matchup_features` | 6 | Weighted expected xwOBA = Σ over pitch types: `pit_pct(pt \| batter_hand) × bat_xwoba_vs(pt)`; weighted expected whiff; platoon advantage; expected PAs (total + vs starter + bullpen share) |

**Switch hitters** are mapped to the opposite of the pitcher's throwing hand throughout
(canonical strategy).

**Models** — train both side-by-side, choose by validation calibration + ROI:
1. L1-regularized logistic regression (baseline, auto-sparsifies)
2. LightGBM (captures nonlinear interactions)

**Cross-validation:**
- Expanding by season for hyperparameter selection (train ≤2023 / val 2024 across folds)
- Monthly expanding folds across 2025 for OOS evaluation (matches "retrain monthly in
  production" cadence)

**EV gate** — NOT hardcoded. Backtest sweeps thresholds (0%, 0.5%, 1%, 2%, 3%, 5%); pick the
one with the best risk-adjusted return (Sharpe-like: mean ROI / std ROI). Expectation is
that frequent low-edge bets compound better than rare high-edge ones.

## What's next

1. **Training script** (`models/mlb/hitter_prop_model.py`): trains all 3 targets through
   expanding-season CV, saves bundles for each.
2. **Backtest script** (`models/mlb/backtest.py`): joins Underdog odds via `bettingpros_props`
   (book_id=36), computes ROI per EV-threshold per market, generates calibration plots.
3. **Live prediction** (when v1 ROI clears its Sharpe threshold): daily script that pulls
   current Underdog lines + computes EV against model P(over), surfaces flagged bets.
4. **Underdog-only markets** (v2): once we have ~60 days of forward Underdog capture, train
   a batter walks model (Underdog offers, BettingPros doesn't) using the same feature stack.

## Lessons from the totals work (preserved in `archive/`)

- Backtest profitability does not guarantee live profitability
- Aggregate calibration can hide systematic bet-selection bias
- Decomposing performance by direction surfaces failure modes aggregate metrics miss
- The "model echoes market consensus" failure mode is real and costs money
- Live test data is the only valid validation
