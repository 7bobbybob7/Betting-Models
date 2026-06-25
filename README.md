# MLB Player Prop Betting Platform

A data + modeling pipeline for **MLB player prop betting** (Hits + Runs + RBIs,
Total Bases, Hits, etc.) targeting Underdog Fantasy's adjusted-odds market.

Pulls Statcast pitch data, MLB box scores, and prop lines from multiple books
into Postgres on a daily schedule. Both strategies below ultimately produce the
same thing — a **`P_true` estimate** that, when it beats the price Underdog
offers, makes a **positive-EV bet**.

## Two strategies for finding +EV

The platform pursues two complementary paths to a profitable edge. They differ
only in *where the "true" probability comes from*:

1. **Own predictive model** (`models/mlb/`) — build `P_true` ourselves from
   matchup-aware features (batter vs pitch type, pitcher arsenal, lineup
   context, swing tendencies). Hardest path: it requires out-forecasting the
   market. In active development + validation.

2. **+EV line-shopping / sharp-vs-soft** (`models/mlb/line_shopping.py`) — let a
   **sharp book be the model**. Take Novig's de-vigged exchange price as fair
   value and bet Underdog (a softer DFS book) wherever its line lags. This is the
   textbook +EV-betting approach used by tools like OddsJam/Outlier/Unabated, and
   our backtest shows **genuine potential edge**: ROI rises monotonically with the
   size of the Novig-vs-Underdog discrepancy — the mechanical signature of a real
   edge, concentrated in large gaps and lower-volume markets. Now being validated
   forward with live dual-book capture (see [Sharp-vs-soft line shopping](#sharp-vs-soft-line-shopping-ev)).

Both feed one betting system; the model and the sharp reference can also be
combined (e.g. blended fair value, or each used to confirm the other).

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

**Data layer**

| Component | Status |
|-----------|--------|
| MLB game/box score scraper | ✅ Running daily (captures runs scored) |
| Statcast pitch data scraper | ✅ Running daily (caught up through current) |
| Player handedness + full names | ✅ Backfilled (4,160 players); cron refresh for rookies |
| Underdog props capture (soft book) | ✅ Running 3x weekday / 4x weekend |
| Novig exchange capture (sharp book, direct GraphQL) | ✅ Running same ticks as Underdog |
| BettingPros historical props (Underdog 36 + Novig 60) | ✅ Backfilled 2024–2026 |
| `mlb_batting_game.runs` column | ✅ Schema + historical backfill complete (100%) |

**Strategy 1 — own predictive model**

| Component | Status |
|-----------|--------|
| Feature stack (batter/pitcher/context/matchup) | ✅ 112 features, 4 modules, leak-safe |
| Dataset assembler | ✅ `models/mlb/hitter_prop_dataset.py` (+ parquet cache) |
| Training (LR-L1 + XGBoost, isotonic calibration, expanding-season CV) | ✅ `models/mlb/hitter_prop_model.py` |
| Backtest (EV threshold sweep vs Underdog odds) | ✅ `models/mlb/backtest.py` |
| Model tuning + live deployment | 🚧 In validation |

**Strategy 2 — +EV line-shopping (sharp-vs-soft)**

| Component | Status |
|-----------|--------|
| Novig vs Underdog discrepancy backtest | ✅ `models/mlb/line_shopping.py` — shows potential edge |
| Live dual-book capture (Novig + Underdog, aligned ticks) | ✅ Running |
| Forward paper-trade harness (persistence + OOS ROI) | 🚧 Next |
| Live betting | ⏸️ Pending forward validation |

## Architecture

```
   MLB Stats API ───►  scrapers/mlb/games.py + boxscores.py  ──► games, mlb_batting_game,
                                                                  mlb_pitching_game, players
   Baseball Savant ──► scrapers/mlb/statcast*.py             ──► mlb_pitches
   Underdog API   ───► scrapers/props/underdog.py            ──► underdog_props   (soft book, live)
   Novig GraphQL  ───► scrapers/props/novig.py               ──► novig_snapshots  (sharp book, live)
   BettingPros API ──► scrapers/props/bettingpros.py         ──► bettingpros_props (Underdog 36 + Novig 60, history)
                                                                          │
                                                                          ▼
                                                  Postgres on Supabase
                                                          │
                  ┌───────────────────────────────────────┴───────────────────────────────┐
                  ▼                                                                          ▼
   STRATEGY 1 — own model                                            STRATEGY 2 — +EV line-shopping
   ────────────────────────                                          ──────────────────────────────
   batter_/pitcher_/context_/matchup_features.py                     models/mlb/line_shopping.py
        │  (112 leak-safe features)                                       │  Novig de-vigged fair price
        ▼                                                                 ▼  vs Underdog offered odds
   hitter_prop_dataset.py  ──►  hitter_prop_model.py  ──► backtest.py     bet where the soft book lags
   (assemble + label)           (LR-L1 + XGBoost,        (EV sweep        (ROI rises with discrepancy)
                                 isotonic calibration)    vs Underdog)          │
        │                                                                       ▼
        └────────────────────────────┬──────────────────────────────►  forward paper-trade harness
                                      ▼                                   (persistence + OOS ROI; planned)
                           one betting system / EV gate
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
│   │   ├── backfill_player_fullname.py    — Fetch canonical full names (cross-book matching)
│   │   ├── backfill_batter_runs.py        — Backfill runs scored (re-fetches box scores)
│   │   └── run_all.py                     — Bootstrap helper for new clones
│   └── props/
│       ├── underdog.py       — Underdog Fantasy MLB props capture (soft book; incl. batter walks)
│       ├── novig.py          — Novig exchange capture (sharp book; direct GraphQL, de-vigged prices)
│       └── bettingpros.py    — BettingPros historical/daily props (Underdog=36, Novig=60)
│
├── models/
│   └── mlb/
│       │  # Strategy 1 — own predictive model
│       ├── batter_arsenal_features.py    — 36 batter features (per pitch type + buckets)
│       ├── pitcher_arsenal_features.py   — 45 pitcher features (mix conditioned on batter hand)
│       ├── context_features.py           — 11 context features (rolling 365D park factors,
│       │                                   weather, lineup OBP-in-front, SLG-behind)
│       ├── matchup_features.py           — 6 cross-features (handedness-aware weighted xwOBA)
│       ├── hitter_prop_dataset.py        — Assembler: 112 features + HRR/TB/RBI labels
│       ├── hitter_prop_model.py          — Train LR-L1 + XGBoost, isotonic calibration, CV
│       ├── backtest.py                   — Model EV sweep vs Underdog odds
│       ├── cache/                        — Cached feature parquets (train / backtest splits)
│       │  # Strategy 2 — +EV line-shopping
│       └── line_shopping.py              — Novig-vs-Underdog discrepancy backtest (sharp-vs-soft)
│                                            (totals-era models live in archive/models/mlb/)
│
├── scripts/
│   └── daily_refresh.py      — Morning cron entry point
│
├── .github/workflows/
│   ├── daily_pipeline.yml    — Morning: games + box scores + Statcast + handedness refresh
│   └── underdog_capture.yml  — 3-4x daily: snapshots BOTH Underdog + Novig prop lines
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
| Underdog Fantasy API | Live prop lines (soft book): both sides' odds + payout multipliers (incl. batter walks) | Free (no auth) | `underdog_props` |
| Novig GraphQL API (`/v1/graphql`) | Live exchange prices (sharp book): de-vigged probabilities + order book + volume | Free (no auth) | `novig_snapshots` |
| BettingPros API (`/v3/props`) | Historical prop lines + outcomes; per-book filter (Underdog=36, Novig=60) back to 2024 | Free (no auth) | `bettingpros_props` |

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
- `players` — 4,160 MLB players with `bats`, `throws`, and `full_name` populated
- `underdog_props` — Underdog (soft book) prop snapshots, multiple per day per line (forward capture)
- `novig_snapshots` — Novig (sharp book) exchange snapshots: de-vigged last/available prices + volume, intraday
- `bettingpros_props` — historical props for Underdog (36) + Consensus (0) + Novig (60), back to Apr 2024
- `predictions` — model predictions log (currently inactive; was used by retired totals models)

## Daily cron schedule

All times in UTC. ET = UTC-4 during summer.

Each `underdog_capture.yml` tick now snapshots **both** Underdog and Novig at the
same timestamp, so discrepancies can be compared time-aligned.

| Cron | Workflow | What it does |
|------|----------|--------------|
| `0 10 * * *` (6 AM ET) | `daily_pipeline.yml` | Box scores + Statcast (last 3 days) + handedness backfill for new players |
| `0 15 * * 1-5` (11 AM ET, weekday) | `underdog_capture.yml` | Underdog + Novig — opening-ish lines |
| `0 19 * * 1-5` (3 PM ET, weekday) | `underdog_capture.yml` | Underdog + Novig — mid-day movement |
| `0 22 * * 1-5` (6 PM ET, weekday) | `underdog_capture.yml` | Underdog + Novig — pre-night-games |
| `0 14 * * 0,6` (10 AM ET, weekend) | `underdog_capture.yml` | Underdog + Novig — opening lines |
| `30 16 * * 0,6` (12:30 PM ET, weekend) | `underdog_capture.yml` | Underdog + Novig — pre-day-games |
| `0 20 * * 0,6` (4 PM ET, weekend) | `underdog_capture.yml` | Underdog + Novig — mid-afternoon |
| `0 22 * * 0,6` (6 PM ET, weekend) | `underdog_capture.yml` | Underdog + Novig — pre-night-games |

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

# Snapshot current Underdog MLB props (soft book; writes to DB + optionally CSV)
python -m scrapers.props.underdog
python -m scrapers.props.underdog --csv-also

# Snapshot current Novig exchange prices (sharp book; direct GraphQL)
python -m scrapers.props.novig
python -m scrapers.props.novig --dry-run   # print summary, no DB write

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
2. XGBoost (captures nonlinear interactions)

Both get an **isotonic calibration** layer fit on a held-out latest season, so the
probabilities fed to the EV gate are honest (no high-confidence overconfidence).

**Cross-validation:**
- Expanding by season for hyperparameter selection (train ≤2023 / val 2024 across folds)
- Monthly expanding folds across 2025 for OOS evaluation (matches "retrain monthly in
  production" cadence)

**EV gate** — NOT hardcoded. Backtest sweeps thresholds (0%, 0.5%, 1%, 2%, 3%, 5%); pick the
one with the best risk-adjusted return (Sharpe-like: mean ROI / std ROI). Expectation is
that frequent low-edge bets compound better than rare high-edge ones.

## Sharp-vs-soft line shopping (+EV)

The second strategy needs no predictive model of our own — it lets the **sharp
market be the model**:

1. Take **Novig**'s exchange price as fair value. Novig is a no-vig exchange, so its
   two sides already sum to ~1.0 — a clean `P_true` with no margin to strip.
2. Compare to **Underdog**'s offered odds on the same prop. Underdog is a
   recreational DFS book; its lines are structurally slower/softer.
3. When Novig's fair probability implies Underdog's price is off by more than
   Underdog's built-in margin, that side is **+EV** — bet it.

**Backtest signal (encouraging).** Across ~84k overlapping props (2024–2026), ROI
rises **monotonically** with the size of the Novig-vs-Underdog discrepancy:
small gaps lose to the vig, but large gaps (≥10%) turn positive, and the edge is
strongest in lower-volume markets books price less carefully (HR, steals). A
monotonic edge-vs-discrepancy curve is the mechanical fingerprint of a *real*
edge rather than noise.

```bash
# Backtest: bet Underdog where Novig implies value, swept by edge threshold
python -m models.mlb.line_shopping                       # all hitter markets
python -m models.mlb.line_shopping --markets 403,293,289 # HRR / TB / RBI only
```

**Validating it's *tradeable*, not just a snapshot artifact.** A backtested
discrepancy is only money if it survives until you can actually bet. The live
dual-book capture (`novig_snapshots` + `underdog_props`, same cron ticks) lets us
measure forward:
- **Persistence** — when a discrepancy appears, is it still there minutes later?
- **Direction of truth** — does Underdog converge toward Novig (confirming Novig
  is the sharper price)?
- **Liquidity** — is there real size behind the Novig price (`volume`, order book)?
- **Out-of-sample ROI** — flag bets live, settle against box scores, tally.

**Honest caveats:** Novig player-prop liquidity is thin pre-game and thickens near
first pitch, so the tradeable edge may concentrate in fewer markets than the
backtest suggests; and soft books limit/ban consistent +EV winners — the practical
risk, not the math.

## What's next

1. **Paper-trade harness** — log live-flagged +EV bets (both strategies) with the
   exact odds available at capture time, settle against box scores, and report
   forward OOS ROI segmented by edge size, liquidity, and discrepancy persistence.
   The decisive test of *tradeable* edge before any real money.
2. **Model tuning + live prediction** — continue developing the predictive model;
   when its backtested ROI clears its risk-adjusted threshold, surface daily flagged
   bets alongside the line-shopping signal.
3. **Combine the two signals** — blended fair value (Novig + model + consensus), or
   use each to confirm the other.
4. **Underdog-only markets** (v2): once we have ~60 days of forward capture, model
   the batter-walks market (Underdog offers it, sharp books barely price it).

## Lessons from the totals work (preserved in `archive/`)

- Backtest profitability does not guarantee live profitability
- Aggregate calibration can hide systematic bet-selection bias
- Decomposing performance by direction surfaces failure modes aggregate metrics miss
- The "model echoes market consensus" failure mode is real and costs money
- Live test data is the only valid validation
