# MLB Player Prop Betting Platform

A data + modeling pipeline for **MLB player prop betting** (Hits + Runs + RBIs,
Total Bases, Hits, etc.) targeting Underdog Fantasy's adjusted-odds market.

Pulls Statcast pitch data, MLB box scores, and prop lines from multiple books
into Postgres on a daily schedule. All three legs below ultimately produce the
same thing — a **`P_true` estimate** that, when it beats the price Underdog
offers, makes a **positive-EV bet**.

## Three legs for finding +EV

The platform pursues three complementary paths to a profitable edge. They differ
only in *where the "true" probability comes from*:

1. **Own predictive model** (`models/mlb/hitter_prop_model.py`) — build `P_true`
   ourselves from matchup-aware features by training on game *outcomes*. Hardest
   path: it requires out-forecasting the market. Currently feature-limited
   (AUC ~0.55 vs the market's ~0.57); in active development.

2. **+EV line-shopping / sharp-vs-soft** (`models/mlb/line_shopping.py`) — let a
   **sharp book be the model**. Take Novig's de-vigged exchange price as fair
   value and bet Underdog (a softer DFS book) wherever its line lags. The textbook
   +EV approach (à la OddsJam/Outlier/Unabated); the backtest shows **genuine
   potential edge** — ROI rises monotonically with the size of the Novig-vs-Underdog
   discrepancy, the mechanical signature of a real edge. The proven near-term edge;
   now being validated forward (see [Sharp-vs-soft line shopping](#sharp-vs-soft-line-shopping-ev)).

3. **Distillation model** (`models/mlb/distill_model.py`) — train our *own* model
   on Novig's price as a **soft target** (learn to reproduce the sharp line, not
   noisy outcomes). The goal: an **owned, book-independent asset** that inherits
   Novig's sharpness, so it survives if Novig vanishes and can be improved over time.
   Sample-efficient (1 season of sharp labels ≈ 6 seasons of outcomes) but currently
   capped at the same ~0.55 feature ceiling as leg 1 — closing the gap to Novig is a
   *feature* problem, not a training-target one.

All three feed one betting system; the legs can also be combined (blended fair
value, or each used to confirm the others). Leg 2 is the edge that clears the vig
today; legs 1 & 3 are the long-term owned models, gated on richer features.

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

**Leg 1 — own predictive model (trained on outcomes)**

| Component | Status |
|-----------|--------|
| Feature stack v1 (batter/pitcher/context/matchup) | ✅ 112 features, 4 modules, leak-safe |
| Statcast pitch-extras backfill (spray, bat tracking, catcher, arm angle) 2019–2026 | ✅ `scrapers/mlb/backfill_pitch_extras.py` (5M pitches) |
| v6 feature stack: + spray/pull, bat-tracking, framing, luck-gap (full coverage) | ✅ 123 features; accepted via batch gates (see `leg1_*_gate.py`) |
| Blend beats market line (residual gate, both time directions) | ✅ TB +0.009/+0.017, HR +0.004/+0.014 AUC |
| Standalone vs vig (90d walk-forward blend, side-balanced) | ✅ Backtest positive both years at ev>2% — pending live confirmation |
| Filter/veto role (improves Leg 2 bet selection) | ✅ Validated both years, 3 model versions |
| Forward tracker (daily scoring, `v3_signals`) | ✅ `models/mlb/v3_tracker.py` on daily cron |
| Rejected honestly: embeddings (×2), swing deltas, swing-path batch | 📕 Gates in repo; BvP chemistry = noise |

**Leg 2 — +EV line-shopping (sharp-vs-soft)**

| Component | Status |
|-----------|--------|
| Novig vs Underdog discrepancy backtest | ✅ `models/mlb/line_shopping.py` — shows potential edge |
| Live dual-book capture (Novig + Underdog, aligned ticks) | ✅ Running |
| Forward paper-trade harness (pre-game filter, settle, ROI) | ✅ `models/mlb/paper_trade.py` — logging live |
| Live betting | ⏸️ Pending forward validation (~weeks of settled bets) |

**Leg 3 — distillation model (trained on Novig's price)**

| Component | Status |
|-----------|--------|
| Distill features → Novig de-vigged price | ✅ `models/mlb/distill_model.py` |
| Walk-forward CV (monthly, embargoed) | ✅ stable ~0.55 AUC, reproduces Novig (MAE ~0.027) |
| Close feature gap so distilled model clears the vig | 🚧 Feature-limited (same ceiling as leg 1) |

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
        ┌─────────────────────────────────────┬──────────────────────────────────┐
        ▼                                      ▼                                    ▼
 LEG 1 — own model (outcomes)        LEG 2 — line-shopping (+EV)        LEG 3 — distillation
 ───────────────────────────        ──────────────────────────        ────────────────────────
 features → predict outcome          Novig fair price vs Underdog       features → predict Novig price
 hitter_prop_dataset.py              line_shopping.py                   distill_model.py
 hitter_prop_model.py                bet where soft book lags           (owned, Novig-independent;
 backtest.py                         (ROI rises w/ discrepancy)          sample-efficient)
 (AUC ~0.55, feature-capped)              │                             (AUC ~0.55, feature-capped)
        │                                 ▼                                       │
        │                     paper_trade.py  (pre-game filter,                   │
        │                      log +EV → settle vs box scores → ROI)              │
        └─────────────────────────────────┴──────────────────────────────────────┘
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
│       ├── feature_sets.py   — SINGLE SOURCE OF TRUTH: accepted feature lists
│       │                       (ADV_FEATS/BATCH1/LUCK), caches, params, build_luck().
│       │                       Research batches that pass their gate get PROMOTED here.
│       ├── features/         — feature builders (batter/pitcher arsenal, context, matchup,
│       │                       game-context, advanced-profile: spray/bat-tracking/framing)
│       ├── hitter/           — Leg 1 + Leg 3 production: hitter_prop_dataset (assembler),
│       │                       hitter_prop_model (base bundles), train_v3 (current v6 bundle),
│       │                       v3_tracker (daily forward scoring -> v3_signals, cron),
│       │                       backtest (odds attach + EV), distill_model
│       ├── trading/          — Leg 2 execution: line_shopping (sharp-vs-soft backtest),
│       │                       paper_trade (log pre-game +EV / settle / report, cron)
│       ├── pitcher/          — pitcher-prop lane (k_gate: market 285 — rejected; flagship
│       │                       K market is sharp)
│       ├── research/         — one-shot gates/audits/backtests, frozen once run
│       │                       (leg1_* attacks, batch gates, embeddings, audits, sims)
│       ├── cache/            — cached feature parquets (train/backtest splits, adv profiles)
│       └── saved/            — model bundles (gitignored)
│                                (totals-era models live in archive/models/mlb/)
│
├── scripts/
│   └── daily_refresh.py      — Morning cron entry point
│
├── .github/workflows/
│   ├── daily_pipeline.yml    — Morning: box scores + Statcast + handedness + paper-trade settle
│   └── underdog_capture.yml  — 5-6x daily: snapshot Underdog + Novig + log paper-trade +EV
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
- `paper_bets` — forward-logged pre-game +EV line-shopping bets, settled vs outcomes (the tradeable-edge test)
- `predictions` — model predictions log (currently inactive; was used by retired totals models)

## Daily cron schedule

All times in UTC. ET = UTC-4 during summer. **Note:** GitHub Actions schedules run
LATE under load (observed ~1-2h delay), so nominal times are set earlier than the
target and ticks are denser for redundancy — the goal is that every game gets at
least one clean PRE-GAME snapshot. Correctness doesn't depend on exact timing: the
paper-trade layer enforces `scheduled_start > capture_time` regardless of when a run lands.

Each `underdog_capture.yml` tick snapshots **both** Underdog and Novig (aligned
within seconds) and then logs pre-game +EV paper bets from those snapshots.

| Cron | Workflow | What it does |
|------|----------|--------------|
| `0 10 * * *` (6 AM ET) | `daily_pipeline.yml` | Box scores + Statcast + handedness backfill + **settle paper bets** |
| `0 12,14,17,20,22 * * 1-5` (weekday) | `underdog_capture.yml` | Snapshot Underdog + Novig, log pre-game +EV |
| `0 13,15,17,19,21,23 * * 0,6` (weekend) | `underdog_capture.yml` | Snapshot Underdog + Novig, log pre-game +EV |

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

## Sharp-vs-soft line shopping (+EV) — Leg 2

Leg 2 needs no predictive model of our own — it lets the **sharp
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

**Validating it's *tradeable* — the paper-trade harness (`paper_trade.py`).** A
backtested discrepancy is only money if it survives until you can actually bet, and
on data the strategy never saw. The harness runs forward on the live dual-book
capture and is the decisive test:

```bash
python -m models.mlb.paper_trade log      # flag + log pre-game +EV (runs each cron tick)
python -m models.mlb.paper_trade settle   # fill outcomes from box scores (runs each morning)
python -m models.mlb.paper_trade report   # realized ROI by edge bucket / market / per-prop
```

It bakes in the **pre-game filter** (`scheduled_start > capture_time`) so live/in-progress
prices can never contaminate it, logs every +EV opportunity with the exact Underdog odds
at flag time, settles against `mlb_batting_game`, and reports realized ROI segmented by
edge size, market, and a one-bet-per-prop view. Accumulating now; a real read needs
hundreds of settled bets (weeks) before the law of large numbers applies.

**Honest caveats:** Novig player-prop liquidity is thin pre-game and thickens near
first pitch, so the tradeable edge may concentrate in fewer markets than the
backtest suggests; and soft books limit/ban consistent +EV winners — the practical
risk, not the math.

## Distillation model (+EV) — Leg 3

Leg 3 builds an **owned, book-independent** version of the sharp price. Instead of
training on noisy game outcomes (leg 1), it trains on **Novig's de-vigged price as a
soft target** — learning to reproduce the sharp line from our features:

```bash
python -m models.mlb.distill_model --target all          # fit + compare vs outcome model
python -m models.mlb.distill_model --mode cv --target all # monthly walk-forward CV (embargoed)
```

Why it matters: if Novig disappears (API closes, book shuts down) or quietly drifts,
the distilled sharpness is already in our weights — and it covers props Novig doesn't
price. Walk-forward CV shows it's **stable (~0.55 AUC) and sample-efficient** (≈1 season
of sharp labels matches 6 seasons of outcomes), reproducing Novig with ~0.027 MAE.

**The key finding:** distillation lands at the *same* ~0.55 ceiling as the outcome
model, short of Novig's ~0.57. Changing the training target didn't help — which proves
the bottleneck is **features, not the loss**. Closing that ~0.02 gap (the difference
between losing to the vig and beating it) is a richer-features problem: pitcher
pitch-shape + batter swing embeddings. Until then, leg 3 is the owned fallback/monitor,
and leg 2 (live Novig directly, zero approximation error) is what actually clears the vig.

## What's next

1. **Forward-validate the standalone candidate** (running automatically) — "v6 +
   90-day walk-forward blend, ev>4%" passed the vig in backtest both years;
   `v3_tracker report` accumulates the live verdict daily. No real money until it
   confirms (~4 weeks).
2. **Pitcher-K market gate** — point the Poisson K model (1.83 MAE, full
   distribution → handles varying lines natively) at BettingPros market 285
   (3 seasons of Underdog K props, 3.4K Novig). The K-suppression mechanism is the
   market's most-confirmed soft spot.
3. **Venue portfolio (Leg 2)** — DK RBI singles (+8.5%, positive 3/3 months, n=243)
   to forward tracking; Fliff (book 39) sweep when backfill lands; shade-corrected
   fair values (books load vig onto overs: measured 0.5–2.5pts, 13/13 cells) into
   paper-trade EVs.
4. **Let data compound** — bat-tracking coverage grows monthly (v6 proved coverage
   is the multiplier: same features at 6× history doubled the edge); monthly
   retrains harvest it with no new code.

## Lessons from the totals work (preserved in `archive/`)

- Backtest profitability does not guarantee live profitability
- Aggregate calibration can hide systematic bet-selection bias
- Decomposing performance by direction surfaces failure modes aggregate metrics miss
- The "model echoes market consensus" failure mode is real and costs money
- Live test data is the only valid validation
