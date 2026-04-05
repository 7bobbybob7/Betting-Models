# Multi-Sport Betting Platform

A predictive betting platform that generates fair-value probabilities across game outcomes, totals, and player prop markets. Built on 7.6M+ Statcast pitches, 25K+ MLB games, and 621K+ odds records. Tracks closing line value (CLV) as the primary signal metric.

## Results

| Model | Market | Key Metric |
|-------|--------|-----------|
| Team-Level (LogReg) | Moneyline | +0.42% CLV, positive in 6/6 seasons |
| Pitcher K (Poisson) | Player Props | 1.83 MAE, calibration within 2% |
| **Totals** | **Over/Under** | **+9.0% ROI, 57.5% win rate (487 bets, 6 seasons)** |

## Architecture

```
DATA LAYER                 MODEL LAYER                 OPERATIONS
──────────                 ───────────                 ──────────
MLB Stats API ──┐          ELO (starter-adjusted)      GitHub Actions cron
pybaseball ─────┤          LogReg + XGBoost (ML)       Daily prediction pipeline
Statcast ───────┤──► PG    Poisson (pitcher K's)       SBR odds scraper
ESPN API ───────┤          Lineup-specific features    FastAPI backend
SBR scraper ────┘          Totals model                Streamlit dashboard
```

## Data

| Table | Rows | Description |
|-------|------|-------------|
| MLB Games | 25,800+ | 2015-2026 schedules and scores |
| Statcast Pitches | 7.6M+ | Pitch-level data (velo, spin, movement, outcomes) |
| Batting Stats | 600K+ | Per-player per-game batting lines |
| Pitching Stats | 221K+ | Per-player per-game pitching lines |
| Odds | 621K+ | Multi-book closing lines (2015-2025, 27+ sportsbooks) |
| CBB Games | 53K+ | College basketball (migrated from prior project) |

## Models

### Phase 1: Team-Level Baseline
- Custom MLB ELO with starter adjustments (pitcher quality shifts team ELO per game)
- Bullpen features (7-day rolling ERA/WHIP, 3-day fatigue)
- 113 features: pitcher, batting, bullpen, ELO, park factor, weather
- Positive CLV in 6 consecutive test seasons (2019-2024)

### Phase 2: Player Prop Models
- **Pitcher K model** (Poisson regression) — Statcast features: whiff rate, chase rate, pitch mix, velocity
- Pitcher IP model — coupled with K model for projected innings
- Hitter prop models — hits, total bases, HR using Statcast batted ball data

### Phase 3: Player-Level Game Model
- Lineup-specific features replace team averages (per-player wOBA, K%, exit velo, xBA)
- Batting order weights — top of order weighted higher
- Totals model — **strongest signal found**: +9.0% ROI on conservative strategy

### Totals Strategy (Validated)
Conservative filters with causal mechanisms only:
- Edge ≥ 1.5 runs vs market total
- May through September (feature quality)
- Regular season only
- Hitter-friendly parks (park factor ≥ 1.0)

Backtested across 6 seasons: 487 bets, 57.5% win rate, +9.0% ROI, 8.7% max drawdown, profitable 5/6 years.

## Stack

- **Database:** PostgreSQL (Supabase)
- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost, statsmodels
- **Data:** MLB Stats API, pybaseball (Statcast), ESPN API, SportsBookReview
- **API:** FastAPI (7 endpoints)
- **Dashboard:** Streamlit (6 tabs)
- **Automation:** GitHub Actions (daily cron)

## Project Structure

```
db/                  Schema and database connection (SimpleConnectionPool)
scrapers/
  mlb/               Games, box scores, Statcast scrapers
  odds/              ESPN historical, Arnav dataset loader, SBR daily scraper
models/
  mlb/               Feature engineering, ELO, training, evaluation
    features.py      113-column feature pipeline
    elo.py           Starter-adjusted ELO system
    train.py         LogReg + XGBoost training
    k_model.py       Pitcher K Poisson model
    hitter_model.py  Hitter prop models
    totals_model.py  Run totals prediction
    lineup_features.py  Per-player lineup aggregation
    statcast_features.py  Pitch-level feature engineering
api/                 FastAPI backend
dashboard/           Streamlit frontend
scripts/             Daily pipeline, data refresh, CBB migration
.github/workflows/   GitHub Actions daily cron
```

## Setup

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Supabase connection string
4. Run `db/schema.sql` in Supabase SQL Editor
5. Run scrapers to load data:
   ```bash
   python -m scrapers.mlb.run_all --start 2015 --end 2026
   python -m scrapers.odds.espn_odds --sport mlb --start 2015 --end 2024
   python -m models.mlb.features --start 2016 --end 2025
   python -m models.mlb.train --save-models
   ```
6. Run the daily pipeline:
   ```bash
   python scripts/daily_pipeline.py
   ```
7. Start the dashboard:
   ```bash
   uvicorn api.app:app --port 8000 &
   streamlit run dashboard/app.py
   ```

## Daily Pipeline

Runs automatically via GitHub Actions at 6:30 PM ET:
1. Pull new game results and box scores
2. Scrape today's odds from SportsBookReview
3. Build features and generate predictions
4. Flag totals bets meeting strategy criteria
5. Log everything to the database

## Sports Roadmap

- **MLB** — Live (Phases 1-4 complete)
- **WNBA** — Next (season starts May 2026, schema ready)
- **CBB** — Data migrated, model from prior project
- **NHL / NFL** — Schema stubbed, build when seasons approach
