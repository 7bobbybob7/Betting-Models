# Multi-Sport Betting Platform

A predictive betting platform that generates fair-value probabilities across game outcomes, totals, and player prop markets. Built on 7.6M+ Statcast pitches, 25K+ MLB games, and 650K+ odds records across 27+ sportsbooks. Tracks decomposed CLV (closing line value) to separate model skill from line shopping value.

## Results

| Model | Market | Key Metric | Notes |
|-------|--------|-----------|-------|
| Team-Level (LogReg) | Moneyline | +0.42% CLV, positive 6/6 seasons | Real signal, unprofitable after vig |
| Pitcher K (Poisson) | Player Props | 1.83 MAE, calibration within 2% | No prop lines yet for CLV |
| **Totals (Regression)** | **Over/Under** | **51.3% win, +0.3% ROI w/ line shopping** | **Live testing 2026** |
| **Totals (Classifier)** | **Over/Under** | **51.1% win, +1.7% ROI w/ line shopping** | **Comparison model 2026** |

**Critical finding:** Without line shopping across 6+ sportsbooks, both strategies lose money. The model provides ~51.3% directional accuracy; profitability depends entirely on getting best-available odds.

## How It Works

### 2-Step Betting Framework

1. **Informational edge:** Multiplicative de-vig on median book odds gives the market consensus P(over). Model generates its own P(over). The gap is the info edge.
2. **Execution edge:** If info edge exceeds threshold, line shop across all books for the best price. Only bet if model probability beats the breakeven at the best available odds.

### CLV Decomposition

Every bet is decomposed into two sources of value:
- **Model CLV** — Did the market consensus move toward our position? (informational edge)
- **Execution CLV** — Was our bet book softer than the consensus? (line shopping value)

This answers the key question: is profit coming from the model being right, or from catching soft lines?

## Architecture

```
DATA LAYER                 MODEL LAYER                 OPERATIONS
----------                 -----------                 ----------
MLB Stats API ---+          ELO (starter-adjusted)      GitHub Actions (3 daily crons)
pybaseball ------+          LogReg (moneyline)          Daily prediction pipeline
Statcast --------+---> PG   LinearReg (totals)          SBR odds scraper (6 books)
ESPN API --------+          LogReg L1 (classifier)      FastAPI backend (8 endpoints)
SBR scraper -----+          Poisson (pitcher K)         Streamlit dashboard (6 tabs)
nba_api ---------+          WNBA ELO + features         Backfill outcomes + CLV
```

## Data

| Table | Rows | Description |
|-------|------|-------------|
| MLB Games | 25,800+ | 2015-2026 schedules and scores |
| Statcast Pitches | 7.6M+ | Pitch-level data (velo, spin, movement, outcomes) |
| Batting Stats | 600K+ | Per-player per-game batting lines |
| Pitching Stats | 221K+ | Per-player per-game pitching lines |
| Odds | 650K+ | Multi-book closing lines (2015-2026, 27+ sportsbooks) |
| WNBA Games | 2,601 | 2015-2024 via ESPN + nba_api |
| Predictions | 14K+ | All model predictions with outcomes, P&L, decomposed CLV |
| CBB Games | 53K+ | College basketball (migrated from prior project) |

## Models

### MLB Moneyline (Phase 1-3)
- Custom ELO with starter adjustments (pitcher quality shifts team ELO per game, K=6)
- 113 features: pitcher, batting, bullpen, ELO, park factor, weather, lineup-specific
- LogReg with L1 regularization outperforms XGBoost at high volume
- +0.42% CLV across 6 seasons, but unprofitable after vig at any threshold

### MLB Totals — Dual Model Live Test (Phase 4)
Two approaches running in parallel for the 2026 season:

**Regression (primary):** LinearRegression predicts total runs, converts to P(over) via normal CDF. Threshold: >=1% info edge + EV gate. ~1,500 bets/year, 51.3% win rate.

**Classifier (comparison):** LogReg L1 directly classifies over/under using game features + market line. Threshold: >=3% info edge + EV gate. ~700 bets/year, 51.1% win rate but higher ROI via odds selection.

Both use: multiplicative de-vig on median book, line shopping across 6+ books, $100 flat sizing, decomposed CLV tracking.

### Pitcher K Props (Phase 2)
- Poisson regression with Statcast features (whiff rate, chase rate, pitch mix, velocity)
- 1.83 MAE, excellent calibration across all K ranges
- No prop line data yet for CLV measurement

### WNBA (Phase 5)
- 2,601 games loaded (2015-2024), ELO trained (K=22, home_advantage=55)
- Cross-validated totals: -2.1% ROI — thin-market thesis did not hold
- Ready to wire into pipeline for May 2026 season start

## Live Pipeline

Runs automatically via GitHub Actions at three times daily:

| Run | Time (ET) | Purpose |
|-----|-----------|---------|
| Backfill | 6:00 AM | Outcomes, edges, P&L, decomposed CLV for completed games |
| Day games | 11:30 AM | Predictions for afternoon slate |
| Night games | 6:30 PM | Predictions for evening slate |

Pipeline steps:
1. Pull new game results and box scores
2. Scrape today's odds from SportsBookReview (6 books: bet365, DraftKings, FanDuel, BetMGM, Caesars, Fanatics)
3. Build features and generate predictions (regression + classifier)
4. De-vig median book, compute info edge, line shop for best price
5. Flag bets meeting threshold criteria, log with bet_book and bet_odds
6. Backfill: compute Model CLV + Execution CLV for resolved bets

### Bankroll
- $10,000 starting bankroll, $100 flat per flagged bet
- One bet per game (deduped when both models flag the same game)
- Per-model tracking for independent ROI/CLV comparison
- Evaluation: October 2026 after regular season ends

## Stack

- **Database:** PostgreSQL (Supabase)
- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost, statsmodels, scipy
- **Data:** MLB Stats API, pybaseball (Statcast), ESPN API, nba_api, SportsBookReview
- **API:** FastAPI (8 endpoints)
- **Dashboard:** Streamlit (6 tabs)
- **Automation:** GitHub Actions (3 daily cron jobs)

## Project Structure

```
db/                  Schema, migrations, and database connection (SimpleConnectionPool)
scrapers/
  mlb/               Games, box scores, Statcast scrapers
  wnba/              ESPN + nba_api scrapers
  odds/              ESPN historical, Arnav dataset, SBR daily scraper
models/
  mlb/
    features.py      113-column feature pipeline
    elo.py           Starter-adjusted ELO system (K=6)
    train.py         LogReg + XGBoost training
    totals_model.py  Run totals regression
    train_totals_classifier.py  Over/under classifier
    k_model.py       Pitcher K Poisson model
    hitter_model.py  Hitter prop models
    lineup_features.py  Per-player lineup aggregation
    statcast_features.py  Pitch-level feature engineering
    threshold_backtest.py  Dual-approach threshold comparison
  wnba/              ELO, features, training
api/                 FastAPI backend (games, predictions, CLV, bankroll, calibration, health)
dashboard/           Streamlit frontend (today, performance, P&L, predictions, calibration, health)
scripts/
  daily_pipeline.py  Main pipeline: scrape, predict, flag, log
  backfill_outcomes.py  Outcomes, P&L, decomposed CLV
  daily_refresh.py   Data refresh utilities
.github/workflows/   GitHub Actions (3 daily cron runs)
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
   python -m models.mlb.train_totals_classifier
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

## Sports Roadmap

- **MLB** — Live (dual model test, decomposed CLV tracking)
- **WNBA** — Next (season starts May 2026, data + models ready)
- **CBB** — Data migrated, model from prior project
- **NHL / NFL** — Schema stubbed, build when seasons approach
