# Sports Betting Platform

Multi-sport betting model platform with PostgreSQL backend, automated data scrapers, and ML prediction models.

## Sports
- **CBB** (College Basketball) — team-level efficiency models, custom ELO
- **MLB** — pitcher matchup models, Statcast pitch-level data
- **WNBA** — efficiency models with player-level data
- **NHL** — planned
- **NFL** — planned

## Stack
- **Database:** PostgreSQL (Supabase)
- **Models:** Python, scikit-learn, XGBoost
- **Data:** ESPN API, pybaseball (Statcast), odds APIs
- **Dashboard:** Streamlit

## Setup

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your Supabase connection string
4. Run `db/schema.sql` against your Supabase SQL Editor
5. Run `python scripts/migrate_cbb.py` to load existing CBB data

## Repo Structure
```
db/              Schema and migrations
scrapers/        Data collection scripts (per sport)
models/          ML models and feature engineering (per sport)
dashboard/       Streamlit app
scripts/         One-time scripts (migrations, backfills)
```
