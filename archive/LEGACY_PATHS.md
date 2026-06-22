# Legacy File Paths & Import Dependencies

This document maps every archived file's original location, its archive location,
its dependencies, and the path adjustments needed to make it runnable again.

**Context:** In June 2026 the project pivoted from MLB game-level betting
(totals/moneyline/spreads) to MLB hitter prop modeling. See
[TOTALS_LIVE_TEST_FINDINGS.md](../TOTALS_LIVE_TEST_FINDINGS.md) for the
reasoning. All totals/moneyline/WNBA code was preserved here rather than
deleted, in case any of it ever needs to be revived.

## How to revive an archived file

Most archived files use this pattern at the top:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
```

After archiving, the file is one directory deeper, so this needs to become:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
```

After fixing the sys.path, all `from db.db import ...`, `from scrapers... import ...`,
and `from models.mlb... import ...` calls work because:
- `db/` and `scrapers/mlb/`, `scrapers/props/` are still in the repo root
- `models/mlb/statcast_features.py`, `lineup_features.py`, `k_model.py`,
  `hitter_model.py` are still in their original location (we kept them as
  reusable infrastructure)

For files that import other archived files (e.g. `archive/models/totals/features.py`
importing `models.mlb.elo`), the import line must be updated:
```python
# OLD:
from models.mlb.elo import MLBElo
# NEW:
from archive.models.totals.elo import MLBElo
```

---

## File-by-file index

### archive/models/totals/

These files made up the totals/moneyline prediction pipeline. They were
trained, deployed via daily cron, and tested live April 11 - June 9, 2026.
All three deployed totals models proved unprofitable in live testing.

| Original path | Archive path | Imports | Imported by |
|---------------|--------------|---------|-------------|
| `models/mlb/elo.py` | `archive/models/totals/elo.py` | `db.db` | `models/mlb/features.py` (also archived) |
| `models/mlb/features.py` | `archive/models/totals/features.py` | `db.db`, `models.mlb.elo` | `scripts/daily_pipeline.py` (archived) |
| `models/mlb/train.py` | `archive/models/totals/train.py` | `db.db` | `investigate_*.py`, `player_model.py` |
| `models/mlb/predict.py` | `archive/models/totals/predict.py` | `db.db` | None — standalone script |
| `models/mlb/evaluate.py` | `archive/models/totals/evaluate.py` | `db.db` | None — standalone calibration tool |
| `models/mlb/totals_model.py` | `archive/models/totals/totals_model.py` | `db.db` | None — trains and saves `totals_model.pkl` |
| `models/mlb/totals_classifier.py` | `archive/models/totals/totals_classifier.py` | `db.db` | None — analysis script |
| `models/mlb/train_totals_classifier.py` | `archive/models/totals/train_totals_classifier.py` | `db.db` | None — trains and saves `totals_classifier.pkl` |
| `models/mlb/totals_v2.py` | `archive/models/totals/totals_v2.py` | `db.db`, `models.mlb.statcast_features`, `models.mlb.lineup_features` | None |
| `models/mlb/totals_v3.py` | `archive/models/totals/totals_v3.py` | `db.db` | None |
| `models/mlb/totals_rolling_retrain.py` | `archive/models/totals/totals_rolling_retrain.py` | `db.db` | None |
| `models/mlb/totals_line_shopping.py` | `archive/models/totals/totals_line_shopping.py` | `db.db` | None |
| `models/mlb/ip_model.py` | `archive/models/totals/ip_model.py` | `db.db`, `models.mlb.statcast_features` | None |
| `models/mlb/player_model.py` | `archive/models/totals/player_model.py` | `db.db`, `models.mlb.train` (archived) | None |

### archive/models/investigations/

Diagnostic / EDA scripts run while investigating model performance. Useful
reference if revisiting similar issues for hitter props.

| Original path | Archive path | Imports | Purpose |
|---------------|--------------|---------|---------|
| `models/mlb/investigate_totals.py` | `archive/models/investigations/investigate_totals.py` | `db.db` | Cross-season totals analysis |
| `models/mlb/investigate_totals_deep.py` | `archive/models/investigations/investigate_totals_deep.py` | `db.db` | Deep totals investigation |
| `models/mlb/investigate_totals_deep2.py` | `archive/models/investigations/investigate_totals_deep2.py` | `db.db` | Refined strategy analysis |
| `models/mlb/investigate_totals_deep3.py` | `archive/models/investigations/investigate_totals_deep3.py` | `db.db` | Compound filters / drawdown |
| `models/mlb/investigate_classifier.py` | `archive/models/investigations/investigate_classifier.py` | `db.db` | Classifier significance testing |
| `models/mlb/investigate_edge.py` | `archive/models/investigations/investigate_edge.py` | `db.db`, `models.mlb.train` | Moneyline CLV investigation |
| `models/mlb/threshold_backtest.py` | `archive/models/investigations/threshold_backtest.py` | `db.db` | Threshold backtest comparison |
| `models/mlb/threshold_odds_check.py` | `archive/models/investigations/threshold_odds_check.py` | `db.db` | Odds dispersion analysis |

### archive/models/wnba/

The WNBA thin-market hypothesis didn't pan out (-2.1% ROI cross-validated
on totals). Models built but never deployed.

| Original path | Archive path | Imports |
|---------------|--------------|---------|
| `models/wnba/elo.py` | `archive/models/wnba/elo.py` | `db.db` |
| `models/wnba/features.py` | `archive/models/wnba/features.py` | `db.db`, `models.wnba.elo` |
| `models/wnba/train.py` | `archive/models/wnba/train.py` | `db.db` |

### archive/scrapers/odds/

Game-level odds scrapers — historical loaders for ESPN (2015-2024) and
Arnav dataset (2021-Aug 2025), plus daily SBR scraper. All produced the
`odds` table data which fed the totals models.

| Original path | Archive path | Purpose |
|---------------|--------------|---------|
| `scrapers/odds/espn_odds.py` | `archive/scrapers/odds/espn_odds.py` | One-shot ESPN historical odds loader |
| `scrapers/odds/load_arnav_odds.py` | `archive/scrapers/odds/load_arnav_odds.py` | One-shot Arnav dataset loader |
| `scrapers/odds/sbr_scraper.py` | `archive/scrapers/odds/sbr_scraper.py` | Daily SBR scrape (was called from old daily_pipeline) |

### archive/scrapers/wnba/

| Original path | Archive path | Notes |
|---------------|--------------|-------|
| `scrapers/wnba/games.py` | `archive/scrapers/wnba/games.py` | ESPN scraper |
| `scrapers/wnba/load_history.py` | `archive/scrapers/wnba/load_history.py` | nba_api historical loader |

### archive/scripts/

| Original path | Archive path | Imports |
|---------------|--------------|---------|
| `scripts/daily_pipeline.py` | `archive/scripts/daily_pipeline.py` | `db.db`, `scrapers.odds.sbr_scraper` (archived), `scripts.daily_refresh` (still active!), `models.mlb.features` (archived), `models.mlb.statcast_features` (still active) |
| `scripts/backfill_outcomes.py` | `archive/scripts/backfill_outcomes.py` | `db.db` |
| `scripts/migrate_cbb.py` | `archive/scripts/migrate_cbb.py` | `db.db` (unused — CBB never used) |

### archive/api/

FastAPI backend serving the totals live test data. 8 endpoints all assumed
totals predictions structure.

| Original path | Archive path | Imports |
|---------------|--------------|---------|
| `api/app.py` | `archive/api/app.py` | `db.db` only — most logic in SQL |
| `api/__init__.py` | `archive/api/__init__.py` | (empty) |

### archive/dashboard/

Streamlit frontend, 6 tabs for the totals workflow.

| Original path | Archive path | Imports |
|---------------|--------------|---------|
| `dashboard/app.py` | `archive/dashboard/app.py` | `db.db` plus calls `api/app.py` over HTTP when running |

### archive/docs/

Old project documentation that referenced the totals direction.

| Original path | Archive path |
|---------------|--------------|
| `README.md` | `archive/docs/README_legacy.md` |
| `PROJECT_OVERVIEW.md` | `archive/docs/PROJECT_OVERVIEW.md` |
| `SESSION_SUMMARY.md` | `archive/docs/SESSION_SUMMARY.md` |
| `PRD.md` | `archive/docs/PRD.md` |

---

## Active code that legacy depends on (DO NOT MOVE)

These files are still in their original location because hitter prop work
will use them too. Archived files that reference them keep working as long as
sys.path is fixed.

| Path | What it provides |
|------|------------------|
| `db/db.py` | DB connection pool, query/execute/bulk_insert helpers |
| `scrapers/mlb/games.py` | MLB Stats API game/team/player loader |
| `scrapers/mlb/boxscores.py` | Box score puller |
| `scrapers/mlb/statcast.py` | Full-season Statcast loader |
| `scrapers/mlb/statcast_daily.py` | Daily Statcast pull (cron) |
| `scrapers/props/underdog.py` | Underdog props capture (cron) |
| `scripts/daily_refresh.py` | Morning data refresh (cron) |
| `models/mlb/statcast_features.py` | Reusable Statcast feature engineering |
| `models/mlb/lineup_features.py` | Reusable lineup feature engineering |
| `models/mlb/k_model.py` | Pitcher K Poisson model (will use for hitter prop pitcher features) |
| `models/mlb/hitter_model.py` | Earlier hitter prop attempt (starting point) |

## Reviving an archived workflow

If you ever want to run an archived script:

1. **Update sys.path.insert** — add one more `"../.."` (since the file is one
   directory deeper now).

2. **Update cross-archive imports** — any `from models.mlb.elo import ...`
   becomes `from archive.models.totals.elo import ...`.

3. **For the old daily_pipeline:**
   - It imports `from scripts.daily_refresh import main as refresh_main` —
     this still works because daily_refresh.py is still in `scripts/`.
   - It imports `from scrapers.odds.sbr_scraper import scrape_date as scrape_odds`
     which is now at `archive/scrapers/odds/sbr_scraper.py` — update the
     import to `from archive.scrapers.odds.sbr_scraper import scrape_date as scrape_odds`.
   - It imports `from models.mlb.features import build_feature_matrix` which
     is now at `archive/models/totals/features.py` — update to
     `from archive.models.totals.features import build_feature_matrix`.
   - `from models.mlb.statcast_features import build_statcast_features` still
     works because `statcast_features.py` is still in `models/mlb/`.

4. **For the old api/dashboard:**
   - `api/app.py` only imports `db.db` so it works as-is after sys.path fix.
   - `dashboard/app.py` only imports `db.db` so it works as-is after sys.path fix.

## DB tables referenced by legacy code

These tables still exist in the database (nothing was dropped during archiving):

| Table | Used by legacy | Used by active |
|-------|---------------|----------------|
| `games`, `teams`, `players`, `seasons` | Yes | Yes |
| `mlb_pitches` | Yes (totals feature pipeline) | Yes (hitter prop matchup features — planned) |
| `mlb_batting_game`, `mlb_pitching_game` | Yes | Yes (essential for hitter props) |
| `mlb_game_info` | Yes | Yes |
| `odds` | Yes (game-level odds for totals) | Maybe (game-level context for hitter props) |
| `predictions` | Yes (live test results stored here) | Will write hitter prop predictions here too |
| `underdog_props` | No (didn't exist during totals era) | Yes — new table for prop snapshots |
| `wnba_*`, `cbb_*` | Yes | No |
