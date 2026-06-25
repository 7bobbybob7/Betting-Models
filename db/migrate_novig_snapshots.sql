-- Intraday Novig exchange snapshots for sharp-vs-soft (Novig vs Underdog) line-shopping.
--
-- Sourced directly from Novig's own GraphQL API (api.novig.us/v1/graphql), NOT BettingPros.
-- Novig prices are de-vigged probabilities (over + under ≈ 1.0). We capture every intraday
-- run (captured_at set once per run) so we can measure whether a Novig-vs-Underdog
-- discrepancy persists long enough to bet — the staleness question the daily backtest
-- can't answer.
--
--   last       = last-traded probability (can be stale if no recent trade)
--   available  = best resting order currently takeable (the real tradeable price; often null)
--   volume     = lifetime $ volume on the market (liquidity/confidence filter)

CREATE TABLE IF NOT EXISTS novig_snapshots (
    captured_at      TIMESTAMPTZ NOT NULL,     -- one value per capture run
    novig_market_id  VARCHAR(40) NOT NULL,     -- unique per (player, type, strike, event)
    game_date        DATE,                     -- ET date of scheduled_start
    scheduled_start  TIMESTAMPTZ,
    market_type      VARCHAR(40),              -- HITS_RUNS_RBIS, TOTAL_BASES, RBIS, ...
    player_name      VARCHAR(100),
    strike           DECIMAL(6,2),             -- the line
    over_last        DECIMAL(7,5),
    under_last       DECIMAL(7,5),
    over_available   DECIMAL(7,5),
    under_available  DECIMAL(7,5),
    volume           DECIMAL(14,2),
    PRIMARY KEY (captured_at, novig_market_id)
);

-- Match a Novig prop to the same Underdog prop at a given time
CREATE INDEX IF NOT EXISTS idx_novig_snap_match
    ON novig_snapshots (game_date, market_type, player_name, strike, captured_at);

-- Track one market's price across the day (persistence)
CREATE INDEX IF NOT EXISTS idx_novig_snap_market
    ON novig_snapshots (novig_market_id, captured_at);
