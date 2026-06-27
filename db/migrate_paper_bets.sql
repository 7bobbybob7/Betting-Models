-- Paper-trade log for the sharp-vs-soft (+EV) line-shopping strategy.
--
-- Each row = a PRE-GAME +EV opportunity flagged at a capture: Novig's de-vigged fair price
-- said Underdog's offered odds were +EV. Logged forward (never backfilled) so realized ROI
-- is a true out-of-sample test — the decisive "is the edge tradeable" measurement.
--
-- Pre-game only: rows are logged with scheduled_start > capture_at. settle() fills actual/
-- won/profit from mlb_batting_game once the game finishes. One row per (capture, prop, side),
-- so a prop flagged at multiple captures is kept (lets us measure persistence).

CREATE TABLE IF NOT EXISTS paper_bets (
    capture_at       TIMESTAMPTZ NOT NULL,     -- the snapshot this bet was flagged at
    game_date        DATE NOT NULL,
    scheduled_start  TIMESTAMPTZ,
    player_id        INT,                      -- our id (for settlement)
    player_name      VARCHAR(100),
    market_type      VARCHAR(40) NOT NULL,     -- HITS_RUNS_RBIS, TOTAL_BASES, ...
    line             DECIMAL(6,2) NOT NULL,
    side             VARCHAR(5) NOT NULL,      -- OVER / UNDER
    ud_odds          INT,                      -- Underdog American odds for the bet side
    ud_payout_dec    DECIMAL(7,4),             -- decimal payout
    novig_fair       DECIMAL(7,5),             -- Novig de-vigged prob for the side
    ev               DECIMAL(7,4),             -- expected value at flag time
    novig_volume     DECIMAL(14,2),            -- liquidity at flag time
    -- settlement (filled after the game)
    actual           DECIMAL(6,2),
    won              BOOLEAN,
    profit           DECIMAL(7,4),             -- +EV units: win -> payout-1, loss -> -1
    settled_at       TIMESTAMPTZ,
    PRIMARY KEY (capture_at, player_id, market_type, line, side)
);

CREATE INDEX IF NOT EXISTS idx_paper_unsettled ON paper_bets (settled_at, game_date) WHERE settled_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_paper_game_date ON paper_bets (game_date);
