-- dk_rbi_signals — forward tracking for the venue-sweep survivor:
-- DraftKings RBI 0.5 singles priced against Novig de-vigged fair.
--
-- Backtest (Apr–Jun 2026): +8.5% ROI on 243 EV>2% bets, positive all 3 months — the only
-- book×market that survived the cross-book sweep. DK + Novig both arrive via the daily
-- BettingPros pull (same bp_player_id), so scoring is retrospective + leak-free: EV is
-- computed from lines that were live pre-game, settled from BettingPros' own `actual`.
--
-- Rows before 2026-07-09 are BACKFILL BASELINE (the backtest sample re-logged for
-- continuity); the clean forward record starts 2026-07-09.

CREATE TABLE IF NOT EXISTS dk_rbi_signals (
    prop_date     DATE NOT NULL,
    bp_player_id  INT NOT NULL,
    player_name   VARCHAR(100),
    dk_line       DECIMAL(4,1) NOT NULL DEFAULT 0.5,
    side          VARCHAR(5) NOT NULL,       -- OVER / UNDER (whichever DK prices soft)
    dk_odds       INT NOT NULL,              -- DraftKings American odds for the bet side
    payout_dec    DECIMAL(7,4) NOT NULL,
    novig_fair    DECIMAL(7,5) NOT NULL,     -- Novig de-vigged prob for the bet side
    ev            DECIMAL(7,4) NOT NULL,     -- (fair * (payout-1)) - (1-fair)
    actual        DECIMAL(6,2),
    won           BOOLEAN,
    profit        DECIMAL(7,4),              -- units: win -> payout-1, loss -> -1
    is_backfill   BOOLEAN NOT NULL DEFAULT false,
    logged_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prop_date, bp_player_id)
);

CREATE INDEX IF NOT EXISTS idx_dk_rbi_date ON dk_rbi_signals (prop_date);
