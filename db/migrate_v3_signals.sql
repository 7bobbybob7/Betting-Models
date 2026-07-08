-- v3_signals — forward out-of-sample tracking for the v3 TB model (Attack 3 winner).
--
-- Each row = one TB 1.5 prop scored retrospectively the morning after the game:
-- p_v3 (calibrated model prob) vs p_mkt (Underdog de-vig, the same anchor the validated
-- residual test used). Logged daily by models/mlb/v3_tracker.py from the pipeline cron.
--
-- Purpose: accumulate genuinely-forward evidence that the Attack 3 edge persists
-- (2026 H1 was the validation year; clean forward OOS starts 2026-07). No real bets —
-- the signal's production use is FILTERING Leg 2 line-shopping bets.

CREATE TABLE IF NOT EXISTS v3_signals (
    game_date    DATE NOT NULL,
    game_id      INT NOT NULL,
    player_id    INT NOT NULL,
    line         DECIMAL(4,1) NOT NULL DEFAULT 1.5,
    p_v3         DECIMAL(7,5) NOT NULL,     -- calibrated model P(over)
    p_mkt        DECIMAL(7,5) NOT NULL,     -- Underdog de-vigged P(over)
    edge         DECIMAL(7,5) NOT NULL,     -- p_v3 - p_mkt
    side         VARCHAR(5) NOT NULL,       -- OVER if edge>0 else UNDER
    odds         INT,                       -- UD American odds for the side
    payout_dec   DECIMAL(7,4),
    actual       DECIMAL(6,2),              -- total bases
    won          BOOLEAN,
    profit       DECIMAL(7,4),              -- hypothetical units at UD odds
    model_version VARCHAR(8),               -- bundle version that scored this row (v3, v4, ...)
    logged_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (game_date, player_id)
);

CREATE INDEX IF NOT EXISTS idx_v3_signals_date ON v3_signals (game_date);
