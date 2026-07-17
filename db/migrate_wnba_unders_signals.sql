-- wnba_unders_signals — forward tracker for the WNBA structural candidate:
-- blanket UNDERS at Novig on threes(390)/assists(391)/points(393).
-- Mechanism: universal ~3pt public over-shade (measured across ALL books) exceeds only
-- the exchange's ~5% vig. Period-split: non-negative 2025H2, +4-5% 2026H1 -> candidate,
-- confirmed only by forward record (clean from 2026-07-12). Odds sanity: |odds|<=2000.
CREATE TABLE IF NOT EXISTS wnba_unders_signals (
    prop_date    DATE NOT NULL,
    bp_player_id INT NOT NULL,
    market_id    INT NOT NULL,             -- 390 threes / 391 assists / 393 points
    line         DECIMAL(5,1) NOT NULL,
    under_odds   INT NOT NULL,
    payout_dec   DECIMAL(8,4) NOT NULL,
    actual       DECIMAL(6,2),
    won          BOOLEAN,
    profit       DECIMAL(8,4),
    is_backfill  BOOLEAN NOT NULL DEFAULT false,
    logged_at    TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (prop_date, bp_player_id, market_id)
);
