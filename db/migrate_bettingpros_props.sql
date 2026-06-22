-- BettingPros historical prop snapshots — primary source for historical Underdog
-- lines + odds. One row per (prop_date, market, book, player). Filtered by
-- book_id at fetch time (Underdog=36, Consensus=0, Novig=60, etc).
--
-- Schema mirrors the BettingPros /v3/props response structure. The `actual`
-- field is included since BettingPros joins outcomes to props for us.

CREATE TABLE IF NOT EXISTS bettingpros_props (
    -- Identity
    prop_date              DATE NOT NULL,            -- game date (cutoff for backtest features)
    event_id               INT,                      -- BettingPros event id
    market_id              INT NOT NULL,
    market_name            VARCHAR(40),
    book_id                INT NOT NULL,             -- 36=Underdog, 0=consensus, 60=Novig, etc
    book_name              VARCHAR(50),

    -- Player
    bp_player_id           VARCHAR(20) NOT NULL,
    player_first_name      VARCHAR(50),
    player_last_name       VARCHAR(50),
    player_team            VARCHAR(10),
    player_position        VARCHAR(20),
    player_slug            VARCHAR(100),

    -- Game context
    opposing_pitcher       VARCHAR(60),
    in_lineup              BOOLEAN,

    -- OVER side (best-available offered by this book)
    over_line              DECIMAL(6,2),
    over_odds              INT,
    over_consensus_line    DECIMAL(6,2),
    over_consensus_odds    INT,
    over_probability       DECIMAL(7,5),

    -- UNDER side
    under_line             DECIMAL(6,2),
    under_odds             INT,
    under_consensus_line   DECIMAL(6,2),
    under_consensus_odds   INT,
    under_probability      DECIMAL(7,5),

    -- BettingPros' own projection (for cross-reference only — not our model)
    bp_projected_value     DECIMAL(6,2),
    bp_recommended_side    VARCHAR(10),
    bp_bet_rating          INT,

    -- Actual outcome
    actual                 DECIMAL(6,2),
    is_scored              BOOLEAN,
    is_push                BOOLEAN,

    -- Provenance
    captured_at            TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (prop_date, market_id, book_id, bp_player_id)
);

CREATE INDEX IF NOT EXISTS idx_bp_player_date
    ON bettingpros_props (bp_player_id, prop_date);

CREATE INDEX IF NOT EXISTS idx_bp_date_market
    ON bettingpros_props (prop_date, market_id);

CREATE INDEX IF NOT EXISTS idx_bp_book_date
    ON bettingpros_props (book_id, prop_date);

CREATE INDEX IF NOT EXISTS idx_bp_market_book
    ON bettingpros_props (market_id, book_id);

CREATE INDEX IF NOT EXISTS idx_bp_player_name
    ON bettingpros_props (player_last_name, player_first_name);
