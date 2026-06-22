-- Underdog Fantasy player prop snapshots
-- One row per (snapshot_ts, line, side) so each option (higher/lower) is stored separately.
-- Captures everything needed to backtest hitter prop models:
--   - The line (stat_value) and stat type
--   - Both sides' odds (american + decimal) and payout multipliers
--   - Player and game context for joining to outcomes

CREATE TABLE IF NOT EXISTS underdog_props (
    snapshot_ts            TIMESTAMPTZ NOT NULL,
    line_id                VARCHAR(50)  NOT NULL,
    stable_id              VARCHAR(100),
    line_type              VARCHAR(30),
    line_status            VARCHAR(20),
    stat_value             DECIMAL(6,2),
    stat_type              VARCHAR(50),
    stat_internal          VARCHAR(100),
    category               VARCHAR(50),
    has_alternates         BOOLEAN,

    -- Player
    underdog_player_id     VARCHAR(50),
    player_first_name      VARCHAR(100),
    player_last_name       VARCHAR(100),
    player_position        VARCHAR(50),
    player_team_id         VARCHAR(50),
    player_jersey          VARCHAR(10),

    -- Game
    underdog_game_id       VARCHAR(50),
    game_title             VARCHAR(200),
    home_team_id           VARCHAR(50),
    away_team_id           VARCHAR(50),
    match_progress         VARCHAR(50),

    -- Option (one row per side)
    choice                 VARCHAR(20),    -- 'higher' or 'lower'
    choice_display         VARCHAR(30),
    american_price         INT,
    decimal_price          DECIMAL(6,3),
    payout_multiplier      DECIMAL(6,4),
    option_status          VARCHAR(20),

    PRIMARY KEY (snapshot_ts, line_id, choice)
);

CREATE INDEX IF NOT EXISTS idx_underdog_player_ts
    ON underdog_props (underdog_player_id, snapshot_ts);

CREATE INDEX IF NOT EXISTS idx_underdog_game
    ON underdog_props (underdog_game_id);

CREATE INDEX IF NOT EXISTS idx_underdog_stat_type
    ON underdog_props (stat_type);

CREATE INDEX IF NOT EXISTS idx_underdog_ts
    ON underdog_props (snapshot_ts);
