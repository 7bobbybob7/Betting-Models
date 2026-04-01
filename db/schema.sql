-- ============================================================
-- MULTI-SPORT BETTING MODEL DATABASE SCHEMA
-- Run against Supabase (PostgreSQL) instance
-- ============================================================

-- ============================================================
-- SHARED / UNIVERSAL TABLES
-- ============================================================

CREATE TABLE sports (
    sport_id    SERIAL PRIMARY KEY,
    name        VARCHAR(20) UNIQUE NOT NULL,  -- 'cbb', 'mlb', 'wnba', 'nhl', 'nfl'
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO sports (name) VALUES ('cbb'), ('mlb'), ('wnba');

-- --------------------------------------------------------

CREATE TABLE seasons (
    season_id   SERIAL PRIMARY KEY,
    sport_id    INT NOT NULL REFERENCES sports(sport_id),
    year        INT NOT NULL,           -- e.g. 2026 (for CBB this is the spring year, so 2025-26 season = 2026)
    start_date  DATE,
    end_date    DATE,
    UNIQUE(sport_id, year)
);

-- --------------------------------------------------------

CREATE TABLE teams (
    team_id     SERIAL PRIMARY KEY,
    sport_id    INT NOT NULL REFERENCES sports(sport_id),
    name        VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10),
    conference  VARCHAR(50),
    division    VARCHAR(50),            -- MLB divisions, NFL divisions, etc.
    active      BOOLEAN DEFAULT TRUE,
    UNIQUE(sport_id, name)
);

-- --------------------------------------------------------

CREATE TABLE players (
    player_id       SERIAL PRIMARY KEY,
    sport_id        INT NOT NULL REFERENCES sports(sport_id),
    external_id     VARCHAR(50),        -- Statcast ID, ESPN ID, NHL API ID, etc.
    name            VARCHAR(100) NOT NULL,
    position        VARCHAR(20),
    bats            VARCHAR(1),         -- 'L', 'R', 'S' (MLB-specific but nullable)
    throws          VARCHAR(1),         -- 'L', 'R' (MLB-specific but nullable)
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sport_id, external_id)
);

-- Maps players to teams by season (handles trades, transfers)
CREATE TABLE player_teams (
    player_team_id  SERIAL PRIMARY KEY,
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),
    season_id       INT NOT NULL REFERENCES seasons(season_id),
    start_date      DATE,
    end_date        DATE,               -- NULL = still on team
    UNIQUE(player_id, team_id, season_id, start_date)
);

-- --------------------------------------------------------

CREATE TABLE games (
    game_id         SERIAL PRIMARY KEY,
    sport_id        INT NOT NULL REFERENCES sports(sport_id),
    season_id       INT NOT NULL REFERENCES seasons(season_id),
    external_id     VARCHAR(50),        -- ESPN game ID, MLB game PK, etc.
    game_date       DATE NOT NULL,
    game_time       TIMESTAMPTZ,        -- actual start time if available
    home_team_id    INT NOT NULL REFERENCES teams(team_id),
    away_team_id    INT NOT NULL REFERENCES teams(team_id),
    home_score      INT,                -- NULL if game hasn't been played
    away_score      INT,
    status          VARCHAR(20) DEFAULT 'scheduled', -- 'scheduled', 'final', 'postponed', 'cancelled'
    venue           VARCHAR(100),
    is_postseason   BOOLEAN DEFAULT FALSE,
    is_neutral_site BOOLEAN DEFAULT FALSE,  -- relevant for CBB tournament
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sport_id, external_id)
);

CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_sport_season ON games(sport_id, season_id);
CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id);

-- --------------------------------------------------------

-- Supports both closing lines only AND line movement snapshots.
-- If you only grab closing lines, there's one row per game/book/market.
-- If you snapshot throughout the day, multiple rows with different timestamps.
CREATE TABLE odds (
    odds_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    sportsbook      VARCHAR(30) NOT NULL,   -- 'draftkings', 'fanduel', 'pinnacle', etc.
    market          VARCHAR(20) NOT NULL,   -- 'moneyline', 'spread', 'total', 'f5_ml', 'f5_total'
    home_line       DECIMAL(8,3),           -- spread value or ML odds (American)
    away_line       DECIMAL(8,3),
    total_line      DECIMAL(6,2),           -- over/under number
    over_odds       DECIMAL(8,3),
    under_odds      DECIMAL(8,3),
    home_implied    DECIMAL(5,4),           -- de-vigged implied probability
    away_implied    DECIMAL(5,4),
    is_closing      BOOLEAN DEFAULT FALSE,  -- flag for closing line
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(game_id, sportsbook, market, recorded_at)
);

CREATE INDEX idx_odds_game ON odds(game_id);
CREATE INDEX idx_odds_closing ON odds(game_id, is_closing) WHERE is_closing = TRUE;

-- --------------------------------------------------------

-- Player props (strikeouts, hits, passing yards, points, SOG, etc.)
-- Works across all sports via the market string column.
CREATE TABLE player_props (
    prop_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    sportsbook      VARCHAR(30) NOT NULL,
    market          VARCHAR(50) NOT NULL,   -- 'strikeouts', 'hits', 'hrs', 'passing_yards', 'points', 'sog', etc.
    line            DECIMAL(6,2) NOT NULL,  -- the number (6.5, 1.5, etc.)
    over_odds       DECIMAL(8,3),
    under_odds      DECIMAL(8,3),
    over_implied    DECIMAL(5,4),
    under_implied   DECIMAL(5,4),
    actual_value    DECIMAL(8,2),           -- what actually happened (for backtesting)
    is_closing      BOOLEAN DEFAULT FALSE,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(game_id, player_id, sportsbook, market, recorded_at)
);

CREATE INDEX idx_props_game ON player_props(game_id);
CREATE INDEX idx_props_player ON player_props(player_id);
CREATE INDEX idx_props_market ON player_props(market);

-- --------------------------------------------------------

CREATE TABLE elo_ratings (
    elo_id      SERIAL PRIMARY KEY,
    team_id     INT NOT NULL REFERENCES teams(team_id),
    game_id     INT REFERENCES games(game_id),  -- NULL for preseason initialization
    rating      DECIMAL(8,2) NOT NULL,
    rating_date DATE NOT NULL,
    UNIQUE(team_id, game_id)
);

CREATE INDEX idx_elo_team_date ON elo_ratings(team_id, rating_date);

-- --------------------------------------------------------

-- Every model prediction lives here. One table to backtest across all sports.
CREATE TABLE predictions (
    prediction_id   SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    model_name      VARCHAR(50) NOT NULL,       -- 'cbb_logreg_v1', 'mlb_xgb_v2', etc.
    market          VARCHAR(20) NOT NULL,       -- 'moneyline', 'spread', 'total'
    predicted_prob  DECIMAL(5,4),               -- predicted win prob (home team)
    predicted_value DECIMAL(6,2),               -- predicted spread or total
    edge            DECIMAL(5,4),               -- predicted_prob - closing_implied
    bet_placed      BOOLEAN DEFAULT FALSE,
    bet_amount      DECIMAL(8,2),
    bet_odds        DECIMAL(8,3),               -- odds at time of bet
    outcome         VARCHAR(10),                -- 'win', 'loss', 'push', NULL if pending
    pnl             DECIMAL(10,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(game_id, model_name, market)
);

CREATE INDEX idx_predictions_model ON predictions(model_name);
CREATE INDEX idx_predictions_game ON predictions(game_id);

-- ============================================================
-- CBB-SPECIFIC TABLES (team-level only)
-- ============================================================

CREATE TABLE cbb_team_game_stats (
    stat_id             SERIAL PRIMARY KEY,
    game_id             INT NOT NULL REFERENCES games(game_id),
    team_id             INT NOT NULL REFERENCES teams(team_id),
    is_home             BOOLEAN NOT NULL,

    -- box score
    points              INT,
    fgm                 INT,
    fga                 INT,
    fg3m                INT,
    fg3a                INT,
    ftm                 INT,
    fta                 INT,
    offensive_rebounds  INT,
    defensive_rebounds  INT,
    assists             INT,
    steals              INT,
    blocks              INT,
    turnovers           INT,
    fouls               INT,

    -- derived efficiency metrics (precomputed for speed)
    offensive_efficiency    DECIMAL(6,2),
    defensive_efficiency    DECIMAL(6,2),
    tempo                   DECIMAL(6,2),
    efg_pct                 DECIMAL(5,4),
    turnover_pct            DECIMAL(5,4),
    orb_pct                 DECIMAL(5,4),
    ft_rate                 DECIMAL(5,4),

    UNIQUE(game_id, team_id)
);

-- ============================================================
-- MLB-SPECIFIC TABLES
-- ============================================================

-- Game-level metadata specific to MLB
CREATE TABLE mlb_game_info (
    game_id         INT PRIMARY KEY REFERENCES games(game_id),
    home_starter_id INT REFERENCES players(player_id),
    away_starter_id INT REFERENCES players(player_id),
    weather_temp    INT,            -- degrees F
    weather_wind    INT,            -- mph
    weather_dir     VARCHAR(20),    -- 'out_to_cf', 'in_from_cf', 'l_to_r', etc.
    weather_cond    VARCHAR(20),    -- 'clear', 'cloudy', 'dome', 'rain'
    umpire_hp       VARCHAR(50),    -- home plate umpire name
    park_factor     DECIMAL(5,2)    -- park factor for that venue/season
);

-- Batter stats per game
CREATE TABLE mlb_batting_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),
    batting_order   INT,            -- 1-9

    pa              INT,
    ab              INT,
    hits            INT,
    doubles         INT,
    triples         INT,
    hr              INT,
    rbi             INT,
    bb              INT,
    so              INT,
    hbp             INT,
    sb              INT,
    cs              INT,

    UNIQUE(game_id, player_id)
);

-- Pitcher stats per game
CREATE TABLE mlb_pitching_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),
    is_starter      BOOLEAN NOT NULL,

    ip              DECIMAL(4,1),   -- innings pitched
    hits_allowed    INT,
    runs            INT,
    earned_runs     INT,
    bb              INT,
    so              INT,
    hr_allowed      INT,
    pitches         INT,
    strikes         INT,
    decision        VARCHAR(5),     -- 'W', 'L', 'S', 'H', NULL

    UNIQUE(game_id, player_id)
);

-- Statcast pitch-level data
CREATE TABLE mlb_pitches (
    pitch_id            SERIAL PRIMARY KEY,
    game_id             INT NOT NULL REFERENCES games(game_id),
    pitcher_id          INT NOT NULL REFERENCES players(player_id),
    batter_id           INT NOT NULL REFERENCES players(player_id),
    inning              INT,
    top_bottom          VARCHAR(3),     -- 'top', 'bot'
    at_bat_number       INT,
    pitch_number        INT,            -- pitch number within at-bat

    -- pitch characteristics
    pitch_type          VARCHAR(5),     -- 'FF', 'SL', 'CU', 'CH', 'SI', etc.
    release_speed       DECIMAL(5,1),   -- mph
    release_spin_rate   INT,            -- rpm
    release_extension   DECIMAL(4,1),   -- feet
    pfx_x               DECIMAL(5,2),   -- horizontal movement (inches)
    pfx_z               DECIMAL(5,2),   -- vertical movement (inches)
    plate_x             DECIMAL(5,2),   -- horizontal plate location
    plate_z             DECIMAL(5,2),   -- vertical plate location
    zone                INT,            -- 1-14 Statcast zone

    -- outcome
    description         VARCHAR(50),    -- 'called_strike', 'ball', 'hit_into_play', etc.
    result              VARCHAR(50),    -- at-bat result if final pitch
    is_strike           BOOLEAN,
    is_swing            BOOLEAN,
    is_whiff            BOOLEAN,
    is_in_play          BOOLEAN,

    -- batted ball data (NULL if not in play)
    launch_speed        DECIMAL(5,1),   -- exit velo
    launch_angle        DECIMAL(5,1),   -- degrees
    hit_distance        DECIMAL(5,1),   -- feet
    xba                 DECIMAL(5,3),   -- expected batting average
    xslg                DECIMAL(5,3),   -- expected slugging
    xwoba               DECIMAL(5,3),   -- expected wOBA

    balls               INT,            -- count before this pitch
    strikes_count       INT             -- count before this pitch
);

-- Pitch table will be BY FAR the largest table. Index aggressively.
CREATE INDEX idx_pitches_game ON mlb_pitches(game_id);
CREATE INDEX idx_pitches_pitcher ON mlb_pitches(pitcher_id);
CREATE INDEX idx_pitches_batter ON mlb_pitches(batter_id);
CREATE INDEX idx_pitches_type ON mlb_pitches(pitch_type);
CREATE INDEX idx_pitches_pitcher_type ON mlb_pitches(pitcher_id, pitch_type);

-- ============================================================
-- WNBA-SPECIFIC TABLES
-- ============================================================

-- Team-level game stats (efficiency metrics, similar to CBB)
CREATE TABLE wnba_team_game_stats (
    stat_id             SERIAL PRIMARY KEY,
    game_id             INT NOT NULL REFERENCES games(game_id),
    team_id             INT NOT NULL REFERENCES teams(team_id),
    is_home             BOOLEAN NOT NULL,

    points              INT,
    fgm                 INT,
    fga                 INT,
    fg3m                INT,
    fg3a                INT,
    ftm                 INT,
    fta                 INT,
    offensive_rebounds  INT,
    defensive_rebounds  INT,
    assists             INT,
    steals              INT,
    blocks              INT,
    turnovers           INT,
    fouls               INT,

    -- derived
    offensive_efficiency    DECIMAL(6,2),
    defensive_efficiency    DECIMAL(6,2),
    tempo                   DECIMAL(6,2),
    efg_pct                 DECIMAL(5,4),
    turnover_pct            DECIMAL(5,4),
    orb_pct                 DECIMAL(5,4),
    ft_rate                 DECIMAL(5,4),

    UNIQUE(game_id, team_id)
);

-- Player-level game stats
CREATE TABLE wnba_player_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),
    is_starter      BOOLEAN,

    minutes         DECIMAL(4,1),
    points          INT,
    fgm             INT,
    fga             INT,
    fg3m            INT,
    fg3a            INT,
    ftm             INT,
    fta             INT,
    orb             INT,
    drb             INT,
    assists         INT,
    steals          INT,
    blocks          INT,
    turnovers       INT,
    fouls           INT,
    plus_minus      INT,

    UNIQUE(game_id, player_id)
);

-- ============================================================
-- NHL-SPECIFIC TABLES (placeholder for future)
-- ============================================================

-- Skater stats per game
CREATE TABLE nhl_skater_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),

    goals           INT,
    assists         INT,
    shots           INT,
    hits            INT,
    blocked_shots   INT,
    pim             INT,           -- penalty minutes
    toi             DECIMAL(5,1),  -- time on ice (minutes)
    plus_minus      INT,
    faceoff_wins    INT,
    faceoff_losses  INT,

    -- advanced (if available from NHL API)
    cf              INT,           -- corsi for
    ca              INT,           -- corsi against
    xgf             DECIMAL(5,2),  -- expected goals for
    xga             DECIMAL(5,2),  -- expected goals against

    UNIQUE(game_id, player_id)
);

-- Goalie stats per game
CREATE TABLE nhl_goalie_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),

    shots_against   INT,
    saves           INT,
    goals_against   INT,
    save_pct        DECIMAL(5,4),
    toi             DECIMAL(5,1),
    decision        VARCHAR(5),    -- 'W', 'L', 'OTL'

    UNIQUE(game_id, player_id)
);

-- ============================================================
-- NFL-SPECIFIC TABLES (placeholder for future)
-- ============================================================

CREATE TABLE nfl_player_game (
    stat_id         SERIAL PRIMARY KEY,
    game_id         INT NOT NULL REFERENCES games(game_id),
    player_id       INT NOT NULL REFERENCES players(player_id),
    team_id         INT NOT NULL REFERENCES teams(team_id),

    -- passing
    pass_att        INT,
    pass_cmp        INT,
    pass_yards      INT,
    pass_td         INT,
    interceptions   INT,
    sacks           INT,
    passer_rating   DECIMAL(5,1),

    -- rushing
    rush_att        INT,
    rush_yards      INT,
    rush_td         INT,

    -- receiving
    targets         INT,
    receptions      INT,
    rec_yards       INT,
    rec_td          INT,

    -- defense
    tackles         INT,
    def_sacks       DECIMAL(3,1),
    def_int         INT,
    forced_fumbles  INT,

    -- kicking
    fg_att          INT,
    fg_made         INT,
    xp_att          INT,
    xp_made         INT,

    UNIQUE(game_id, player_id)
);

-- ============================================================
-- USEFUL VIEWS
-- ============================================================

-- Quick view: game results with team names
CREATE VIEW v_game_results AS
SELECT
    g.game_id,
    s.name AS sport,
    g.game_date,
    ht.name AS home_team,
    at.name AS away_team,
    g.home_score,
    g.away_score,
    g.status,
    g.is_postseason,
    g.is_neutral_site
FROM games g
JOIN sports s ON g.sport_id = s.sport_id
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id;

-- Quick view: predictions with outcomes for backtesting
CREATE VIEW v_backtest AS
SELECT
    p.model_name,
    s.name AS sport,
    g.game_date,
    ht.name AS home_team,
    at.name AS away_team,
    p.market,
    p.predicted_prob,
    p.edge,
    p.outcome,
    p.pnl,
    g.home_score,
    g.away_score
FROM predictions p
JOIN games g ON p.game_id = g.game_id
JOIN sports s ON g.sport_id = s.sport_id
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id;
