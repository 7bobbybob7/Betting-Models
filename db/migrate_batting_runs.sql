-- Add runs scored column to mlb_batting_game. Required for the HRR (Hits+Runs+RBIs)
-- prop label. Default NULL — historical rows backfilled by re-running the boxscores
-- scraper; new rows captured by the updated scraper going forward.

ALTER TABLE mlb_batting_game
    ADD COLUMN IF NOT EXISTS runs INTEGER;

CREATE INDEX IF NOT EXISTS idx_mlb_batting_game_runs
    ON mlb_batting_game (player_id, game_id)
    WHERE runs IS NOT NULL;
