-- mlb_pitch_extras — Statcast columns not stored in mlb_pitches, backfilled 2024+.
-- Companion table (1:1 join on game_id/at_bat_number/pitch_number) so the 7.9M-row
-- mlb_pitches table is untouched. Populated by scrapers/mlb/backfill_pitch_extras.py
-- and topped up daily by the daily_pipeline.yml cron.
--
-- Feeds Leg 1 v2 / Attack 3 features (models/mlb/advanced_profile_features.py):
--   catcher_mlbam           -> opposing-catcher framing
--   hc_x, hc_y              -> true pull% / spray
--   bat_speed, swing_length,
--   attack_angle            -> bat-tracking swing profile
-- arm_angle / attack_direction / swing_path_tilt captured for future use.

CREATE TABLE IF NOT EXISTS mlb_pitch_extras (
    game_id           INT NOT NULL,
    at_bat_number     INT NOT NULL,
    pitch_number      INT NOT NULL,
    batter_id         INT,
    catcher_mlbam     INT,
    hc_x              NUMERIC,
    hc_y              NUMERIC,
    bat_speed         NUMERIC,
    swing_length      NUMERIC,
    arm_angle         NUMERIC,
    attack_angle      NUMERIC,
    attack_direction  NUMERIC,
    swing_path_tilt   NUMERIC,
    PRIMARY KEY (game_id, at_bat_number, pitch_number)
);

CREATE INDEX IF NOT EXISTS idx_pitch_extras_batter ON mlb_pitch_extras (batter_id);
