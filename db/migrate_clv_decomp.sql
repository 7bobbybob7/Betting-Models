-- CLV Decomposition: add bet_book and decomposed CLV columns
-- Run this against the Supabase database before deploying the updated pipeline.

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bet_book VARCHAR(30);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS clv_model DECIMAL(8,6);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS clv_execution DECIMAL(8,6);
