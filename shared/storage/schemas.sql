CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    venue TEXT,
    race_start_time TIMESTAMP,
    race_name TEXT,
    market_type TEXT,
    country_code TEXT,
    event_type TEXT
);

CREATE TABLE IF NOT EXISTS runners (
    market_id TEXT,
    selection_id BIGINT,
    runner_name TEXT,
    stall_draw INTEGER,
    PRIMARY KEY (market_id, selection_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    market_id TEXT,
    selection_id BIGINT,
    snapshot_time TIMESTAMP,
    seconds_to_start INTEGER,
    best_back_price DOUBLE,
    best_back_size DOUBLE,
    best_lay_price DOUBLE,
    best_lay_size DOUBLE,
    last_traded_price DOUBLE,
    total_matched DOUBLE,
    runner_status TEXT,
    venue TEXT,
    race_start_time TIMESTAMP,
    race_name TEXT
);

CREATE TABLE IF NOT EXISTS results (
    market_id TEXT,
    selection_id BIGINT,
    win_flag BOOLEAN,
    bsp DOUBLE,
    place_position INTEGER,
    PRIMARY KEY (market_id, selection_id)
);

CREATE TABLE IF NOT EXISTS bets (
    market_id TEXT,
    selection_id BIGINT,
    run_id TEXT,
    bet_time TIMESTAMP,
    stake DOUBLE,
    price DOUBLE,
    bet_type TEXT,
    expected_value DOUBLE,
    commission_rate DOUBLE,
    result_profit DOUBLE
);

CREATE TABLE IF NOT EXISTS model_runs (
    created_at TIMESTAMP DEFAULT now(),
    run_id TEXT,
    model_path TEXT,
    calibrator_path TEXT,
    cutoff_minutes INTEGER,
    notes TEXT,
    metrics JSON
);

CREATE TABLE IF NOT EXISTS oof_predictions (
    created_at TIMESTAMP DEFAULT now(),
    run_id TEXT,
    cutoff_minutes INTEGER,
    market_id TEXT,
    selection_id BIGINT,
    p_hat DOUBLE
);

ALTER TABLE markets ADD COLUMN IF NOT EXISTS market_type TEXT;
ALTER TABLE bets ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS run_id TEXT;
