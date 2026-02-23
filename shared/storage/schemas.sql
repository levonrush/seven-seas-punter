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

CREATE TABLE IF NOT EXISTS runner_metadata_snapshots (
    market_id TEXT,
    selection_id BIGINT,
    snapshot_time TIMESTAMP,
    race_start_time TIMESTAMP,
    seconds_to_start INTEGER,
    source TEXT,
    runner_name TEXT,
    jockey_name TEXT,
    trainer_name TEXT,
    age INTEGER,
    official_rating DOUBLE,
    adjusted_rating DOUBLE,
    days_since_last_run INTEGER,
    weight_value DOUBLE,
    weight_units TEXT,
    jockey_claim DOUBLE,
    stall_draw INTEGER,
    form_string TEXT,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS external_runner_form_runs (
    source TEXT,
    market_id TEXT,
    selection_id BIGINT,
    snapshot_time TIMESTAMP,
    race_start_time TIMESTAMP,
    seconds_to_start INTEGER,
    run_index INTEGER,
    runner_name TEXT,
    horse_name TEXT,
    jockey_name TEXT,
    trainer_name TEXT,
    track TEXT,
    surface TEXT,
    distance_m INTEGER,
    class_label TEXT,
    class_index DOUBLE,
    track_condition TEXT,
    run_date TIMESTAMP,
    run_finish_pos INTEGER,
    run_field_size INTEGER,
    run_distance_m INTEGER,
    run_surface TEXT,
    run_track TEXT,
    run_class_label TEXT,
    run_class_index DOUBLE,
    run_track_condition TEXT,
    run_sectional_time DOUBLE,
    run_speed_rating DOUBLE,
    run_weight_value DOUBLE,
    run_barrier INTEGER,
    run_jockey_name TEXT,
    run_trainer_name TEXT,
    run_won BOOLEAN,
    run_placed BOOLEAN,
    metadata JSON
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

CREATE TABLE IF NOT EXISTS tab_quotes (
    market_id TEXT,
    selection_id BIGINT,
    quote_time TIMESTAMP,
    source_channel TEXT,
    product_type TEXT,
    display_odds DOUBLE,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS tab_executions (
    market_id TEXT,
    selection_id BIGINT,
    placed_time TIMESTAMP,
    source_channel TEXT,
    product_type TEXT,
    stake DOUBLE,
    accepted_odds DOUBLE,
    was_repriced BOOLEAN,
    was_refused BOOLEAN,
    notes TEXT
);

ALTER TABLE markets ADD COLUMN IF NOT EXISTS market_type TEXT;
ALTER TABLE bets ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE model_runs ADD COLUMN IF NOT EXISTS run_id TEXT;
