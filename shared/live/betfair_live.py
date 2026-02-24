from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from shared.backtest.engine import (
    DEFAULT_LONGSHOT_PRICE_FLOOR,
    DEFAULT_LONGSHOT_PROBABILITY_CAP,
    DEFAULT_MARKET_ANCHOR_WEIGHT,
    DEFAULT_PROBABILITY_CLIP_EPSILON,
    compute_expected_value,
    sanitize_probability_for_decision,
)
from shared.betfair.client import BetfairClient
from shared.features.builder import TICK_LADDER, build_features_from_store
from shared.model.training import load_model_and_calibrator, predict_probabilities
from shared.utils.market_types import api_market_type_codes, market_type_matches, tokens_to_filter_set
from shared.utils.progress import log

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guard
    yaml = None


DEFAULT_LIVE_CONFIG: Dict[str, Any] = {
    "dry_run": True,
    "poll_interval_seconds": 10,
    "max_iterations": None,
    "model": {
        "cutoff_minutes": 10,
        "model_path": "artifacts/model_cutoff_10.joblib",
        "calibrator_path": "artifacts/calibrator_cutoff_10.joblib",
    },
    "markets": {
        "event_type_id": "7",
        "market_countries": ["AU"],
        "market_type_codes": ["ALL"],
        "start_in_minutes": 5,
        "horizon_minutes": 240,
        "max_results": 200,
        "sort": "FIRST_TO_START",
    },
    "strategy": {
        "stake_per_bet": 1.0,
        "min_ev": 0.02,
        "min_edge": 0.1,
        "min_prob": None,
        "commission": 0.05,
        "per_market_limit": 1,
        "persistence_type": "LAPSE",
    },
    "safety": {
        "max_stake_per_market": 5.0,
        "max_daily_exposure": 50.0,
        "ignore_within_minutes": 5,
    },
    "state": {
        "exposure_path": "artifacts/live_exposure_state.json",
        "decision_log_path": "artifacts/live_decisions.csv",
    },
}


class InMemoryFeatureStore:
    """Provide the store interface expected by feature builder without writing transient live data to DuckDB."""

    def __init__(self, snapshots: list[dict], markets: list[dict]) -> None:
        """Capture current snapshot and market state for one live iteration."""
        self._snapshots = pd.DataFrame(snapshots)
        self._markets = pd.DataFrame(markets)

    def load_snapshots(self) -> pd.DataFrame:
        """Return in-memory snapshots so feature generation reuses the existing pipeline logic."""
        return self._snapshots.copy()

    def load_markets(self) -> pd.DataFrame:
        """Return in-memory market metadata, including market_type used by model bucketing."""
        return self._markets.copy()

    def load_results(self) -> pd.DataFrame:
        """Return an empty frame because live inference has no settled outcomes yet."""
        return pd.DataFrame(columns=["market_id", "selection_id", "win_flag"])


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge config dictionaries recursively so users only set keys they need to change."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_live_config(config_path: str) -> dict:
    """Load and validate live YAML config so execution behavior is explicit and reproducible."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for live config parsing. Install `pyyaml`.")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Live config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Live config must be a YAML mapping at the top level.")
    config = _deep_merge(DEFAULT_LIVE_CONFIG, payload)
    _validate_live_config(config)
    return config


def write_live_config_template(path: str, overwrite: bool = False) -> Path:
    """Write a starter live config so operators can bootstrap safely from known defaults."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for live config parsing. Install `pyyaml`.")
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Live config already exists at {target}. Use overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(DEFAULT_LIVE_CONFIG, sort_keys=False),
        encoding="utf-8",
    )
    return target


def _validate_live_config(config: dict) -> None:
    """Fail early on invalid risk settings so no execution loop starts with unsafe parameters."""
    cutoff = int(config["model"]["cutoff_minutes"])
    if cutoff not in {60, 30, 10, 5, 2, 1}:
        raise ValueError("model.cutoff_minutes must be one of 60, 30, 10, 5, 2, 1.")
    stake_per_bet = float(config["strategy"]["stake_per_bet"])
    max_stake_market = float(config["safety"]["max_stake_per_market"])
    max_daily = float(config["safety"]["max_daily_exposure"])
    if stake_per_bet <= 0:
        raise ValueError("strategy.stake_per_bet must be > 0.")
    if max_stake_market <= 0 or max_daily <= 0:
        raise ValueError("Safety caps must be > 0.")
    if stake_per_bet > max_stake_market:
        raise ValueError("strategy.stake_per_bet cannot exceed safety.max_stake_per_market.")
    if float(config["poll_interval_seconds"]) <= 0:
        raise ValueError("poll_interval_seconds must be > 0.")


def _utc_now() -> dt.datetime:
    """Return timezone-aware UTC now so all time comparisons are consistent."""
    return dt.datetime.now(dt.timezone.utc)


def build_market_filter(config: dict, now_utc: dt.datetime) -> dict:
    """Build a tight market filter so polling avoids excessive payloads and API weight."""
    market_cfg = config["markets"]
    start_in_minutes = float(market_cfg.get("start_in_minutes", 0))
    horizon_minutes = float(market_cfg.get("horizon_minutes", 240))
    from_ts = now_utc + dt.timedelta(minutes=start_in_minutes)
    to_ts = now_utc + dt.timedelta(minutes=horizon_minutes)
    market_filter = {
        "event_type_ids": [str(market_cfg.get("event_type_id", "7"))],
        "market_countries": list(market_cfg.get("market_countries", ["AU"])),
        "market_start_time": {
            "from": from_ts.isoformat(),
            "to": to_ts.isoformat(),
        },
    }
    market_type_codes = api_market_type_codes(market_cfg.get("market_type_codes", ["ALL"]))
    if market_type_codes:
        market_filter["market_type_codes"] = market_type_codes
    return market_filter


def discover_markets(client: BetfairClient, config: dict, now_utc: dt.datetime) -> list[dict]:
    """Fetch markets for live polling from config-driven filters."""
    market_cfg = config["markets"]
    market_filter = build_market_filter(config, now_utc)
    markets = client.list_market_catalogue(
        market_filter=market_filter,
        market_projection=["MARKET_START_TIME", "EVENT", "RUNNER_METADATA", "MARKET_DESCRIPTION"],
        max_results=int(market_cfg.get("max_results", 200)),
        sort=str(market_cfg.get("sort", "FIRST_TO_START")),
        country_code=(market_cfg.get("market_countries") or [None])[0],
        event_type="horse_racing",
    )
    selected = tokens_to_filter_set(market_cfg.get("market_type_codes", ["ALL"]))
    if selected is None:
        return markets
    filtered = [market for market in markets if market_type_matches(market.get("market_type"), selected)]
    log(f"Live: market-type filter kept {len(filtered)}/{len(markets)} markets.")
    return filtered


def _market_lookup(markets: list[dict]) -> dict:
    """Build a market metadata lookup to fill occasional gaps in market book payloads."""
    lookup = {}
    for market in markets:
        lookup[market["market_id"]] = market
    return lookup


def enrich_snapshots_with_market_data(snapshots: list[dict], markets: list[dict]) -> list[dict]:
    """Backfill market-level fields from catalogue so feature generation has stable inputs."""
    lookup = _market_lookup(markets)
    enriched = []
    for row in snapshots:
        market = lookup.get(row.get("market_id"), {})
        enriched_row = dict(row)
        if enriched_row.get("race_start_time") is None:
            enriched_row["race_start_time"] = market.get("race_start_time")
        if enriched_row.get("venue") is None:
            enriched_row["venue"] = market.get("venue")
        if enriched_row.get("race_name") is None:
            enriched_row["race_name"] = market.get("race_name")
        enriched.append(enriched_row)
    return enriched


def prune_snapshot_history(history: list[dict], now_utc: dt.datetime) -> list[dict]:
    """Drop stale rows so in-memory history stays bounded while retaining useful pre-jump context."""
    if not history:
        return history
    minimum_snapshot_time = now_utc - dt.timedelta(hours=24)
    keep = []
    for row in history:
        snapshot_time = pd.to_datetime(row.get("snapshot_time"), utc=True, errors="coerce")
        race_start = pd.to_datetime(row.get("race_start_time"), utc=True, errors="coerce")
        if pd.notna(snapshot_time) and snapshot_time >= minimum_snapshot_time:
            keep.append(row)
            continue
        if pd.notna(race_start) and race_start >= now_utc - dt.timedelta(hours=1):
            keep.append(row)
    return keep


def _select_price(row: pd.Series, cutoff_minutes: int) -> Optional[float]:
    """Pick cutoff price with fallback so execution can proceed when a single offset is missing."""
    candidate = row.get(f"back_price_t{cutoff_minutes}")
    if pd.notnull(candidate):
        return float(candidate)
    for offset in [60, 30, 10, 5, 2, 1]:
        fallback = row.get(f"back_price_t{offset}")
        if pd.notnull(fallback):
            return float(fallback)
    return None


def build_candidate_bets(
    features: pd.DataFrame,
    probs: pd.Series,
    config: dict,
    now_utc: dt.datetime,
) -> pd.DataFrame:
    """Rank value candidates while applying pre-trade filters before safety and execution checks."""
    if features.empty or probs.empty:
        return pd.DataFrame()
    strategy_cfg = config["strategy"]
    safety_cfg = config["safety"]
    cutoff = int(config["model"]["cutoff_minutes"])
    min_prob = strategy_cfg.get("min_prob")

    scored = features.copy()
    scored["p_hat"] = probs.reindex(scored.index).values
    scored["price"] = scored.apply(lambda row: _select_price(row, cutoff), axis=1)
    scored = scored.dropna(subset=["p_hat", "price"])
    if scored.empty:
        return pd.DataFrame()
    scored = scored[scored["price"] > 0].copy()
    if scored.empty:
        return pd.DataFrame()

    scored["race_start_time"] = pd.to_datetime(scored["race_start_time"], utc=True, errors="coerce")
    scored = scored.dropna(subset=["race_start_time"])
    if scored.empty:
        return pd.DataFrame()

    scored["minutes_to_start"] = (
        scored["race_start_time"] - now_utc
    ).dt.total_seconds() / 60.0
    scored = scored[scored["minutes_to_start"] >= float(safety_cfg["ignore_within_minutes"])].copy()
    if scored.empty:
        return pd.DataFrame()

    if min_prob is not None:
        scored = scored[scored["p_hat"] >= float(min_prob)].copy()
        if scored.empty:
            return pd.DataFrame()

    scored["p_hat"] = scored.apply(
        lambda row: sanitize_probability_for_decision(
            prob=float(row["p_hat"]),
            price=float(row["price"]),
            clip_epsilon=DEFAULT_PROBABILITY_CLIP_EPSILON,
            market_anchor_weight=DEFAULT_MARKET_ANCHOR_WEIGHT,
            longshot_price_floor=DEFAULT_LONGSHOT_PRICE_FLOOR,
            longshot_probability_cap=DEFAULT_LONGSHOT_PROBABILITY_CAP,
        ),
        axis=1,
    )
    scored["implied_prob"] = 1.0 / scored["price"]
    scored["edge_pct"] = (scored["p_hat"] - scored["implied_prob"]) / scored["implied_prob"]
    scored["expected_value"] = scored.apply(
        lambda row: compute_expected_value(
            prob=float(row["p_hat"]),
            price=float(row["price"]),
            commission=float(strategy_cfg.get("commission", 0.05)),
        ),
        axis=1,
    )
    scored = scored[scored["edge_pct"] >= float(strategy_cfg["min_edge"])]
    scored = scored[scored["expected_value"] >= float(strategy_cfg["min_ev"])]
    if scored.empty:
        return pd.DataFrame()

    per_market_limit = int(strategy_cfg.get("per_market_limit", 1))
    scored = scored.sort_values("expected_value", ascending=False)
    if per_market_limit > 0:
        scored = scored.groupby("market_id").head(per_market_limit)
    scored = scored.sort_values("expected_value", ascending=False).reset_index(drop=True)
    scored["stake"] = float(strategy_cfg["stake_per_bet"])
    return scored


def _default_exposure_state(today: dt.date) -> dict:
    """Create a day-scoped exposure state so caps apply consistently across polling iterations."""
    return {
        "date": today.isoformat(),
        "total_stake": 0.0,
        "market_stake": {},
    }


def load_exposure_state(path: str, today: dt.date) -> dict:
    """Load persisted day exposure so safety caps survive restarts during the trading day."""
    state_path = Path(path)
    if not state_path.exists():
        return _default_exposure_state(today)
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(f"Live: exposure state file is invalid JSON ({state_path}); resetting.")
        return _default_exposure_state(today)
    if payload.get("date") != today.isoformat():
        return _default_exposure_state(today)
    payload.setdefault("market_stake", {})
    payload.setdefault("total_stake", 0.0)
    return payload


def save_exposure_state(path: str, state: dict) -> None:
    """Persist exposure state so process restarts do not bypass daily stake limits."""
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def apply_safety_gates(
    candidates: pd.DataFrame,
    config: dict,
    exposure_state: dict,
    now_utc: Optional[dt.datetime] = None,
) -> list[dict]:
    """Apply hard stake caps before execution to guarantee safe rejection on limit breaches."""
    if candidates.empty:
        return []
    now_utc = now_utc or _utc_now()
    today = now_utc.date()
    if exposure_state.get("date") != today.isoformat():
        # Reset daily exposure automatically at UTC day boundary so caps remain day-scoped.
        exposure_state.clear()
        exposure_state.update(_default_exposure_state(today))
    strategy_cfg = config["strategy"]
    safety_cfg = config["safety"]
    dry_run = bool(config.get("dry_run", True))

    decisions = []
    market_exposure = dict(exposure_state.get("market_stake", {}))
    total_exposure = float(exposure_state.get("total_stake", 0.0))
    max_market = float(safety_cfg["max_stake_per_market"])
    max_daily = float(safety_cfg["max_daily_exposure"])
    default_stake = float(strategy_cfg["stake_per_bet"])

    for _, row in candidates.iterrows():
        market_id = str(row["market_id"])
        selection_id = int(row["selection_id"])
        stake = float(row.get("stake", default_stake))
        market_total = float(market_exposure.get(market_id, 0.0))

        reason = None
        if stake > max_market:
            reason = "stake_exceeds_market_cap"
        elif market_total + stake > max_market:
            reason = "market_cap_exceeded"
        elif total_exposure + stake > max_daily:
            reason = "daily_cap_exceeded"

        decision = {
            "decision_time": _utc_now().isoformat(),
            "market_id": market_id,
            "selection_id": selection_id,
            "price": float(row["price"]),
            "stake": stake,
            "p_hat": float(row["p_hat"]),
            "edge_pct": float(row["edge_pct"]),
            "expected_value": float(row["expected_value"]),
            "minutes_to_start": float(row["minutes_to_start"]),
            "reason": reason,
        }
        if reason is not None:
            decision["action"] = "REJECT"
            decision["approved"] = False
            decisions.append(decision)
            continue

        decision["action"] = "DRY_RUN" if dry_run else "PLACE"
        decision["approved"] = True
        decisions.append(decision)
        market_exposure[market_id] = market_total + stake
        total_exposure += stake

    exposure_state["market_stake"] = market_exposure
    exposure_state["total_stake"] = total_exposure
    return decisions


def _round_to_valid_odds(price: float) -> float:
    """Round price to Betfair odds ladder so orders are valid and deterministic."""
    clipped = max(1.01, min(1000.0, float(price)))
    for lower, upper, step in TICK_LADDER:
        if clipped <= upper:
            step_count = round((clipped - lower) / step)
            rounded = lower + (step_count * step)
            return round(rounded, 2)
    return round(clipped, 2)


def execute_decisions(client: BetfairClient, decisions: list[dict], config: dict) -> list[dict]:
    """Execute approved decisions, or simulate them in dry-run mode, while capturing API outcomes."""
    strategy_cfg = config["strategy"]
    executed = []
    for decision in decisions:
        row = dict(decision)
        if row["action"] == "REJECT":
            row["execution_status"] = "REJECTED"
            executed.append(row)
            continue
        if row["action"] == "DRY_RUN":
            row["execution_status"] = "SIMULATED"
            row["order_response"] = {"status": "DRY_RUN"}
            executed.append(row)
            continue

        instruction = {
            "selectionId": int(row["selection_id"]),
            "side": "BACK",
            "orderType": "LIMIT",
            "limitOrder": {
                "size": float(row["stake"]),
                "price": _round_to_valid_odds(float(row["price"])),
                "persistenceType": str(strategy_cfg.get("persistence_type", "LAPSE")),
            },
        }
        customer_ref = (
            f"live-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
            f"{row['market_id']}-{row['selection_id']}"
        )
        try:
            response = client.place_orders(
                market_id=row["market_id"],
                instructions=[instruction],
                customer_ref=customer_ref,
            )
            row["order_response"] = response
            row["execution_status"] = response.get("status") or "UNKNOWN"
        except Exception as exc:  # pragma: no cover - external API behavior
            row["order_response"] = {"status": "EXCEPTION", "error": str(exc)}
            row["execution_status"] = "EXCEPTION"
            row["reason"] = "order_api_exception"
        executed.append(row)
    return executed


def append_decision_log(path: str, decisions: list[dict]) -> None:
    """Append decision audit rows so dry-run and live behavior are reviewable after execution."""
    if not decisions:
        return
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(decisions)
    if log_path.exists():
        existing = pd.read_csv(log_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(log_path, index=False)


def log_decisions(decisions: list[dict]) -> None:
    """Write concise decision logs so operators can see why bets were placed or rejected."""
    if not decisions:
        log("Live: no candidate bets this iteration.")
        return
    for decision in decisions:
        log(
            "Live decision: "
            f"market={decision['market_id']} selection={decision['selection_id']} "
            f"action={decision['action']} status={decision.get('execution_status')} "
            f"stake={decision['stake']:.2f} edge={decision['edge_pct']:.3f} "
            f"ev={decision['expected_value']:.4f} reason={decision.get('reason')}"
        )


def run_live_iteration(
    client: BetfairClient,
    model: Any,
    calibrator: Any,
    config: dict,
    runtime_state: dict,
    now_utc: Optional[dt.datetime] = None,
) -> list[dict]:
    """Run one polling cycle from market discovery through decision logging."""
    now_utc = now_utc or _utc_now()
    markets = discover_markets(client, config, now_utc)
    if not markets:
        log("Live: no markets returned by discovery filter.")
        return []

    market_ids = [market["market_id"] for market in markets]
    books = client.fetch_market_books(market_ids)
    snapshots = enrich_snapshots_with_market_data(books, markets)
    history = runtime_state.setdefault("snapshot_history", [])
    history.extend(snapshots)
    runtime_state["snapshot_history"] = prune_snapshot_history(history, now_utc)

    store = InMemoryFeatureStore(runtime_state["snapshot_history"], markets)
    cutoff = int(config["model"]["cutoff_minutes"])
    features = build_features_from_store(store, cutoff_minutes=cutoff)
    if features.empty:
        log("Live: no feature rows available from current snapshot history.")
        return []
    features = features[features["market_id"].isin(market_ids)].copy()
    if features.empty:
        log("Live: no feature rows match discovered markets in this iteration.")
        return []

    probs = predict_probabilities(model, calibrator, features)
    candidates = build_candidate_bets(features, probs, config, now_utc)
    decisions = apply_safety_gates(candidates, config, runtime_state["exposure_state"], now_utc=now_utc)
    executed = execute_decisions(client, decisions, config)
    log_decisions(executed)
    return executed


def run_live_loop(config_path: str, overrides: Optional[dict] = None) -> None:
    """Run the polling loop so the trained model can drive dry-run or live execution continuously."""
    config = load_live_config(config_path)
    if overrides:
        config = _deep_merge(config, overrides)
        _validate_live_config(config)
    client = BetfairClient()
    if not config.get("dry_run", True) and client.dry_run:
        raise RuntimeError("Live mode requested but Betfair authentication failed; refusing to place bets.")

    model_cfg = config["model"]
    model, calibrator = load_model_and_calibrator(
        model_cfg["model_path"],
        model_cfg.get("calibrator_path"),
    )

    today = _utc_now().date()
    state_cfg = config["state"]
    exposure_state = load_exposure_state(state_cfg["exposure_path"], today)
    runtime_state = {
        "snapshot_history": [],
        "exposure_state": exposure_state,
    }

    poll_interval = float(config["poll_interval_seconds"])
    max_iterations = config.get("max_iterations")
    iteration = 0
    while True:
        decisions = run_live_iteration(
            client=client,
            model=model,
            calibrator=calibrator,
            config=config,
            runtime_state=runtime_state,
        )
        append_decision_log(state_cfg["decision_log_path"], decisions)
        save_exposure_state(state_cfg["exposure_path"], runtime_state["exposure_state"])

        iteration += 1
        if max_iterations is not None and iteration >= int(max_iterations):
            log("Live: reached max_iterations; exiting.")
            break
        time.sleep(poll_interval)
