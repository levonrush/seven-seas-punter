from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd

MARKET_TYPE_LABELS = {
    "WIN": "WIN (pick the winner)",
    "PLACE": "PLACE (finish in the placings)",
    "EACH_WAY": "EACH_WAY (win + place combined)",
    "FORECAST": "FORECAST/EXACTA (first two in exact order)",
    "REVERSE_FORECAST": "REVERSE_FORECAST (first two in any order)",
    "EXACTA": "EXACTA (first two in exact order)",
    "QUINELLA": "QUINELLA (first two in any order)",
    "TRIFECTA": "TRIFECTA (first three in exact order)",
    "TRICAST": "TRICAST (first three in exact order)",
    "DUEL": "DUEL (head-to-head match bet)",
    "MATCH_BET": "MATCH_BET (head-to-head match bet)",
}

BET_TYPE_LABELS = {
    "BACK": "BACK (you win if the selection lands)",
    "LAY": "LAY (you win if the selection loses)",
}

RUNNER_NAME_PATTERN = re.compile(r"^\s*(\d+)\.\s*(.+)$")
REVERSE_COMBO_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*/\s*(\d+)\s*-\s*(\d+)\s*$")
COMBO_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
NUMERIC_MARKET_PATTERN = re.compile(r"^\s*\d+\s*$")


def _truncate_text(value: Any, max_len: int) -> str | None:
    """Clamp long strings to a fixed width so console tables remain aligned."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def describe_market_type(market_type: str | None) -> str:
    """Translate a Betfair market type into a plain-English label for preview tables."""
    if not market_type:
        return "UNKNOWN (market type missing)"
    market_type = str(market_type).strip().upper()
    if not market_type:
        return "UNKNOWN (market type missing)"
    if NUMERIC_MARKET_PATTERN.match(market_type):
        if market_type == "7":
            return "UNKNOWN (event type id 7)"
        return "UNKNOWN (numeric market id)"
    return MARKET_TYPE_LABELS.get(market_type, f"{market_type} (see Betfair market type)")


def describe_bet_type(bet_type: str | None) -> str:
    """Translate a bet direction into a plain-English label for preview tables."""
    if not bet_type:
        return "UNKNOWN (bet type missing)"
    bet_type = str(bet_type).strip().upper()
    if not bet_type:
        return "UNKNOWN (bet type missing)"
    return BET_TYPE_LABELS.get(bet_type, bet_type)


def infer_bet_guidance(
    market_type: str | None,
    selection_kind: str | None,
    bet_type: str | None,
) -> str:
    """Return plain-English betting guidance based on market, selection, and bet type."""
    market_raw = str(market_type or "").strip().upper()
    selection_kind = str(selection_kind or "").strip().upper()
    bet_type = str(bet_type or "BACK").strip().upper()

    exotic_types = {"FORECAST", "REVERSE_FORECAST", "EXACTA", "QUINELLA", "TRIFECTA", "TRICAST"}
    if selection_kind == "COMBO":
        if market_raw in {"REVERSE_FORECAST", "QUINELLA"}:
            return "Exotic combo: runners can finish in either order."
        if market_raw in {"FORECAST", "EXACTA"}:
            return "Exotic combo: runners must finish in exact order."
        if market_raw in {"TRIFECTA", "TRICAST"}:
            return "Exotic combo: top three must finish in exact order."
        if bet_type == "LAY":
            return "Lay this exotic combo (it must lose to win)."
        return "Exotic combo selection; not a WIN bet."

    if market_raw == "WIN":
        return "Lay this runner to lose." if bet_type == "LAY" else "Back this runner to WIN."
    if market_raw == "PLACE":
        return "Lay this runner to miss the placings." if bet_type == "LAY" else "Back this runner to PLACE."
    if market_raw == "EACH_WAY":
        return "Each-way: win + place combined (two bets)."
    if market_raw in exotic_types:
        return "Exotic market; usually needs combo selections."
    if market_raw in {"DUEL", "MATCH_BET"}:
        return "Head-to-head market; pick the runner to beat the other."
    if NUMERIC_MARKET_PATTERN.match(market_raw):
        return "Market type missing; check Betfair market type before betting."
    return "Market type unknown; check exchange market before betting."


def explain_selection(selection: str | None, market_type: str | None) -> Dict[str, Any]:
    """Explain a selection string so previews are readable for new bettors."""
    if selection is None:
        return {
            "selection_kind": "UNKNOWN",
            "selection_label": None,
            "runner_number": None,
            "selection_notes": "Selection name missing.",
        }
    selection_str = str(selection).strip()
    if not selection_str:
        return {
            "selection_kind": "UNKNOWN",
            "selection_label": None,
            "runner_number": None,
            "selection_notes": "Selection name missing.",
        }

    reverse_match = REVERSE_COMBO_PATTERN.match(selection_str)
    if reverse_match:
        first, second, third, fourth = reverse_match.groups()
        if first == fourth and second == third:
            note = f"Combo of runners {first} and {second} in either order."
        else:
            note = f"Combo order options: {first}-{second} / {third}-{fourth}."
        return {
            "selection_kind": "COMBO",
            "selection_label": selection_str,
            "runner_number": None,
            "selection_notes": note,
        }

    combo_match = COMBO_PATTERN.match(selection_str)
    if combo_match:
        first, second = combo_match.groups()
        note = f"Combo in exact order: runner {first} then {second}."
        return {
            "selection_kind": "COMBO",
            "selection_label": selection_str,
            "runner_number": None,
            "selection_notes": note,
        }

    runner_match = RUNNER_NAME_PATTERN.match(selection_str)
    if runner_match:
        runner_number, runner_name = runner_match.groups()
        clean_name = runner_name.strip()
        note = f"Runner #{runner_number} {clean_name}."
        return {
            "selection_kind": "RUNNER",
            "selection_label": clean_name,
            "runner_number": int(runner_number),
            "selection_notes": note,
        }

    label = selection_str
    market_hint = describe_market_type(market_type)
    note = f"Runner name (number unknown). Market: {market_hint}."
    return {
        "selection_kind": "RUNNER",
        "selection_label": label,
        "runner_number": None,
        "selection_notes": note,
    }


def annotate_preview_frame(
    df: pd.DataFrame,
    selection_col: str = "selection",
    market_type_col: str = "market_type",
    bet_type_col: str | None = None,
    default_bet_type: str | None = None,
) -> pd.DataFrame:
    """Add plain-English market/bet/selection explanations to preview tables."""
    if df.empty:
        return df
    preview = df.copy()
    if market_type_col not in preview.columns:
        preview[market_type_col] = None
    if selection_col not in preview.columns:
        preview[selection_col] = None
    if bet_type_col is None and default_bet_type is not None:
        preview["bet_type"] = default_bet_type
        bet_type_col = "bet_type"

    preview["market_type_label"] = preview[market_type_col].apply(describe_market_type)
    if bet_type_col and bet_type_col in preview.columns:
        preview["bet_type_label"] = preview[bet_type_col].apply(describe_bet_type)

    selection_info = preview.apply(
        lambda row: explain_selection(row.get(selection_col), row.get(market_type_col)),
        axis=1,
        result_type="expand",
    )
    for col in selection_info.columns:
        preview[col] = selection_info[col]
    preview["bet_guidance"] = preview.apply(
        lambda row: infer_bet_guidance(
            row.get(market_type_col),
            row.get("selection_kind"),
            row.get(bet_type_col) if bet_type_col else None,
        ),
        axis=1,
    )
    preview["market_type_label_full"] = preview["market_type_label"]
    preview["selection_notes_full"] = preview["selection_notes"]
    preview["bet_guidance_full"] = preview["bet_guidance"]
    preview["market_type_label"] = preview["market_type_label"].apply(lambda v: _truncate_text(v, 36))
    preview["selection_notes"] = preview["selection_notes"].apply(lambda v: _truncate_text(v, 70))
    preview["bet_guidance"] = preview["bet_guidance"].apply(lambda v: _truncate_text(v, 60))
    return preview


def preview_legend_lines() -> List[str]:
    """Return short legend lines so preview output is self-explanatory."""
    return [
        "Legend: market_type_label explains the market (WIN/PLACE/EXOTIC).",
        "Legend: selection_notes explains runner numbers or combo selections.",
        "Legend: bet_guidance suggests how to interpret the selection.",
        "Legend: expected_value is expected profit per $1 after commission; edge_pct is edge vs market.",
        "Legend: *_full columns keep the un-truncated text for exports.",
    ]
