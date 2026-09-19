"""Pure validation and scoring functions used by the UI and database layer."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any
from urllib.parse import urlparse

from questionnaire_definitions import (
    BACKGROUND_OPTIONS,
    COMPENSATION_TYPES,
    CONDITIONS,
    FLEET_FINAL_ITEMS,
    FLEET_FINAL_OPTIONS,
    FLEET_FINAL_SUSTAINABLE_OPTIONS,
    FLEET_POST_CONDITION_ITEMS,
    FLEET_TLX_DIMENSIONS,
    POST_CONDITION_ITEMS,
    TASK_ORDERS,
    TLX_DIMENSIONS,
)


PARTICIPANT_ID_PATTERN = re.compile(r"^P1([ABC])-(\d{3})$")
STUDY_PARTICIPANT_LETTERS = {
    "1A": "A",
    "1B": "B",
    "Fleet": "C",
}


class ValidationError(ValueError):
    """Raised when user-entered study data fail validation."""


def validate_participant_id(participant_id: str, study: str) -> str:
    normalized = participant_id.strip().upper()
    match = PARTICIPANT_ID_PATTERN.fullmatch(normalized)
    if not match:
        raise ValidationError("Participant ID must use the format P1A-001, P1B-001, or P1C-001.")
    expected_letter = STUDY_PARTICIPANT_LETTERS.get(study)
    if expected_letter is None:
        raise ValidationError("Study must be 1A, 1B, or Fleet.")
    if match.group(1) != expected_letter:
        raise ValidationError(f"Participant ID {normalized} does not match Study {study}.")
    number = int(match.group(2))
    if not 1 <= number <= 100:
        raise ValidationError("Participant number must be between 001 and 100.")
    return normalized


def validate_task_order(task_order_code: str) -> str:
    if task_order_code not in TASK_ORDERS:
        raise ValidationError("Task order must be T-O1, T-O2, or N/A.")
    return task_order_code


def validate_tlx_20(values: Mapping[str, Any]) -> dict[str, int]:
    """Validate actively confirmed integer TLX responses on the 0–20 scale."""

    required = [dimension["code_20"] for dimension in TLX_DIMENSIONS]
    missing = [code for code in required if values.get(code) is None]
    if missing:
        raise ValidationError("Every NASA-TLX slider must be actively confirmed.")
    validated: dict[str, int] = {}
    for code in required:
        value = values[code]
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) <= 20:
            raise ValidationError("NASA-TLX values must be whole numbers from 0 to 20.")
        validated[code] = int(value)
    return validated


def validate_fleet_tlx_20(values: Mapping[str, Any]) -> dict[str, int]:
    """Validate actively confirmed Fleet NASA-TLX responses on the 0-20 scale."""

    required = [dimension["code"] for dimension in FLEET_TLX_DIMENSIONS]
    missing = [code for code in required if values.get(code) is None]
    if missing:
        raise ValidationError("Every NASA-TLX slider must be actively confirmed.")
    validated: dict[str, int] = {}
    for code in required:
        value = values[code]
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) <= 20:
            raise ValidationError("Fleet NASA-TLX values must be whole numbers from 0 to 20.")
        validated[code] = int(value)
    return validated


def convert_tlx_to_100(values: Mapping[str, Any]) -> dict[str, int]:
    validated = validate_tlx_20(values)
    return {
        dimension["code_100"]: validated[dimension["code_20"]] * 5
        for dimension in TLX_DIMENSIONS
    }


def calculate_raw_tlx(values: Mapping[str, Any]) -> float:
    """Return the unweighted Raw NASA-TLX mean on the original 0–20 scale."""

    validated = validate_tlx_20(values)
    return sum(validated.values()) / len(validated)


def calculate_raw_tlx_scores(values: Mapping[str, Any]) -> tuple[float, float]:
    raw_20 = calculate_raw_tlx(values)
    return raw_20, raw_20 * 5


def calculate_spatial_understanding(a2: Any, a3: Any) -> float:
    try:
        values = (float(a2), float(a3))
    except (TypeError, ValueError) as exc:
        raise ValidationError("A2 and A3 must be numeric.") from exc
    if any(value not in range(1, 8) for value in values):
        raise ValidationError("A2 and A3 must be agreement ratings from 1 to 7.")
    return sum(values) / 2


def validate_ranking(ranking: Sequence[str], expected_codes: Sequence[str]) -> list[str]:
    ranking_list = list(ranking)
    expected = list(expected_codes)
    if len(ranking_list) != len(expected) or len(set(ranking_list)) != len(expected):
        raise ValidationError("The ranking must contain each condition exactly once.")
    if set(ranking_list) != set(expected):
        raise ValidationError("The ranking contains an invalid condition.")
    return ranking_list


def validate_background(data: Mapping[str, Any]) -> None:
    try:
        age = int(data.get("age"))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Age is required and must be a whole number.") from exc
    if not 18 <= age <= 100:
        raise ValidationError("Age must be between 18 and 100.")

    for field in (
        "handedness",
        "robotics_experience",
        "vr_experience_last_12_months",
        "gaming_controller_experience_last_12_months",
        "teleoperation_experience",
    ):
        value = data.get(field)
        if value not in BACKGROUND_OPTIONS[field]:
            raise ValidationError(f"A valid response for {field.replace('_', ' ')} is required.")

    choices = list(data.get("kinesthetic_experience") or [])
    if not choices or any(choice not in BACKGROUND_OPTIONS["kinesthetic_experience"] for choice in choices):
        raise ValidationError("Select at least one valid previous kinesthetic-controller response.")
    if "None of the above" in choices and len(choices) > 1:
        raise ValidationError("‘None of the above’ cannot be combined with another kinesthetic response.")

    susceptibility = data.get("motion_sickness_susceptibility")
    if susceptibility not in range(1, 8):
        raise ValidationError("Motion-sickness susceptibility must be rated from 1 to 7.")


def validate_understanding_check(data: Mapping[str, Any]) -> None:
    safety_fields = (
        "risk_indicator_correct",
        "unattended_robots_correct",
        "multiple_paused_correct",
        "release_decision_correct",
        "ready_state_correct",
        "safety_response_correct",
    )
    if not all(data.get(field) is True for field in safety_fields):
        raise ValidationError("Every safety-critical understanding item must be marked correct.")
    if data.get("retraining_required") not in (True, False):
        raise ValidationError("Record whether retraining was required.")
    if data.get("successful_practice_intervention") is not True:
        raise ValidationError("A successful practice intervention is required before proceeding.")
    try:
        count = int(data.get("practice_intervention_count"))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Practice intervention count must be a whole number.") from exc
    if count < 1:
        raise ValidationError("Practice intervention count must be at least 1.")


def validate_condition_responses(
    study: str, agreement: Mapping[str, Any], tlx: Mapping[str, Any]
) -> None:
    if study == "Fleet":
        validate_fleet_condition_responses(agreement, tlx)
        return
    try:
        required_codes = [item["code"] for item in POST_CONDITION_ITEMS[study]]
    except KeyError as exc:
        raise ValidationError("Study must be 1A, 1B, or Fleet.") from exc
    for code in required_codes:
        if agreement.get(code) not in range(1, 8):
            raise ValidationError(f"{code} is required and must be rated from 1 to 7.")
    calculate_raw_tlx(tlx)


def validate_fleet_condition_responses(
    agreement: Mapping[str, Any], tlx: Mapping[str, Any], fleet_size: int | None = None
) -> None:
    for item in FLEET_POST_CONDITION_ITEMS:
        code = item["code"]
        if fleet_size in item.get("na_for_fleet_sizes", []):
            continue
        if agreement.get(code) not in range(1, 8):
            raise ValidationError(f"{code} is required and must be rated from 1 to 7.")
    validate_fleet_tlx_20(tlx)


def validate_compensation(compensation_type: str, sona_code: str | None = None) -> tuple[str, str | None]:
    if compensation_type not in COMPENSATION_TYPES:
        raise ValidationError("Choose either €15 payment or SONA / study-pool credit.")
    if compensation_type == "sona_credit":
        normalized = (sona_code or "").strip()
        if not re.fullmatch(r"\d{4}", normalized):
            raise ValidationError("The anonymous SONA code must contain exactly four numeric digits.")
        return compensation_type, normalized
    return compensation_type, None


def validate_payment_url(url: str | None) -> str | None:
    """Return a secure configured URL, None when unset, or reject unsafe URLs."""

    normalized = (url or "").strip()
    if not normalized:
        return None
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValidationError("PAYMENT_QUESTIONNAIRE_URL must be a complete HTTPS URL.")
    return normalized


def validate_final_comparison(
    study: str,
    ranking: Sequence[str],
    comparison_rankings: Mapping[str, Sequence[str]],
    reason: str,
) -> None:
    expected_codes = list(CONDITIONS.get(study, {}))
    if not expected_codes:
        raise ValidationError("Study must be 1A, 1B, or Fleet.")
    if study == "Fleet":
        raise ValidationError("Fleet uses the final fleet-size questionnaire, not final comparison rankings.")
    validate_ranking(ranking, expected_codes)
    for code in ("q2", "q3", "q4", "q5"):
        direct_ranking = comparison_rankings.get(code)
        if direct_ranking is None:
            raise ValidationError("Every direct comparison ranking is required.")
        try:
            validate_ranking(direct_ranking, expected_codes)
        except ValidationError as exc:
            raise ValidationError(
                "Every direct comparison must rank each condition exactly once."
            ) from exc
    if not reason or not reason.strip():
        raise ValidationError("The main reason for the preferred condition is required.")


def validate_fleet_final_response(data: Mapping[str, Any]) -> dict[str, str]:
    preferred = data.get("preferred_sustainable_fleet_size")
    if preferred not in FLEET_FINAL_SUSTAINABLE_OPTIONS:
        raise ValidationError(FLEET_FINAL_ITEMS["preferred_sustainable_fleet_size"]["question"] + " is required.")

    best_balance = data.get("best_balance_fleet_size")
    if best_balance in (None, ""):
        best_balance = ""
    elif best_balance not in FLEET_FINAL_OPTIONS:
        raise ValidationError("Choose a valid fleet size for the best balance question.")

    strategy = str(data.get("prioritization_strategy_comment") or "").strip()
    if len(strategy) > 4000:
        raise ValidationError("The prioritization strategy comment must be 4000 characters or fewer.")

    comment = str(data.get("fleet_overload_comment") or "").strip()
    if len(comment) > 4000:
        raise ValidationError("The fleet overload comment must be 4000 characters or fewer.")

    return {
        "preferred_sustainable_fleet_size": str(preferred),
        "best_balance_fleet_size": str(best_balance),
        "prioritization_strategy_comment": strategy,
        "fleet_overload_comment": comment,
    }
