"""Creation of analysis-ready local exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from database import Database
from questionnaire_definitions import (
    BACKGROUND_OPTIONS,
    CONDITIONS,
    FLEET_POST_CONDITION_ITEMS,
    FLEET_SIZES,
    FLEET_TLX_DIMENSIONS,
    LEGACY_BACKGROUND_OPTIONS,
)


class ExportError(RuntimeError):
    """Raised when a session is incomplete or files cannot be exported."""


LONG_COLUMNS = [
    "participant_id",
    "session_id",
    "study",
    "block_id",
    "block_number",
    "condition_code",
    "condition_name",
    "condition_position",
    "task_order_code",
    "measure_code",
    "measure_label",
    "category",
    "value",
    "submitted_at",
]

LEGACY_TLX_CODES = {
    "TLX_MENTAL",
    "TLX_PHYSICAL",
    "TLX_TEMPORAL",
    "TLX_PERFORMANCE",
    "TLX_EFFORT",
    "TLX_FRUSTRATION",
    "TLX_RAW_MEAN",
}

FLEET_CONDITION_COLUMNS = [
    "participant_id",
    "session_id",
    "study",
    "questionnaire_type",
    "fleet_size",
    "condition_order",
    "condition_order_code",
    "condition_code",
    "condition_name",
    "block_id",
    "nasa_mental",
    "nasa_physical",
    "nasa_temporal",
    "nasa_performance",
    "nasa_effort",
    "nasa_frustration",
    "fleet_awareness",
    "fleet_attention_switch",
    "fleet_prioritization",
    "fleet_response_time",
    "fleet_keep_up",
    "submitted_at",
]

FLEET_FINAL_COLUMNS = [
    "participant_id",
    "session_id",
    "study",
    "questionnaire_type",
    "preferred_sustainable_fleet_size",
    "best_balance_fleet_size",
    "prioritization_strategy_comment",
    "fleet_overload_comment",
    "submitted_at",
]

FLEET_BACKGROUND_COLUMNS = [
    "participant_id",
    "session_id",
    "study",
    "questionnaire_type",
    "age",
    "handedness",
    "robotics_experience",
    "robotics_experience_score",
    "vr_experience_last_12_months",
    "vr_experience_last_12_months_score",
    "gaming_controller_experience_last_12_months",
    "gaming_controller_experience_last_12_months_score",
    "teleoperation_experience",
    "teleoperation_experience_score",
    "kinesthetic_experience",
    "motion_sickness_susceptibility",
    "submitted_at",
]


def _require_complete_session(db: Database, session_id: str) -> dict[str, Any]:
    session = db.get_session(session_id)
    blocks = db.get_condition_blocks(session_id)
    if session["current_workflow_state"] != "complete":
        raise ExportError("Only a complete session can be exported.")
    expected_blocks = len(CONDITIONS[session["study"]])
    if len(blocks) != expected_blocks or any(not block["questionnaire_submitted_at"] for block in blocks):
        raise ExportError("The session is missing one or more condition submissions.")
    if session["study"] == "Fleet":
        if db.get_background(session_id) is None:
            raise ExportError("The session is missing required background data.")
        if db.get_fleet_final_response(session_id) is None:
            raise ExportError("The session is missing the final Fleet questionnaire.")
        return session
    if db.get_background(session_id) is None or db.get_understanding_check(session_id) is None:
        raise ExportError("The session is missing required pre-condition data.")
    if db.get_final_comparison(session_id) is None:
        raise ExportError("The session is missing the final comparison.")
    return session


def _background_row(
    session: dict[str, Any], background: dict[str, Any], questionnaire_type: str
) -> dict[str, Any]:
    row = {
        "participant_id": session["participant_id"],
        "session_id": session["session_id"],
        "study": session["study"],
        "questionnaire_type": questionnaire_type,
        "age": background["age"],
        "handedness": background["handedness"],
        "kinesthetic_experience": " | ".join(background["kinesthetic_experience"]),
        "motion_sickness_susceptibility": background["motion_sickness_susceptibility"],
        "submitted_at": background["submitted_at"],
    }
    for key in (
        "robotics_experience",
        "vr_experience_last_12_months",
        "gaming_controller_experience_last_12_months",
        "teleoperation_experience",
    ):
        value = background[key]
        options = BACKGROUND_OPTIONS.get(key, [])
        legacy_options = LEGACY_BACKGROUND_OPTIONS.get(key, [])
        row[key] = value
        if value in options:
            row[f"{key}_score"] = options.index(value) + 1
        elif value in legacy_options:
            row[f"{key}_score"] = legacy_options.index(value) + 1
        else:
            row[f"{key}_score"] = None
    return row


def _wide_row(db: Database, session_id: str, session: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = dict(session)
    row["task_order_sequence"] = " → ".join(session.get("task_order_sequence", []))
    background = db.get_background(session_id) or {}
    understanding = db.get_understanding_check(session_id) or {}
    blocks = db.get_condition_blocks(session_id)
    final = db.get_final_comparison(session_id) or {}

    for key, value in background.items():
        if key not in {"session_id", "participant_id"}:
            row[f"background_{key}"] = (
                " | ".join(value) if key == "kinesthetic_experience" else value
            )
            if key in {
                "robotics_experience",
                "vr_experience_last_12_months",
                "gaming_controller_experience_last_12_months",
                "teleoperation_experience",
            }:
                options = BACKGROUND_OPTIONS.get(key, [])
                legacy_options = LEGACY_BACKGROUND_OPTIONS.get(key, [])
                if value in options:
                    row[f"background_{key}_score"] = options.index(value) + 1
                elif value in legacy_options:
                    row[f"background_{key}_score"] = legacy_options.index(value) + 1
    for key, value in understanding.items():
        if key not in {"session_id", "participant_id"}:
            row[f"understanding_{key}"] = value

    for block in blocks:
        prefix = block["block_id"]
        row[f"{prefix}_block_id"] = block["block_id"]
        row[f"{prefix}_block_number"] = block["block_number"]
        row[f"{prefix}_condition_code"] = block["condition_code"]
        row[f"{prefix}_condition_name"] = block["condition_name"]
        row[f"{prefix}_condition_position"] = block["condition_position"]
        row[f"{prefix}_task_order_code"] = block["task_order_code"]
        row[f"{prefix}_opened_at"] = block["questionnaire_opened_at"]
        row[f"{prefix}_submitted_at"] = block["questionnaire_submitted_at"]

    for response in db.get_condition_responses(session_id):
        if response["measure_code"] in LEGACY_TLX_CODES:
            continue
        row[f"{response['block_id']}_{response['measure_code']}"] = response["value"]

    ranking = final.get("ranking", [])
    for position, condition_code in enumerate(ranking, start=1):
        row[f"final_rank_{position}"] = condition_code
    for key, value in final.items():
        if key not in {"session_id", "participant_id", "study", "ranking"}:
            if key.endswith("_ranking"):
                if value:
                    ranking_prefix = key.removesuffix("_ranking")
                    for position, condition_code in enumerate(value, start=1):
                        row[f"final_{ranking_prefix}_rank_{position}"] = condition_code
            else:
                row[f"final_{key}"] = value
    return row


def export_session(
    db: Database,
    session_id: str,
    export_root: str | Path,
) -> dict[str, Path]:
    """Write long CSV, wide CSV, and metadata JSON for a complete session."""

    session = _require_complete_session(db, session_id)
    if session["study"] == "Fleet":
        return _export_fleet_session(db, session_id, session, export_root)

    export_directory = Path(export_root) / session_id
    export_directory.mkdir(parents=True, exist_ok=True)

    response_rows = [
        row
        for row in db.get_condition_responses(session_id)
        if row["measure_code"] not in LEGACY_TLX_CODES
    ]
    long_frame = pd.DataFrame(response_rows)
    if long_frame.empty:
        raise ExportError("The session contains no condition response data.")
    long_frame = long_frame[LONG_COLUMNS]
    long_path = export_directory / "questionnaire_long.csv"
    long_frame.to_csv(long_path, index=False)

    wide_path = export_directory / "questionnaire_wide.csv"
    pd.DataFrame([_wide_row(db, session_id, session)]).to_csv(wide_path, index=False)

    blocks = db.get_condition_blocks(session_id)
    metadata = {
        "participant_id": session["participant_id"],
        "session_id": session["session_id"],
        "study": session["study"],
        "session_date": session["session_date"],
        "condition_order_code": session["condition_order_code"],
        "task_order_code": session["task_order_code"],
        "task_order_sequence": session["task_order_sequence"],
        "experimenter_initials": session["experimenter_initials"],
        "written_consent_confirmed": bool(session["written_consent_confirmed"]),
        "current_workflow_state": session["current_workflow_state"],
        "session_created_at": session["session_created_at"],
        "session_completed_at": session["session_completed_at"],
        "condition_blocks": [
            {
                "participant_id": block["participant_id"],
                "session_id": block["session_id"],
                "study": block["study"],
                "block_number": block["block_number"],
                "block_id": block["block_id"],
                "condition_code": block["condition_code"],
                "condition_name": block["condition_name"],
                "condition_position": block["condition_position"],
                "task_order_code": block["task_order_code"],
                "questionnaire_opened_at": block["questionnaire_opened_at"],
                "questionnaire_submitted_at": block["questionnaire_submitted_at"],
            }
            for block in blocks
        ],
    }
    metadata_path = export_directory / "session_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    expected_paths = (long_path, wide_path, metadata_path)
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_paths):
        raise ExportError("One or more export files could not be verified.")
    db.record_export(session_id, export_directory)
    return {"long": long_path, "wide": wide_path, "metadata": metadata_path}


def _export_fleet_session(
    db: Database,
    session_id: str,
    session: dict[str, Any],
    export_root: str | Path,
) -> dict[str, Path]:
    export_directory = Path(export_root) / session_id
    export_directory.mkdir(parents=True, exist_ok=True)

    responses_by_block: dict[str, dict[str, Any]] = {}
    for response in db.get_condition_responses(session_id):
        responses_by_block.setdefault(response["block_id"], {})[response["measure_code"]] = response["value"]

    condition_rows = []
    for block in db.get_condition_blocks(session_id):
        measures = responses_by_block.get(block["block_id"], {})
        condition_rows.append(
            {
                "participant_id": session["participant_id"],
                "session_id": session_id,
                "study": "Fleet",
                "questionnaire_type": "post_condition",
                "fleet_size": FLEET_SIZES[block["condition_code"]],
                "condition_order": block["block_number"],
                "condition_order_code": session["condition_order_code"],
                "condition_code": block["condition_code"],
                "condition_name": block["condition_name"],
                "block_id": block["block_id"],
                **{dimension["code"]: measures.get(dimension["code"]) for dimension in FLEET_TLX_DIMENSIONS},
                **{item["code"]: measures.get(item["code"]) for item in FLEET_POST_CONDITION_ITEMS},
                "submitted_at": block["questionnaire_submitted_at"],
            }
        )

    condition_path = export_directory / "fleet_condition_questionnaires.csv"
    pd.DataFrame(condition_rows)[FLEET_CONDITION_COLUMNS].to_csv(condition_path, index=False)

    final = db.get_fleet_final_response(session_id)
    if final is None:
        raise ExportError("The session is missing the final Fleet questionnaire.")
    final_row = {
        "participant_id": session["participant_id"],
        "session_id": session_id,
        "study": "Fleet",
        "questionnaire_type": "fleet_final",
        "preferred_sustainable_fleet_size": final["preferred_sustainable_fleet_size"],
        "best_balance_fleet_size": final["best_balance_fleet_size"] or "",
        "prioritization_strategy_comment": final["prioritization_strategy_comment"],
        "fleet_overload_comment": final["fleet_overload_comment"],
        "submitted_at": final["submitted_at"],
    }
    final_path = export_directory / "fleet_final_questionnaire.csv"
    pd.DataFrame([final_row])[FLEET_FINAL_COLUMNS].to_csv(final_path, index=False)

    background = db.get_background(session_id)
    if background is None:
        raise ExportError("The session is missing required background data.")
    background_path = export_directory / "fleet_background.csv"
    pd.DataFrame([_background_row(session, background, "background")])[
        FLEET_BACKGROUND_COLUMNS
    ].to_csv(background_path, index=False)

    blocks = db.get_condition_blocks(session_id)
    metadata = {
        "participant_id": session["participant_id"],
        "session_id": session["session_id"],
        "study": session["study"],
        "session_date": session["session_date"],
        "condition_order_code": session["condition_order_code"],
        "task_order_code": session["task_order_code"],
        "task_order_sequence": session["task_order_sequence"],
        "experimenter_initials": session["experimenter_initials"],
        "written_consent_confirmed": bool(session["written_consent_confirmed"]),
        "current_workflow_state": session["current_workflow_state"],
        "session_created_at": session["session_created_at"],
        "session_completed_at": session["session_completed_at"],
        "condition_blocks": [
            {
                "participant_id": block["participant_id"],
                "session_id": block["session_id"],
                "study": block["study"],
                "block_number": block["block_number"],
                "block_id": block["block_id"],
                "condition_code": block["condition_code"],
                "condition_name": block["condition_name"],
                "fleet_size": FLEET_SIZES[block["condition_code"]],
                "condition_position": block["condition_position"],
                "task_order_code": block["task_order_code"],
                "questionnaire_opened_at": block["questionnaire_opened_at"],
                "questionnaire_submitted_at": block["questionnaire_submitted_at"],
            }
            for block in blocks
        ],
        "background": _background_row(session, background, "background"),
        "fleet_final_response": final_row,
    }
    metadata_path = export_directory / "session_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    expected_paths = (background_path, condition_path, final_path, metadata_path)
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected_paths):
        raise ExportError("One or more export files could not be verified.")
    db.record_export(session_id, export_directory)
    return {
        "fleet_background": background_path,
        "fleet_condition": condition_path,
        "fleet_final": final_path,
        "metadata": metadata_path,
    }
