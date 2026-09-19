from __future__ import annotations

import pytest

from database import Database
from questionnaire_definitions import (
    FLEET_POST_CONDITION_ITEMS,
    FLEET_TLX_DIMENSIONS,
    POST_CONDITION_ITEMS,
    TLX_DIMENSIONS,
)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "data" / "test.sqlite3", tmp_path / "backups")


def valid_background():
    return {
        "age": 29,
        "handedness": "Right",
        "robotics_experience": "Less than 10 hours of hands-on experience",
        "vr_experience_last_12_months": "2-5 times",
        "gaming_controller_experience_last_12_months": "Approximately weekly",
        "teleoperation_experience": "Observed a demonstration only",
        "kinesthetic_experience": ["Franka robot", "FACTR"],
        "motion_sickness_susceptibility": 3,
    }


def valid_understanding():
    return {
        "risk_indicator_correct": True,
        "unattended_robots_correct": True,
        "multiple_paused_correct": True,
        "release_decision_correct": True,
        "ready_state_correct": True,
        "safety_response_correct": True,
        "retraining_required": False,
        "successful_practice_intervention": True,
        "practice_intervention_count": 1,
        "experimenter_notes": "",
    }


def valid_agreement(study):
    return {item["code"]: 6 for item in POST_CONDITION_ITEMS[study]}


def valid_fleet_agreement():
    return {item["code"]: 6 for item in FLEET_POST_CONDITION_ITEMS}


def valid_tlx():
    return {
        dimension["code_20"]: value
        for value, dimension in zip((4, 5, 6, 3, 7, 2), TLX_DIMENSIONS)
    }


def valid_fleet_tlx():
    return {
        dimension["code"]: value
        for value, dimension in zip((4, 5, 6, 2, 7, 3), FLEET_TLX_DIMENSIONS)
    }


def create_ready_session(
    db, study="1A", participant_id=None, order_code=None, task_order_code="T-O1"
):
    participant_id = participant_id or ("P1A-001" if study == "1A" else "P1B-001")
    order_code = order_code or ("S-O1" if study == "1A" else "C-O1")
    session = db.create_session(
        participant_id=participant_id,
        study=study,
        condition_order_code=order_code,
        task_order_code=task_order_code,
        experimenter_initials="AB",
        session_date="2026-08-03",
    )
    db.confirm_written_consent(session["session_id"], True)
    db.submit_background(session["session_id"], valid_background())
    db.submit_understanding_check(session["session_id"], valid_understanding())
    return db.get_session(session["session_id"])


def create_fleet_session(db, participant_id="P1C-001", order_code="F-O1"):
    return db.create_session(
        participant_id=participant_id,
        study="Fleet",
        condition_order_code=order_code,
        task_order_code="N/A",
        experimenter_initials="AB",
        session_date="2026-08-03",
    )


def create_fleet_ready_session(db, participant_id="P1C-001", order_code="F-O1"):
    session = create_fleet_session(db, participant_id, order_code)
    db.submit_background(session["session_id"], valid_background())
    return db.get_session(session["session_id"])


def complete_session(
    db, study="1A", participant_id=None, order_code=None, task_order_code="T-O1"
):
    session = create_ready_session(db, study, participant_id, order_code, task_order_code)
    for block_number in (1, 2, 3):
        db.unlock_condition(session["session_id"], block_number)
        db.submit_condition_questionnaire(
            session["session_id"], block_number, valid_agreement(study), valid_tlx()
        )
    condition_codes = list(("S1", "S2", "S3") if study == "1A" else ("C1", "C2", "C3"))
    db.submit_final_comparison(
        session["session_id"],
        ranking=condition_codes,
        comparison_rankings={
            "q2": condition_codes,
            "q3": condition_codes[1:] + condition_codes[:1],
            "q4": condition_codes[2:] + condition_codes[:2],
            "q5": condition_codes,
        },
        reason="It provided the clearest control.",
        comment="",
    )
    db.submit_compensation(session["session_id"], "payment")
    return db.get_session(session["session_id"])


def complete_fleet_session(db, participant_id="P1C-001", order_code="F-O1"):
    session = create_fleet_ready_session(db, participant_id, order_code)
    for block_number in (1, 2, 3, 4):
        db.unlock_condition(session["session_id"], block_number)
        db.submit_condition_questionnaire(
            session["session_id"],
            block_number,
            valid_fleet_agreement(),
            valid_fleet_tlx(),
        )
    db.submit_fleet_final_response(
        session["session_id"],
        {
            "preferred_sustainable_fleet_size": "6 robots",
            "best_balance_fleet_size": "3 robots",
            "prioritization_strategy_comment": "I used urgency first.",
            "fleet_overload_comment": "Nine robots felt too demanding.",
        },
    )
    db.submit_compensation(session["session_id"], "payment")
    return db.get_session(session["session_id"])
