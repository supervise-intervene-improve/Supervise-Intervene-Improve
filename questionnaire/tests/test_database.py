from __future__ import annotations

import sqlite3

import pytest

from database import DuplicateSubmissionError, InvalidTransitionError
from tests.conftest import (
    complete_session,
    create_fleet_session,
    create_fleet_ready_session,
    create_ready_session,
    valid_agreement,
    valid_background,
    valid_fleet_agreement,
    valid_fleet_tlx,
    valid_tlx,
)


def test_workflow_state_transitions_and_audit_log(db):
    session = db.create_session(
        "P1A-001", "1A", "S-O4", "T-O2", "XY", "2026-08-03"
    )
    session_id = session["session_id"]
    assert session["current_workflow_state"] == "paper_consent"

    db.confirm_written_consent(session_id, True)
    assert db.get_session(session_id)["current_workflow_state"] == "participant_background"
    db.submit_background(session_id, valid_background())
    assert db.get_session(session_id)["current_workflow_state"] == "understanding_check"

    from tests.conftest import valid_understanding

    db.submit_understanding_check(session_id, valid_understanding())
    assert db.get_session(session_id)["current_workflow_state"] == "waiting_condition_1"

    for block_number in (1, 2, 3):
        db.unlock_condition(session_id, block_number)
        assert db.get_session(session_id)["current_workflow_state"] == f"condition_{block_number}_questionnaire"
        db.submit_condition_questionnaire(
            session_id, block_number, valid_agreement("1A"), valid_tlx()
        )
        expected = f"waiting_condition_{block_number + 1}" if block_number < 3 else "final_comparison"
        assert db.get_session(session_id)["current_workflow_state"] == expected

    db.submit_final_comparison(
        session_id,
        ["S2", "S3", "S1"],
        {
            "q2": ["S2", "S3", "S1"],
            "q3": ["S3", "S1", "S2"],
            "q4": ["S1", "S2", "S3"],
            "q5": ["S2", "S1", "S3"],
        },
        "Clear spatial relationships.",
    )
    final = db.get_final_comparison(session_id)
    assert final and final["comparison_q2"] == "S2"
    assert final["comparison_q2_ranking"] == ["S2", "S3", "S1"]
    assert final["comparison_q5_ranking"] == ["S2", "S1", "S3"]
    assert db.get_session(session_id)["current_workflow_state"] == "compensation"
    db.submit_compensation(session_id, "sona_credit", "1234")
    assert db.get_session(session_id)["current_workflow_state"] == "complete"
    events = [row["event_type"] for row in db.get_audit_log(session_id)]
    assert "session_creation" in events
    assert events.count("questionnaire_unlock") == 3
    assert events.count("questionnaire_submission") == 3
    assert events[-3:] == [
        "final_comparison_submission",
        "compensation_submission",
        "session_completion",
    ]


def test_wrong_or_skipped_condition_is_prevented(db):
    session = create_ready_session(db)
    with pytest.raises(InvalidTransitionError):
        db.unlock_condition(session["session_id"], 2)
    db.unlock_condition(session["session_id"], 1)
    with pytest.raises(InvalidTransitionError):
        db.submit_condition_questionnaire(
            session["session_id"], 2, valid_agreement("1A"), valid_tlx()
        )


def test_duplicate_condition_submission_is_prevented(db):
    session = create_ready_session(db)
    db.unlock_condition(session["session_id"], 1)
    db.submit_condition_questionnaire(
        session["session_id"], 1, valid_agreement("1A"), valid_tlx()
    )
    before = len(db.get_condition_responses(session["session_id"]))
    with pytest.raises(InvalidTransitionError):
        db.submit_condition_questionnaire(
            session["session_id"], 1, valid_agreement("1A"), valid_tlx()
        )
    assert len(db.get_condition_responses(session["session_id"])) == before


def test_tlx_performance_and_both_scales_are_stored(db):
    session = create_ready_session(db)
    db.unlock_condition(session["session_id"], 1)
    db.submit_condition_questionnaire(
        session["session_id"], 1, valid_agreement("1A"), valid_tlx()
    )
    responses = {
        row["measure_code"]: row["value"]
        for row in db.get_condition_responses(session["session_id"])
        if row["block_id"] == "B01"
    }
    assert responses["tlx_performance_20"] == 3
    assert responses["tlx_performance_100"] == 15
    assert responses["raw_tlx_20"] == pytest.approx(4.5)
    assert responses["raw_tlx_100"] == pytest.approx(22.5)


def test_condition_order_is_persisted_in_blocks(db):
    session = db.create_session(
        "P1B-001", "1B", "C-O6", "T-O1", "AB", "2026-08-03"
    )
    blocks = db.get_condition_blocks(session["session_id"])
    assert [block["condition_code"] for block in blocks] == ["C3", "C2", "C1"]
    assert [block["block_id"] for block in blocks] == ["B01", "B02", "B03"]


def test_task_order_is_fixed_across_all_condition_blocks(db):
    session = db.create_session(
        "P1A-001", "1A", "S-O2", "T-O2", "AB", "2026-08-03"
    )
    assert session["task_order_code"] == "T-O2"
    assert session["task_order_sequence"] == ["Cups", "T-shape"]
    blocks = db.get_condition_blocks(session["session_id"])
    assert [block["task_order_code"] for block in blocks] == ["T-O2", "T-O2", "T-O2"]


def test_fleet_session_uses_four_conditions_and_final_fleet_questionnaire(db):
    session = create_fleet_session(db, "P1C-001", "F-O1")
    session_id = session["session_id"]
    assert session["current_workflow_state"] == "participant_background"
    db.submit_background(session_id, valid_background())
    assert db.get_session(session_id)["current_workflow_state"] == "waiting_condition_1"
    assert session["task_order_code"] == "N/A"
    assert session["task_order_sequence"] == []

    blocks = db.get_condition_blocks(session_id)
    assert [block["condition_code"] for block in blocks] == ["F1", "F3", "F6", "F9"]
    assert [block["condition_name"] for block in blocks] == [
        "1 robot",
        "3 robots",
        "6 robots",
        "9 robots",
    ]

    for block_number in (1, 2, 3, 4):
        db.unlock_condition(session_id, block_number)
        db.submit_condition_questionnaire(
            session_id,
            block_number,
            valid_fleet_agreement(),
            valid_fleet_tlx(),
        )
    assert db.get_session(session_id)["current_workflow_state"] == "fleet_final"

    responses = {
        row["measure_code"]: row["value"]
        for row in db.get_condition_responses(session_id)
        if row["block_id"] == "B01"
    }
    assert responses["nasa_performance"] == 2
    assert "raw_tlx_100" not in responses
    assert responses["fleet_awareness"] == 6
    assert "fleet_attention_switch" not in responses
    assert "fleet_prioritization" not in responses
    assert responses["fleet_response_time"] == 6

    db.submit_fleet_final_response(
        session_id,
        {
            "preferred_sustainable_fleet_size": "6 robots",
            "best_balance_fleet_size": "3 robots",
            "prioritization_strategy_comment": "I used urgency first.",
            "fleet_overload_comment": "Nine robots felt too demanding.",
        },
    )
    assert db.get_session(session_id)["current_workflow_state"] == "compensation"
    final = db.get_fleet_final_response(session_id)
    assert final and final["preferred_sustainable_fleet_size"] == "6 robots"
    assert final["prioritization_strategy_comment"] == "I used urgency first."
    db.submit_compensation(session_id, "payment")
    assert db.get_session(session_id)["current_workflow_state"] == "complete"
    events = [row["event_type"] for row in db.get_audit_log(session_id)]
    assert events.count("questionnaire_submission") == 4
    assert events[-3:] == [
        "fleet_final_submission",
        "compensation_submission",
        "session_completion",
    ]


def test_fleet_duplicate_participant_and_duplicate_condition_submission_are_prevented(db):
    session = create_fleet_ready_session(db, "P1C-001", "F-O1")
    with pytest.raises(DuplicateSubmissionError):
        create_fleet_session(db, "P1C-001", "F-O2")
    db.unlock_condition(session["session_id"], 1)
    db.submit_condition_questionnaire(
        session["session_id"], 1, valid_fleet_agreement(), valid_fleet_tlx()
    )
    with pytest.raises(InvalidTransitionError):
        db.submit_condition_questionnaire(
            session["session_id"], 1, valid_fleet_agreement(), valid_fleet_tlx()
        )


def test_compensation_is_stored_separately_after_scientific_final(db):
    session = create_ready_session(db)
    for block_number in (1, 2, 3):
        db.unlock_condition(session["session_id"], block_number)
        db.submit_condition_questionnaire(
            session["session_id"], block_number, valid_agreement("1A"), valid_tlx()
        )
    db.submit_final_comparison(
        session["session_id"],
        ["S1", "S2", "S3"],
        {
            "q2": ["S1", "S2", "S3"],
            "q3": ["S2", "S3", "S1"],
            "q4": ["S3", "S1", "S2"],
            "q5": ["S1", "S3", "S2"],
        },
        "Synthetic test reason.",
    )
    assert db.get_session(session["session_id"])["current_workflow_state"] == "compensation"
    db.submit_compensation(session["session_id"], "sona_credit", "0042")
    record = db.get_compensation(session["session_id"])
    assert record and record["compensation_type"] == "sona_credit"
    assert record["sona_code"] == "0042"
    assert db.get_session(session["session_id"])["current_workflow_state"] == "complete"


def test_sqlite_persistence_and_recovery_after_new_database_instance(db):
    session = create_ready_session(db)
    reopened = type(db)(db.db_path, db.backup_dir)
    recovered = reopened.recover_session(session["participant_id"])
    assert recovered["session_id"] == session["session_id"]
    assert recovered["current_workflow_state"] == "waiting_condition_1"
    with reopened.connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("study", "participant_id", "order_code", "expected_response_count"),
    [
        ("1A", "P1A-001", "S-O2", 69),
        ("1B", "P1B-001", "C-O5", 63),
    ],
)
def test_complete_sessions_for_both_studies(
    db, study, participant_id, order_code, expected_response_count
):
    session = complete_session(db, study, participant_id, order_code)
    assert session["current_workflow_state"] == "complete"
    assert session["session_completed_at"]
    assert len(db.get_condition_responses(session["session_id"])) == expected_response_count


def test_backup_is_created_verified_and_retained(db):
    session = db.create_session(
        "P1A-001", "1A", "S-O1", "T-O1", "AB", "2026-08-03"
    )
    latest = db.latest_backup()
    assert latest and latest.is_file() and latest.stat().st_size > 0
    with sqlite3.connect(latest) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT session_id FROM sessions").fetchone()[0] == session["session_id"]
    for _ in range(22):
        db.backup_database()
    assert len(list(db.backup_dir.glob("test_*.sqlite3"))) == 20
