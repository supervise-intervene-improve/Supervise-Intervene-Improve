from __future__ import annotations

import sqlite3

from database import CURRENT_SCHEMA_VERSION, Database


def create_legacy_database(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL UNIQUE,
                study TEXT NOT NULL,
                session_date TEXT NOT NULL,
                condition_order_code TEXT NOT NULL,
                experimenter_initials TEXT NOT NULL,
                written_consent_confirmed INTEGER NOT NULL DEFAULT 0,
                current_workflow_state TEXT NOT NULL,
                session_created_at TEXT NOT NULL,
                session_completed_at TEXT
            );
            CREATE TABLE participant_background (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                age INTEGER NOT NULL,
                handedness TEXT NOT NULL,
                robotics_experience TEXT NOT NULL,
                vr_experience TEXT NOT NULL,
                gaming_experience TEXT NOT NULL,
                teleoperation_experience TEXT NOT NULL,
                kinesthetic_experience_json TEXT NOT NULL,
                motion_sickness_susceptibility INTEGER NOT NULL,
                submitted_at TEXT NOT NULL
            );
            CREATE TABLE condition_blocks (
                session_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                study TEXT NOT NULL,
                block_number INTEGER NOT NULL,
                block_id TEXT NOT NULL,
                condition_code TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                condition_position INTEGER NOT NULL,
                questionnaire_opened_at TEXT,
                questionnaire_submitted_at TEXT,
                PRIMARY KEY(session_id, block_id),
                UNIQUE(session_id, block_number)
            );
            CREATE TABLE condition_responses (
                response_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                study TEXT NOT NULL,
                block_id TEXT NOT NULL,
                measure_code TEXT NOT NULL,
                measure_label TEXT NOT NULL,
                category TEXT NOT NULL,
                value REAL NOT NULL,
                submitted_at TEXT NOT NULL,
                UNIQUE(session_id, block_id, measure_code)
            );
            CREATE TABLE final_comparisons (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                study TEXT NOT NULL,
                ranking_json TEXT NOT NULL,
                comparison_q2 TEXT NOT NULL,
                comparison_q3 TEXT NOT NULL,
                comparison_q4 TEXT NOT NULL,
                comparison_q5 TEXT NOT NULL,
                preference_reason TEXT NOT NULL,
                additional_comment TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL
            );
            """
        )
        session_id = "P1A-001_20260803_120000"
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, '1A', '2026-08-03', 'S-O1', 'AB', 1, 'complete', ?, ?)",
            (session_id, "P1A-001", "2026-08-03T10:00:00+00:00", "2026-08-03T11:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO participant_background VALUES (?, ?, 30, 'Right', 'None', ?, ?, 'None', '[]', 1, ?)",
            (
                session_id,
                "P1A-001",
                "Used VR 2–5 times",
                "Several times per year",
                "2026-08-03T10:05:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO final_comparisons VALUES(
                ?, ?, '1A', '["S1", "S2", "S3"]',
                'S1', 'S2', 'S3', 'S1', 'Legacy winner reason', '', ?
            )
            """,
            (session_id, "P1A-001", "2026-08-03T10:55:00+00:00"),
        )
        for number, code, name in (
            (1, "S1", "Desktop-RGB + Kinesthetic Teaching"),
            (2, "S2", "VR-RGB + Kinesthetic Teaching"),
            (3, "S3", "VR-PointCloud + Kinesthetic Teaching"),
        ):
            conn.execute(
                "INSERT INTO condition_blocks VALUES (?, ?, '1A', ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    "P1A-001",
                    number,
                    f"B{number:02d}",
                    code,
                    name,
                    number,
                    "2026-08-03T10:10:00+00:00",
                    "2026-08-03T10:20:00+00:00",
                ),
            )
        for code, label, value in (
            ("TLX_PERFORMANCE", "Performance", 35.0),
            ("TLX_RAW_MEAN", "Raw NASA-TLX mean", 25.0),
        ):
            conn.execute(
                """
                INSERT INTO condition_responses(
                    session_id, participant_id, study, block_id, measure_code,
                    measure_label, category, value, submitted_at
                ) VALUES (?, ?, '1A', 'B01', ?, ?, 'Raw NASA-TLX', ?, ?)
                """,
                (session_id, "P1A-001", code, label, value, "2026-08-03T10:20:00+00:00"),
            )
    return session_id


def test_legacy_database_is_backed_up_and_migrated_without_recreation(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    backup_dir = tmp_path / "backups"
    session_id = create_legacy_database(db_path)

    db = Database(db_path, backup_dir)

    assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
    session = db.get_session(session_id)
    assert session["current_workflow_state"] == "complete"
    assert session["task_order_code"] is None
    assert session["task_order_sequence"] == []
    assert [block["task_order_code"] for block in db.get_condition_blocks(session_id)] == [
        None,
        None,
        None,
    ]
    background = db.get_background(session_id)
    assert background["vr_experience_last_12_months"] == "Used VR 2–5 times"
    assert background["gaming_controller_experience_last_12_months"] == "Several times per year"
    responses = {row["measure_code"]: row["value"] for row in db.get_condition_responses(session_id)}
    assert responses["tlx_performance_20"] == 7
    assert responses["tlx_performance_100"] == 35
    assert responses["raw_tlx_20"] == 5
    assert responses["raw_tlx_100"] == 25
    assert db.get_compensation(session_id) is None
    assert db.get_fleet_final_response(session_id) is None
    final = db.get_final_comparison(session_id)
    assert final and final["comparison_q2"] == "S1"
    assert final["comparison_q2_ranking"] is None
    assert list(backup_dir.glob("legacy_*.sqlite3"))

    reopened = Database(db_path, backup_dir)
    assert reopened.recover_session(session_id)["current_workflow_state"] == "complete"

    fleet = reopened.create_session(
        "P1C-001", "Fleet", "F-O1", "N/A", "AB", "2026-08-03"
    )
    assert fleet["current_workflow_state"] == "participant_background"
    assert len(reopened.get_condition_blocks(fleet["session_id"])) == 4
