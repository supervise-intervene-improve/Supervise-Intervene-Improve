"""SQLite persistence, workflow transitions, audit logging, and backups."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from config import BACKUP_RETENTION
from questionnaire_definitions import (
    CONDITIONS,
    CONDITION_ORDERS,
    FLEET_FINAL_ITEMS,
    FLEET_POST_CONDITION_ITEMS,
    FLEET_SIZES,
    FLEET_TLX_DIMENSIONS,
    POST_CONDITION_ITEMS,
    TASK_ORDERS,
    TLX_DIMENSIONS,
    resolve_condition_order,
    resolve_task_order,
)
from validation import (
    ValidationError,
    calculate_raw_tlx,
    calculate_raw_tlx_scores,
    calculate_spatial_understanding,
    convert_tlx_to_100,
    validate_background,
    validate_compensation,
    validate_condition_responses,
    validate_fleet_final_response,
    validate_fleet_condition_responses,
    validate_fleet_tlx_20,
    validate_final_comparison,
    validate_participant_id,
    validate_task_order,
    validate_understanding_check,
)


CURRENT_SCHEMA_VERSION = 5


class DatabaseError(RuntimeError):
    """Base class for application persistence errors."""


class SessionNotFoundError(DatabaseError):
    """Raised when a requested session cannot be found."""


class DuplicateSubmissionError(DatabaseError):
    """Raised when a submitted form is submitted again."""


class InvalidTransitionError(DatabaseError):
    """Raised when a workflow action is attempted out of sequence."""


class BackupError(DatabaseError):
    """Raised when a committed database change cannot be backed up."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Small transactional repository for a single local SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        backup_dir: str | Path,
        backup_retention: int = BACKUP_RETENTION,
    ) -> None:
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_retention = max(20, backup_retention)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        migration_needed = self._migration_needed()
        if migration_needed:
            self.backup_database()
        self.initialize()
        if migration_needed:
            self.backup_database()

    def _migration_needed(self) -> bool:
        if not self.db_path.is_file() or self.db_path.stat().st_size == 0:
            return False
        conn = sqlite3.connect(self.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if "sessions" not in tables:
                return False
            session_sql = self._table_sql(conn, "sessions") or ""
            block_sql = self._table_sql(conn, "condition_blocks") or ""
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            block_columns = {row[1] for row in conn.execute("PRAGMA table_info(condition_blocks)")}
            background_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(participant_background)")
            }
            final_comparison_columns = (
                {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(final_comparisons)")
                }
                if "final_comparisons" in tables
                else set()
            )
            fleet_final_columns = (
                {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(fleet_final_responses)")
                }
                if "fleet_final_responses" in tables
                else set()
            )
            return (
                "task_order_code" not in session_columns
                or "task_order_sequence" not in session_columns
                or "task_order_code" not in block_columns
                or "vr_experience_last_12_months" not in background_columns
                or "gaming_controller_experience_last_12_months" not in background_columns
                or "compensation_records" not in tables
                or "schema_migrations" not in tables
                or "Fleet" not in session_sql
                or "BETWEEN 1 AND 4" not in block_sql
                or "fleet_final_responses" not in tables
                or (
                    "fleet_final_responses" in tables
                    and "prioritization_strategy_comment" not in fleet_final_columns
                )
                or (
                    "final_comparisons" in tables
                    and "comparison_q2_ranking_json" not in final_comparison_columns
                )
            )
        finally:
            conn.close()

    @staticmethod
    def _table_sql(conn: sqlite3.Connection, table: str) -> str | None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        return row[0] if row else None

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    participant_id TEXT NOT NULL UNIQUE,
                    study TEXT NOT NULL CHECK(study IN ('1A', '1B', 'Fleet')),
                    session_date TEXT NOT NULL,
                    condition_order_code TEXT NOT NULL,
                    task_order_code TEXT,
                    task_order_sequence TEXT,
                    experimenter_initials TEXT NOT NULL,
                    written_consent_confirmed INTEGER NOT NULL DEFAULT 0 CHECK(written_consent_confirmed IN (0, 1)),
                    current_workflow_state TEXT NOT NULL,
                    session_created_at TEXT NOT NULL,
                    session_completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS participant_background (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    age INTEGER NOT NULL CHECK(age BETWEEN 18 AND 100),
                    handedness TEXT NOT NULL,
                    robotics_experience TEXT NOT NULL,
                    vr_experience TEXT NOT NULL,
                    gaming_experience TEXT NOT NULL,
                    vr_experience_last_12_months TEXT NOT NULL,
                    gaming_controller_experience_last_12_months TEXT NOT NULL,
                    teleoperation_experience TEXT NOT NULL,
                    kinesthetic_experience_json TEXT NOT NULL,
                    motion_sickness_susceptibility INTEGER NOT NULL CHECK(motion_sickness_susceptibility BETWEEN 1 AND 7),
                    submitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS understanding_checks (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    risk_indicator_correct INTEGER NOT NULL,
                    unattended_robots_correct INTEGER NOT NULL,
                    multiple_paused_correct INTEGER NOT NULL,
                    release_decision_correct INTEGER NOT NULL,
                    ready_state_correct INTEGER NOT NULL,
                    safety_response_correct INTEGER NOT NULL,
                    retraining_required INTEGER NOT NULL,
                    successful_practice_intervention INTEGER NOT NULL,
                    practice_intervention_count INTEGER NOT NULL,
                    experimenter_notes TEXT NOT NULL DEFAULT '',
                    submitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS condition_blocks (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    study TEXT NOT NULL,
                    block_number INTEGER NOT NULL CHECK(block_number BETWEEN 1 AND 4),
                    block_id TEXT NOT NULL,
                    condition_code TEXT NOT NULL,
                    condition_name TEXT NOT NULL,
                    condition_position INTEGER NOT NULL CHECK(condition_position BETWEEN 1 AND 4),
                    task_order_code TEXT,
                    questionnaire_opened_at TEXT,
                    questionnaire_submitted_at TEXT,
                    PRIMARY KEY(session_id, block_id),
                    UNIQUE(session_id, block_number)
                );

                CREATE TABLE IF NOT EXISTS condition_responses (
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
                    FOREIGN KEY(session_id, block_id) REFERENCES condition_blocks(session_id, block_id) ON DELETE RESTRICT,
                    UNIQUE(session_id, block_id, measure_code)
                );

                CREATE TABLE IF NOT EXISTS final_comparisons (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    study TEXT NOT NULL,
                    ranking_json TEXT NOT NULL,
                    comparison_q2 TEXT NOT NULL,
                    comparison_q3 TEXT NOT NULL,
                    comparison_q4 TEXT NOT NULL,
                    comparison_q5 TEXT NOT NULL,
                    comparison_q2_ranking_json TEXT,
                    comparison_q3_ranking_json TEXT,
                    comparison_q4_ranking_json TEXT,
                    comparison_q5_ranking_json TEXT,
                    preference_reason TEXT NOT NULL,
                    additional_comment TEXT NOT NULL DEFAULT '',
                    submitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fleet_final_responses (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    study TEXT NOT NULL CHECK(study = 'Fleet'),
                    preferred_sustainable_fleet_size TEXT NOT NULL,
                    best_balance_fleet_size TEXT,
                    prioritization_strategy_comment TEXT NOT NULL DEFAULT '',
                    fleet_overload_comment TEXT NOT NULL DEFAULT '',
                    submitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    event_type TEXT NOT NULL,
                    event_at TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS compensation_records (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE RESTRICT,
                    participant_id TEXT NOT NULL,
                    compensation_type TEXT NOT NULL CHECK(compensation_type IN ('payment', 'sona_credit')),
                    sona_code TEXT,
                    created_at TEXT NOT NULL,
                    CHECK(
                        (compensation_type = 'payment' AND sona_code IS NULL)
                        OR (compensation_type = 'sona_credit' AND length(sona_code) = 4)
                    )
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_participant ON sessions(participant_id);
                CREATE INDEX IF NOT EXISTS idx_responses_session ON condition_responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
                """
            )
            self._migrate_schema(conn)

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    def _rebuild_core_tables_for_fleet_if_needed(self, conn: sqlite3.Connection) -> None:
        session_sql = self._table_sql(conn, "sessions") or ""
        block_sql = self._table_sql(conn, "condition_blocks") or ""
        rebuild_sessions = "Fleet" not in session_sql
        rebuild_blocks = "BETWEEN 1 AND 4" not in block_sql
        if not rebuild_sessions and not rebuild_blocks:
            return

        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA legacy_alter_table = ON")
        try:
            if rebuild_sessions:
                conn.executescript(
                    """
                    CREATE TABLE sessions_new (
                        session_id TEXT PRIMARY KEY,
                        participant_id TEXT NOT NULL UNIQUE,
                        study TEXT NOT NULL CHECK(study IN ('1A', '1B', 'Fleet')),
                        session_date TEXT NOT NULL,
                        condition_order_code TEXT NOT NULL,
                        task_order_code TEXT,
                        task_order_sequence TEXT,
                        experimenter_initials TEXT NOT NULL,
                        written_consent_confirmed INTEGER NOT NULL DEFAULT 0 CHECK(written_consent_confirmed IN (0, 1)),
                        current_workflow_state TEXT NOT NULL,
                        session_created_at TEXT NOT NULL,
                        session_completed_at TEXT
                    );
                    INSERT INTO sessions_new(
                        session_id, participant_id, study, session_date,
                        condition_order_code, task_order_code, task_order_sequence,
                        experimenter_initials, written_consent_confirmed,
                        current_workflow_state, session_created_at, session_completed_at
                    )
                    SELECT
                        session_id, participant_id, study, session_date,
                        condition_order_code, task_order_code, task_order_sequence,
                        experimenter_initials, written_consent_confirmed,
                        current_workflow_state, session_created_at, session_completed_at
                    FROM sessions;
                    DROP TABLE sessions;
                    ALTER TABLE sessions_new RENAME TO sessions;
                    """
                )
            if rebuild_blocks:
                conn.executescript(
                    """
                    CREATE TABLE condition_blocks_new (
                        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE RESTRICT,
                        participant_id TEXT NOT NULL,
                        study TEXT NOT NULL,
                        block_number INTEGER NOT NULL CHECK(block_number BETWEEN 1 AND 4),
                        block_id TEXT NOT NULL,
                        condition_code TEXT NOT NULL,
                        condition_name TEXT NOT NULL,
                        condition_position INTEGER NOT NULL CHECK(condition_position BETWEEN 1 AND 4),
                        task_order_code TEXT,
                        questionnaire_opened_at TEXT,
                        questionnaire_submitted_at TEXT,
                        PRIMARY KEY(session_id, block_id),
                        UNIQUE(session_id, block_number)
                    );
                    INSERT INTO condition_blocks_new(
                        session_id, participant_id, study, block_number, block_id,
                        condition_code, condition_name, condition_position,
                        task_order_code, questionnaire_opened_at, questionnaire_submitted_at
                    )
                    SELECT
                        session_id, participant_id, study, block_number, block_id,
                        condition_code, condition_name, condition_position,
                        task_order_code, questionnaire_opened_at, questionnaire_submitted_at
                    FROM condition_blocks;
                    DROP TABLE condition_blocks;
                    ALTER TABLE condition_blocks_new RENAME TO condition_blocks;
                    """
                )
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_participant ON sessions(participant_id);
                CREATE INDEX IF NOT EXISTS idx_responses_session ON condition_responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
                """
            )
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise DatabaseError("Fleet schema migration failed foreign-key verification.")
        finally:
            conn.execute("PRAGMA legacy_alter_table = OFF")
            conn.execute("PRAGMA foreign_keys = ON")

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Apply additive, idempotent migrations without deleting legacy data."""

        session_columns = self._columns(conn, "sessions")
        if "task_order_code" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN task_order_code TEXT")
        if "task_order_sequence" not in session_columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN task_order_sequence TEXT")

        block_columns = self._columns(conn, "condition_blocks")
        if "task_order_code" not in block_columns:
            conn.execute("ALTER TABLE condition_blocks ADD COLUMN task_order_code TEXT")

        background_columns = self._columns(conn, "participant_background")
        if "vr_experience_last_12_months" not in background_columns:
            conn.execute(
                "ALTER TABLE participant_background ADD COLUMN vr_experience_last_12_months TEXT"
            )
        if "gaming_controller_experience_last_12_months" not in background_columns:
            conn.execute(
                "ALTER TABLE participant_background "
                "ADD COLUMN gaming_controller_experience_last_12_months TEXT"
            )

        final_comparison_columns = self._columns(conn, "final_comparisons")
        for code in ("q2", "q3", "q4", "q5"):
            column = f"comparison_{code}_ranking_json"
            if column not in final_comparison_columns:
                conn.execute(f"ALTER TABLE final_comparisons ADD COLUMN {column} TEXT")

        fleet_final_columns = self._columns(conn, "fleet_final_responses")
        if "prioritization_strategy_comment" not in fleet_final_columns:
            conn.execute(
                "ALTER TABLE fleet_final_responses "
                "ADD COLUMN prioritization_strategy_comment TEXT NOT NULL DEFAULT ''"
            )

        # Existing development sessions retain their original categorical labels.
        # The new columns identify the updated variable while preserving old values.
        conn.execute(
            """
            UPDATE participant_background
            SET vr_experience_last_12_months = COALESCE(vr_experience_last_12_months, vr_experience),
                gaming_controller_experience_last_12_months =
                    COALESCE(gaming_controller_experience_last_12_months, gaming_experience)
            """
        )
        conn.execute(
            """
            UPDATE condition_blocks
            SET task_order_code = (
                SELECT sessions.task_order_code FROM sessions
                WHERE sessions.session_id = condition_blocks.session_id
            )
            WHERE task_order_code IS NULL
            """
        )
        self._rebuild_core_tables_for_fleet_if_needed(conn)
        self._migrate_legacy_tlx_responses(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO schema_migrations(version, applied_at, description)
            VALUES (?, ?, ?)
            """,
            (
                CURRENT_SCHEMA_VERSION,
                utc_now(),
                "Direct-comparison rankings, task order, background variables, Raw TLX aliases, compensation, and Fleet study",
            ),
        )

    @staticmethod
    def _migrate_legacy_tlx_responses(conn: sqlite3.Connection) -> None:
        legacy_dimensions = {
            "TLX_MENTAL": ("tlx_mental_20", "tlx_mental_100", "Mental Demand"),
            "TLX_PHYSICAL": ("tlx_physical_20", "tlx_physical_100", "Physical Demand"),
            "TLX_TEMPORAL": ("tlx_temporal_20", "tlx_temporal_100", "Temporal Demand"),
            "TLX_PERFORMANCE": ("tlx_performance_20", "tlx_performance_100", "Performance"),
            "TLX_EFFORT": ("tlx_effort_20", "tlx_effort_100", "Effort"),
            "TLX_FRUSTRATION": ("tlx_frustration_20", "tlx_frustration_100", "Frustration"),
        }
        for legacy_code, (code_20, code_100, label) in legacy_dimensions.items():
            for new_code, suffix, factor in (
                (code_20, "0–20", 0.2),
                (code_100, "0–100 equivalent", 1.0),
            ):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO condition_responses(
                        session_id, participant_id, study, block_id, measure_code,
                        measure_label, category, value, submitted_at
                    )
                    SELECT session_id, participant_id, study, block_id, ?, ?,
                           'NASA-TLX', value * ?, submitted_at
                    FROM condition_responses WHERE measure_code = ?
                    """,
                    (new_code, f"{label} ({suffix})", factor, legacy_code),
                )
        for legacy_code, new_code, factor, label in (
            ("TLX_RAW_MEAN", "raw_tlx_20", 0.2, "NASA-TLX mean (0–20)"),
            ("TLX_RAW_MEAN", "raw_tlx_100", 1.0, "NASA-TLX mean (0–100 equivalent)"),
        ):
            conn.execute(
                """
                INSERT OR IGNORE INTO condition_responses(
                    session_id, participant_id, study, block_id, measure_code,
                    measure_label, category, value, submitted_at
                )
                SELECT session_id, participant_id, study, block_id, ?, ?,
                       'Calculated score', value * ?, submitted_at
                FROM condition_responses WHERE measure_code = ?
                """,
                (new_code, label, factor, legacy_code),
            )

    @staticmethod
    def _audit(
        conn: sqlite3.Connection,
        session_id: str,
        event_type: str,
        details: Mapping[str, Any] | None = None,
        event_at: str | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO audit_log(session_id, event_type, event_at, details_json) VALUES (?, ?, ?, ?)",
            (session_id, event_type, event_at or utc_now(), json.dumps(details or {}, sort_keys=True)),
        )

    @staticmethod
    def _session_row(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found.")
        return row

    @staticmethod
    def _require_state(row: sqlite3.Row, expected: str) -> None:
        actual = row["current_workflow_state"]
        if actual != expected:
            raise InvalidTransitionError(
                f"This action requires workflow state {expected!r}; the session is currently {actual!r}."
            )

    def _backup_after_commit(self) -> Path:
        try:
            return self.backup_database()
        except Exception as exc:
            if isinstance(exc, BackupError):
                raise
            raise BackupError(
                "The data were saved, but the required database backup failed. "
                "Stop data collection and ask the experimenter to inspect storage."
            ) from exc

    def create_session(
        self,
        participant_id: str,
        study: str,
        condition_order_code: str,
        task_order_code: str,
        experimenter_initials: str,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        participant_id = validate_participant_id(participant_id, study)
        order = resolve_condition_order(study, condition_order_code)
        if study == "Fleet":
            task_order_code = "N/A"
        else:
            validate_task_order(task_order_code)
            if task_order_code == "N/A":
                raise ValidationError("Study 1A and 1B sessions require T-O1 or T-O2.")
        task_sequence = resolve_task_order(task_order_code)
        if not experimenter_initials.strip():
            raise ValidationError("Experimenter initials are required.")

        created_at = utc_now()
        local_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{participant_id}_{local_stamp}"
        chosen_date = session_date or date.today().isoformat()
        initial_state = "participant_background" if study == "Fleet" else "paper_consent"
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions(
                        session_id, participant_id, study, session_date,
                        condition_order_code, task_order_code, task_order_sequence,
                        experimenter_initials,
                        current_workflow_state, session_created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        participant_id,
                        study,
                        chosen_date,
                        condition_order_code,
                        task_order_code,
                        json.dumps(list(task_sequence)),
                        experimenter_initials.strip().upper(),
                        initial_state,
                        created_at,
                    ),
                )
                for block_number, condition_code in enumerate(order, start=1):
                    conn.execute(
                        """
                        INSERT INTO condition_blocks(
                            session_id, participant_id, study, block_number, block_id,
                            condition_code, condition_name, condition_position, task_order_code
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            participant_id,
                            study,
                            block_number,
                            f"B{block_number:02d}",
                            condition_code,
                            CONDITIONS[study][condition_code],
                            block_number,
                            task_order_code,
                        ),
                    )
                self._audit(
                    conn,
                    session_id,
                    "session_creation",
                    {
                        "condition_order_code": condition_order_code,
                        "task_order_code": task_order_code,
                        "task_order_sequence": list(task_sequence),
                        "fleet_sizes": (
                            [FLEET_SIZES[code] for code in order]
                            if study == "Fleet"
                            else None
                        ),
                    },
                    created_at,
                )
        except sqlite3.IntegrityError as exc:
            if "participant_id" in str(exc):
                raise DuplicateSubmissionError(
                    f"A session already exists for participant {participant_id}. Resume that session instead."
                ) from exc
            raise DatabaseError("The session could not be created because it conflicts with existing data.") from exc
        self._backup_after_commit()
        return self.get_session(session_id)

    def confirm_written_consent(self, session_id: str, confirmed: bool) -> None:
        if confirmed is not True:
            raise ValidationError("Written paper consent must be confirmed before proceeding.")
        submitted_at = utc_now()
        with self.transaction() as conn:
            session = self._session_row(conn, session_id)
            self._require_state(session, "paper_consent")
            conn.execute(
                """
                UPDATE sessions
                SET written_consent_confirmed = 1, current_workflow_state = 'participant_background'
                WHERE session_id = ?
                """,
                (session_id,),
            )
            self._audit(conn, session_id, "paper_consent_confirmation", event_at=submitted_at)
        self._backup_after_commit()

    def submit_background(self, session_id: str, data: Mapping[str, Any]) -> None:
        validate_background(data)
        submitted_at = utc_now()
        try:
            with self.transaction() as conn:
                session = self._session_row(conn, session_id)
                self._require_state(session, "participant_background")
                conn.execute(
                    """
                    INSERT INTO participant_background(
                        session_id, participant_id, age, handedness, robotics_experience,
                        vr_experience, gaming_experience, teleoperation_experience,
                        vr_experience_last_12_months,
                        gaming_controller_experience_last_12_months,
                        kinesthetic_experience_json, motion_sickness_susceptibility, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["participant_id"],
                        int(data["age"]),
                        data["handedness"],
                        data["robotics_experience"],
                        data["vr_experience_last_12_months"],
                        data["gaming_controller_experience_last_12_months"],
                        data["teleoperation_experience"],
                        data["vr_experience_last_12_months"],
                        data["gaming_controller_experience_last_12_months"],
                        json.dumps(list(data["kinesthetic_experience"])),
                        int(data["motion_sickness_susceptibility"]),
                        submitted_at,
                    ),
                )
                next_state = (
                    "waiting_condition_1" if session["study"] == "Fleet" else "understanding_check"
                )
                conn.execute(
                    "UPDATE sessions SET current_workflow_state = ? WHERE session_id = ?",
                    (next_state, session_id),
                )
                self._audit(conn, session_id, "background_submission", event_at=submitted_at)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSubmissionError("The participant background has already been submitted.") from exc
        self._backup_after_commit()

    def submit_understanding_check(self, session_id: str, data: Mapping[str, Any]) -> None:
        validate_understanding_check(data)
        submitted_at = utc_now()
        boolean_fields = (
            "risk_indicator_correct",
            "unattended_robots_correct",
            "multiple_paused_correct",
            "release_decision_correct",
            "ready_state_correct",
            "safety_response_correct",
            "retraining_required",
            "successful_practice_intervention",
        )
        try:
            with self.transaction() as conn:
                session = self._session_row(conn, session_id)
                self._require_state(session, "understanding_check")
                conn.execute(
                    """
                    INSERT INTO understanding_checks(
                        session_id, participant_id, risk_indicator_correct,
                        unattended_robots_correct, multiple_paused_correct,
                        release_decision_correct, ready_state_correct, safety_response_correct,
                        retraining_required, successful_practice_intervention,
                        practice_intervention_count, experimenter_notes, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["participant_id"],
                        *(int(bool(data[field])) for field in boolean_fields),
                        int(data["practice_intervention_count"]),
                        str(data.get("experimenter_notes", "")).strip(),
                        submitted_at,
                    ),
                )
                conn.execute(
                    "UPDATE sessions SET current_workflow_state = 'waiting_condition_1' WHERE session_id = ?",
                    (session_id,),
                )
                self._audit(conn, session_id, "understanding_check_completion", event_at=submitted_at)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSubmissionError("The understanding check has already been submitted.") from exc
        self._backup_after_commit()

    def unlock_condition(self, session_id: str, block_number: int) -> None:
        opened_at = utc_now()
        with self.transaction() as conn:
            session = self._session_row(conn, session_id)
            block_count = len(CONDITIONS[session["study"]])
            if not 1 <= block_number <= block_count:
                raise ValidationError(f"Block number must be between 1 and {block_count}.")
            self._require_state(session, f"waiting_condition_{block_number}")
            block = conn.execute(
                "SELECT * FROM condition_blocks WHERE session_id = ? AND block_number = ?",
                (session_id, block_number),
            ).fetchone()
            if block is None:
                raise DatabaseError("The assigned condition block is missing.")
            if block["questionnaire_opened_at"] is not None:
                raise InvalidTransitionError("This questionnaire has already been unlocked.")
            conn.execute(
                "UPDATE condition_blocks SET questionnaire_opened_at = ? WHERE session_id = ? AND block_number = ?",
                (opened_at, session_id, block_number),
            )
            conn.execute(
                "UPDATE sessions SET current_workflow_state = ? WHERE session_id = ?",
                (f"condition_{block_number}_questionnaire", session_id),
            )
            self._audit(
                conn,
                session_id,
                "questionnaire_unlock",
                {"block_id": block["block_id"], "condition_code": block["condition_code"]},
                opened_at,
            )
        self._backup_after_commit()

    def submit_condition_questionnaire(
        self,
        session_id: str,
        block_number: int,
        agreement: Mapping[str, Any],
        tlx: Mapping[str, Any],
    ) -> None:
        submitted_at = utc_now()
        with self.transaction() as conn:
            session = self._session_row(conn, session_id)
            block_count = len(CONDITIONS[session["study"]])
            if not 1 <= block_number <= block_count:
                raise ValidationError(f"Block number must be between 1 and {block_count}.")
            self._require_state(session, f"condition_{block_number}_questionnaire")
            block = conn.execute(
                "SELECT * FROM condition_blocks WHERE session_id = ? AND block_number = ?",
                (session_id, block_number),
            ).fetchone()
            if block is None:
                raise DatabaseError("The assigned condition block is missing.")
            if block["questionnaire_opened_at"] is None:
                raise InvalidTransitionError("The condition questionnaire has not been unlocked.")
            if block["questionnaire_submitted_at"] is not None:
                raise DuplicateSubmissionError("This condition questionnaire has already been submitted.")

            measures: list[tuple[str, str, str, float]] = []
            if session["study"] == "Fleet":
                fleet_size = FLEET_SIZES[block["condition_code"]]
                validate_fleet_condition_responses(agreement, tlx, fleet_size)
                validated_tlx_20 = validate_fleet_tlx_20(tlx)
                for dimension in FLEET_TLX_DIMENSIONS:
                    code = dimension["code"]
                    measures.append(
                        (code, f"{dimension['label']} (0-20)", "NASA-TLX", float(validated_tlx_20[code]))
                    )
                for item in FLEET_POST_CONDITION_ITEMS:
                    if fleet_size in item.get("na_for_fleet_sizes", []):
                        continue
                    measures.append(
                        (item["code"], item["label"], item["category"], float(agreement[item["code"]]))
                    )
            else:
                validate_condition_responses(session["study"], agreement, tlx)
                for item in POST_CONDITION_ITEMS[session["study"]]:
                    measures.append(
                        (item["code"], item["label"], item["category"], float(agreement[item["code"]]))
                    )
                validated_tlx = {
                    dimension["code_20"]: int(tlx[dimension["code_20"]])
                    for dimension in TLX_DIMENSIONS
                }
                converted_tlx = convert_tlx_to_100(validated_tlx)
                for dimension in TLX_DIMENSIONS:
                    code_20 = dimension["code_20"]
                    code_100 = dimension["code_100"]
                    label = dimension["label"]
                    measures.append(
                        (code_20, f"{label} (0–20)", "NASA-TLX", float(validated_tlx[code_20]))
                    )
                    measures.append(
                        (
                            code_100,
                            f"{label} (0–100 equivalent)",
                            "NASA-TLX",
                            float(converted_tlx[code_100]),
                        )
                    )
                raw_tlx_20, raw_tlx_100 = calculate_raw_tlx_scores(validated_tlx)
                measures.extend(
                    [
                        ("raw_tlx_20", "NASA-TLX mean (0–20)", "Calculated score", raw_tlx_20),
                        (
                            "raw_tlx_100",
                            "NASA-TLX mean (0–100 equivalent)",
                            "Calculated score",
                            raw_tlx_100,
                        ),
                    ]
                )
                if session["study"] == "1A":
                    measures.append(
                        (
                            "SPATIAL_UNDERSTANDING_MEAN",
                            "Spatial understanding mean",
                            "Calculated score",
                            calculate_spatial_understanding(agreement["A2"], agreement["A3"]),
                        )
                    )

            try:
                conn.executemany(
                    """
                    INSERT INTO condition_responses(
                        session_id, participant_id, study, block_id, measure_code,
                        measure_label, category, value, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            session["participant_id"],
                            session["study"],
                            block["block_id"],
                            code,
                            label,
                            category,
                            value,
                            submitted_at,
                        )
                        for code, label, category, value in measures
                    ],
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateSubmissionError("This condition questionnaire has already been submitted.") from exc

            conn.execute(
                """
                UPDATE condition_blocks SET questionnaire_submitted_at = ?
                WHERE session_id = ? AND block_number = ?
                """,
                (submitted_at, session_id, block_number),
            )
            if block_number < block_count:
                next_state = f"waiting_condition_{block_number + 1}"
            elif session["study"] == "Fleet":
                next_state = "fleet_final"
            else:
                next_state = "final_comparison"
            conn.execute(
                "UPDATE sessions SET current_workflow_state = ? WHERE session_id = ?",
                (next_state, session_id),
            )
            self._audit(
                conn,
                session_id,
                "questionnaire_submission",
                {"block_id": block["block_id"], "condition_code": block["condition_code"]},
                submitted_at,
            )
        self._backup_after_commit()

    def submit_final_comparison(
        self,
        session_id: str,
        ranking: Sequence[str],
        comparison_rankings: Mapping[str, Sequence[str]],
        reason: str,
        comment: str = "",
    ) -> None:
        submitted_at = utc_now()
        try:
            with self.transaction() as conn:
                session = self._session_row(conn, session_id)
                self._require_state(session, "final_comparison")
                validate_final_comparison(
                    session["study"], ranking, comparison_rankings, reason
                )
                completed_blocks = conn.execute(
                    """
                    SELECT COUNT(*) FROM condition_blocks
                    WHERE session_id = ? AND questionnaire_submitted_at IS NOT NULL
                    """,
                    (session_id,),
                ).fetchone()[0]
                expected_blocks = len(CONDITIONS[session["study"]])
                if completed_blocks != expected_blocks:
                    raise InvalidTransitionError(
                        f"All {expected_blocks} condition questionnaires must be complete first."
                    )
                conn.execute(
                    """
                    INSERT INTO final_comparisons(
                        session_id, participant_id, study, ranking_json,
                        comparison_q2, comparison_q3, comparison_q4, comparison_q5,
                        comparison_q2_ranking_json, comparison_q3_ranking_json,
                        comparison_q4_ranking_json, comparison_q5_ranking_json,
                        preference_reason, additional_comment, submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["participant_id"],
                        session["study"],
                        json.dumps(list(ranking)),
                        *(comparison_rankings[code][0] for code in ("q2", "q3", "q4", "q5")),
                        *(
                            json.dumps(list(comparison_rankings[code]))
                            for code in ("q2", "q3", "q4", "q5")
                        ),
                        reason.strip(),
                        comment.strip(),
                        submitted_at,
                    ),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET current_workflow_state = 'compensation'
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                self._audit(conn, session_id, "final_comparison_submission", event_at=submitted_at)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSubmissionError("The final comparison has already been submitted.") from exc
        self._backup_after_commit()

    def submit_fleet_final_response(self, session_id: str, data: Mapping[str, Any]) -> None:
        validated = validate_fleet_final_response(data)
        submitted_at = utc_now()
        try:
            with self.transaction() as conn:
                session = self._session_row(conn, session_id)
                self._require_state(session, "fleet_final")
                if session["study"] != "Fleet":
                    raise InvalidTransitionError("The final Fleet questionnaire is only for Fleet sessions.")
                completed_blocks = conn.execute(
                    """
                    SELECT COUNT(*) FROM condition_blocks
                    WHERE session_id = ? AND questionnaire_submitted_at IS NOT NULL
                    """,
                    (session_id,),
                ).fetchone()[0]
                expected_blocks = len(CONDITIONS["Fleet"])
                if completed_blocks != expected_blocks:
                    raise InvalidTransitionError(
                        f"All {expected_blocks} Fleet condition questionnaires must be complete first."
                    )
                conn.execute(
                    """
                    INSERT INTO fleet_final_responses(
                        session_id, participant_id, study,
                        preferred_sustainable_fleet_size, best_balance_fleet_size,
                        prioritization_strategy_comment, fleet_overload_comment,
                        submitted_at
                    ) VALUES (?, ?, 'Fleet', ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["participant_id"],
                        validated["preferred_sustainable_fleet_size"],
                        validated["best_balance_fleet_size"] or None,
                        validated["prioritization_strategy_comment"],
                        validated["fleet_overload_comment"],
                        submitted_at,
                    ),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET current_workflow_state = 'compensation'
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                self._audit(conn, session_id, "fleet_final_submission", event_at=submitted_at)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSubmissionError("The final Fleet questionnaire has already been submitted.") from exc
        self._backup_after_commit()

    def submit_compensation(
        self,
        session_id: str,
        compensation_type: str,
        sona_code: str | None = None,
    ) -> None:
        compensation_type, normalized_sona_code = validate_compensation(
            compensation_type, sona_code
        )
        created_at = utc_now()
        try:
            with self.transaction() as conn:
                session = self._session_row(conn, session_id)
                self._require_state(session, "compensation")
                conn.execute(
                    """
                    INSERT INTO compensation_records(
                        session_id, participant_id, compensation_type, sona_code, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session["participant_id"],
                        compensation_type,
                        normalized_sona_code,
                        created_at,
                    ),
                )
                conn.execute(
                    """
                    UPDATE sessions
                    SET current_workflow_state = 'complete', session_completed_at = ?
                    WHERE session_id = ?
                    """,
                    (created_at, session_id),
                )
                self._audit(
                    conn,
                    session_id,
                    "compensation_submission",
                    {"compensation_type": compensation_type},
                    created_at,
                )
                self._audit(conn, session_id, "session_completion", event_at=created_at)
        except sqlite3.IntegrityError as exc:
            raise DuplicateSubmissionError("The compensation choice has already been submitted.") from exc
        self._backup_after_commit()

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            return self._decoded_session(self._session_row(conn, session_id))

    @staticmethod
    def _decoded_session(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        raw_sequence = result.get("task_order_sequence")
        if raw_sequence:
            try:
                result["task_order_sequence"] = json.loads(raw_sequence)
            except (TypeError, json.JSONDecodeError):
                result["task_order_sequence"] = []
        else:
            result["task_order_sequence"] = []
        return result

    def recover_session(self, identifier: str) -> dict[str, Any]:
        normalized = identifier.strip().upper()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions
                WHERE UPPER(session_id) = ? OR UPPER(participant_id) = ?
                ORDER BY session_created_at DESC LIMIT 1
                """,
                (normalized, normalized),
            ).fetchone()
            if row is None:
                raise SessionNotFoundError("No session matches that participant ID or session ID.")
            return self._decoded_session(row)

    def list_incomplete_sessions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions WHERE current_workflow_state != 'complete'
                ORDER BY session_created_at DESC
                """
            ).fetchall()
            return [self._decoded_session(row) for row in rows]

    def get_task_order_counts(self, study: str) -> dict[str, int]:
        counts = {code: 0 for code in TASK_ORDERS}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT task_order_code, COUNT(*) AS count
                FROM sessions WHERE study = ? AND task_order_code IS NOT NULL
                GROUP BY task_order_code
                """,
                (study,),
            ).fetchall()
        for row in rows:
            if row["task_order_code"] in counts:
                counts[row["task_order_code"]] = row["count"]
        return counts

    def get_condition_order_counts(self, study: str) -> dict[str, int]:
        counts = {code: 0 for code in CONDITION_ORDERS[study]}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT condition_order_code, COUNT(*) AS count
                FROM sessions WHERE study = ?
                GROUP BY condition_order_code
                """,
                (study,),
            ).fetchall()
        for row in rows:
            if row["condition_order_code"] in counts:
                counts[row["condition_order_code"]] = row["count"]
        return counts

    def get_condition_blocks(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            self._session_row(conn, session_id)
            rows = conn.execute(
                "SELECT * FROM condition_blocks WHERE session_id = ? ORDER BY block_number",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_background(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM participant_background WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["kinesthetic_experience"] = json.loads(result.pop("kinesthetic_experience_json"))
            result.pop("vr_experience", None)
            result.pop("gaming_experience", None)
            return result

    def get_understanding_check(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM understanding_checks WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_condition_responses(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.*, b.block_number, b.condition_code, b.condition_name,
                       b.condition_position, b.task_order_code
                FROM condition_responses r
                JOIN condition_blocks b ON b.session_id = r.session_id AND b.block_id = r.block_id
                WHERE r.session_id = ?
                ORDER BY b.block_number, r.response_id
                """,
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_compensation(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM compensation_records WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
            return int(row[0] or 0)

    def get_final_comparison(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM final_comparisons WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["ranking"] = json.loads(result.pop("ranking_json"))
            for code in ("q2", "q3", "q4", "q5"):
                raw_ranking = result.pop(f"comparison_{code}_ranking_json", None)
                result[f"comparison_{code}_ranking"] = (
                    json.loads(raw_ranking) if raw_ranking else None
                )
            return result

    def get_fleet_final_response(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM fleet_final_responses WHERE session_id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_audit_log(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE session_id = ? ORDER BY audit_id", (session_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def record_export(self, session_id: str, export_directory: str | Path) -> None:
        with self.transaction() as conn:
            self._session_row(conn, session_id)
            self._audit(
                conn,
                session_id,
                "export",
                {"directory": str(export_directory)},
            )
        self._backup_after_commit()

    def backup_database(self) -> Path:
        """Create and verify a consistent timestamped SQLite backup."""

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = self.backup_dir / f"{self.db_path.stem}_{stamp}.sqlite3"
        try:
            source = self.connect()
            destination = sqlite3.connect(backup_path)
            try:
                source.backup(destination)
                destination.commit()
                integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                destination.close()
                source.close()
            if integrity != "ok" or not backup_path.is_file() or backup_path.stat().st_size == 0:
                raise BackupError("The database backup could not be verified.")
            backups = sorted(
                self.backup_dir.glob(f"{self.db_path.stem}_*.sqlite3"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[self.backup_retention :]:
                old_backup.unlink()
        except Exception as exc:
            if backup_path.exists():
                backup_path.unlink()
            if isinstance(exc, BackupError):
                raise
            raise BackupError("The database backup could not be created or verified.") from exc
        return backup_path

    def latest_backup(self) -> Path | None:
        backups = sorted(
            self.backup_dir.glob(f"{self.db_path.stem}_*.sqlite3"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return backups[0] if backups else None
