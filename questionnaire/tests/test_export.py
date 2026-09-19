from __future__ import annotations

import json

import pandas as pd
import pytest

from export import ExportError, FLEET_CONDITION_COLUMNS, LONG_COLUMNS, export_session
from tests.conftest import complete_fleet_session, complete_session, create_ready_session


def test_export_creation_for_complete_session(db, tmp_path):
    session = complete_session(db, "1A", "P1A-001", "S-O3")
    paths = export_session(db, session["session_id"], tmp_path / "exports")

    assert set(paths) == {"long", "wide", "metadata"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())

    long_frame = pd.read_csv(paths["long"])
    assert list(long_frame.columns) == LONG_COLUMNS
    assert set(long_frame["block_id"]) == {"B01", "B02", "B03"}
    assert set(long_frame["task_order_code"]) == {"T-O1"}
    assert "raw_tlx_20" in set(long_frame["measure_code"])
    assert "raw_tlx_100" in set(long_frame["measure_code"])
    assert "tlx_performance_20" in set(long_frame["measure_code"])
    assert "tlx_performance_100" in set(long_frame["measure_code"])
    assert "SPATIAL_UNDERSTANDING_MEAN" in set(long_frame["measure_code"])

    wide_frame = pd.read_csv(paths["wide"])
    assert len(wide_frame) == 1
    assert wide_frame.loc[0, "session_id"] == session["session_id"]
    assert wide_frame.loc[0, "task_order_code"] == "T-O1"
    assert wide_frame.loc[0, "B01_block_id"] == "B01"
    assert wide_frame.loc[0, "B01_block_number"] == 1
    assert wide_frame.loc[0, "B01_condition_position"] == 1
    assert wide_frame.loc[0, "B01_task_order_code"] == "T-O1"
    assert "B01_A1" in wide_frame.columns
    assert "B01_tlx_mental_20" in wide_frame.columns
    assert "B01_tlx_mental_100" in wide_frame.columns
    assert wide_frame.loc[0, "final_comparison_q2_rank_1"] == "S1"
    assert wide_frame.loc[0, "final_comparison_q2_rank_3"] == "S3"

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["condition_order_code"] == "S-O3"
    assert metadata["task_order_code"] == "T-O1"
    assert metadata["task_order_sequence"] == ["T-shape", "Cups"]
    assert {block["task_order_code"] for block in metadata["condition_blocks"]} == {"T-O1"}
    assert {block["session_id"] for block in metadata["condition_blocks"]} == {
        session["session_id"]
    }
    assert len(metadata["condition_blocks"]) == 3
    assert metadata["current_workflow_state"] == "complete"
    assert db.get_audit_log(session["session_id"])[-1]["event_type"] == "export"

    combined_headers = " ".join(long_frame.columns) + " " + " ".join(wide_frame.columns)
    assert "sona_code" not in combined_headers.lower()
    assert "compensation" not in combined_headers.lower()
    assert "sona_code" not in paths["metadata"].read_text(encoding="utf-8").lower()


def test_incomplete_session_export_is_rejected(db, tmp_path):
    session = create_ready_session(db)
    with pytest.raises(ExportError):
        export_session(db, session["session_id"], tmp_path / "exports")


def test_fleet_export_creation_for_complete_session_is_pandas_readable(db, tmp_path):
    session = complete_fleet_session(db, "P1C-001", "F-O1")
    paths = export_session(db, session["session_id"], tmp_path / "exports")

    assert set(paths) == {"fleet_background", "fleet_condition", "fleet_final", "metadata"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())

    condition_frame = pd.read_csv(paths["fleet_condition"])
    assert list(condition_frame.columns) == FLEET_CONDITION_COLUMNS
    assert condition_frame["participant_id"].tolist() == ["P1C-001"] * 4
    assert condition_frame["fleet_size"].tolist() == [1, 3, 6, 9]
    assert condition_frame["condition_order"].tolist() == [1, 2, 3, 4]
    assert set(condition_frame["questionnaire_type"]) == {"post_condition"}
    assert condition_frame.loc[0, "nasa_performance"] == 2
    assert pd.isna(condition_frame.loc[0, "fleet_attention_switch"])
    assert pd.isna(condition_frame.loc[0, "fleet_prioritization"])
    assert condition_frame.loc[1, "fleet_attention_switch"] == 6
    assert condition_frame.loc[1, "fleet_prioritization"] == 6
    assert "raw_tlx_100" not in condition_frame.columns

    final_frame = pd.read_csv(paths["fleet_final"])
    assert final_frame.loc[0, "participant_id"] == "P1C-001"
    assert final_frame.loc[0, "preferred_sustainable_fleet_size"] == "6 robots"
    assert final_frame.loc[0, "best_balance_fleet_size"] == "3 robots"
    assert final_frame.loc[0, "prioritization_strategy_comment"] == "I used urgency first."

    background_frame = pd.read_csv(paths["fleet_background"])
    assert background_frame.loc[0, "participant_id"] == "P1C-001"
    assert background_frame.loc[0, "questionnaire_type"] == "background"
    assert background_frame.loc[0, "age"] == 29
    assert "robotics_experience_score" in background_frame.columns

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["study"] == "Fleet"
    assert [block["fleet_size"] for block in metadata["condition_blocks"]] == [1, 3, 6, 9]
    assert metadata["background"]["questionnaire_type"] == "background"
    assert metadata["fleet_final_response"]["questionnaire_type"] == "fleet_final"
