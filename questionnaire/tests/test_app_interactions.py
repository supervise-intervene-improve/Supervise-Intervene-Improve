from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from database import Database
from tests.conftest import create_ready_session, valid_agreement, valid_tlx


def configured_database(tmp_path, monkeypatch):
    st.cache_resource.clear()
    database_path = tmp_path / "data" / "questionnaire.sqlite3"
    backup_path = tmp_path / "backups"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("HRI_BACKUP_DIR", str(backup_path))
    monkeypatch.setenv("HRI_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setattr(config, "DATA_DIR", database_path.parent)
    monkeypatch.setattr(config, "DATABASE_PATH", database_path)
    monkeypatch.setattr(config, "BACKUP_DIR", backup_path)
    monkeypatch.setattr(config, "EXPORT_DIR", tmp_path / "exports")
    return Database(database_path, backup_path)


def test_nasa_tlx_slider_change_confirms_matching_rating(tmp_path, monkeypatch):
    database = configured_database(tmp_path, monkeypatch)
    session = create_ready_session(database)
    database.unlock_condition(session["session_id"], 1)

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    )
    app.session_state["active_session_id"] = session["session_id"]
    app.run()

    assert not app.exception
    assert len(app.slider) == 6
    assert len(app.checkbox) == 6
    assert app.checkbox[0].value is False

    app.slider[0].set_value(7).run()

    assert not app.exception
    assert app.checkbox[0].value is True
    assert "NASA-TLX workload" in [heading.value for heading in app.subheader]


def test_final_comparison_uses_drag_rankings_and_requires_reason(tmp_path, monkeypatch):
    database = configured_database(tmp_path, monkeypatch)
    session = create_ready_session(database)
    for block_number in (1, 2, 3):
        database.unlock_condition(session["session_id"], block_number)
        database.submit_condition_questionnaire(
            session["session_id"],
            block_number,
            valid_agreement("1A"),
            valid_tlx(),
        )

    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    )
    app.session_state["active_session_id"] = session["session_id"]
    app.run()

    instruction = "Drag to reorder the list. Top is Best, middle is Second, bottom is Third."
    assert [caption.value for caption in app.caption].count(instruction) == 5
    confirmation_label = (
        "Please confirm that the rankings above reflect your final choices before saving. *"
    )
    assert [checkbox.label for checkbox in app.checkbox].count(confirmation_label) == 5
    assert all(checkbox.value is False for checkbox in app.checkbox)

    app.button[0].click().run()

    assert not app.exception
    assert any(
        "Please confirm that the rankings above reflect your final choices" in message.value
        for message in app.error
    )
    assert database.get_session(session["session_id"])["current_workflow_state"] == (
        "final_comparison"
    )

    for index in range(4):
        app.checkbox[index].set_value(True).run()
    app.button[0].click().run()

    assert not app.exception
    assert any(
        "Unconfirmed: Direct comparison 4" in message.value for message in app.error
    )

    app.checkbox[4].set_value(True).run()
    app.button[0].click().run()

    assert not app.exception
    assert any(
        "main reason for the preferred condition is required" in message.value
        for message in app.error
    )

    app.text_area[0].set_value("The first-ranked option felt easiest to use.").run()
    app.button[0].click().run()

    assert not app.exception
    assert database.get_session(session["session_id"])["current_workflow_state"] == "compensation"
