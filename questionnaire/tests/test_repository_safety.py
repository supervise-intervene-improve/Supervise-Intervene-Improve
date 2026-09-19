from __future__ import annotations

from scripts.check_before_commit import content_violations, path_violations


def test_repository_safety_rejects_sensitive_paths():
    assert path_violations("data/questionnaire.sqlite3")
    assert path_violations("exports/session/results.csv")
    assert path_violations("participant_data/P1A-001.txt")
    assert path_violations("consent/signed-form.pdf")
    assert path_violations("compensation_exports/sona_codes.txt")
    assert path_violations("administrative_exports/payment_status.json")
    assert path_violations("sona_codes.json")
    assert path_violations("study_compensation.json")
    assert path_violations(".env")
    assert path_violations(".streamlit/secrets.toml")


def test_repository_safety_allows_runtime_gitkeep_and_example_environment():
    assert not path_violations("data/.gitkeep")
    assert not path_violations("exports/.gitkeep")
    assert not path_violations("backups/.gitkeep")
    assert not path_violations("logs/.gitkeep")
    assert not path_violations(".env.example")


def test_repository_safety_detects_credentials_but_allows_placeholders():
    assert content_violations(b"API_KEY=replace_with_local_value") == []
    assert content_violations(b"API_KEY=actual-secret-value")
    private_key_marker = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert content_violations(private_key_marker)
