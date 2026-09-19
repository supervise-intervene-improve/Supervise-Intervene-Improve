from __future__ import annotations

import pytest

from questionnaire_definitions import BACKGROUND_OPTIONS, FLEET_TLX_DIMENSIONS, resolve_condition_order
from validation import (
    ValidationError,
    calculate_raw_tlx,
    calculate_raw_tlx_scores,
    calculate_spatial_understanding,
    convert_tlx_to_100,
    validate_compensation,
    validate_fleet_final_response,
    validate_fleet_tlx_20,
    validate_final_comparison,
    validate_participant_id,
    validate_payment_url,
    validate_ranking,
    validate_task_order,
    validate_tlx_20,
)


@pytest.mark.parametrize(
    ("participant_id", "study", "expected"),
    [
        ("P1A-001", "1A", "P1A-001"),
        ("p1a-100", "1A", "P1A-100"),
        ("P1B-001", "1B", "P1B-001"),
        ("P1B-100", "1B", "P1B-100"),
        ("P1C-001", "Fleet", "P1C-001"),
        ("p1c-100", "Fleet", "P1C-100"),
    ],
)
def test_participant_id_validation_accepts_valid_ids(participant_id, study, expected):
    assert validate_participant_id(participant_id, study) == expected


@pytest.mark.parametrize(
    ("participant_id", "study"),
    [
        ("P1A-000", "1A"),
        ("P1A-101", "1A"),
        ("P1B-001", "1A"),
        ("P1A-001", "1B"),
        ("P1C-001", "1A"),
        ("P1A-001", "Fleet"),
        ("P2A-001", "1A"),
        ("Jane", "1A"),
    ],
)
def test_participant_id_validation_rejects_invalid_or_mismatched_ids(participant_id, study):
    with pytest.raises(ValidationError):
        validate_participant_id(participant_id, study)


def test_all_study_1a_condition_orders_resolve():
    assert resolve_condition_order("1A", "S-O1") == ("S1", "S2", "S3")
    assert resolve_condition_order("1A", "S-O2") == ("S1", "S3", "S2")
    assert resolve_condition_order("1A", "S-O3") == ("S2", "S1", "S3")
    assert resolve_condition_order("1A", "S-O4") == ("S2", "S3", "S1")
    assert resolve_condition_order("1A", "S-O5") == ("S3", "S1", "S2")
    assert resolve_condition_order("1A", "S-O6") == ("S3", "S2", "S1")


def test_all_study_1b_condition_orders_resolve():
    assert resolve_condition_order("1B", "C-O1") == ("C1", "C2", "C3")
    assert resolve_condition_order("1B", "C-O2") == ("C1", "C3", "C2")
    assert resolve_condition_order("1B", "C-O3") == ("C2", "C1", "C3")
    assert resolve_condition_order("1B", "C-O4") == ("C2", "C3", "C1")
    assert resolve_condition_order("1B", "C-O5") == ("C3", "C1", "C2")
    assert resolve_condition_order("1B", "C-O6") == ("C3", "C2", "C1")


def test_fleet_condition_orders_resolve_with_four_fleet_sizes():
    assert resolve_condition_order("Fleet", "F-O1") == ("F1", "F3", "F6", "F9")
    assert resolve_condition_order("Fleet", "F-O24") == ("F9", "F6", "F3", "F1")


def test_task_order_validation():
    assert validate_task_order("T-O1") == "T-O1"
    assert validate_task_order("T-O2") == "T-O2"
    assert validate_task_order("N/A") == "N/A"
    with pytest.raises(ValidationError):
        validate_task_order("T-O3")


def test_fleet_tlx_20_validation_and_performance_direction():
    values = {dimension["code"]: 10 for dimension in FLEET_TLX_DIMENSIONS}
    values["nasa_performance"] = 0
    assert validate_fleet_tlx_20(values)["nasa_performance"] == 0
    with pytest.raises(ValidationError):
        validate_fleet_tlx_20({**values, "nasa_effort": 21})


def test_fleet_final_response_validation():
    validated = validate_fleet_final_response(
        {
            "preferred_sustainable_fleet_size": "6 robots",
            "best_balance_fleet_size": "",
            "prioritization_strategy_comment": "I used urgency first.",
            "fleet_overload_comment": "Nine robots felt too demanding.",
        }
    )
    assert validated["preferred_sustainable_fleet_size"] == "6 robots"
    assert validated["best_balance_fleet_size"] == ""
    assert validated["prioritization_strategy_comment"] == "I used urgency first."
    with pytest.raises(ValidationError):
        validate_fleet_final_response({"preferred_sustainable_fleet_size": ""})


def test_raw_nasa_tlx_calculation_is_unweighted_mean():
    values = {
        "tlx_mental_20": 0,
        "tlx_physical_20": 2,
        "tlx_temporal_20": 4,
        "tlx_performance_20": 6,
        "tlx_effort_20": 8,
        "tlx_frustration_20": 10,
    }
    assert calculate_raw_tlx(values) == 5.0
    assert calculate_raw_tlx_scores(values) == (5.0, 25.0)
    converted = convert_tlx_to_100(values)
    assert converted["tlx_mental_100"] == 0
    assert converted["tlx_performance_100"] == 30
    assert converted["tlx_frustration_100"] == 50


def test_raw_nasa_tlx_rejects_missing_unconfirmed_or_out_of_range_values():
    with pytest.raises(ValidationError):
        calculate_raw_tlx({})
    with pytest.raises(ValidationError):
        validate_tlx_20(
            {
                "tlx_mental_20": 21,
                "tlx_physical_20": 2,
                "tlx_temporal_20": 4,
                "tlx_performance_20": 6,
                "tlx_effort_20": 8,
                "tlx_frustration_20": 10,
            }
        )
    with pytest.raises(ValidationError):
        validate_tlx_20(
            {
                "tlx_mental_20": 1.5,
                "tlx_physical_20": 2,
                "tlx_temporal_20": 4,
                "tlx_performance_20": 6,
                "tlx_effort_20": 8,
                "tlx_frustration_20": 10,
            }
        )
    with pytest.raises(ValidationError):
        validate_tlx_20(
            {
                "tlx_mental_20": None,
                "tlx_physical_20": 2,
                "tlx_temporal_20": 4,
                "tlx_performance_20": 6,
                "tlx_effort_20": 8,
                "tlx_frustration_20": 10,
            }
        )


def test_spatial_understanding_calculation():
    assert calculate_spatial_understanding(5, 7) == 6.0


def test_ranking_requires_every_condition_exactly_once():
    assert validate_ranking(["S3", "S1", "S2"], ["S1", "S2", "S3"]) == ["S3", "S1", "S2"]
    with pytest.raises(ValidationError):
        validate_ranking(["S1", "S1", "S3"], ["S1", "S2", "S3"])
    with pytest.raises(ValidationError):
        validate_ranking(["S1", "S2", "C1"], ["S1", "S2", "S3"])


def test_final_comparison_requires_complete_rankings_for_every_question():
    rankings = {
        code: ["S1", "S2", "S3"] for code in ("q2", "q3", "q4", "q5")
    }
    validate_final_comparison(
        "1A", ["S3", "S1", "S2"], rankings, "It was easiest to use."
    )

    rankings["q4"] = ["S1", "S1", "S3"]
    with pytest.raises(ValidationError, match="rank each condition exactly once"):
        validate_final_comparison(
            "1A", ["S3", "S1", "S2"], rankings, "It was easiest to use."
        )


def test_last_12_month_vr_and_gaming_options_are_authoritative():
    assert BACKGROUND_OPTIONS["vr_experience_last_12_months"] == [
        "Never",
        "Once",
        "2-5 times",
        "Several times during the year",
        "Approximately monthly",
        "Approximately weekly",
        "Several times per week or more",
    ]
    assert BACKGROUND_OPTIONS["gaming_controller_experience_last_12_months"] == [
        "Never",
        "Once or twice",
        "Several times during the year",
        "Approximately monthly",
        "Approximately weekly",
        "Several times per week",
        "Daily or almost daily",
    ]


def test_compensation_choice_and_sona_code_validation():
    assert validate_compensation("payment", "ignored") == ("payment", None)
    assert validate_compensation("sona_credit", "0427") == ("sona_credit", "0427")
    for invalid in (None, "123", "12345", "12A4", " 123 "):
        with pytest.raises(ValidationError):
            validate_compensation("sona_credit", invalid)
    with pytest.raises(ValidationError):
        validate_compensation("cash")


def test_payment_url_handling_requires_https_and_allows_empty_configuration():
    assert validate_payment_url("") is None
    assert validate_payment_url("https://payments.example.edu/form") == (
        "https://payments.example.edu/form"
    )
    with pytest.raises(ValidationError):
        validate_payment_url("http://payments.example.edu/form")
