from __future__ import annotations

from questionnaire_definitions import (
    BACKGROUND_QUESTIONS,
    FINAL_COMPARISON_INTRODUCTION,
    POST_CONDITION_INTRODUCTION,
    TASK_ORDERS,
    TLX_DIMENSIONS,
    FLEET_TLX_DIMENSIONS,
    condition_order_display_name,
)


def test_task_orders_and_introductions_are_authoritative():
    assert TASK_ORDERS == {
        "T-O1": ("T-shape", "Cups"),
        "T-O2": ("Cups", "T-shape"),
        "N/A": (),
    }
    assert POST_CONDITION_INTRODUCTION == (
        "Thinking about the condition you just experienced, how was your overall experience? "
        "Please indicate how strongly you agree or disagree with each of the following statements. "
        "There are no right or wrong answers. We are interested in your individual experience."
    )
    assert FINAL_COMPARISON_INTRODUCTION.startswith(
        "You have now completed all three conditions."
    )


def test_last_12_month_question_wording_and_tlx_performance_direction():
    assert BACKGROUND_QUESTIONS["vr_experience_last_12_months"] == (
        "During the last 12 months, how often have you used a virtual-reality headset?"
    )
    assert BACKGROUND_QUESTIONS["gaming_controller_experience_last_12_months"] == (
        "During the last 12 months, how often have you played video games or used handheld motion/game controllers?"
    )
    performance = next(item for item in TLX_DIMENSIONS if item["name"] == "performance")
    assert performance["low"] == "Bad"
    assert performance["high"] == "Perfect"
    assert "description" not in performance


def test_assignment_order_names_omit_the_fixed_part_of_each_study_condition():
    assert condition_order_display_name("1A", "S1") == "Desktop-RGB"
    assert condition_order_display_name("1A", "S3") == "VR-PointCloud"
    assert condition_order_display_name("1B", "C1") == "Kinesthetic Teaching"
    assert condition_order_display_name("1B", "C2") == "Motion Controller"
    assert condition_order_display_name("Fleet", "F6") == "6 robots"


def test_fleet_nasa_performance_keeps_standard_tlx_direction():
    performance = next(item for item in FLEET_TLX_DIMENSIONS if item["name"] == "performance")
    assert performance["code"] == "nasa_performance"
    assert performance["low"] == "Failure / completely unsuccessful"
    assert performance["high"] == "Perfect / completely successful"
