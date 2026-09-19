"""Authoritative questionnaire wording and study configuration."""

from __future__ import annotations

from itertools import permutations


STUDY_LABELS = {
    "1A": "Study 1A — Supervision Interface Comparison",
    "1B": "Study 1B — Control Interface Comparison",
    "Fleet": "Fleet — Fleet-Size Robot Supervision",
}

CONDITIONS = {
    "1A": {
        "S1": "Desktop-RGB + Kinesthetic Teaching",
        "S2": "VR-RGB + Kinesthetic Teaching",
        "S3": "VR-PointCloud + Kinesthetic Teaching",
    },
    "1B": {
        "C1": "VR-PointCloud + Kinesthetic Teaching",
        "C2": "VR-PointCloud + Motion Controller",
        "C3": "VR-PointCloud + FACTR",
    },
    "Fleet": {
        "F1": "1 robot",
        "F3": "3 robots",
        "F6": "6 robots",
        "F9": "9 robots",
    },
}

FLEET_SIZES = {
    "F1": 1,
    "F3": 3,
    "F6": 6,
    "F9": 9,
}


def condition_order_display_name(study: str, condition_code: str) -> str:
    """Return the concise condition name used in the assignment-order picker."""

    name = CONDITIONS[study][condition_code]
    if study == "1A":
        return name.removesuffix(" + Kinesthetic Teaching")
    if study == "1B":
        return name.removeprefix("VR-PointCloud + ")
    return name


def _orders(prefix: str, condition_codes: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        f"{prefix}-O{number}": order
        for number, order in enumerate(permutations(condition_codes), start=1)
    }


CONDITION_ORDERS = {
    "1A": _orders("S", ("S1", "S2", "S3")),
    "1B": _orders("C", ("C1", "C2", "C3")),
    "Fleet": _orders("F", ("F1", "F3", "F6", "F9")),
}

TASK_ORDERS = {
    "T-O1": ("T-shape", "Cups"),
    "T-O2": ("Cups", "T-shape"),
    "N/A": (),
}

AGREEMENT_OPTIONS = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Somewhat disagree",
    4: "Neither agree nor disagree",
    5: "Somewhat agree",
    6: "Agree",
    7: "Strongly agree",
}

POST_CONDITION_ITEMS = {
    "1A": [
        {
            "code": "A1",
            "label": "Selected-robot task state",
            "category": "Scene and spatial understanding",
            "text": "I could clearly understand what the selected robot was doing and what correction was required.",
        },
        {
            "code": "A2",
            "label": "Spatial relationships",
            "category": "Scene and spatial understanding",
            "text": "The task-relevant objects and their spatial relationships were easy to understand.",
        },
        {
            "code": "A3",
            "label": "Relative pose judgment",
            "category": "Scene and spatial understanding",
            "text": "I could accurately judge the relative position and orientation of the robot, the manipulated object, and the target.",
        },
        {
            "code": "A4",
            "label": "Attention identification",
            "category": "Supervisory attention and awareness",
            "text": "It was easy to identify which robot required my attention.",
        },
        {
            "code": "A5",
            "label": "Situation awareness",
            "category": "Supervisory attention and awareness",
            "text": "I remained aware of the other robots while focusing on the selected robot.",
        },
        {
            "code": "A6",
            "label": "Risk-indicator usefulness",
            "category": "Intervention decision support",
            "text": "The risk indicator helped me decide when to intervene.",
        },
        {
            "code": "A7",
            "label": "Intervention confidence",
            "category": "Intervention decision support",
            "text": "I was confident that my correction was sufficient before returning control to autonomy.",
        },
        {
            "code": "A8",
            "label": "Visual comfort",
            "category": "Visual comfort",
            "text": "The visual display was comfortable to view throughout the condition.",
        },
    ],
    "1B": [
        {
            "code": "B1",
            "label": "Takeover clarity",
            "category": "Takeover transition",
            "text": "I clearly understood when manual control had begun.",
        },
        {
            "code": "B2",
            "label": "Controller mapping",
            "category": "Control quality",
            "text": "The robot moved in the direction I expected.",
        },
        {
            "code": "B3",
            "label": "Correction precision",
            "category": "Control quality",
            "text": "I could make precise corrections with this controller.",
        },
        {
            "code": "B4",
            "label": "Gripper control",
            "category": "Control quality",
            "text": "It was easy to control the gripper.",
        },
        {
            "code": "B5",
            "label": "Perceived control",
            "category": "Perceived agency",
            "text": "I felt in control while correcting the robot.",
        },
        {
            "code": "B6",
            "label": "Release ease",
            "category": "Interaction usability",
            "text": "It was easy to return control to the autonomous policy.",
        },
        {
            "code": "B7",
            "label": "Postural comfort",
            "category": "Interaction usability",
            "text": "I could maintain a comfortable posture while using the controller.",
        },
    ],
}

TLX_DIMENSIONS = [
    {
        "name": "mental",
        "code_20": "tlx_mental_20",
        "code_100": "tlx_mental_100",
        "label": "Mental Demand",
        "question": "How mentally demanding was the condition?",
        "description": "Consider how much thinking, deciding, remembering, looking, or searching was required.",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "physical",
        "code_20": "tlx_physical_20",
        "code_100": "tlx_physical_100",
        "label": "Physical Demand",
        "question": "How physically demanding was the condition?",
        "description": "Consider how much physical activity, movement, force, or bodily effort was required.",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "temporal",
        "code_20": "tlx_temporal_20",
        "code_100": "tlx_temporal_100",
        "label": "Temporal Demand",
        "question": "How much time pressure did you feel during the condition?",
        "description": "Consider whether the pace felt slow and relaxed or rapid and rushed.",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "performance",
        "code_20": "tlx_performance_20",
        "code_100": "tlx_performance_100",
        "label": "Performance",
        "question": "How successful do you think you were in accomplishing the task goals?",
        "low": "Bad",
        "high": "Perfect",
    },
    {
        "name": "effort",
        "code_20": "tlx_effort_20",
        "code_100": "tlx_effort_100",
        "label": "Effort",
        "question": "How hard did you have to work to achieve your level of performance?",
        "description": "Consider both mental and physical effort.",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "frustration",
        "code_20": "tlx_frustration_20",
        "code_100": "tlx_frustration_100",
        "label": "Frustration",
        "question": "How frustrated did you feel during the condition?",
        "description": "Consider feelings such as irritation, stress, discouragement, or annoyance.",
        "low": "Very low",
        "high": "Very high",
    },
]
TLX_VALUES_20 = list(range(0, 21))

FLEET_TLX_DIMENSIONS = [
    {
        "name": "mental",
        "code": "nasa_mental",
        "label": "Mental Demand",
        "question": "How mentally demanding was this condition?",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "physical",
        "code": "nasa_physical",
        "label": "Physical Demand",
        "question": "How physically demanding was this condition?",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "temporal",
        "code": "nasa_temporal",
        "label": "Temporal Demand",
        "question": "How much time pressure did you feel while supervising and intervening?",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "performance",
        "code": "nasa_performance",
        "label": "Performance",
        "question": "How successful do you think you were in accomplishing the supervision and intervention task?",
        "low": "Failure / completely unsuccessful",
        "high": "Perfect / completely successful",
    },
    {
        "name": "effort",
        "code": "nasa_effort",
        "label": "Effort",
        "question": "How hard did you have to work to achieve your level of performance?",
        "low": "Very low",
        "high": "Very high",
    },
    {
        "name": "frustration",
        "code": "nasa_frustration",
        "label": "Frustration",
        "question": "How insecure, discouraged, irritated, stressed, or annoyed did you feel?",
        "low": "Very low",
        "high": "Very high",
    },
]

FLEET_POST_CONDITION_ITEMS = [
    {
        "code": "fleet_awareness",
        "label": "Fleet Awareness",
        "category": "Fleet supervision",
        "text": "I was able to maintain awareness of what was happening across the robot fleet.",
    },
    {
        "code": "fleet_attention_switch",
        "label": "Attention Switching",
        "category": "Fleet supervision",
        "text": "I was able to shift my attention between robots when necessary.",
        "na_for_fleet_sizes": [1],
    },
    {
        "code": "fleet_prioritization",
        "label": "Prioritization",
        "category": "Fleet supervision",
        "text": "When multiple robots required attention, I was able to decide which robot to address first.",
        "na_for_fleet_sizes": [1],
    },
    {
        "code": "fleet_response_time",
        "label": "Time to Respond",
        "category": "Fleet supervision",
        "text": "I had enough time to respond to robots that required intervention.",
    },
    {
        "code": "fleet_keep_up",
        "label": "Keeping Up",
        "category": "Fleet supervision",
        "text": "I felt able to keep up with the intervention demands during this condition.",
    },
]

FLEET_FINAL_OPTIONS = ["1 robot", "3 robots", "6 robots", "9 robots"]
FLEET_FINAL_SUSTAINABLE_OPTIONS = [*FLEET_FINAL_OPTIONS, "None of these"]
FLEET_FINAL_ITEMS = {
    "preferred_sustainable_fleet_size": {
        "label": "Comfortable Fleet Size",
        "question": "Which fleet size would you feel comfortable supervising for an extended period?",
        "options": FLEET_FINAL_SUSTAINABLE_OPTIONS,
        "required": True,
    },
    "best_balance_fleet_size": {
        "label": "Best Balance",
        "question": "Which fleet size provided the best balance between overall productivity and your ability to supervise the robots effectively?",
        "options": FLEET_FINAL_OPTIONS,
        "required": False,
    },
    "prioritization_strategy_comment": {
        "label": "Prioritization Strategy",
        "question": "When several robots required your attention close together, how did you decide which robot to address first?",
        "required": False,
    },
    "fleet_overload_comment": {
        "label": "Fleet Difficulty",
        "question": "Was there a point at which it became difficult to keep track of the robot fleet? If so, what made it difficult?",
        "required": False,
    },
}

FLEET_CONDITION_INTRODUCTION = (
    "Thinking about the fleet-size condition you just completed, please rate your "
    "workload and ability to supervise the robot fleet."
)

FLEET_FINAL_INTRODUCTION = (
    "You have now completed all fleet-size conditions. Please answer the final "
    "fleet supervision questions."
)

POST_CONDITION_INTRODUCTION = (
    "Thinking about the condition you just experienced, how was your overall experience? "
    "Please indicate how strongly you agree or disagree with each of the following statements. "
    "There are no right or wrong answers. We are interested in your individual experience."
)

FINAL_COMPARISON_INTRODUCTION = (
    "You have now completed all three conditions. We are interested in your comparative opinion. "
    "Please think back to your experience with each condition and answer the following questions."
)

BACKGROUND_OPTIONS = {
    "handedness": ["Right", "Left", "Ambidextrous", "Prefer not to say"],
    "robotics_experience": [
        "None",
        "Seen demonstrations or attended introductory lectures only",
        "Completed coursework or tutorials, but little hands-on use",
        "Less than 10 hours of hands-on experience",
        "10–50 hours of hands-on experience",
        "More than 50 hours or regular research/project use",
        "Professional or advanced research experience",
    ],
    "vr_experience_last_12_months": [
        "Never",
        "Once",
        "2-5 times",
        "Several times during the year",
        "Approximately monthly",
        "Approximately weekly",
        "Several times per week or more",
    ],
    "gaming_controller_experience_last_12_months": [
        "Never",
        "Once or twice",
        "Several times during the year",
        "Approximately monthly",
        "Approximately weekly",
        "Several times per week",
        "Daily or almost daily",
    ],
    "teleoperation_experience": [
        "None",
        "Observed a demonstration only",
        "Used teleoperation once or twice",
        "Less than 10 hours of hands-on use",
        "10–50 hours of hands-on use",
        "More than 50 hours or regular research/project use",
        "Professional or advanced research experience",
    ],
    "kinesthetic_experience": [
        "Franka robot",
        "FACTR",
        "Another kinesthetic or physically guided robot",
        "None of the above",
    ],
}

BACKGROUND_QUESTIONS = {
    "vr_experience_last_12_months": (
        "During the last 12 months, how often have you used a virtual-reality headset?"
    ),
    "gaming_controller_experience_last_12_months": (
        "During the last 12 months, how often have you played video games or used handheld motion/game controllers?"
    ),
}

# Used only to interpret existing development databases created before the
# last-12-month wording migration. New submissions use BACKGROUND_OPTIONS.
LEGACY_BACKGROUND_OPTIONS = {
    "vr_experience_last_12_months": [
        "Never used VR",
        "Used VR once",
        "Used VR 2–5 times",
        "Used VR occasionally",
        "Use VR approximately monthly",
        "Use VR approximately weekly",
        "Use VR several times per week or professionally",
    ],
    "gaming_controller_experience_last_12_months": [
        "Never",
        "Less than once per year",
        "Several times per year",
        "Approximately monthly",
        "Approximately weekly",
        "Several times per week",
        "Daily or competitive/professional use",
    ],
}

COMPENSATION_TYPES = {
    "payment": "€15 monetary compensation",
    "sona_credit": "SONA / study-pool credit",
}

UNDERSTANDING_CHECKS = [
    (
        "risk_indicator_correct",
        "Does a red risk indicator guarantee that the robot will fail?",
        "No. It indicates higher estimated risk, but it may be imperfect.",
    ),
    (
        "unattended_robots_correct",
        "What happens to the other robots while you are controlling one robot?",
        "They continue operating unless they have been paused.",
    ),
    (
        "multiple_paused_correct",
        "Can more than one robot be paused?",
        "Yes.",
    ),
    (
        "release_decision_correct",
        "When should you release the robot back to autonomy?",
        "When you believe that the required correction has been completed.",
    ),
    (
        "ready_state_correct",
        "What should you do when the system displays READY?",
        "Manual control may begin.",
    ),
    (
        "safety_response_correct",
        "What should you do if the robot behaves unexpectedly or unsafely?",
        "Stop moving and inform the experimenter.",
    ),
]

FINAL_COMPARISON_ITEMS = {
    "1A": {
        "ranking": "Rank the three supervision interfaces from best to worst overall.",
        "q2": "Rank the supervision interfaces from best to worst for understanding the task-relevant spatial relationships.",
        "q3": "Rank the supervision interfaces from best to worst for deciding when intervention was necessary.",
        "q4": "Rank the supervision interfaces from best to worst for visual comfort.",
        "q5": "Rank the supervision interfaces from most to least preferred for supervising multiple robots over a longer period.",
        "reason": "What was the main reason for your preferred supervision interface?",
        "comment": "Optional additional comment.",
    },
    "1B": {
        "ranking": "Rank the three controllers from best to worst overall.",
        "q2": "Rank the controllers from best to worst for making precise corrections.",
        "q3": "Rank the controllers from best to worst for an intuitive relationship between your movement and the robot’s movement.",
        "q4": "Rank the controllers from most to least natural to use.",
        "q5": "Rank the controllers from most to least preferred for repeated corrective interventions.",
        "reason": "What was the main reason for your preferred controller?",
        "comment": "Optional additional comment.",
    },
}

WORKFLOW_STATES = (
    "paper_consent",
    "participant_background",
    "understanding_check",
    "waiting_condition_1",
    "condition_1_questionnaire",
    "waiting_condition_2",
    "condition_2_questionnaire",
    "waiting_condition_3",
    "condition_3_questionnaire",
    "waiting_condition_4",
    "condition_4_questionnaire",
    "final_comparison",
    "fleet_final",
    "compensation",
    "complete",
)


def resolve_condition_order(study: str, order_code: str) -> tuple[str, ...]:
    """Return the ordered condition codes for a study/order combination."""

    try:
        return CONDITION_ORDERS[study][order_code]
    except KeyError as exc:
        raise ValueError(f"Invalid condition order {order_code!r} for Study {study}.") from exc


def resolve_task_order(order_code: str) -> tuple[str, ...]:
    """Return the two-task sequence for a task-order code."""

    try:
        return TASK_ORDERS[order_code]
    except KeyError as exc:
        raise ValueError(f"Invalid task order {order_code!r}.") from exc
