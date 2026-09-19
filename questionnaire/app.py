"""Streamlit entry point for the local ACM HRI questionnaire application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import streamlit as st
from streamlit_sortables import sort_items

from config import (
    BACKUP_DIR,
    DATABASE_PATH,
    EXPERIMENTER_PIN_HASH,
    EXPORT_DIR,
    PAYMENT_QUESTIONNAIRE_URL,
    ensure_directories,
)
from condition_images import condition_image_references, missing_condition_images
from database import BackupError, Database, DatabaseError
from export import ExportError, export_session
from questionnaire_definitions import (
    AGREEMENT_OPTIONS,
    BACKGROUND_OPTIONS,
    BACKGROUND_QUESTIONS,
    COMPENSATION_TYPES,
    CONDITIONS,
    CONDITION_ORDERS,
    FLEET_CONDITION_INTRODUCTION,
    FLEET_FINAL_ITEMS,
    FLEET_FINAL_INTRODUCTION,
    FLEET_POST_CONDITION_ITEMS,
    FLEET_SIZES,
    FLEET_TLX_DIMENSIONS,
    FINAL_COMPARISON_INTRODUCTION,
    FINAL_COMPARISON_ITEMS,
    POST_CONDITION_INTRODUCTION,
    POST_CONDITION_ITEMS,
    STUDY_LABELS,
    TASK_ORDERS,
    TLX_DIMENSIONS,
    UNDERSTANDING_CHECKS,
    condition_order_display_name,
)
from security import is_supported_pin_hash, verify_pin
from validation import ValidationError, validate_payment_url


st.set_page_config(
    page_title="ACM HRI Questionnaire",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1040px; padding-top: 2rem; padding-bottom: 4rem;}
    h1 {letter-spacing: -0.025em;}
    h2, h3 {margin-top: 1.7rem;}
    [data-testid="stForm"] {border: 1px solid #d9e1e8; border-radius: 12px; padding: 1.25rem;}
    [data-testid="stMetric"] {background: #f6f8fa; border-radius: 10px; padding: .7rem;}
    .participant-lead {font-size: 1.16rem; line-height: 1.6; color: #263746;}
    .required-note {color: #526575; font-size: .94rem;}
    .privacy-note {background: #f6f8fa; border-left: 4px solid #457b9d; padding: .75rem 1rem;}
    .tlx-block {background: #f8fafb; border: 1px solid #d9e1e8; border-radius: 12px; padding: 1rem 1rem .35rem; margin: 1rem 0;}
    .tlx-anchors {display: flex; justify-content: space-between; color: #415465; font-weight: 600; margin-top: -.55rem;}
    .condition-image-placeholder {aspect-ratio: 4 / 3; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .55rem; text-align: center; border: 2px dashed #9aaab7; border-radius: 10px; background: #f4f7f9; color: #526575; padding: 1rem;}
    .condition-image-placeholder strong {color: #263746;}
    </style>
    """,
    unsafe_allow_html=True,
)


ensure_directories()


@st.cache_resource
def get_database() -> Database:
    return Database(DATABASE_PATH, BACKUP_DIR)


db = get_database()


def set_flash(level: str, message: str) -> None:
    st.session_state["flash"] = (level, message)


def show_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        level, message = flash
        getattr(st, level, st.info)(message)


def active_session() -> dict[str, Any] | None:
    session_id = st.session_state.get("active_session_id")
    if not session_id:
        return None
    try:
        return db.get_session(session_id)
    except DatabaseError:
        st.session_state.pop("active_session_id", None)
        return None


def perform_action(action: Callable[[], Any], success_message: str) -> bool:
    try:
        action()
    except BackupError as exc:
        set_flash("error", str(exc))
        st.rerun()
    except (ValidationError, DatabaseError, ExportError) as exc:
        st.error(str(exc))
        return False
    except Exception:
        st.error("The operation failed unexpectedly. Your entered responses remain on this page.")
        return False
    set_flash("success", success_message)
    st.rerun()
    return True


def confirm_tlx_rating(confirmation_key: str) -> None:
    """Mark a NASA-TLX dimension as confirmed after its slider is changed."""

    st.session_state[confirmation_key] = True


def state_title(state: str) -> str:
    return state.replace("_", " ").title()


def state_block_number(state: str) -> int | None:
    match = re.search(r"condition_(\d)", state)
    return int(match.group(1)) if match else None


def experimenter_login() -> bool:
    if st.session_state.get("experimenter_authenticated"):
        return True
    st.subheader("Experimenter sign in")
    if not is_supported_pin_hash(EXPERIMENTER_PIN_HASH):
        st.error("Experimenter access is not configured on this computer.")
        st.write("Generate a secure PIN hash, then save it in the ignored local `.env` file.")
        st.code("python scripts/generate_pin_hash.py", language="bash")
        return False
    st.caption("Enter the locally configured experimenter PIN.")
    with st.form("experimenter_login", clear_on_submit=True):
        pin = st.text_input("PIN", type="password")
        submitted = st.form_submit_button("Unlock experimenter mode", type="primary")
    if submitted:
        if verify_pin(pin, EXPERIMENTER_PIN_HASH):
            st.session_state["experimenter_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect PIN.")
    return False


def render_create_or_resume() -> None:
    create_tab, resume_tab = st.tabs(["Create session", "Resume session"])
    with create_tab:
        study = st.selectbox(
            "Study *",
            options=list(STUDY_LABELS),
            format_func=lambda value: STUDY_LABELS[value],
            key="create_study",
        )
        order_codes = list(CONDITION_ORDERS[study])
        if study == "Fleet":
            condition_order_counts = db.get_condition_order_counts(study)
            st.caption(
                "Fleet condition-order allocation: "
                + " · ".join(f"{code}: {condition_order_counts[code]}" for code in order_codes)
            )
        else:
            task_order_counts = db.get_task_order_counts(study)
            st.caption(
                "Task-order allocation for this study: "
                + " · ".join(
                    f"{code}: {task_order_counts[code]} of intended 10"
                    for code in ("T-O1", "T-O2")
                )
            )
        with st.form("create_session_form"):
            placeholder = {"1A": "P1A-001", "1B": "P1B-001", "Fleet": "P1C-001"}[study]
            participant_id = st.text_input(
                "Participant ID *",
                placeholder=placeholder,
            )
            condition_order_code = st.selectbox(
                "Assigned condition order *",
                options=order_codes,
                format_func=lambda code: (
                    f"{code}: "
                    + " → ".join(
                        condition_order_display_name(study, condition)
                        for condition in CONDITION_ORDERS[study][code]
                    )
                ),
            )
            if study == "Fleet":
                task_order_code = "N/A"
                st.caption("Fleet has no T-shape/Cups task order.")
            else:
                task_order_code = st.selectbox(
                    "Task Order *",
                    options=["T-O1", "T-O2"],
                    index=None,
                    placeholder="Please select the assigned task order",
                    format_func=lambda code: f"{code}: {' → '.join(TASK_ORDERS[code])}",
                )
            experimenter_initials = st.text_input("Experimenter initials *", max_chars=12)
            session_date = st.date_input("Session date")
            submitted = st.form_submit_button("Create session", type="primary")
        if submitted:
            created: dict[str, Any] = {}

            def create() -> None:
                created.update(
                    db.create_session(
                        participant_id=participant_id,
                        study=study,
                        condition_order_code=condition_order_code,
                        task_order_code=task_order_code,
                        experimenter_initials=experimenter_initials,
                        session_date=session_date.isoformat(),
                    )
                )
                st.session_state["active_session_id"] = created["session_id"]

            perform_action(create, "Session created and backed up.")

    with resume_tab:
        incomplete = db.list_incomplete_sessions()
        if incomplete:
            st.caption(f"{len(incomplete)} incomplete local session(s) available.")
        with st.form("resume_session_form"):
            identifier = st.text_input(
                "Participant ID or session ID",
                placeholder="P1A-001 or P1A-001_20260818_143500",
            )
            submitted = st.form_submit_button("Resume session", type="primary")
        if submitted:
            try:
                recovered = db.recover_session(identifier)
            except DatabaseError as exc:
                st.error(str(exc))
            else:
                st.session_state["active_session_id"] = recovered["session_id"]
                set_flash("success", "Session restored from permanent local storage.")
                st.rerun()


def render_status(session: dict[str, Any]) -> None:
    blocks = db.get_condition_blocks(session["session_id"])
    completed = sum(bool(block["questionnaire_submitted_at"]) for block in blocks)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Participant", session["participant_id"])
    col2.metric("Study", session["study"])
    col3.metric("Conditions saved", f"{completed} / {len(blocks)}")
    col4.metric("State", state_title(session["current_workflow_state"]))

    with st.expander("Session assignment and save status", expanded=False):
        st.write(f"**Session ID:** `{session['session_id']}`")
        st.write(f"**Condition order:** {session['condition_order_code']}")
        if session.get("task_order_code") and session["task_order_code"] != "N/A":
            st.write(
                f"**Task order:** {session['task_order_code']} — "
                f"{' → '.join(session.get('task_order_sequence', []))}"
            )
        elif session["study"] == "Fleet":
            st.write("**Task order:** N/A")
        else:
            st.warning("This legacy session has no recorded task order; no assignment was guessed during migration.")
        status_rows = []
        for block in blocks:
            if block["questionnaire_submitted_at"]:
                status = "Saved"
            elif block["questionnaire_opened_at"]:
                status = "Unlocked"
            else:
                status = "Pending"
            status_rows.append(
                {
                    "Block": block["block_id"],
                    "Position": block["block_number"],
                    "Condition": f"{block['condition_code']} — {block['condition_name']}",
                    "Fleet size": FLEET_SIZES.get(block["condition_code"], ""),
                    "Task order": block.get("task_order_code") or "Legacy: not recorded",
                    "Status": status,
                }
            )
        st.dataframe(status_rows, hide_index=True, use_container_width=True)
        latest = db.latest_backup()
        if latest:
            st.success(f"Latest verified backup: {latest.name}")
        else:
            st.warning("No verified backup is present.")

    missing_images = [] if session["study"] == "Fleet" else missing_condition_images(session["study"])
    if missing_images:
        st.warning(
            "Condition reference image(s) have not yet been provided: "
            + ", ".join(path.name for path in missing_images)
            + ". Neutral placeholders will be shown in the final comparison."
        )
    try:
        payment_url = validate_payment_url(PAYMENT_QUESTIONNAIRE_URL)
    except ValidationError as exc:
        st.warning(str(exc))
    else:
        if payment_url is None:
            st.warning(
                "PAYMENT_QUESTIONNAIRE_URL is not configured. Participants may record the payment choice, "
                "but the external secure payment questionnaire link will be unavailable."
            )

    compensation = db.get_compensation(session["session_id"])
    if compensation:
        st.success(f"Compensation choice recorded: {COMPENSATION_TYPES[compensation['compensation_type']]}")


def render_understanding_check(session: dict[str, Any]) -> None:
    st.subheader("Verbal understanding check")
    st.write("Ask each question verbally, then record whether the participant answered correctly.")
    with st.form("understanding_check_form"):
        recorded: dict[str, Any] = {}
        for field, question, answer in UNDERSTANDING_CHECKS:
            st.markdown(f"**{question}**")
            st.caption(f"Correct answer: {answer}")
            response = st.radio(
                "Experimenter record *",
                ["Correct", "Incorrect"],
                index=None,
                horizontal=True,
                key=f"understanding_{field}",
            )
            recorded[field] = True if response == "Correct" else False if response == "Incorrect" else None
            st.divider()
        retraining = st.radio(
            "Was retraining required? *",
            ["Yes", "No"],
            index=None,
            horizontal=True,
        )
        successful_practice = st.radio(
            "Was at least one practice intervention completed successfully? *",
            ["Yes", "No"],
            index=None,
            horizontal=True,
        )
        practice_count = st.number_input(
            "Practice intervention count *", min_value=0, max_value=100, step=1, value=0
        )
        notes = st.text_area("Experimenter notes", max_chars=4000)
        st.caption("Do not enter the participant’s name or other direct identifying information in notes.")
        submitted = st.form_submit_button("Save understanding check", type="primary")
    if submitted:
        recorded.update(
            {
                "retraining_required": (
                    True if retraining == "Yes" else False if retraining == "No" else None
                ),
                "successful_practice_intervention": (
                    True
                    if successful_practice == "Yes"
                    else False if successful_practice == "No" else None
                ),
                "practice_intervention_count": practice_count,
                "experimenter_notes": notes,
            }
        )
        perform_action(
            lambda: db.submit_understanding_check(session["session_id"], recorded),
            "Understanding check saved. Condition 1 is waiting for experimenter unlock.",
        )


def render_exports(session: dict[str, Any]) -> None:
    st.subheader("Session complete")
    st.success("All required participant data have been saved. The session is locked against resubmission.")
    if st.button("Create verified CSV and JSON exports", type="primary"):
        try:
            paths = export_session(db, session["session_id"], EXPORT_DIR)
        except (ExportError, DatabaseError, OSError) as exc:
            st.error(str(exc))
        else:
            st.session_state["export_paths"] = {key: str(path) for key, path in paths.items()}
            set_flash("success", f"Exports saved under exports/{session['session_id']}/.")
            st.rerun()

    stored_paths = st.session_state.get("export_paths", {})
    if stored_paths:
        labels = {
            "long": ("Download long-format CSV", "text/csv"),
            "wide": ("Download wide-format CSV", "text/csv"),
            "fleet_background": ("Download Fleet background CSV", "text/csv"),
            "fleet_condition": ("Download Fleet condition CSV", "text/csv"),
            "fleet_final": ("Download Fleet final CSV", "text/csv"),
            "metadata": ("Download session metadata JSON", "application/json"),
        }
        for key, path_text in stored_paths.items():
            path = Path(path_text)
            if path.is_file():
                label, mime = labels[key]
                st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)


def render_experimenter() -> None:
    st.title("ACM HRI Questionnaire")
    st.caption("Experimenter console · local storage only")
    show_flash()
    if not experimenter_login():
        return

    session = active_session()
    if session is None:
        render_create_or_resume()
        return

    if st.button("Close active session view"):
        st.session_state.pop("active_session_id", None)
        st.session_state.pop("export_paths", None)
        st.rerun()

    render_status(session)
    state = session["current_workflow_state"]
    if state == "paper_consent":
        st.subheader("Paper consent confirmation")
        st.warning("Do not proceed until the participant has provided written paper consent.")
        with st.form("paper_consent_form"):
            consent = st.checkbox("I confirm that written paper consent was obtained. *")
            submitted = st.form_submit_button("Confirm consent", type="primary")
        if submitted:
            perform_action(
                lambda: db.confirm_written_consent(session["session_id"], consent),
                "Written consent confirmation saved. Hand the device to the participant for background questions.",
            )
    elif state == "participant_background":
        st.info("Participant background is ready. Switch to Participant mode and hand over the device.")
    elif state == "understanding_check":
        render_understanding_check(session)
    elif state.startswith("waiting_condition_"):
        block_number = state_block_number(state)
        assert block_number is not None
        block = db.get_condition_blocks(session["session_id"])[block_number - 1]
        st.subheader(f"Condition {block_number} is ready")
        st.write(f"Next assigned condition: **{block['condition_code']} — {block['condition_name']}**")
        if session["study"] == "Fleet":
            st.write(f"Fleet size for this condition: **{FLEET_SIZES[block['condition_code']]}**")
        st.warning("Unlock only after this condition has been completed in the laboratory task.")
        if st.button(f"Unlock Condition {block_number} questionnaire", type="primary"):
            perform_action(
                lambda: db.unlock_condition(session["session_id"], block_number),
                f"Condition {block_number} questionnaire unlocked. Switch to Participant mode.",
            )
    elif state.startswith("condition_"):
        block_number = state_block_number(state)
        st.info(
            f"Condition {block_number} questionnaire is unlocked and waiting for participant submission. "
            "Switch to Participant mode and hand over the device."
        )
    elif state == "final_comparison":
        st.info("All three conditions are saved. Switch to Participant mode for the final comparison.")
    elif state == "fleet_final":
        st.info("All four Fleet conditions are saved. Switch to Participant mode for the final Fleet questionnaire.")
    elif state == "compensation":
        st.info(
            "The scientific questionnaire is complete. Switch to Participant mode for the separate compensation choice."
        )
    elif state == "complete":
        render_exports(session)


def option_selectbox(
    label: str, options: list[str], key: str, numbered: bool = False
) -> str | None:
    return st.selectbox(
        label,
        options,
        index=None,
        placeholder="Please select",
        key=key,
        format_func=(
            (lambda value: f"{options.index(value) + 1} — {value}")
            if numbered
            else (lambda value: value)
        ),
    )


def render_background(session: dict[str, Any]) -> None:
    st.title("Participant background")
    st.markdown(
        '<p class="participant-lead">Please answer each required question. Your responses are saved only on this laboratory computer.</p>',
        unsafe_allow_html=True,
    )
    st.caption("* Required")
    with st.form("participant_background_form"):
        age = st.number_input(
            "Age *", min_value=18, max_value=100, step=1, value=None, placeholder="Enter age"
        )
        handedness = option_selectbox(
            "Handedness *", BACKGROUND_OPTIONS["handedness"], "background_handedness"
        )
        robotics = option_selectbox(
            "Robotics experience *",
            BACKGROUND_OPTIONS["robotics_experience"],
            "background_robotics",
            numbered=True,
        )
        vr = option_selectbox(
            f"{BACKGROUND_QUESTIONS['vr_experience_last_12_months']} *",
            BACKGROUND_OPTIONS["vr_experience_last_12_months"],
            "background_vr",
            numbered=True,
        )
        gaming = option_selectbox(
            f"{BACKGROUND_QUESTIONS['gaming_controller_experience_last_12_months']} *",
            BACKGROUND_OPTIONS["gaming_controller_experience_last_12_months"],
            "background_gaming",
            numbered=True,
        )
        teleoperation = option_selectbox(
            "Robot-teleoperation experience *",
            BACKGROUND_OPTIONS["teleoperation_experience"],
            "background_teleoperation",
            numbered=True,
        )
        kinesthetic = st.multiselect(
            "Previous kinesthetic-controller experience *",
            BACKGROUND_OPTIONS["kinesthetic_experience"],
            placeholder="Select all that apply",
        )
        motion_sickness = st.radio(
            "Motion-sickness susceptibility *",
            options=list(range(1, 8)),
            index=None,
            horizontal=True,
            captions=["Not susceptible", "", "", "", "", "", "Very susceptible"],
        )
        submitted = st.form_submit_button("Save background responses", type="primary")
    if submitted:
        data = {
            "age": age,
            "handedness": handedness,
            "robotics_experience": robotics,
            "vr_experience_last_12_months": vr,
            "gaming_controller_experience_last_12_months": gaming,
            "teleoperation_experience": teleoperation,
            "kinesthetic_experience": kinesthetic,
            "motion_sickness_susceptibility": motion_sickness,
        }
        perform_action(
            lambda: db.submit_background(session["session_id"], data),
            "Your background responses were saved. Please return the device to the experimenter.",
        )


def render_condition_questionnaire(session: dict[str, Any], block_number: int) -> None:
    study = session["study"]
    st.title(f"Condition {block_number} of {len(CONDITIONS[study])}")
    st.markdown(
        f'<p class="participant-lead">{POST_CONDITION_INTRODUCTION}</p>',
        unsafe_allow_html=True,
    )
    st.caption("Agreement scale: 1 = Strongly disagree · 7 = Strongly agree")
    agreement: dict[str, int | None] = {}
    tlx: dict[str, int | None] = {}
    current_category = None
    for item in POST_CONDITION_ITEMS[study]:
        if item["category"] != current_category:
            current_category = item["category"]
            st.subheader(current_category)
        st.markdown(f"**{item['code']} — {item['label']}**")
        st.write(f'“{item["text"]}”')
        agreement[item["code"]] = st.radio(
            f"Response for {item['code']} *",
            list(AGREEMENT_OPTIONS),
            index=None,
            format_func=lambda value: f"{value} — {AGREEMENT_OPTIONS[value]}",
            key=f"block_{block_number}_{item['code']}",
            label_visibility="collapsed",
        )
        st.divider()

    st.subheader("NASA-TLX workload")
    st.write(
        "Move each slider to your rating from 0 to 20. Changing a slider immediately "
        "checks its confirmation box; all six confirmations are required."
    )
    for dimension in TLX_DIMENSIONS:
        code = dimension["code_20"]
        confirmation_key = f"block_{block_number}_{code}_confirmed"
        with st.container(border=True):
            st.markdown(f"### {dimension['label']}")
            st.markdown(f"**{dimension['question']}**")
            if description := dimension.get("description"):
                st.write(description)
            rating = st.slider(
                f"{dimension['label']} rating *",
                min_value=0,
                max_value=20,
                value=10,
                step=1,
                key=f"block_{block_number}_{code}_slider",
                on_change=confirm_tlx_rating,
                args=(confirmation_key,),
            )
            st.markdown(
                '<div class="tlx-anchors">'
                f"<span>0 = {dimension['low']}</span>"
                f"<span>20 = {dimension['high']}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            confirmed = st.checkbox(
                f"I confirm my {dimension['label'].lower()} rating shown above. *",
                key=confirmation_key,
            )
            tlx[code] = rating if confirmed else None
    submitted = st.button(
        "Save Condition questionnaire",
        type="primary",
        key=f"save_condition_{block_number}",
    )
    if submitted:
        perform_action(
            lambda: db.submit_condition_questionnaire(
                session["session_id"], block_number, agreement, tlx
            ),
            f"Condition {block_number} responses were saved. Please return the device to the experimenter.",
        )


def render_fleet_condition_questionnaire(session: dict[str, Any], block_number: int) -> None:
    blocks = db.get_condition_blocks(session["session_id"])
    block = blocks[block_number - 1]
    st.title(f"Fleet condition {block_number} of {len(blocks)}")
    st.subheader(f"{block['condition_name']} condition")
    st.markdown(
        f'<p class="participant-lead">{FLEET_CONDITION_INTRODUCTION}</p>',
        unsafe_allow_html=True,
    )
    st.caption("NASA-TLX scale: 0 = low endpoint · 20 = high endpoint")

    tlx: dict[str, int | None] = {}
    for dimension in FLEET_TLX_DIMENSIONS:
        code = dimension["code"]
        confirmation_key = f"fleet_block_{block_number}_{code}_confirmed"
        with st.container(border=True):
            st.markdown(f"### {dimension['label']}")
            st.markdown(f"**{dimension['question']}**")
            rating = st.slider(
                f"{dimension['label']} rating *",
                min_value=0,
                max_value=20,
                value=10,
                step=1,
                key=f"fleet_block_{block_number}_{code}_slider",
                on_change=confirm_tlx_rating,
                args=(confirmation_key,),
            )
            st.markdown(
                '<div class="tlx-anchors">'
                f"<span>0 = {dimension['low']}</span>"
                f"<span>20 = {dimension['high']}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            confirmed = st.checkbox(
                f"I confirm my {dimension['label'].lower()} rating shown above. *",
                key=confirmation_key,
            )
            tlx[code] = rating if confirmed else None

    st.subheader("Fleet supervision")
    st.caption("Agreement scale: 1 = Strongly disagree · 7 = Strongly agree")
    agreement: dict[str, int | None] = {}
    fleet_size = FLEET_SIZES[block["condition_code"]]
    for item in FLEET_POST_CONDITION_ITEMS:
        st.markdown(f"**{item['label']}**")
        st.write(f'“{item["text"]}”')
        if fleet_size in item.get("na_for_fleet_sizes", []):
            st.info("Not applicable for the 1-robot condition. This will be saved as NA.")
            agreement[item["code"]] = None
            st.divider()
            continue
        agreement[item["code"]] = st.radio(
            f"Response for {item['label']} *",
            list(AGREEMENT_OPTIONS),
            index=None,
            format_func=lambda value: f"{value} — {AGREEMENT_OPTIONS[value]}",
            key=f"fleet_block_{block_number}_{item['code']}",
            label_visibility="collapsed",
        )
        st.divider()

    if st.button("Save Fleet condition questionnaire", type="primary", key=f"save_fleet_condition_{block_number}"):
        perform_action(
            lambda: db.submit_condition_questionnaire(
                session["session_id"], block_number, agreement, tlx
            ),
            f"Fleet condition {block_number} responses were saved. Please return the device to the experimenter.",
        )


def render_fleet_final(session: dict[str, Any]) -> None:
    st.title("Final Fleet questionnaire")
    st.markdown(
        f'<p class="participant-lead">{FLEET_FINAL_INTRODUCTION}</p>',
        unsafe_allow_html=True,
    )
    preferred_item = FLEET_FINAL_ITEMS["preferred_sustainable_fleet_size"]
    balance_item = FLEET_FINAL_ITEMS["best_balance_fleet_size"]
    strategy_item = FLEET_FINAL_ITEMS["prioritization_strategy_comment"]
    comment_item = FLEET_FINAL_ITEMS["fleet_overload_comment"]

    preferred = st.selectbox(
        f"{preferred_item['question']} *",
        preferred_item["options"],
        index=None,
        placeholder="Please select",
    )
    best_balance = st.selectbox(
        balance_item["question"],
        balance_item["options"],
        index=None,
        placeholder="Optional",
    )
    strategy = st.text_area(strategy_item["question"], max_chars=4000)
    comment = st.text_area(comment_item["question"], max_chars=4000)

    if st.button("Save final Fleet questionnaire", type="primary"):
        perform_action(
            lambda: db.submit_fleet_final_response(
                session["session_id"],
                {
                    "preferred_sustainable_fleet_size": preferred,
                    "best_balance_fleet_size": best_balance,
                    "prioritization_strategy_comment": strategy,
                    "fleet_overload_comment": comment,
                },
            ),
            "Your Fleet questionnaire responses were saved. Please complete the separate compensation step.",
        )


def render_final_comparison(session: dict[str, Any]) -> None:
    study = session["study"]
    items = FINAL_COMPARISON_ITEMS[study]
    condition_codes = list(CONDITIONS[study])

    def condition_label(code: str) -> str:
        return f"{code} — {CONDITIONS[study][code]}"

    condition_labels = [condition_label(code) for code in condition_codes]
    codes_by_label = dict(zip(condition_labels, condition_codes))
    sortable_style = """
    .sortable-component {
        padding: 0.25rem 0 0.5rem;
    }
    .sortable-component.vertical {
        align-items: stretch;
        display: block !important;
    }
    .sortable-container {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }
    .sortable-container-header {
        color: #263746;
        font-weight: 700;
        padding: 0.35rem 0.2rem;
    }
    .sortable-container-body {
        border: 1px solid #b8c5cf;
        border-radius: 8px;
        display: flex !important;
        flex-direction: column !important;
        gap: 0.45rem;
        padding: 0.35rem;
    }
    .sortable-item, .sortable-item:hover {
        background: #f6f8fa;
        border: 1px solid #b8c5cf;
        border-radius: 8px;
        color: #263746;
        cursor: grab;
        display: block !important;
        font-weight: 600;
        margin: 0 !important;
        padding: 0.7rem 0.85rem;
        width: 100% !important;
        box-sizing: border-box;
    }
    .sortable-item:nth-child(1)::before { content: "1. Best  "; color: #536879; }
    .sortable-item:nth-child(2)::before { content: "2. Second  "; color: #536879; }
    .sortable-item:nth-child(3)::before { content: "3. Third  "; color: #536879; }
    .sortable-item:active { cursor: grabbing; }
    """

    def drag_ranking(prompt: str, key: str) -> tuple[list[str], bool]:
        st.markdown(f"**{prompt}** · Required")
        st.caption(
            "Drag to reorder the list. Top is Best, middle is Second, bottom is Third."
        )
        ranked_labels = sort_items(
            condition_labels,
            header="Ranking",
            direction="vertical",
            custom_style=sortable_style,
            key=key,
        )
        confirmed = st.checkbox(
            "Please confirm that the rankings above reflect your final choices before saving. *",
            key=f"{key}_confirmed",
        )
        return [codes_by_label[label] for label in ranked_labels], confirmed

    st.title("Final comparison")
    image_columns = st.columns(3)
    for column, reference in zip(image_columns, condition_image_references(study)):
        with column:
            if reference.exists:
                st.image(str(reference.path), use_container_width=True)
            else:
                st.markdown(reference.placeholder_html(), unsafe_allow_html=True)
            st.caption(reference.condition_name)
    st.markdown(
        f'<p class="participant-lead">{FINAL_COMPARISON_INTRODUCTION}</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.subheader("Overall ranking")
        ranking, ranking_confirmed = drag_ranking(
            items["ranking"], "final_overall_ranking"
        )

        st.subheader("Direct comparisons")
        comparison_rankings: dict[str, list[str]] = {}
        comparison_confirmations: dict[str, bool] = {}
        for code in ("q2", "q3", "q4", "q5"):
            comparison_rankings[code], comparison_confirmations[code] = drag_ranking(
                items[code], f"final_{code}_ranking"
            )
            if code != "q5":
                st.divider()

        st.markdown(
            '<p class="privacy-note">Please do not enter your name or other identifying personal information.</p>',
            unsafe_allow_html=True,
        )
        reason = st.text_area(
            f"{items['reason']} *", max_chars=4000, key="final_preference_reason"
        )
        comment = st.text_area(
            items["comment"], max_chars=4000, key="final_additional_comment"
        )
        submitted = st.button("Save final comparison", type="primary")
    if submitted:
        incomplete_sections = [
            label
            for label, value in (
                ("Overall ranking", ranking),
                *(
                    (f"Direct comparison {number}", comparison_rankings[code])
                    for number, code in enumerate(("q2", "q3", "q4", "q5"), start=1)
                ),
            )
            if len(value) != len(condition_codes) or set(value) != set(condition_codes)
        ]
        if incomplete_sections:
            st.error(
                "Complete every ranking by ordering each condition exactly once. "
                "Incomplete: " + ", ".join(incomplete_sections) + "."
            )
        elif not ranking_confirmed or not all(comparison_confirmations.values()):
            unconfirmed_sections = [
                label
                for label, confirmed in (
                    ("Overall ranking", ranking_confirmed),
                    *(
                        (
                            f"Direct comparison {number}",
                            comparison_confirmations[code],
                        )
                        for number, code in enumerate(("q2", "q3", "q4", "q5"), start=1)
                    ),
                )
                if not confirmed
            ]
            st.error(
                "Please confirm that the rankings above reflect your final choices before saving. "
                "Unconfirmed: " + ", ".join(unconfirmed_sections) + "."
            )
        else:
            perform_action(
                lambda: db.submit_final_comparison(
                    session["session_id"],
                    ranking,
                    comparison_rankings,
                    reason,
                    comment,
                ),
                "Your scientific questionnaire responses were saved. Please complete the separate compensation step.",
            )


def render_compensation(session: dict[str, Any]) -> None:
    st.title("Compensation")
    st.markdown(
        '<p class="participant-lead">The scientific questionnaire is complete. Please choose one compensation option.</p>',
        unsafe_allow_html=True,
    )
    choice = st.radio(
        "Compensation option *",
        options=list(COMPENSATION_TYPES),
        index=None,
        format_func=lambda value: COMPENSATION_TYPES[value],
    )
    sona_code: str | None = None
    if choice == "payment":
        st.info(
            "No name, address, IBAN, bank-account details, or financial information are collected in this application."
        )
        try:
            payment_url = validate_payment_url(PAYMENT_QUESTIONNAIRE_URL)
        except ValidationError:
            payment_url = None
        if payment_url:
            st.link_button("Open secure payment questionnaire", payment_url, type="primary")
        else:
            st.info("The experimenter will provide the approved secure payment procedure separately.")
    elif choice == "sona_credit":
        sona_code = st.text_input(
            "Please enter your four-digit anonymous SONA code.",
            max_chars=4,
            type="password",
            help="Enter exactly four numeric digits.",
        )

    if st.button("Save compensation choice", type="primary", disabled=choice is None):
        perform_action(
            lambda: db.submit_compensation(session["session_id"], choice, sona_code),  # type: ignore[arg-type]
            "Thank you. Your compensation choice was saved and the session is complete.",
        )


def render_participant() -> None:
    show_flash()
    session = active_session()
    if session is None:
        st.title("Questionnaire not ready")
        st.info("Please return the device to the experimenter so they can open or resume a session.")
        return
    state = session["current_workflow_state"]
    if state == "participant_background":
        render_background(session)
    elif state.startswith("condition_"):
        block_number = state_block_number(state)
        assert block_number is not None
        if session["study"] == "Fleet":
            render_fleet_condition_questionnaire(session, block_number)
        else:
            render_condition_questionnaire(session, block_number)
    elif state == "final_comparison":
        render_final_comparison(session)
    elif state == "fleet_final":
        render_fleet_final(session)
    elif state == "compensation":
        render_compensation(session)
    elif state == "complete":
        st.title("Responses saved")
        st.success("Thank you. Your responses have been saved and the questionnaire is complete.")
        st.write("Please return the device to the experimenter.")
    else:
        st.title("Please wait")
        st.info("There is no participant questionnaire open right now. Please return the device to the experimenter.")


with st.sidebar:
    st.header("ACM HRI study")
    role = st.radio("Mode", ["Participant", "Experimenter"])
    st.divider()
    st.caption("Runs locally · no network services · SQLite storage")

if role == "Experimenter":
    render_experimenter()
else:
    st.session_state.pop("experimenter_authenticated", None)
    render_participant()
