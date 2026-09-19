# ACM HRI Local Questionnaire

A complete offline Streamlit application used to collect the questionnaires for all three user studies in the paper:

- **Study 1A — Supervision Interface Comparison** (Supervision Study; conditions S1–S3, participant IDs `P1A-###`)
- **Study 1B — Control Interface Comparison** (Control Study; conditions C1–C3, participant IDs `P1B-###`)
- **Fleet — Fleet-Size Robot Supervision** (Fleet Study; conditions F1, F3, F6, F9, participant IDs `P1C-###`)

It uses SQLite for permanent local storage, creates a verified backup after each saved workflow action, and exports analysis-ready CSV and JSON files. It does not require an internet connection or send data to an external service.

> [!IMPORTANT]
> This directory contains application source code only; no participant data are included. When running a study, keep runtime databases, exports, backups, and logs in the ignored local directories and in approved study-data storage, and inspect `git status` and the staged diff before every commit.

## Repository privacy rules

- Keep any repository that is used during data collection private.
- Never commit participant responses, consent or recruitment records, compensation records, robot logs containing participant data, databases, exports, PINs, passwords, tokens, or API keys.
- Runtime data are stored in ignored local directories: `data/`, `exports/`, `backups/`, and `logs/`.
- Use institutionally approved encrypted storage, access controls, retention rules, and transfer procedures for all study data.
- Run `git status` and review `git diff --cached` before every push.
- Install the local safety hook with `python scripts/install_git_hooks.py` after cloning.

## Requirements

- Python 3.11 or newer
- A local macOS or Linux laboratory computer
- A modern browser (opened by Streamlit on the same computer)

## Install and run

From this project directory on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create the ignored local configuration file and generate a PIN hash:

```bash
cp .env.example .env
python scripts/generate_pin_hash.py
```

Copy the generated `EXPERIMENTER_PIN_HASH=...` line into `.env`, replacing the placeholder. Then start the application:

```bash
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`. The app uses no external services; researchers remain responsible for keeping local data on institutionally approved storage.

## Experimenter PIN

There is intentionally no default PIN. The app accepts only a salted PBKDF2-SHA256 hash loaded from `EXPERIMENTER_PIN_HASH`; it never stores the plaintext PIN. Generate a hash interactively:

```bash
python scripts/generate_pin_hash.py
```

Place the generated line in the local `.env` file. Both `.env` and `.streamlit/secrets.toml` are ignored. Never add a real PIN, PIN hash, password, token, or API key to source code, `.env.example`, documentation, or Git history. The PIN is a local role-separation control, not file encryption; access to the computer and study-data directories must still be restricted.

Optional path variables are `DATABASE_PATH`, `HRI_DATA_DIR`, `HRI_EXPORT_DIR`, and `HRI_BACKUP_DIR`.

To enable the €15 payment option's external button, set an institutionally approved HTTPS form in the ignored `.env` file:

```dotenv
PAYMENT_QUESTIONNAIRE_URL=https://approved.example.edu/secure-payment-form
```

The questionnaire app never collects names, addresses, IBANs, bank-account details, or other payment information.

## Storage locations

- Primary SQLite database: `data/hri_questionnaires.sqlite3`
- Per-session exports: `exports/<session_id>/`
- Timestamped SQLite backups: `backups/`

SQLite foreign keys and WAL mode are enabled. Backups are made with SQLite's consistent backup API, verified with `PRAGMA integrity_check`, and rotated so at least the latest 20 are retained. Do not edit the database, exports, or backups while a session is in progress.

These directories are excluded from Git, but `.gitignore` is not a substitute for institutionally approved study-data storage and access controls.

## Experimenter workflow

1. Choose **Experimenter** mode and enter the local PIN.
2. Create a session with participant ID, study, assigned condition order, required task order, and experimenter initials. The three condition blocks and mergeable IDs `B01`, `B02`, and `B03` are created automatically.
3. Confirm that written paper consent was obtained.
4. Switch to **Participant** mode and let the participant complete the background form.
5. Return to **Experimenter** mode and record the verbal understanding check. Every safety-critical answer and a successful practice intervention are required.
6. After each laboratory condition, unlock exactly the next assigned questionnaire, switch to Participant mode, and let the participant submit it.
7. After all three condition questionnaires, switch to Participant mode for the final comparison and condition-reference images.
8. After the scientific final comparison, the participant chooses either €15 payment or SONA/study-pool credit in a separate compensation step.
9. When the session is complete, return to Experimenter mode and generate the scientific long CSV, wide CSV, and metadata JSON exports.

The experimenter console displays the persistent workflow state, assigned order, completed/pending blocks, and latest verified backup.

### Task-order procedure

Task order is counterbalanced **between participants** and fixed **within a participant** across all three conditions:

- `T-O1`: T-shape → Cups
- `T-O2`: Cups → T-shape

The intended 20-participant allocation is 10 participants per task order. The creation screen shows current counts as an allocation aid; the experimenter remains responsible for following the approved counterbalancing schedule. Task order cannot be changed by participants and is stored in the session and all three block records.

## Participant workflow

The participant uses only **Participant** mode on the active session prepared by the experimenter. Depending on the persistent state, the participant can complete:

- background questions;
- the currently unlocked Condition 1, 2, or 3 questionnaire; or
- the final comparison after all three conditions; and
- the separate compensation choice after the scientific questionnaire.

The participant cannot choose a participant ID, study, condition, or order. Future questionnaires stay locked. As soon as a form is submitted, its values are saved transactionally and the state advances, so previous answers are no longer displayed or editable. No names, email addresses, street addresses, or other direct identifiers are collected.

## Resume an interrupted session

After a browser closure, Streamlit restart, or computer restart:

1. Run `streamlit run app.py` again.
2. Open Experimenter mode and enter the PIN.
3. Select **Resume session**.
4. Enter either the coded participant ID (for example, `P1A-001`) or the full session ID.

The application reloads the last committed state from SQLite. It never deletes or resets an existing session from the user interface. If a backup warning appears after a submission, the primary database commit may already have succeeded; stop collection, inspect local storage, then resume the session instead of resubmitting blindly.

## Scoring and exports

Agreement items use the required 1–7 scale without a preselected response. NASA-TLX uses six integer sliders from 0 to 20. Changing a slider immediately checks its separate confirmation box, while the initial position remains unconfirmed until the participant moves the slider or confirms it manually. The application stores every original `*_20` value, the equivalent `*_100 = *_20 × 5` value, `raw_tlx_20`, and `raw_tlx_100 = raw_tlx_20 × 5`. The score is the unweighted (Raw TLX) arithmetic mean; the 15 pairwise weighting comparisons are not implemented. Performance runs from 0 = Bad to 20 = Perfect. Study 1A also stores A2 and A3 individually and calculates `SPATIAL_UNDERSTANDING_MEAN`. Study 1B stores B2, B3, and B4 individually and does not create a controller-quality average.

The VR-headset and gaming/controller background questions explicitly ask about frequency during the **last 12 months** and store their updated variable labels separately from legacy development columns.

The final comparison uses drag-and-drop best-to-worst lists for the overall ranking and all four direct-comparison questions. Each ranking is a single draggable list, so every condition appears exactly once; the top item is Best, the middle item is Second, and the bottom item is Third. Participants must actively confirm each ranking before saving. The wide export retains each direct comparison's first-place winner and also includes explicit `final_comparison_q*_rank_1`, `_rank_2`, and `_rank_3` columns.

The long-format export contains one measurement per row. The wide-format export contains one row per participant/session. The metadata JSON contains condition/task assignments, timestamps, workflow state, and all block metadata. Scientific exports deliberately exclude compensation type and anonymous SONA code.

### Merge with robot logs

Use these stable keys in both questionnaire exports and external robot logs:

- `participant_id`: coded participant, such as `P1A-001`;
- `session_id`: specific study visit, such as `P1A-001_20260818_143500`;
- `block_id`: within-session condition block, `B01`, `B02`, or `B03`.

The long CSV also includes `condition_code`, `condition_name`, `block_number`, and `condition_position`. Prefer the three-key merge above rather than inferring a condition from row order.

The robot logger must write the exact same `participant_id`, `session_id`, and `block_id` supplied by this application. The expected episode/intervention schema and a synthetic example are documented in [docs/robot_data_schema.md](docs/robot_data_schema.md) and [examples/example_condition_data.json](examples/example_condition_data.json).

## Condition reference images

The final comparison shows three equal-width reference images. Until approved screenshots are installed, neutral named placeholders are shown and Experimenter Mode reports the missing files. Place the six real PNGs at the exact paths documented in [docs/condition_images.md](docs/condition_images.md), and track completion in [TODO.md](TODO.md). Do not use screenshots containing participant data, results, or condition-performance information.

## Compensation privacy

Compensation is collected only after the scientific final comparison:

- `payment` opens the configured external secure payment questionnaire; no financial or identity data enter this application;
- `sona_credit` stores exactly one four-digit anonymous SONA code.

Compensation records use the separate `compensation_records` table. SONA codes are excluded from scientific CSV/JSON exports, and the repository provides no routine compensation export. Protect the local SQLite database as administrative research data.

## Database migration

Schema version 3 is an additive, idempotent migration. Before changing an existing database, the application creates and verifies a timestamped SQLite backup and creates another verified backup after migration. It adds direct-comparison ranking fields while retaining the original winner fields, along with the earlier task-order, background, TLX-alias, migration-ledger, and compensation additions. Legacy sessions remain resumable; data that cannot be reconstructed safely are left blank rather than guessed. Full details are in [docs/database_migration.md](docs/database_migration.md).

## Run tests

With the virtual environment active:

```bash
pytest
```

The tests use temporary databases and cover ID/study/task-order validation, all condition orders, TLX conversion, rankings, state transitions, condition sequencing, compensation validation/privacy, image fallback, additive migration, persistence/recovery, both full study workflows, backup verification/retention, and scientific export creation.

## Git safety checks

These scripts assume that this directory is the root of its own Git repository. To use them, copy `questionnaire/` into a separate repository first.

Install the repository-owned pre-commit hook:

```bash
python scripts/install_git_hooks.py
```

The hook runs `scripts/check_before_commit.py` and blocks staged databases, exports, participant/consent data, local secrets, known credential patterns, and oversized files. You can also run it manually:

```bash
python scripts/check_before_commit.py
```

The default maximum staged-file size is 5 MiB. Override it locally with `MAX_STAGED_FILE_BYTES` or the script's `--max-bytes` option. Always manually inspect `git status` and `git diff --cached`; automated checks reduce risk but cannot identify every sensitive value.

Security and privacy incidents must be reported privately according to [SECURITY.md](SECURITY.md). Do not put sensitive data in a public issue.

## Project files

- `app.py` — Streamlit participant and experimenter interface
- `config.py` — local paths, PIN, and backup retention configuration
- `security.py` — salted PIN hashing and verification
- `condition_images.py` — approved condition-image resolution and neutral fallback
- `questionnaire_definitions.py` — condition orders, scales, and exact study wording
- `validation.py` — input validation and calculated scores
- `database.py` — normalized SQLite schema, transactions, state machine, audit log, and backups
- `export.py` — long CSV, wide CSV, and session metadata JSON generation
- `tests/` — pytest suite using isolated temporary databases
- `scripts/check_before_commit.py` — staged-file privacy and credential check
- `scripts/install_git_hooks.py` — local pre-commit hook installer
- `scripts/generate_pin_hash.py` — interactive local PIN-hash generator
- `docs/` — robot-log schema, migration behavior, and condition-image instructions
- `examples/` — synthetic robot-data structure only
- `.streamlit/config.toml` — low-distraction local UI theme and disabled usage telemetry

## Design limitations

- This is a single-computer, single-active-session laboratory application. Do not run multiple Streamlit server processes against the same database during collection.
- The experimenter PIN gates UI functions but does not encrypt database files. Protect the computer account and storage directory using institutional procedures.
- Paper consent itself is not digitized; only the experimenter's confirmation is recorded.
- Exports are deliberately blocked until a session is complete, preventing partial datasets from being mistaken for final study data.
