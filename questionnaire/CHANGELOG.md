# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Repository checklist for the six missing condition-reference images.
- Required between-participant task-order assignment (`T-O1`/`T-O2`) persisted across all blocks and exports.
- External robot-data schema documentation and a synthetic example dataset.
- Condition-reference image resolution with neutral missing-image placeholders.
- Separate post-questionnaire compensation workflow and `compensation_records` table.
- Configurable HTTPS `PAYMENT_QUESTIONNAIRE_URL` without local payment-detail collection.
- Additive schema version 2 migration with a verified pre-migration backup.
- Additive schema version 3 fields for complete direct-comparison rankings.

### Changed

- Participant IDs now accept participant numbers from `001` through `100`.
- Participant-facing workload labels now use “NASA-TLX”; stable `raw_tlx_*` data fields retain the unweighted scoring method for compatibility.
- Changing a NASA-TLX slider now immediately checks its matching confirmation box, while an untouched default remains unconfirmed.
- VR and gaming/controller experience now explicitly cover the last 12 months.
- Final comparison and post-condition pages now include standardized introductions.
- Scientific exports include task-order/block merge metadata and exclude compensation administration data.
- Assigned-order labels omit the condition component that is fixed within each study.
- Overall and criterion-specific final comparisons now use single-list drag-and-drop rankings, so every condition appears exactly once without duplicate rank choices, and participants must confirm each ranking before saving.
- Removed the incorrect explanatory sentence beneath NASA-TLX Performance.

### Security

- Replaced plaintext/default PIN configuration with salted PBKDF2-SHA256 hashes loaded from an ignored local environment file.
- Added staged-file privacy checks and an optional local pre-commit hook.
- Added repository guidance for participant-data and credential incidents.

## [1.0.0] - 2026-08-03

### Added

- Offline Study 1A and Study 1B questionnaire workflows.
- Persistent SQLite state machine, audit logging, verified backups, and recovery.
- Raw NASA-TLX scoring and Study 1A spatial-understanding scoring.
- Long-format CSV, wide-format CSV, and session metadata JSON exports.
- Automated validation, persistence, workflow, backup, and export tests.
