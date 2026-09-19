# Database migration: schema version 3

The application checks its SQLite schema when `Database` is initialized. Migration is automatic, additive, and idempotent; it does not delete or recreate the existing database.

Before applying a migration to an existing database, the application creates a consistent timestamped backup in `backups/` and verifies it with SQLite `PRAGMA integrity_check`. It creates and verifies a second backup after the migration succeeds.

Version 3 adds:

- `final_comparisons.comparison_q2_ranking_json`
- `final_comparisons.comparison_q3_ranking_json`
- `final_comparisons.comparison_q4_ranking_json`
- `final_comparisons.comparison_q5_ranking_json`

Each new field stores all three condition codes from best to worst. The existing `comparison_q2` through `comparison_q5` columns remain and store each ranking's first-place condition for compatibility with older analysis code. Historical winner-only responses are preserved, while their unavailable second and third positions remain `NULL` rather than being guessed.

Version 2 added:

- `sessions.task_order_code`
- `sessions.task_order_sequence` (JSON array text in SQLite; decoded to a list by the application)
- `condition_blocks.task_order_code`
- `participant_background.vr_experience_last_12_months`
- `participant_background.gaming_controller_experience_last_12_months`
- `compensation_records`
- `schema_migrations`
- new condition-response aliases for all Raw NASA-TLX 0–20 values, converted 0–100 values, and both means

Legacy background values are copied into the newly named last-12-month columns so old development sessions continue to open and export. Original columns and values remain intact for compatibility.

Historical task order cannot be inferred safely, so migrated legacy sessions use `NULL`/blank task order. The experimenter console labels this explicitly. No assignment is guessed.

Historical Raw NASA-TLX values were stored on a 0–100 scale in increments of five. Migration preserves those records and adds corresponding values using:

```text
new_20 = legacy_100 / 5
new_100 = legacy_100
```

Scientific exports omit the deprecated legacy TLX measure names when the new aliases are available.

Compensation is not fabricated for legacy completed sessions. Such sessions remain complete and their scientific data remain exportable without a `compensation_records` row.
