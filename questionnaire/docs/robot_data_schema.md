# External robot-data integration schema

The questionnaire database stores questionnaire and session metadata only. Full robot trajectories and event streams belong in the robot-logging system on institutionally approved study-data storage.

## Required join identifiers

The robot logger must receive and record the exact identifiers created by the questionnaire application:

- `participant_id` — coded participant identifier
- `session_id` — unique questionnaire session identifier
- `block_id` — `B01`, `B02`, or `B03` within that session

Join questionnaire and robot data using all three fields. Do not infer a block from timestamps or condition order alone. Each block manifest should also record `study`, `block_number`, `condition_code`, `condition_name`, `condition_position`, and `task_order_code` for independent verification.

The task order is assigned between participants and remains fixed across all three blocks. Within each episode, `task` should be exactly `T-shape` or `Cups` and should follow the session's `task_order_code`:

- `T-O1`: `T-shape` then `Cups`
- `T-O2`: `Cups` then `T-shape`

## Episode-level fields

| Field | Suggested type | Meaning |
|---|---|---|
| `episode_id` | string | Unique episode identifier |
| `participant_id` | string | Questionnaire participant identifier |
| `session_id` | string | Questionnaire session identifier |
| `block_id` | string | Questionnaire block identifier |
| `condition_code` | string | `S1`–`S3` or `C1`–`C3` |
| `task` | string | `T-shape` or `Cups` |
| `scenario_id` | string | Scenario presented in this episode |
| `id_ood_label` | string | In-distribution/out-of-distribution label |
| `robot_cell_id` | string | Robot or cell identifier |
| `episode_start_time` | ISO 8601 timestamp | Episode start, including timezone |
| `episode_end_time` | ISO 8601 timestamp | Episode end, including timezone |
| `success` | boolean | Whether the episode succeeded |
| `completion_time_s` | number | Episode duration in seconds |
| `number_of_interventions` | integer | Total corrective interventions |
| `success_without_intervention` | boolean | Success with no intervention |
| `success_after_intervention` | boolean | Success following intervention |

## Intervention-level fields

| Field | Suggested type | Meaning |
|---|---|---|
| `intervention_id` | string | Unique intervention identifier |
| `episode_id` | string | Parent episode identifier |
| `participant_id` | string | Questionnaire participant identifier |
| `session_id` | string | Questionnaire session identifier |
| `block_id` | string | Questionnaire block identifier |
| `robot_cell_id` | string | Robot or cell identifier |
| `request_timestamp` | ISO 8601 timestamp | Intervention request time |
| `pause_timestamp` | ISO 8601 timestamp | Robot pause time |
| `ready_timestamp` | ISO 8601 timestamp | READY state time |
| `manual_control_start_timestamp` | ISO 8601 timestamp | Manual-control start time |
| `release_timestamp` | ISO 8601 timestamp | Return-to-autonomy time |
| `synchronization_time_s` | number | Request-to-ready synchronization duration |
| `correction_time_s` | number | Manual correction duration |
| `queue_waiting_time_s` | number | Time waiting in an intervention queue |
| `robot_state_at_takeover` | object/string | Versioned robot-state snapshot/reference |
| `robot_state_at_release` | object/string | Versioned robot-state snapshot/reference |
| `outcome_after_intervention` | string | Outcome category after release |
| `repeated_intervention_index` | integer | One-based intervention index for the episode |

## Logging guidance

- Use timezone-aware ISO 8601 timestamps, preferably UTC.
- Keep units in field names (`_s`) and avoid mixed units.
- Validate foreign keys between intervention and episode records before analysis.
- Store schema/version metadata with every robot-log dataset.
- Never commit real robot logs or participant-linked data to this source repository.

See [`examples/example_condition_data.json`](../examples/example_condition_data.json) for synthetic structure only.
