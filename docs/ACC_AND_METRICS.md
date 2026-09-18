# ACC Risk Cue and Performance Metrics

## ACC risk cue

`intervene_base/playback/acc_scorer.py` computes the Action Chunking Consistency cue
shown on each cell's border. Select the method with `STUDY_ACC_METHOD`. The studies
used `chunk_residual`.

| `STUDY_ACC_METHOD` | Measures |
|---|---|
| `chunk_residual` | Commanded-vs-achieved tracking residual + re-plan discontinuity + action jerk (weighted composite) |
| `ensemble_disagreement` | Spread across overlapping action chunks predicting the same timestep |
| `none` | Always 0 |

`STUDY_ACC_SCALE` maps the raw value to the 0–1 cue. Its defaults are 0.15 for
residual and 0.03 for ensemble. To score recorded episodes offline, run:

```bash
python intervene_base/utils/rescore_acc.py <episode_dir_or_npz> [--method chunk_residual] [--overwrite]
```

## Performance metrics (optional)

`METRICS=1 bash ./run_multi_window_robot.sh` writes windowed latency and publish-rate
statistics to `session_logs/metrics_S<ii>_<ts>.jsonl`. `tools/quest_telemetry.py`
collects headset-side rates over `adb logcat`, and `tools/perf_report.py` joins both
into one report.
