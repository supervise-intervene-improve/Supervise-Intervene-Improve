# Supervise, Intervene, Improve

**Interface Design for Human Oversight of Multiple Autonomous Robot Manipulators**

Project website: https://supervise-intervene-improve.github.io/supervise-intervene-improve-webpage/

---

## Overview

Learned manipulation policies fail in states that their training data does not cover.
Selective human intervention lets autonomous execution continue during routine operation
while one operator takes over a robot only when it needs help, and the corrections
double as targeted demonstrations for later policy updates. How well this works depends
on the interfaces the operator uses to *see* what a robot is doing, to *correct* it, and
to *spread attention* over several robots at once.

**Supervise–Intervene–Improve** is a risk-informed framework that connects these three
stages:

- **Supervise.** A shared overview shows every robot cell with a green-to-red
  *Action Chunking Consistency* (ACC) risk cue on its border. The operator selects a cell
  to inspect it in more detail while it keeps running autonomously.
- **Intervene.** Requesting an intervention pauses only the selected cell, prepares the
  control device, and then hands control to the human. The other cells keep running.
- **Improve.** Active corrections are recorded together with the robot state and
  exported as demonstrations for offline fine-tuning of the policy.

Each cell is a MuJoCo simulation of a Franka Panda running a pretrained
Action Chunking Transformer (ACT) policy. The system has two task scenes, **T-Shape
Assembly** and **Cup Insertion**, each with in-distribution and out-of-distribution scene
banks.

### Interfaces

**Supervision interfaces** (how the operator inspects a cell)

| Interface | Description |
|---|---|
| **Desktop-RGB** | Desktop grid of all cells; selecting a cell enlarges its left, right, top and wrist camera views. |
| **VR-RGB** | The same four camera views as panels in a Meta Quest 3, in AR passthrough. |
| **VR-PointCloud** | GPU-fused point cloud from three virtual cameras, registered to the workspace in the Meta Quest 3 and inspectable from any viewpoint. |

**Corrective-control interfaces** (how the operator corrects the robot)

| Interface | Description |
|---|---|
| **Kinesthetic Teaching (KT)** | A physical Franka Panda is aligned to the selected simulated robot and then guided by hand in gravity compensation. |
| **Motion Controller (MC)** | The Quest right controller's tracked pose drives the end effector through relative IK; buttons control the gripper. |
| **FACTR** | An articulated force-feedback leader arm is aligned to the robot and guided by hand. |

The paper reports three independent within-subjects user studies (Supervision, Control
and Fleet, with 24 participants each) and an offline evaluation of policies fine-tuned on the
collected corrections.

---

## System architecture

```
 ┌──────────────── Linux PC ────────────────────────────────────────────┐        Wi-Fi         ┌──── Meta Quest 3 ────┐
 │ intervene_base/utils/run_main_policy.sh                               │                      │ SII-Meta-Quest3 app  │
 │   ├─ N × ACT policy processes (intervene_base/app.py, one per cell)   │                      │  • session selector  │
 │   ├─ ACC risk scoring, OOD scene scheduling, episode recording        │   ZMQ topics +       │  • point cloud / RGB │
 │   ├─ control sidecars (KT robot mirror / MC IK / FACTR service)       │   UDP discovery      │    single view       │
 │   └─ run_multi_window_robot.sh  ── VR streaming backend ─────────────┼─────────────────────▶│  • controller input  │
 │        └─ multi_session_launcher.py → N × intervention_vr_runtime.py │◀─────────────────────┼─ (A/B/X/Y, trigger)  │
 └───────────────────────────────────────────────────────────────────────┘                      └──────────────────────┘
```

- `intervene_base/utils/run_main_policy.sh` is the **full system**: policies, study
  logging and the selected control interface. When `START_VR=1` (default), it starts the
  VR backend below.
- `run_multi_window_robot.sh` is the **VR streaming backend**. It starts one MuJoCo
  runtime per cell, streams thumbnails, RGB views and point clouds to the headset,
  and forwards controller commands. It can also run alone (sim replay only, no
  policies), which is the quickest way to check the PC–headset link.

---

## Requirements

**Hardware**

- Linux PC (tested on Ubuntu 22.04) with an NVIDIA GPU for CUDA point-cloud generation.
  Nine concurrent cells were run on a single workstation.
- Meta Quest 3 on the **same Wi-Fi subnet** as the PC.
- KT condition: a Franka Panda reachable through Polymetis.
  FACTR condition: a FACTR leader arm (Dynamixel) on a local serial port.
  MC condition: only the Quest controllers.

**Software**

The system uses **two Python environments**. See [`env/ENVIRONMENT_SETUP.md`](env/ENVIRONMENT_SETUP.md)
for lock files, Polymetis, CUDA Open3D and hardware configuration.

```bash
# 1) Policy / robot environment (Python 3.10, conda; lerobot, torch, Polymetis)
conda create -n polymetis --file env/conda_policy_polymetis_linux64_explicit.txt
conda activate polymetis
pip install -r env/requirements_policy_polymetis_py310_portable.txt
pip install -e SimPublisher

# 2) VR streaming environment (Python 3.13 venv at ./.venv; MuJoCo, ZMQ, Open3D)
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r env/requirements_vr_py313_portable.txt
pip install -e SimPublisher
# Open3D is not in the requirements file. GPU point clouds need Open3D 0.19 built with
# CUDA: install that wheel into the venv or point OPEN3D_PYTHON_PACKAGE at the build.
# CPU-only alternative (the launcher then uses the "legacy" backend automatically):
pip install open3d==0.19.0
```

**Headset app.** Open `SII-Meta-Quest3/` in Unity (see
[`SII-Meta-Quest3/README.md`](SII-Meta-Quest3/README.md)), build the Android APK with the
`SII-Meta-Quest3-Release` build profile, and install it with `adb install -r`. The start-up
scene is `SIIScene_SessionSelector` (build index 0).

**Policy checkpoints.** `run_main_policy.sh` expects the ACT checkpoints under
`intervene_base/MODEL_WEIGHTS/` (for example `MODEL_WEIGHTS/tshape/0800000/pretrained_model`).
Pass `CHECKPOINT=<dir>` to use a different location. Reset trajectories for both tasks
ship in `intervene_base/RESET_NPZs/`.

---

## Quick start

### 1. VR backend only (simulation replay, no policies)

```bash
# Nine cells, VR-PointCloud single view, T-Shape scene
WINDOWS=9 bash ./run_multi_window_robot.sh

# Same grid, VR-RGB single view (four camera panels instead of the point cloud)
WINDOWS=9 RGB=1 bash ./run_multi_window_robot.sh

# One cell, skip the selector grid (the headset goes straight into single view)
SINGLE=1 bash ./run_multi_window_robot.sh
```

Then start the app on the Quest. It finds the PC through a UDP discovery beacon, shows
the grid of cells, and you select a cell with the right controller ray + index trigger.

### 2. Full system (policies + VR + control interface)

Run `run_main_policy.sh` from `intervene_base/`. It starts one ACT policy per cell,
the desktop grid, the VR backend (`run_multi_window_robot.sh`) and the selected control
interface:

```bash
cd intervene_base
WINDOWS=9 LAB=tshape bash utils/run_main_policy.sh   # T-Shape Assembly, VR-PointCloud + KT
WINDOWS=9 LAB=cups   bash utils/run_main_policy.sh   # Cup Insertion
```

### 3. Reproducing the user studies

These are the commands used for each condition, run from `intervene_base/`. Ctrl+C
stops a run.

Each command below is written for T-Shape Assembly. Every Supervision and Control
condition was run once per task; for Cup Insertion, change `LAB=tshape` →
`LAB=cups`.

Every command shares `FPS=60 STUDY_ACC_METHOD=chunk_residual`.
The arguments are explained in the tables below.

**Supervision study** (KT fixed; the supervision interface varies)

```bash
# S1  Desktop-RGB + KT
START_VR=0 WINDOWS=9 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
# S2  VR-RGB + KT
RGB=1 WINDOWS=9 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
# S3  VR-PointCloud + KT
WINDOWS=9 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
```

**Control study** (VR-PointCloud fixed; the corrective-control interface varies)

```bash
# C1  KT
WINDOWS=9 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
# C2  Motion Controller (simulated arm driven through IK)
MC_ACTIVE=1 MC_SIM=1 WINDOWS=9 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
# C3  FACTR (set the USB-serial latency timer to 1 ms first, see env/ENVIRONMENT_SETUP.md)
FACTR_ACTIVE=1 FACTR_POSITION_TOLERANCE=0.08 WINDOWS=9 LAB=tshape FPS=60 \
  STUDY_ACC_METHOD=chunk_residual bash utils/run_main_policy.sh
```

**Fleet study** (VR-PointCloud + KT, T-Shape Assembly; the number of active robots varies)

The Fleet conditions use the VR-PointCloud + KT command with a different number of cells.
Set `WINDOWS` to `1`, `3`, `6` or `9` for conditions F1, F3, F6 and F9:

```bash
WINDOWS=1 LAB=tshape FPS=60 STUDY_ACC_METHOD=chunk_residual \
  bash utils/run_main_policy.sh
```

**Questionnaires.** The post-condition and final-comparison questionnaires for all three
studies were collected with the offline Streamlit app in
[`questionnaire/`](questionnaire/README.md) (Study 1A = Supervision, Study 1B = Control, Fleet = Fleet):

```bash
cd questionnaire
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && python scripts/generate_pin_hash.py   # paste the hash into .env
streamlit run app.py
```

#### `run_main_policy.sh` arguments used above

| Variable | Value in studies | Meaning |
|---|---|---|
| `WINDOWS` | `9` (Fleet: `1/3/6/9`) | Number of concurrent robot cells, each running its own ACT policy. Unset = a single cell with no grid. |
| `LAB` | `tshape` / `cups` | Task. Selects the scene XML, reset trajectory, ACT checkpoint and OOD object set together. |
| `FPS` | `60` | Sensor publish rate of the selected cell in the VR view (forwarded to `run_multi_window_robot.sh`). |
| `STUDY_ACC_METHOD` | `chunk_residual` | How the ACC risk cue is computed (see [`docs/ACC_AND_METRICS.md`](docs/ACC_AND_METRICS.md)). |
| `START_VR` | `0` for S1 | `0` = do not start the VR backend; the operator uses the desktop grid only (Desktop-RGB). Default `1`. |
| `RGB` | `1` for S2 | VR single view shows four RGB camera panels instead of the point cloud (VR-RGB). |
| `MC_ACTIVE`, `MC_SIM` | `1`, `1` for C2 | Motion-controller condition. The Quest right controller drives the **simulated** arm through local IK. |
| `FACTR_ACTIVE` | `1` for C3 | Start the FACTR leader-arm service; the leader drives the simulated arm during an intervention. |
| `FACTR_POSITION_TOLERANCE` | `0.08` | Joint-space tolerance (rad) for aligning FACTR with the robot before human control begins. |

Other useful `run_main_policy.sh` variables are `CHECKPOINT` (ACT checkpoint directory),
`XML` / `RESET_NPZ` (override the scene or reset trajectory), `ROBOT_KEY` (real Franka
for KT), `OOD_MIN_STATES` / `OOD_MAX_STATES` (number of concurrent OOD cells in
Supervision and Control; default 2–3), and `MAX_STEPS`.

---

## `run_multi_window_robot.sh` reference

The script takes all of its options as **environment variables**
(`VAR=value bash ./run_multi_window_robot.sh`). Boolean options accept `1` / `true` /
`TRUE`. It turns them into a `multi_session_launcher.py` call (printed at start-up)
and runs it with `exec`.

### Network and headset

| Variable | Default | Meaning |
|---|---|---|
| `HOST_IP` | auto: first `192.168.x.y` address | IP of the PC on the Quest's Wi-Fi. The script exits with an explanation if it cannot find one. Set it explicitly when your Wi-Fi uses another subnet. |
| `UNITY_NODE` | `MQ3-2` | Name of the headset node the runtimes publish to. |
| `BASE_TOPIC_PORT` | `7741` | ZMQ topic port for session 0; session *i* uses `+10·i`. Raise it (e.g. `+200`) to get around ports still held by an earlier run. |

### Sessions and scene

| Variable | Default | Meaning |
|---|---|---|
| `WINDOWS` | `3` | Number of active cells: `1, 3, 6, 9, 12, 15`. Each group of three fills one column of the selector grid. |
| `SINGLE` | `0` | `1` = launch one session and skip the selector grid (overrides `WINDOWS`). |
| `GRID_CAPACITY` | *(active count)* | Number of selector cells shown on the headset, independent of the number of active sessions (max 15). |
| `LAB_XML` | `intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml` | MuJoCo scene. Use `sii_scene_table_boxes_cups.xml` for Cup Insertion. |
| `TRAJ` | *(see below)* | One or more space-separated `.npz` trajectories replayed in the cells; they are cycled to fill `WINDOWS`. |
| `DEFAULT_TRAJ_DIR` / `DEFAULT_TRAJ_GLOB` | `intervene_base/INTERVENTION_DATA` / `policy_episode_*.npz` | Where trajectories are picked from when `TRAJ` is empty. If the directory is missing, `FALLBACK_TRAJ` (the bundled T-shape reset trajectory) is used. |
| `MAX_RESTARTS` | `3` | How often a crashed session is respawned (`0` = never). |
| `POST_REPLAN_MODE` | `preview_replay` | After an intervention: `preview_replay` replays the stitched trajectory; `policy_handoff` stops and hands back to the policy. |

### Supervision view

| Variable | Default | Meaning |
|---|---|---|
| `WITH_PC` | `1` | VR-PointCloud: stream fused point clouds (top/left/right cameras) in single view. |
| `RGB` | `0` | VR-RGB: show four camera panels (top/left/right/wrist) instead. Mutually exclusive with `WITH_PC=1`; setting both aborts. |
| `FPS` | `60` | Sensor publish rate of the **selected** session. |
| `FPS_IDLE` | `10` | Publish rate of the unselected (grid) sessions. |
| `RGB_THUMBNAIL_FPS` | `10` | Cap on the selector-thumbnail feed (`0` = uncapped). |
| `JPG_QUALITY` | `75` | JPEG quality of streamed RGB images. |

### Point-cloud pipeline (`WITH_PC=1`)

| Variable | Default | Meaning |
|---|---|---|
| `PC_BACKEND` | auto: `gpu` if CUDA Open3D imports, else `legacy` | Point-cloud builder. |
| `PC_MAX_POINTS` | `200000` (`80000` when `WINDOWS ≥ 9`) | Maximum points per camera per frame. Must not exceed the Unity loader's `maxPointsPerSource`. |
| `PC_STRIDE` | `3` | Use every *n*-th depth pixel (higher = sparser and faster). |
| `PC_WIDTH` / `PC_HEIGHT` | `640` / `480` | Resolution of the depth render. |
| `PC_SAMPLING` | `grid` | `grid` (deterministic, cheap) or `stable_random` (jittered). |
| `PC_ACTIVE_ONLY` | `1` | Build point clouds only for the selected session; the others stream thumbnails. |
| `PC_ROUND_ROBIN` | `0` | Render one camera per tick (top → right → left) to free main-loop time. |
| `PC_MAX_SOURCE_AGE_S` | `0.4` | With round-robin: maximum age of a camera's cloud before it is forced to refresh. |
| `PC_WORKER_THREADS` | `1` if `WINDOWS ≥ 4`, else `3` | Parallel GPU build threads (more threads need more GPU memory). |
| `SELECTED_SENSOR_ACTIVATION` | `1` | Headset `ENTER_SINGLE` / `EXIT_SINGLE` commands (not the subscriber count) decide which session is active. |
| `PEER_FALLBACK_GRACE_S` | `0` | Seconds a subscriber-count fallback must hold before an unselected session is promoted (`0` = strict selection). |
| `SELECTED_PEER_IDLE_GRACE_S` | `5` | A selected session without subscribers drops back to idle after this many seconds (guards against a lost `EXIT_SINGLE`). |

### Control and policy wiring

| Variable | Default | Meaning |
|---|---|---|
| `ROBOT` | `0` | `1` = mirror a real Franka (used by KT); the arm is only engaged for the selected session. Same as `run_multi_window_robot_mirror.sh`. |
| `ROBOT_KEY` | `INTERVENE_ROBOT_KEY` or `p4` | Robot entry in `intervene_base/record.py` (`ROBOTS`) that sets IP and ports. |
| `CONTROL_HZ` | `120` | Real-robot command rate. |
| `REPLAN_OUTPUT_DIR` | `intervene_base/teleop_logs/replans` | Where intervention recordings are written. |
| `MC_ACTIVE` | `0` | Motion-controller condition: advertised to the headset so it reserves the right controller during an intervention. |
| `POLICY_STATE_BASE_PORT` / `POLICY_CMD_BASE_PORT` | *(unset)* | Session *i* mirrors the policy instance at `base + i·POLICY_PORT_STEP`. `run_main_policy.sh` sets these. |
| `POLICY_PORT_STEP` / `POLICY_HOST` | `10` / `127.0.0.1` | Port stride and host of the policy instances. |
| `EXTRA_RUNTIME_ARGS` | *(empty)* | Extra flags forwarded to every `intervention_vr_runtime.py` session. |

### Environment and logging

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON_BIN` | `python` of the activated env | Interpreter for the launcher. |
| `VR_VENV` | `./.venv` | Virtualenv the script activates (after `conda activate base` if conda is found). |
| `OPEN3D_PYTHON_PACKAGE` | *(unset)* | Optional path to a locally built CUDA Open3D Python package, prepended to `PYTHONPATH`. |
| `POLYMETIS_LIB_DIR` | `~/miniforge3` or `~/miniconda3` `envs/polymetis/lib` | Native Polymetis libraries (only needed with `ROBOT=1`). |
| `PERF_LOG` | `0` | Print per-frame `[Perf] render/total/budget` timings. |
| `METRICS` | `0` | Write windowed latency/rate metrics to `session_logs/metrics_S<ii>_<ts>.jsonl` (read by `tools/perf_report.py`). |
| `METRICS_WINDOW_S` / `METRICS_SUMMARY` | `10` / `0` | Metrics window length; also print a `[PerfSummary]` block. |

All session output goes to a single `session_logs/multi_session_<timestamp>.log`, with
lines prefixed by session (`[S<i>]`).

### Helper scripts

| Script | Purpose |
|---|---|
| `run_multi_window_robot_mirror.sh` | `run_multi_window_robot.sh` with the real-robot mirror arguments preset (`ROBOT_KEY`, `CONTROL_HZ`, `REPLAN_OUTPUT_DIR`). |
| `send_multi_window_command.py --command {A,B,X,Y}` | Send a controller command to every running session, e.g. `B` = start/pause all. |
| `reset_all_multi_windows.sh` | Shortcut for `send_multi_window_command.py --command B`. |

---

## Headset controls

A/B are on the **right** controller, X/Y on the **left**.

| Where | Control | Action |
|---|---|---|
| Selector | Right controller ray | Hover a cell |
| Selector | Right index trigger | Select the cell → single view |
| Selector | Right **B** short / hold | Start/pause the hovered cell / all cells |
| Selector | Left **Y** short / hold | Reset the hovered cell / all cells |
| Selector | Right grip | Re-center the grid in front of the head |
| Single view | Left **X** | Request intervention: pause → prepare controller → human control → record. Press again to release. |
| Single view | Left grip | Cancel the intervention in progress |
| Single view | Right **B** | Start/pause the simulation |
| Single view | Left **Y** | Reset the episode |
| Single view | Right index trigger | Back to the selector |
| Single view | Left Menu (hold 3 s) | Unlock/lock scene-anchor adjustment (left stick = XY, left trigger/grip = Z, A/B = pitch, X/Y = yaw, stick click = roll). The pose is saved on the headset. |

In the MC condition the right controller is reserved for the motion controller while an
intervention is live.

---

## Repository structure

```
.
├── intervene_base/                  # Policies, tasks, control interfaces, episode recording
│   ├── app.py                       # Per-cell ACT policy process (ACC scoring, OOD scenes, interventions)
│   ├── utils/run_main_policy.sh     # Full-system launcher (see Quick start)
│   ├── utils/train_act_*.sh         # ACT training / fine-tuning (convert_npz_to_lerobot.py for data)
│   ├── utils/evaluate_act_*.sh      # Offline policy evaluation
│   ├── data_io/                     # Experiment blocks, study logger, OOD scene variants, stitching
│   ├── robot/                       # KT robot mirror, sim MC driver, live replanning
│   ├── factr/                       # FACTR leader-arm service and configuration
│   ├── playback/                    # ACC risk scorer, trajectory playback
│   ├── mujoco_scenes/               # T-Shape and Cup Insertion scenes (+ OOD variants)
│   └── RESET_NPZs/                  # Reset trajectories for both tasks
├── SimPublisher/                    # XR publishing framework (third-party, Apache-2.0) + our runtime
│   └── sii/integration_v1/          # multi_session_launcher.py, intervention_vr_runtime.py, runtime_impl.py
├── SII-Meta-Quest3/                 # Unity project for the Quest 3 app
│   └── Assets/SIIMetaQuest3/Core/   # Selector grid, GPU point-cloud renderer, RGB panels, HUD, input forwarding
├── GPU/                             # Standalone GPU point-cloud streamer / renderer sources
├── questionnaire/                   # Offline Streamlit questionnaire app used in the user studies
├── tools/                           # Performance and forensics reports (perf_report.py, quest_telemetry.py, …)
├── env/                             # Environment lock files and setup guide (ENVIRONMENT_SETUP.md)
├── run_multi_window_robot.sh        # VR streaming backend (see reference above)
├── run_multi_window_robot_mirror.sh # … with real-robot mirroring
├── send_multi_window_command.py     # Broadcast controller commands to all sessions
└── docs/ACC_AND_METRICS.md          # ACC risk cue and performance metrics
```

---

## Troubleshooting

**The Quest does not find the PC.** The launcher prints
`Discovery beacon broadcasting on UDP 8720`. The PC and the headset must be on the same
Wi-Fi subnet, and the access point must allow broadcast (no client isolation).

**`--host X is not assigned to any local interface`.** The PC is not on the Wi-Fi.
Connect it, or pass the address explicitly: `HOST_IP=192.168.0.x bash ./run_multi_window_robot.sh`.

**`Address in use` at start-up.** An earlier run still holds the ports. Stop it
(`pgrep -af intervention_vr_runtime.py`) or start with a different `BASE_TOPIC_PORT`.

**No point cloud in single view.** Check the log for `[CmdListener] RX cmd='ENTER_SINGLE'`
followed by `First PC publish`. If Python publishes but the headset draws nothing, check
that `PC_MAX_POINTS` does not exceed `maxPointsPerSource` in the Unity scene.

**The point cloud is in the wrong place.** Hold Left Menu for 3 s, move the anchor, and hold
Left Menu again to lock it.

**Sessions fail to start at high `WINDOWS`.** Lower `PC_MAX_POINTS` / `PC_WORKER_THREADS`,
or look for `MemoryError` / CUDA out-of-memory in the session log.

---

## Citation

```bibtex
@inproceedings{anonymous2027sii,
  title     = {Supervise, Intervene, Improve: Interface Design for Human Oversight of
               Multiple Autonomous Robot Manipulators},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

Third-party components keep their original licenses (e.g. `SimPublisher/LICENSE`,
`SII-Meta-Quest3/LICENSE`, Apache-2.0).
