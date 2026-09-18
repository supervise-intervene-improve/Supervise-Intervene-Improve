import time
from pathlib import Path
import numpy as np

# MuJoCo physics + interactive viewer
import mujoco
import mujoco.viewer

# Your previous recording/teleop module (contains ROBOTS, ControlType, hold_then_replan_record, etc.)
import record

# ============================================================
# User configuration
# ============================================================

# Path to a previously recorded trajectory (npz) produced by TrajectoryRecorder.save()
LOG_PATH = "/path/to/Intervention_IL_AR/intervene_base/demonstrations/demo_pick_place/T_shape/p1_ep_0001_1776868120.npz"

# MuJoCo model used for replay (must match the one used during recording)
MUJOCO_XML_PATH = "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"

# Replay loop frequency (viewer update rate)
VIEW_HZ = 60.0
DT = 1.0 / VIEW_HZ

# If True, send the replayed commands to a REAL robot as well (dangerous if not careful!)
MIRROR_ROBOT = True
MIRROR_ROBOT_KEY = "p1"

# Control mode used when mirroring commands to the real robot
MIRROR_MODE = record.ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL

# Optional joint velocity limit (rad/s) when mirroring to real robot
# This prevents large jumps / unsafe motion due to command discontinuities.
MAX_DQ = 0.8


# ============================================================
# GLFW is used to capture key presses in the MuJoCo viewer
# (space to pause, esc to quit)
# ============================================================
try:
    import glfw
except Exception as e:
    raise RuntimeError("Need glfw for key handling (pip install glfw).") from e


# ============================================================
# Utility: Stitch two npz files (original prefix + new suffix)
# ============================================================
def stitch_npz(original_path: str, suffix_path: str, cut_idx: int, out_path: str):
    """
    Create a new trajectory file by:
      - taking original data up to cut_idx (inclusive)
      - appending all suffix data
    This is used after replanning: keep the replay prefix, then append the replan recording.
    """
    orig = np.load(original_path)
    suf = np.load(suffix_path)

    out = {}

    # For each key in the original file, try to concatenate if the suffix also contains it
    for k in orig.files:
        if k in suf.files and orig[k].ndim >= 1:
            # concatenate along time axis
            out[k] = np.concatenate([orig[k][: cut_idx + 1], suf[k]], axis=0)
        else:
            # keep original value as-is (e.g., scalars or missing keys)
            out[k] = orig[k]

    # Add any keys that exist only in the suffix
    for k in suf.files:
        if k not in out:
            out[k] = suf[k]

    # Save stitched trajectory
    np.savez_compressed(out_path, **out)
    print(f"[INFO] Stitched saved: {out_path}")


# ============================================================
# Utility: Command rate-limiting for real robot safety
# ============================================================
def clamp_step(q_cmd, q_prev, max_dq, dt):
    """
    Clamp the joint velocity implied by q_cmd to +/- max_dq.
    This is a simple per-joint rate limiter:
        dq = (q_cmd - q_prev)/dt
        dq = clip(dq, -max_dq, max_dq)
        q_next = q_prev + dq*dt
    """
    dq = (q_cmd - q_prev) / dt
    dq = np.clip(dq, -max_dq, max_dq)
    return q_prev + dq * dt


def main():
    # ------------------------------------------------------------
    # Load MuJoCo model and initialize sim
    # ------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)

    # Load recorded log
    log = np.load(LOG_PATH)

    # ctrl_sim: sequence of control vectors applied in sim during recording
    ctrl_seq = log["ctrl_sim"]

    # sim_t: simulation timestamps corresponding to the recording
    sim_t = log["sim_t"]

    # Initialize simulation with the first recorded control command
    data.ctrl[:] = ctrl_seq[0]
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------
    # Optional: connect to real robot and mirror replay commands
    # ------------------------------------------------------------
    robot = None
    q_prev = None

    if MIRROR_ROBOT:
        robot = record.ROBOTS[MIRROR_ROBOT_KEY]
        print(f"[MIRROR] Connecting real robot {MIRROR_ROBOT_KEY} in {MIRROR_MODE}...")
        robot.connect(MIRROR_MODE)

        # Reset robot state
        robot.reset()

        # Store current real robot joint position for velocity clamping
        q_prev = robot.robot_arm.get_state().joint_pos.detach().cpu().numpy().astype(np.float64)
        print("[MIRROR] Connected. Real robot will mirror replay.")

    # ------------------------------------------------------------
    # Input state flags driven by keyboard callback
    # ------------------------------------------------------------
    request_pause = False
    request_quit = False

    # Index into ctrl_seq (the recorded command timeline)
    i = 0

    # ------------------------------------------------------------
    # Viewer key callback (GLFW)
    # ------------------------------------------------------------
    def key_cb(*args):
        """
        MuJoCo viewer calls this with key events.
        We interpret:
          - SPACE: request pause (open terminal prompt for action)
          - ESC: request quit
        """
        nonlocal request_pause, request_quit
        if not args:
            return

        key = args[0]

        # args typically include (key, scancode, action, mods)
        if len(args) >= 3:
            act = args[2]
            if act not in (glfw.PRESS, glfw.REPEAT):
                return

        if key == glfw.KEY_SPACE:
            request_pause = True
        elif key == glfw.KEY_ESCAPE:
            request_quit = True

    print(
        "\nControls:\n"
        "  SPACE: pause and choose (r=replan, c=continue, q=quit)\n"
        "  ESC: quit\n"
        f"  MIRROR_ROBOT={'ON' if MIRROR_ROBOT else 'OFF'}\n"
    )
    FIRST_IN = False
    try:
        # If user chooses "replan", we store seeds here and then exit viewer loop
        pending_replan = False
        pending = {}

        # Launch passive viewer (we step simulation ourselves)
        with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:

            # ------------------------------------------------------------
            # Main replay loop
            # ------------------------------------------------------------
            while viewer.is_running() and i < len(ctrl_seq):
                t0 = time.time()

                # Hard quit if requested
                if request_quit:
                    print("[INFO] Quit requested.")
                    break

                # --------------------------------------------------------
                # Pause menu: user chooses continue / quit / replan
                # --------------------------------------------------------
                if request_pause:
                    request_pause = False

                    # Compute where we are in the recording timeline:
                    # find largest index where sim_t[idx] <= current data.time
                    cut_idx = int(np.searchsorted(sim_t, data.time, side="right") - 1)
                    cut_idx = int(np.clip(cut_idx, 0, len(sim_t) - 1))

                    print(f"\n[PAUSE] paused at sim_time={data.time:.3f}, cut_idx={cut_idx}")
                    choice = input("Choose: r=replan from here, c=continue, q=quit: ").strip().lower()

                    if choice == "q":
                        print("[INFO] Quitting.")
                        break

                    if choice == "c":
                        print("[PAUSE] continuing replay.")

                    elif choice == "r":
                        # User wants to stop replay and start a replanning segment.
                        pending_replan = True
                        FIRST_IN = True

                        # Save the exact sim state at pause time, so replan recording continues smoothly
                        pending["cut_idx"] = cut_idx
                        pending["qpos_seed"] = data.qpos.copy()
                        pending["qvel_seed"] = data.qvel.copy()

                        # Use current control as the "hold pose" and seed pose for replan
                        pending["q_seed"] = np.asarray(data.ctrl[:7], dtype=np.float64).copy()
                        pending["finger_seed"] = float(data.ctrl[7]) if model.nu >= 8 else 0.0

                        print("[INFO] Closing replay window to start replanning...")
                        break
                    else:
                        print("[PAUSE] unknown choice; continuing.")

                # --------------------------------------------------------
                # Apply the next recorded command to the simulation
                # --------------------------------------------------------
                data.ctrl[:] = ctrl_seq[i]
                grip_width = log['grip_real'][i]
                # --------------------------------------------------------
                # Optional: send the same command to the real robot
                # --------------------------------------------------------
                if robot is not None:
                    q_cmd = np.asarray(data.ctrl[:7], dtype=np.float64)

                    # Safety: clamp the step (limit joint velocity)
                    if MAX_DQ is not None and q_prev is not None:
                        q_cmd = clamp_step(q_cmd, q_prev, MAX_DQ, DT)

                    q_prev = q_cmd.copy()

                    robot.robot_arm.go_to_within_limits(q_cmd)

                    if grip_width >= (robot.robot_gripper.max_width + robot.robot_gripper.min_width)/2:
                        robot.robot_gripper.apply_commands(1.0)
                    else:
                        robot.robot_gripper.apply_commands(-1.0)

                    # Apply gripper command if model has gripper actuator in ctrl[7]
                    # if model.nu >= 8:
                    #     # recorded ctrl[7] is per-finger pos (0..0.04), convert to width (0..0.08)
                    #     width_cmd = float(np.clip(2.0 * float(data.ctrl[7]), 0.0, 0.08))
                    #     robot.robot_gripper.apply_commands(width_cmd)

                # --------------------------------------------------------
                # Step MuJoCo for one display tick (DT)
                # --------------------------------------------------------
                steps = max(1, int(DT / model.opt.timestep))
                for _ in range(steps):
                    mujoco.mj_step(model, data)

                # Advance recorded index "i" when sim time passes next timestamp
                if i + 1 < len(sim_t) and data.time >= sim_t[i + 1]:
                    i += 1

                # Update viewer visuals
                viewer.sync()

                # Maintain VIEW_HZ in wall-time
                sleep_time = DT - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # ------------------------------------------------------------
        # If user requested replanning, run hold + record and stitch
        # ------------------------------------------------------------
        if pending_replan:
            qpos_seed = pending["qpos_seed"]
            qvel_seed = pending["qvel_seed"]
            q_seed = pending["q_seed"]
            finger_seed = pending["finger_seed"]
            cut_idx = pending["cut_idx"]

            # Use the default record.ROBOT_KEY for replanning record
            # (you might want to use MIRROR_ROBOT_KEY instead, depending on your workflow)
            robot_replan = record.ROBOTS[record.ROBOT_KEY]

            # Where to save the replanned suffix recording
            suffix_path = Path(LOG_PATH).with_name(f"suffix_{int(time.time())}.npz")

            # This function:
            # 1) holds the robot at q_hold (impedance control),
            # 2) asks the user when ready,
            # 3) switches to human control (if possible),
            # 4) records a new segment while user guides.
            record.hold_then_replan_record(
                robot=robot_replan,
                q_hold=q_seed,
                finger_hold=finger_seed,
                save_path=suffix_path,
                mujoco_xml_path=MUJOCO_XML_PATH,
                view_hz=VIEW_HZ,
                log_hz=60.0,
                alpha=record.ALPHA,
                qpos_seed=qpos_seed,
                qvel_seed=qvel_seed,
            )

            # Output stitched file = original prefix up to cut_idx + new suffix
            out_path = str(Path(LOG_PATH).with_name(Path(LOG_PATH).stem + f"_replanned_{int(time.time())}.npz"))
            stitch_npz(LOG_PATH, str(suffix_path), cut_idx, out_path)

            print("[INFO] Done replanning+stitching.")

    finally:
        # Always close robot if connected
        if robot is not None:
            print("[MIRROR] Closing robot connection...")
            robot.close()


if __name__ == "__main__":
    main()