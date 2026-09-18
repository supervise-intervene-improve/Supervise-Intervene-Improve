import time
import mujoco
import numpy as np

from config import (
    ALPHA,
    CAMERA_NAMES,
    LAB_ID,
    LOG_HZ,
    MUJOCO_XML_PATH,
    OBJECT_RANDOM_SEED,
    OBJECT_EULER_RANDOM_RANGE_DEG,
    OBJECT_XY_BOUNDS,
    OBJECT_XY_RANDOM_RANGE,
    RANDOMIZE_OBJECT_EULER,
    RANDOMIZE_OBJECT_EULER_NAMES,
    PANDA_JOINT_NAMES,
    RANDOMIZE_OBJECT_NAMES,
    RANDOMIZE_OBJECT_XY,
    RGB_HEIGHT,
    RGB_WIDTH,
    ROBOT_KEY,
    SAVE_DEPTH,
    SAVE_RGB,
    SAVE_ROOT,
    TASK_ID,
    VIEW_HZ,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    WINDOW_WIDTH,
)
from collector.commands import CommandState
from collector.metadata import DemoMetadata
from collector.recorder import TrajectoryRecorder
from collector.renderer import MujocoRenderer
from collector.utils import ensure_actuator_size, get_qpos_indices_for_joints
from collector.window import install_callbacks
from robot_io.robot_worker import RobotWorker

from simpub.sim.mj_publisher import MujocoPublisher


GRIPPER_CTRL_INDEX = 7


class DemoCollectorApp:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
        self.data = mujoco.MjData(self.model)

        print("center:", self.model.stat.center)
        print("extent:", self.model.stat.extent)
        print("nq:", self.model.nq, "nv:", self.model.nv, "nu:", self.model.nu)

        self.arm_qpos_idxs = get_qpos_indices_for_joints(self.model, PANDA_JOINT_NAMES)
        self.rng = np.random.default_rng(OBJECT_RANDOM_SEED)
        self.randomized_object_qpos_idxs = self._get_randomized_object_qpos_idxs()
        self.randomized_object_euler_qpos_idxs = self._get_randomized_object_euler_qpos_idxs()

        self.renderer = MujocoRenderer(
            self.model,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WINDOW_TITLE,
        )

        self.cmd_state = CommandState()
        install_callbacks(
            self.renderer.window,
            self.model,
            self.renderer.scn,
            self.renderer.cam,
            self.cmd_state,
            self.renderer,
        )

        self.robot_worker = RobotWorker(ROBOT_KEY, poll_hz=100.0)

        self.current_recorder = None
        self.episode_idx = 0
        self.initialized_from_robot = False

        self.initial_robot_q = None
        self.initial_robot_grip = None
        self.initial_qpos = None
        self.initial_qvel = None

        self.waiting_for_reset = False
        self.pending_reset_move_seq = None
        self.reset_request_time = None

    def initialize(self):
        self.robot_worker.start()

    def _get_randomized_object_qpos_idxs(self):
        if not RANDOMIZE_OBJECT_XY:
            return {}

        return self._get_free_joint_qpos_idxs(RANDOMIZE_OBJECT_NAMES, "xy randomize")

    def _get_randomized_object_euler_qpos_idxs(self):
        if not RANDOMIZE_OBJECT_EULER:
            return {}

        return self._get_free_joint_qpos_idxs(RANDOMIZE_OBJECT_EULER_NAMES, "euler randomize")

    def _get_free_joint_qpos_idxs(self, body_names, label):
        qpos_idxs = {}
        for body_name in body_names:
            joint_name = f"{body_name}_free"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                print(f"[WARN] Cannot {label} {body_name}: missing free joint {joint_name!r}.")
                continue

            joint_type = self.model.jnt_type[joint_id]
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                print(f"[WARN] Cannot {label} {body_name}: joint {joint_name!r} is not free.")
                continue

            qpos_idxs[body_name] = int(self.model.jnt_qposadr[joint_id])

        return qpos_idxs

    @staticmethod
    def _euler_xyz_to_quat(roll, pitch, yaw):
        cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
        cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
        cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _apply_object_euler_randomization(self):
        if not self.randomized_object_euler_qpos_idxs or self.initial_qpos is None:
            return

        ranges = np.radians(
            [
                OBJECT_EULER_RANDOM_RANGE_DEG["roll"],
                OBJECT_EULER_RANDOM_RANGE_DEG["pitch"],
                OBJECT_EULER_RANDOM_RANGE_DEG["yaw"],
            ]
        )
        offsets = []

        for body_name, qpos_idx in self.randomized_object_euler_qpos_idxs.items():
            euler_offset = self.rng.uniform(-ranges, ranges)
            delta_quat = self._euler_xyz_to_quat(*euler_offset)
            base_quat = self.initial_qpos[qpos_idx + 3:qpos_idx + 7]
            randomized_quat = self._quat_mul(delta_quat, base_quat)
            randomized_quat /= np.linalg.norm(randomized_quat)
            self.data.qpos[qpos_idx + 3:qpos_idx + 7] = randomized_quat
            offsets.append(
                f"{body_name}=({np.degrees(euler_offset[0]):+.1f}, "
                f"{np.degrees(euler_offset[1]):+.1f}, {np.degrees(euler_offset[2]):+.1f})deg"
            )

        print("[RANDOMIZE] object euler:", ", ".join(offsets))

    def _apply_object_xy_randomization(self):
        if not self.randomized_object_qpos_idxs or self.initial_qpos is None:
            return

        x_min, x_max = OBJECT_XY_BOUNDS["x"]
        y_min, y_max = OBJECT_XY_BOUNDS["y"]
        offsets = []

        for body_name, qpos_idx in self.randomized_object_qpos_idxs.items():
            xy_offset = self.rng.uniform(
                -OBJECT_XY_RANDOM_RANGE,
                OBJECT_XY_RANDOM_RANGE,
                size=2,
            )
            base_xy = self.initial_qpos[qpos_idx:qpos_idx + 2]
            randomized_xy = np.array(
                [
                    np.clip(base_xy[0] + xy_offset[0], x_min, x_max),
                    np.clip(base_xy[1] + xy_offset[1], y_min, y_max),
                ],
                dtype=np.float64,
            )
            self.data.qpos[qpos_idx:qpos_idx + 2] = randomized_xy
            offsets.append(f"{body_name}=({randomized_xy[0]:.3f}, {randomized_xy[1]:.3f})")

        print("[RANDOMIZE] object xy:", ", ".join(offsets))

    def _apply_object_randomization(self):
        self._apply_object_xy_randomization()
        self._apply_object_euler_randomization()

    def _set_gripper_ctrl_from_robot(self, grip_width):
        if self.data.ctrl.shape[0] <= GRIPPER_CTRL_INDEX:
            return

        ctrl = float(grip_width)
        if self.model.actuator_ctrllimited[GRIPPER_CTRL_INDEX]:
            ctrlrange = self.model.actuator_ctrlrange[GRIPPER_CTRL_INDEX]
            ctrl = float(np.clip(ctrl, float(ctrlrange[0]), float(ctrlrange[1])))

        self.data.ctrl[GRIPPER_CTRL_INDEX] = ctrl

    def _try_initialize_sim_from_robot(self):
        if self.initialized_from_robot:
            return

        rs = self.robot_worker.get_latest_state()
        if not rs["connected"]:
            return
        if rs["q"] is None or rs["grip"] is None:
            return

        q0 = rs["q"]
        grip0 = rs["grip"]

        print("q0 =", q0)
        print("grip0 =", grip0)

        for i, qpos_idx in enumerate(self.arm_qpos_idxs):
            self.data.qpos[qpos_idx] = q0[i]

        mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[:7] = q0.copy()
        self._set_gripper_ctrl_from_robot(grip0)

        self.initial_robot_q = q0.copy()
        self.initial_robot_grip = float(grip0)
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self._apply_object_randomization()
        mujoco.mj_forward(self.model, self.data)

        print("data.qpos =", self.data.qpos)
        print("data.ctrl =", self.data.ctrl)

        print(f"[INIT] initial_robot_grip = {self.initial_robot_grip:.4f}")

        self.initialized_from_robot = True
        print("[INFO] MuJoCo initialized from robot state.")
        print("[INFO] Saved initial pose for all episodes.")

    def _create_recorder(self):
        self.episode_idx += 1
        ts = int(time.time())
        save_path = SAVE_ROOT / TASK_ID / LAB_ID / f"{ROBOT_KEY}_ep_{self.episode_idx:04d}_{ts}.npz"

        meta = DemoMetadata(
            task_id=TASK_ID,
            lab_id=LAB_ID,
            robot_key=ROBOT_KEY,
            robot_name=ROBOT_KEY,
            mujoco_xml_path=MUJOCO_XML_PATH,
            view_hz=VIEW_HZ,
            log_hz=LOG_HZ,
            alpha=ALPHA,
            panda_joint_names=PANDA_JOINT_NAMES,
            camera_names=CAMERA_NAMES,
            rgb_width=RGB_WIDTH,
            rgb_height=RGB_HEIGHT,
            save_rgb=SAVE_RGB,
            save_depth=SAVE_DEPTH,
            start_wall_time=time.time(),
            episode_idx=self.episode_idx,
            notes="Kinesthetic teaching with fixed robot reset and randomized object xy pose.",
        )

        return TrajectoryRecorder(
            save_path=save_path,
            metadata=meta,
            log_hz=LOG_HZ,
            view_hz=VIEW_HZ,
            camera_names=CAMERA_NAMES,
            save_rgb=SAVE_RGB,
            save_depth=SAVE_DEPTH,
        )

    def _request_reset(self):
        if self.initial_robot_q is None:
            print("[WARN] Initial robot pose not available yet.")
            return

        print("[INFO] Requesting robot reset to saved initial pose...")
        self.waiting_for_reset = True
        self.reset_request_time = time.time()
        self.pending_reset_move_seq = self.robot_worker.request_go_to_joint_and_gripper(
            self.initial_robot_q,
            self.initial_robot_grip,
        )

    def _finish_reset_if_ready(self):
        if not self.waiting_for_reset:
            return

        rs = self.robot_worker.get_latest_state()
        if rs["q"] is None or rs["grip"] is None:
            return

        # Motion not completed yet
        if self.pending_reset_move_seq is not None:
            if rs["last_completed_move_seq"] < self.pending_reset_move_seq:
                return

        # Prefer a decent closeness check, but don't get stuck forever
        err = np.linalg.norm(rs["q"] - self.initial_robot_q)
        time_since_request = time.time() - self.reset_request_time if self.reset_request_time else 0.0

        # Accept if close enough, or if the move completed and we waited long enough
        if err > 0.12 and time_since_request < 1.0:
            return

        print(f"[INFO] Finishing reset. joint error to saved pose = {err:.4f}")

        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()

        for i, qpos_idx in enumerate(self.arm_qpos_idxs):
            self.data.qpos[qpos_idx] = rs["q"][i]

        self.data.ctrl[:7] = self.initial_robot_q.copy()
        self._set_gripper_ctrl_from_robot(rs["grip"])

        self._apply_object_xy_randomization()
        mujoco.mj_forward(self.model, self.data)

        self.waiting_for_reset = False
        self.pending_reset_move_seq = None
        self.reset_request_time = None
        print("[INFO] Reset complete. Ready for next episode.")
        rs = self.robot_worker.get_latest_state()
        print(f"[RESET] current_grip = {rs['grip']:.4f}")

    def _handle_commands(self):
        if self.cmd_state.consume_start():
            if self.current_recorder is None and not self.waiting_for_reset:
                self.current_recorder = self._create_recorder()
                print(f"[INFO] Started episode {self.episode_idx}")
            elif self.waiting_for_reset:
                print("[WARN] Reset in progress. Wait before starting a new episode.")
            else:
                print("[WARN] Episode already active.")

        if self.cmd_state.consume_stop():
            if self.current_recorder is not None:
                if self.current_recorder.num_frames() > 0:
                    self.current_recorder.save()
                else:
                    print("[WARN] Episode empty. Not saving.")

                print(f"[INFO] Saved episode {self.episode_idx}")
                self.current_recorder = None
                self._request_reset()
            else:
                print("[WARN] No active episode.")

        if self.cmd_state.consume_discard():
            if self.current_recorder is not None:
                print(f"[INFO] Discarded episode {self.episode_idx}")
                self.current_recorder = None
                self._request_reset()
            else:
                print("[WARN] No active episode to discard.")

    def _build_overlay_lines(self):
        rs = self.robot_worker.get_latest_state()

        if rs["connected"]:
            robot_status = "BUSY" if rs["busy"] else "CONNECTED"
        elif rs["connect_error"] is not None:
            robot_status = f"ERROR: {rs['connect_error'][:40]}"
        else:
            robot_status = "CONNECTING..."

        if self.waiting_for_reset:
            rec_status = "RESETTING"
            frame_count = 0
        elif self.current_recorder is None:
            rec_status = "IDLE"
            frame_count = 0
        else:
            rec_status = "RECORDING"
            frame_count = self.current_recorder.num_frames()

        return [
            f"Task: {TASK_ID}",
            f"Lab: {LAB_ID}",
            f"Robot: {ROBOT_KEY}",
            f"Robot status: {robot_status}",
            f"Grip real: {rs['grip']:.4f}" if rs["grip"] is not None else "Grip real: --",
            f"Grip sim ctrl: {self.data.ctrl[GRIPPER_CTRL_INDEX]:.4f}"
            if self.data.ctrl.shape[0] > GRIPPER_CTRL_INDEX
            else "Grip sim ctrl: --",
            f"Episode: {self.episode_idx}",
            f"Status: {rec_status}",
            f"Frames: {frame_count}",
            "",
            "Keys:",
            "S - start episode",
            "E - end episode + save + reset",
            "X - discard episode + reset",
            "O - toggle overlay",
            "Q / ESC - quit",
            "",
            "Mouse:",
            "Left drag   - rotate",
            "Right drag  - move",
            "Scroll      - zoom",
            "Shift+drag  - horizontal mode",
        ]

    def step(self):
        dt = 1.0 / VIEW_HZ
        ensure_actuator_size(self.data, 8)

        loop_start = time.time()

        self.renderer.poll_events()
        self._try_initialize_sim_from_robot()

        rs = self.robot_worker.get_latest_state()

        if (
            rs["connected"]
            and rs["q"] is not None
            and rs["grip"] is not None
            and not self.waiting_for_reset
        ):
            q_real = rs["q"]
            dq_real = rs["dq"]
            grip_real = rs["grip"]

            self.data.ctrl[:7] = (1.0 - ALPHA) * self.data.ctrl[:7] + ALPHA * q_real
            self._set_gripper_ctrl_from_robot(grip_real)
        else:
            q_real = np.zeros(7, dtype=np.float64)
            dq_real = None
            grip_real = 0.0

        n_substeps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._handle_commands()
        self._finish_reset_if_ready()

        if (
            self.current_recorder is not None
            and not self.waiting_for_reset
            and self.current_recorder.should_log_this_step()
        ):
            self.current_recorder.record_state(
                now_wall=time.time(),
                data=self.data,
                q_real=q_real,
                dq_real=dq_real,
                gripper_width=grip_real,
            )

            rgb_by_cam = {}
            depth_by_cam = {}

            for cam_name in CAMERA_NAMES:
                rgb, depth = self.renderer.render_rgbd_from_camera(
                    self.data, cam_name, RGB_WIDTH, RGB_HEIGHT
                )
                if SAVE_RGB:
                    rgb_by_cam[cam_name] = rgb
                if SAVE_DEPTH:
                    depth_by_cam[cam_name] = depth

            self.current_recorder.record_images(
                rgb_by_cam=rgb_by_cam if SAVE_RGB else None,
                depth_by_cam=depth_by_cam if SAVE_DEPTH else None,
            )

        overlay_lines = self._build_overlay_lines()
        self.renderer.render_viewer(self.data, overlay_lines=overlay_lines)

        elapsed = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    def run(self):
        self.initialize()
        print("[INFO] Controls: S=start, E=end, Q=quit")
        publisher = MujocoPublisher(self.model, self.data, host="192.168.0.38", visible_geoms_groups=[0, 2])
        try:
            while not self.renderer.window_should_close():
                if self.cmd_state.is_quit_requested():
                    break
                self.step()
        finally:
            if self.current_recorder is not None and self.current_recorder.num_frames() > 0:
                self.current_recorder.save()

            self.renderer.close()
            self.robot_worker.close()
