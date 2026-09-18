import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

def infer_arm_indices_from_qpos(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """
    Return the qpos indices in MuJoCo corresponding to the given joint names.

    MuJoCo stores joint positions in the global vector `data.qpos`.
    Each joint has an index (address) inside that vector.
    This function maps joint names -> qpos indices.
    """
    idxs = []
    for name in joint_names:
        j = model.joint(name)              # Access joint object
        joint_id = j.id                    # Internal MuJoCo joint ID
        qpos_adr = model.jnt_qposadr[joint_id]  # Address inside qpos
        idxs.append(int(qpos_adr))
    return idxs


def _episode_scene_xml(ep: dict, npz_path: Path) -> str:
    """Which MuJoCo scene this episode was recorded with, or "" if unknown.

    Prefers the NPZ (self-describing, survives a file being moved on its own) and falls back
    to the sidecar .json, which is where provenance lived before it was promoted into the NPZ.
    """
    value = ep.get("mujoco_xml_path")
    if value is not None:
        try:
            return str(np.asarray(value).item())
        except Exception:
            return str(value)
    sidecar = npz_path.with_suffix(".json")
    if sidecar.is_file():
        try:
            return str(json.loads(sidecar.read_text()).get("mujoco_xml_path", "") or "")
        except Exception:
            return ""
    return ""


def load_episode(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def collect_episode_files(episodes_dir: Path) -> list[Path]:
    episode_candidates = sorted(episodes_dir.glob("*.npz"))
    if not episode_candidates:
        raise FileNotFoundError(f"No .npz files found in {episodes_dir}")

    episode_files = []
    missing = []
    for ep_path in episode_candidates:
        if ep_path.exists() and ep_path.is_file():
            episode_files.append(ep_path)
        else:
            missing.append(ep_path)

    for ep_path in missing:
        if ep_path.is_symlink():
            print(f"[WARN] Skipping broken episode symlink: {ep_path} -> {ep_path.readlink()}")
        else:
            print(f"[WARN] Skipping missing episode file: {ep_path}")

    if not episode_files:
        raise FileNotFoundError(f"No readable .npz files found in {episodes_dir}")

    if missing:
        print(f"[INFO] Using {len(episode_files)} readable episodes; skipped {len(missing)} missing/broken entries.")

    return episode_files


def to_chw_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {img.shape}")

    if img.shape[-1] == 3:
        return np.transpose(img, (2, 0, 1))

    if img.shape[0] == 3:
        return img

    raise ValueError(f"Cannot infer image format from shape {img.shape}")


def parse_camera_names(value: str) -> list[str]:
    cameras = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not cameras:
        raise ValueError("At least one camera name is required")
    return cameras


class MujocoEpisodeRenderer:
    def __init__(self, xml_path: str, height: int = 224, width: int = 224):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

    def render(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray, camera: str) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        n_ctrl = min(self.data.ctrl.shape[0], ctrl.shape[0])
        self.data.ctrl[:n_ctrl] = ctrl[:n_ctrl]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        # Mirror the LIVE recorder's render flags (rendering/viewer.py::set_fast_visuals).
        # mujoco.Renderer enables shadows by default; the live capture does not. That single
        # flag put re-rendered frames 13-25/255 away from the live ones — with it cleared the
        # difference is 0.67/255 (see utils/verify_rerender_fidelity.py). REFLECTION/SKYBOX/FOG
        # make no difference in the current scenes but are mirrored so a future XML that adds
        # a skybox or a reflective floor cannot silently reintroduce the mismatch.
        # MUST be set after update_scene(), which repopulates the scene each call.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        return self.renderer.render().astype(np.uint8)

    def close(self) -> None:
        self.renderer.close()


def build_state(qpos_sim_t: np.ndarray, qvel_sim_t: np.ndarray, ctrl_sim_t: np.ndarray, arm_qpos_idxs: list[int]) -> np.ndarray:
    q_arm = qpos_sim_t[arm_qpos_idxs]
    dq_arm = qvel_sim_t[:7]  # assuming first 7 qvel are arm dofs
    grip = np.array([ctrl_sim_t[7]], dtype=np.float32)
    state = np.concatenate([q_arm, dq_arm, grip], axis=0).astype(np.float32)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--episodes_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Local dataset name, e.g. anonymous/tshape_sim_bc")
    parser.add_argument("--root", type=str, default="lerobot_local_data",
                        help="Root directory for local LeRobot datasets")
    parser.add_argument("--task", type=str, default="T-shape pick and place")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--action_shift",
        type=int,
        default=0,
        help=(
            "Use ctrl_sim[t + action_shift] as the action for observation t. "
            "For demos recorded after stepping the simulator, --action_shift 1 "
            "trains the policy to predict the next control target instead of "
            "repeating the control already active in the observation."
        ),
    )
    parser.add_argument("--use_videos", action="store_true",
                        help="Store videos instead of per-frame images if supported by your install")
    parser.add_argument("--arm_qpos_idxs_json", type=str, default="",
                        help='JSON list of true arm qpos indices, e.g. "[0,1,2,3,4,5,6]"')
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing dataset at root/repo_id before creating.")
    parser.add_argument(
        "--rerender_images",
        action="store_true",
        help=(
            "Ignore rgb_right/rgb_left in the NPZ and render camera images from "
            "qpos_sim/qvel_sim/ctrl_sim using the provided MuJoCo XML. This keeps "
            "training images aligned with rollout-time MuJoCo rendering."
        ),
    )
    parser.add_argument(
        "--camera_names",
        default="right,left,wrist",
        help="Comma/space separated MuJoCo camera names to store as observation.images.<name>. "
             "Default matches what the ACT checkpoints declare as required inputs "
             "(observation.images.{right,left,wrist}); the old 'right,left' default silently "
             "produced a 2-camera dataset the policy cannot consume.",
    )
    args = parser.parse_args()
    if args.action_shift < 0:
        raise ValueError("--action_shift must be non-negative")
    camera_names = parse_camera_names(args.camera_names)

    episodes_dir = Path(args.episodes_dir)
    root = Path(args.root)

    if args.overwrite and root.exists():
        print(f"[INFO] Removing existing dataset root: {root}")
        shutil.rmtree(root)

    if args.arm_qpos_idxs_json:
        arm_qpos_idxs = json.loads(args.arm_qpos_idxs_json)
    else:
        arm_model = mujoco.MjModel.from_xml_path(args.xml)
        arm_qpos_idxs = infer_arm_indices_from_qpos(arm_model, PANDA_JOINT_NAMES)

    print("Inferred arm qpos indices:", arm_qpos_idxs)

    episode_files = collect_episode_files(episodes_dir)

    # LeRobot v3 creation pattern shown in docs/examples:
    #   dataset = LeRobotDataset.create(...)
    #   dataset.add_frame(frame)
    #   dataset.save_episode()
    #   dataset.finalize()
    #
    # This script uses a simple feature schema:
    # - observation.state         -> float32[15]
    # - observation.images.<camera> -> uint8[3,H,W]
    # - action                    -> float32[8]
    image_features = {
        f"observation.images.{camera_name}": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }
        for camera_name in camera_names
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="franka_panda_mujoco",
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (15,),
                "names": {
                    "axes": [
                        "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                        "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                        "grip",
                    ]
                },
            },
            **image_features,
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": {
                    "axes": ["ctrl1", "ctrl2", "ctrl3", "ctrl4", "ctrl5", "ctrl6", "ctrl7", "grip_ctrl"]
                },
            },
            "intervention": {
                "dtype": "int64",
                "shape": (1,),
                "names": {"axes": ["is_human_intervention"]},
            },
            "action_source": {
                "dtype": "int64",
                "shape": (1,),
                "names": {"axes": ["0_policy_1_human"]},
            },
        },
        use_videos=args.use_videos,
    )

    all_actions = []
    renderer = MujocoEpisodeRenderer(args.xml) if args.rerender_images else None

    try:
        for ep_idx, ep_path in enumerate(episode_files):
            ep = load_episode(ep_path)

            required = ["qpos_sim", "qvel_sim", "ctrl_sim"]
            if not args.rerender_images:
                required += [f"rgb_{camera_name}" for camera_name in camera_names]
            for key in required:
                if key not in ep:
                    raise KeyError(f"{ep_path.name} missing key: {key}")

            # When re-rendering, --xml is the ONLY thing that determines what the training
            # images contain — the missing-key guard above no longer protects us, and a wrong
            # scene silently produces a plausible-looking but wrong dataset. Each episode
            # records the scene it actually ran under; refuse to convert against a different
            # one rather than fail quietly.
            if args.rerender_images:
                recorded_xml = _episode_scene_xml(ep, ep_path)
                if recorded_xml and Path(recorded_xml).name != Path(args.xml).name:
                    raise ValueError(
                        f"{ep_path.name} was recorded with scene '{recorded_xml}' but "
                        f"--xml is '{args.xml}'. Re-rendering against the wrong scene would "
                        f"produce wrong images. Convert this episode separately, or pass the "
                        f"matching --xml."
                    )

            n = ep["ctrl_sim"].shape[0]
            n_obs = n - args.action_shift
            if n_obs <= 0:
                raise ValueError(
                    f"{ep_path.name} has {n} frames, too short for action_shift={args.action_shift}"
                )
            print(f"\n[INFO] Episode {ep_idx}: {ep_path.name}")
            print("  frames:", n)
            print("  usable observation/action pairs:", n_obs)
            intervention = np.asarray(
                ep.get("intervention", np.zeros((n,), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            action_source = np.asarray(
                ep.get("action_source", intervention),
                dtype=np.int64,
            ).reshape(-1)

            for t in range(n_obs):
                action_t = t + args.action_shift
                state = build_state(
                    qpos_sim_t=ep["qpos_sim"][t],
                    qvel_sim_t=ep["qvel_sim"][t],
                    ctrl_sim_t=ep["ctrl_sim"][t],
                    arm_qpos_idxs=arm_qpos_idxs,
                )
                action = ep["ctrl_sim"][action_t].astype(np.float32)
                all_actions.append(action)

                frame = {
                    "observation.state": torch.from_numpy(state),
                    "action": torch.from_numpy(action),
                    "intervention": torch.tensor([int(intervention[t])], dtype=torch.int64),
                    "action_source": torch.tensor([int(action_source[t])], dtype=torch.int64),
                    "task": args.task,
                }
                for camera_name in camera_names:
                    if args.rerender_images:
                        rgb = renderer.render(
                            ep["qpos_sim"][t],
                            ep["qvel_sim"][t],
                            ep["ctrl_sim"][t],
                            camera_name,
                        )
                    else:
                        rgb = ep[f"rgb_{camera_name}"][t]
                    frame[f"observation.images.{camera_name}"] = torch.from_numpy(to_chw_uint8(rgb))
                dataset.add_frame(frame)

            dataset.save_episode()
            print(f"[OK] saved episode {ep_idx}: {ep_path.name}")
    finally:
        if renderer is not None:
            renderer.close()

    dataset.finalize()

    all_actions = np.stack(all_actions)
    print(f"[DONE] LeRobot absolute-action dataset written to: {root}")
    print("\n[DEBUG] Absolute action stats:")
    print("  shape:", all_actions.shape)
    print("  first:", all_actions[0])
    print("  mean:", all_actions.mean(axis=0))
    print("  std: ", all_actions.std(axis=0))
    print("  min: ", all_actions.min(axis=0))
    print("  max: ", all_actions.max(axis=0))


if __name__ == "__main__":
    main()
