import numpy as np
import mujoco


def safe_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def ensure_actuator_size(data: mujoco.MjData, min_size: int):
    if data.ctrl.shape[0] < min_size:
        raise RuntimeError(
            f"Expected at least {min_size} actuators, got {data.ctrl.shape[0]}"
        )


def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    idxs = []
    for name in joint_names:
        joint = model.joint(name)
        idxs.append(int(model.jnt_qposadr[joint.id]))
    return idxs


def get_camera_id(model: mujoco.MjModel, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found.")
    return cam_id


def depth_buffer_to_meters(model: mujoco.MjModel, depth_buffer: np.ndarray) -> np.ndarray:
    znear = model.vis.map.znear * model.stat.extent
    zfar = model.vis.map.zfar * model.stat.extent

    depth = depth_buffer.astype(np.float32)
    depth_m = znear / (1.0 - depth * (1.0 - znear / zfar))

    invalid = ~np.isfinite(depth_m)
    depth_m[invalid] = 0.0
    return depth_m