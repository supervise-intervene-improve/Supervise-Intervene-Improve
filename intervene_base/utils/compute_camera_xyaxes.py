import numpy as np

def camera_xyaxes_from_lookat(cam_pos, target, world_up=(0, 0, 1)):
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    # camera forward: from camera to target
    f = target - cam_pos
    f /= np.linalg.norm(f)

    # MuJoCo camera looks along -Z
    z = -f

    # camera X = image right
    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    # camera Y = image up
    y = np.cross(z, x)
    y /= np.linalg.norm(y)

    return np.concatenate([x, y])

target = np.array([0.6, 0.0, 0.22])
mid_target = np.array([0.6, 0.0, 0.25])

front = camera_xyaxes_from_lookat((1.5, 0.0, 0.8), target)
right = camera_xyaxes_from_lookat((1.1, +0.55, 0.6), target)
left  = camera_xyaxes_from_lookat((1.1, -0.55, 0.6), target)
VIS_RIGHT = camera_xyaxes_from_lookat((0.6, +0.75, +0.5), mid_target)
VIS_LEFT  = camera_xyaxes_from_lookat((0.6, -0.75, +0.5), mid_target)


print("VIS_RIGHT:", " ".join(f"{v:.3f}" for v in VIS_RIGHT))
print("VIS_LEFT:", " ".join(f"{v:.3f}" for v in VIS_LEFT))

print("front:", " ".join(f"{v:.3f}" for v in front))
print("right:", " ".join(f"{v:.3f}" for v in right))
print("left: ", " ".join(f"{v:.3f}" for v in left))