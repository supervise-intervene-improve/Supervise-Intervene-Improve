import numpy as np
import mujoco
import matplotlib.pyplot as plt


MUJOCO_XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml"

HEIGHT = 224
WIDTH = 224


def get_display_limits(depth: np.ndarray):
    valid = np.isfinite(depth)
    if not np.any(valid):
        return 0.0, 1.0

    vmin = np.min(depth[valid])
    vmax = np.percentile(depth[valid], 99)
    if vmax <= vmin:
        vmax = np.max(depth[valid])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


# Load model
model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# ✅ Automatically get cameras
CAM_NAMES = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    for i in range(model.ncam)
    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) is not None
]

print("Detected cameras:", CAM_NAMES)

# Prepare figure
num_cams = len(CAM_NAMES)
fig, axes = plt.subplots(num_cams, 2, figsize=(10, 6 * num_cams), dpi=120)

# Handle case with 1 camera
if num_cams == 1:
    axes = np.expand_dims(axes, axis=0)

for i, cam_name in enumerate(CAM_NAMES):
    # RGB
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render()

    # Depth
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    vmin, vmax = get_display_limits(depth)

    # Plot RGB
    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f"{cam_name} | RGB")
    axes[i, 0].axis("off")

    # Plot Depth
    im = axes[i, 1].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f"{cam_name} | Depth")
    axes[i, 1].axis("off")

    # Optional: add colorbar per row
    plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()