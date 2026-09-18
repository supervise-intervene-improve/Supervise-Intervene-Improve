from pathlib import Path

# cameras only for viewer
VIEWER_MODE = "multicam"   # "freecam" or "multicam"
VIEWER_CAMERA_NAMES = {
    "front": "front",
    "right": "VIS_RIGHT",
    "left": "VIS_LEFT",
}

LAB_ID = "boxes_cups" # "T_shape", "boxes_cups", "wire_base_and_spoon"
TASK_ID = "demo_pick_place_workspace" # "demo_pick_place" "demo_pick_place_CUPS" "WG"
ROBOT_KEY = "p1"
# mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml
MUJOCO_XML_PATH = (
    "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_" + LAB_ID + ".xml"
)

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

VIEW_HZ = 60.0
LOG_HZ = 10.0
ALPHA = 0.9

RANDOMIZE_OBJECT_XY = LAB_ID == "boxes_cups"
RANDOMIZE_OBJECT_NAMES = [
    "cracker_box",
    "sugar_box",
    "cup1",
    "cup2",
    "cup3",
    "cup4",
]
OBJECT_XY_RANDOM_RANGE = 0.01
OBJECT_XY_BOUNDS = {
    "x": (0.40, 0.80),
    "y": (-0.25, 0.25),
}

RANDOMIZE_OBJECT_EULER = LAB_ID == "boxes_cups"
RANDOMIZE_OBJECT_EULER_NAMES = [
    "cracker_box",
    "sugar_box",
]
# Small random yaw around the table normal. Keep roll/pitch at 0 so boxes stay standing.
OBJECT_EULER_RANDOM_RANGE_DEG = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 5.0,
}
OBJECT_RANDOM_SEED = None

RGB_WIDTH = 224
RGB_HEIGHT = 224

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "MuJoCo Demo Collector"

SAVE_ROOT = Path("workspace_Test_demonstrations")

# cameras only for recording
CAMERA_NAMES = [
    # "front",
    "right",
    "left",
    "wrist",
]

SAVE_RGB = True
SAVE_DEPTH = True
