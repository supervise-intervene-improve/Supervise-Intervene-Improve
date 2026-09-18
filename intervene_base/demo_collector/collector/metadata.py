from dataclasses import dataclass


@dataclass
class DemoMetadata:
    task_id: str
    lab_id: str
    robot_key: str
    robot_name: str
    mujoco_xml_path: str
    view_hz: float
    log_hz: float
    alpha: float
    panda_joint_names: list[str]
    camera_names: list[str]
    rgb_width: int
    rgb_height: int
    save_rgb: bool
    save_depth: bool
    start_wall_time: float
    episode_idx: int
    notes: str = ""