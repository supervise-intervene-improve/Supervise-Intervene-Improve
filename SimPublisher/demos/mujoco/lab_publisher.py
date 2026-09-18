import argparse
import os
import time
import asyncio
import sys
import mujoco

from simpub.sim.mj_publisher import MujocoPublisher

# Windows: avoid zmq/asyncio Proactor issues
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def resolve_xml(lab_id: int) -> str:
    # Adjust these to your actual filenames/paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    lab_dir = os.path.join(repo_root, "LAB")

    mapping = {
        1: os.path.join(lab_dir, "lab1_T_stack_vizCollisionSeparated_v3_groups_named_v4_noFloorVisual.xml"),
        2: os.path.join(lab_dir, "lab2_boxes_cups_vizCollisionSeparated_v3_groups_named_v4_noFloorVisual.xml"),
        3: os.path.join(lab_dir, "lab3_stick_maze_vizCollisionSeparated_v3_groups_named_v4_noFloorVisual.xml"),
        4: os.path.join(lab_dir, "lab1_T_stack.xml"),
        5: os.path.join(lab_dir, "lab2_boxes_cups.xml"),
        6: os.path.join(lab_dir, "lab3_stick_maze.xml"),
    }
    if lab_id not in mapping:
        raise ValueError(f"Unknown lab_id={lab_id}. Use 1/2/3.")
    return mapping[lab_id]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", type=int, default=2, help="1, 2, or 3")
    ap.add_argument("--host", type=str, required=True, help="PC LAN IP, e.g. 192.168.0.208")
    ap.add_argument("--hz", type=float, default=60.0)
    args = ap.parse_args()

    xml_path = resolve_xml(args.lab)
    print(f"[LabPublisher] Loading MJCF: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # IMPORTANT: show all geom groups unless you deliberately want filtering
    publisher = MujocoPublisher(
        model,
        data,
        host=args.host,
        visible_geoms_groups=[2],
    )

    # Ensure xpos/xquat valid
    mujoco.mj_step(model, data)

    dt = 1.0 / args.hz
    print(f"[LabPublisher] Streaming at ~{args.hz} Hz on host {args.host}")

    while True:
        mujoco.mj_step(model, data)
        time.sleep(dt)

if __name__ == "__main__":
    main()
#python franka_quest_teleop.py --xml lab1_T_stack_vizCollisionSeparated_v3_groups_named_v4_noFloorVisual.xml --host 192.168.0.208 --unity_node MQ3-2
