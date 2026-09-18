import argparse, time, os, sys, asyncio
import numpy as np
import mujoco

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.core.node_manager import init_xr_node_manager
from simpub.xr_device.meta_quest3 import MetaQuest3

# Windows ZMQ asyncio stability
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def find_first_existing(model, names, objtype):
    for n in names:
        try:
            _id = mujoco.mj_name2id(model, objtype, n)
            if _id != -1:
                return n, _id
        except Exception:
            pass
    return None, -1

# --- add near the top of franka_quest_teleop.py ---
import zmq

def _get_pub_socket(node_mgr):
    """
    Try common attribute names used by SimPub node managers.
    """
    candidates = [
        "pub_socket", "_pub_socket", "_pubSocket",
        "topic_socket", "_topic_socket", "_topicSocket",
        "_pub", "pub",
    ]
    for name in candidates:
        sock = getattr(node_mgr, name, None)
        if sock is not None:
            return sock
    raise RuntimeError(
        "Could not find a PUB socket on node manager. "
        "Print dir(node_mgr) and look for a zmq/NetMQ Publisher socket."
    )

def publish_topic(pub_sock, topic: str, payload: bytes):
    # Unity side expects [topic][payload] multipart
    pub_sock.send_multipart([topic.encode("utf-8"), payload])

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1,0,0,0], dtype=np.float64)

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(a, b):
    # wxyz convention for mujoco
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=np.float64)

def quat_to_rotvec(q):
    # small-angle rotvec from quaternion (wxyz)
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w*w))
    axis = q[1:] / s
    if angle < 1e-6:
        return np.zeros(3)
    return axis * angle

def unity_to_mj_pos(p_unity):
    # MetaQuest3 class typically gives Unity coords (x,y,z).
    # Your pipeline earlier uses mj2unity_pos: [-y, z, x]
    # Invert it: unity(x,y,z) -> mj(z, -x, y)
    x,y,z = p_unity
    return np.array([z, -x, y], dtype=np.float64)

def unity_to_mj_quat(q_unity_xyzw):
    # Your mj2unity_quat: [quat[2], -quat[3], -quat[1], quat[0]]  (xyzw in Unity)
    # Invert mapping to get mujoco wxyz.
    x,y,z,w = q_unity_xyzw
    # Solve:
    # unity.x = mj.z
    # unity.y = -mj.w? (careful: mj2unity used [z, -w, -y, x] when mj is wxyz)
    # From your code: return [quat[2], -quat[3], -quat[1], quat[0]]
    # Let mj = [w, x, y, z]
    # unity = [z, -? actually unity_x = mj_y? wait:
    # unity_x = mj[2]? no -> unity_x = quat[2] => mj.y
    # unity_y = -quat[3] => -mj.z
    # unity_z = -quat[1] => -mj.x
    # unity_w = quat[0] => mj.w
    # So:
    # mj.w = unity_w
    # mj.x = -unity_z
    # mj.y = unity_x
    # mj.z = -unity_y
    return quat_normalize(np.array([w, -z, x, -y], dtype=np.float64))

def mj_to_unity_pos(p_mj: np.ndarray) -> np.ndarray:
    x, y, z = p_mj
    return np.array([-y, z, x], dtype=np.float32)

def damped_ls(J, e, lam=0.05):
    # solve dq = J^T (J J^T + lam^2 I)^-1 e
    JJt = J @ J.T
    A = JJt + (lam*lam) * np.eye(JJt.shape[0])
    return J.T @ np.linalg.solve(A, e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to MJCF with Franka + lab")
    ap.add_argument("--host", required=True, help="PC LAN IP, e.g. 192.168.0.208")
    ap.add_argument("--unity_node", required=True, help="Quest Unity node name from dashboard, e.g. MQ3-2")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Publish visuals only
    publisher = MujocoPublisher(
        model,
        data,
        host=args.host,
        visible_geoms_groups=[2],
        preferred_xr_name=args.unity_node,
    )

    # XR input
    #net_manager = init_xr_node_manager(args.host)
   # net_manager.start_discover_node_loop()
    mq3 = MetaQuest3(args.unity_node)

    # Find EE site/body
    site_name, site_id = find_first_existing(
        model,
        ["ee_site", "grasp_site", "panda_hand_site", "tcp", "tool0"],
        mujoco.mjtObj.mjOBJ_SITE
    )
    if site_id == -1:
        # fallback to common body names
        body_name, body_id = find_first_existing(
            model,
            ["panda_hand", "hand", "eef", "panda_link8"],
            mujoco.mjtObj.mjOBJ_BODY
        )
        if body_id == -1:
            raise RuntimeError("Could not find EE site/body. Add a site named ee_site to your robot MJCF.")
        use_site = False
        ee_id = body_id
    else:
        use_site = True
        ee_id = site_id

    # Initial target = current EE
    mujoco.mj_step(model, data)
    if use_site:
        cur_pos = data.site_xpos[ee_id].copy()
        cur_quat = data.site_xquat[ee_id].copy()  # wxyz
    else:
        cur_pos = data.xpos[ee_id].copy()
        cur_quat = data.xquat[ee_id].copy()
    target_pos = cur_pos.copy()
    target_quat = cur_quat.copy()

    clutch_prev = False
    pos_offset = np.zeros(3)
    quat_offset = np.array([1,0,0,0], dtype=np.float64)

    dt = 1.0 / args.fps

    while True:
        # --- XR input ---
        inp = mq3.get_controller_data()
        """
        if inp is None:
            print("No controller data yet...")
        else:
            print("Controller keys:", inp.keys())
            if "right" in inp:
                print("Right keys:", inp["right"].keys())
        """
        if inp and "right" in inp:
            right = inp["right"]
            # these keys match your pick_and_place example: pos/rot/index_trigger
            u_pos = np.array(right["pos"], dtype=np.float64)
            u_rot = np.array(right["rot"], dtype=np.float64)  # xyzw

            clutch = (right.get("hand_trigger", 0.0) > 0.6) or (right.get("grip", 0.0) > 0.6) or bool(right.get("thumbstick", False))
            close = right.get("index_trigger", 0.0) > 0.6


            mj_pos = unity_to_mj_pos(u_pos)
            mj_quat = unity_to_mj_quat(u_rot)

            # clutch logic: when clutch first pressed, lock relative offset
            if clutch and not clutch_prev:
                # recompute current EE
                mujoco.mj_forward(model, data)
                if use_site:
                    ee_pos = data.site_xpos[ee_id].copy()
                    ee_quat = data.site_xquat[ee_id].copy()
                else:
                    ee_pos = data.xpos[ee_id].copy()
                    ee_quat = data.xquat[ee_id].copy()

                pos_offset = ee_pos - mj_pos
                quat_offset = quat_mul(ee_quat, quat_conj(mj_quat))

            clutch_prev = clutch

            if clutch:
                target_pos = mj_pos + pos_offset
                target_quat = quat_mul(quat_offset, mj_quat)

            # gripper
            data.ctrl[7] = 0.0 if close else 0.04  # actuator8 finger_joint1

        # --- IK step ---
        mujoco.mj_forward(model, data)

        if use_site:
            ee_pos = data.site_xpos[ee_id].copy()
            ee_quat = data.site_xquat[ee_id].copy()
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, Jp, Jr, ee_id)
        else:
            ee_pos = data.xpos[ee_id].copy()
            ee_quat = data.xquat[ee_id].copy()
            Jp = np.zeros((3, model.nv))
            Jr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, Jp, Jr, ee_id)

        pos_err = (target_pos - ee_pos)
        q_err = quat_mul(target_quat, quat_conj(ee_quat))
        rot_err = quat_to_rotvec(q_err)

        e = np.hstack([pos_err, rot_err])  # 6x1

        J = np.vstack([Jp, Jr])  # 6 x nv

        dq = damped_ls(J, e, lam=0.08)

        # only apply to first 7 joints (Franka arm)
        q = data.qpos.copy()
        q[0:7] = q[0:7] + 0.25 * dq[0:7]  # step size

        # send joint targets as ctrl for your position-like general actuators
        data.ctrl[0:7] = q[0:7]

        # step sim
        mujoco.mj_step(model, data)
        time.sleep(dt)

if __name__ == "__main__":
    main()

# python .\franka_quest_teleop.py --xml ..\LAB\lab1_T_stack_vizCollisionSeparated_v3_groups_named_v4_noFloorVisual.xml --host 192.168.0.208 --unity_node MQ3-2
#python .\SimPublisher\sii\franka_quest_teleop.py --xml .\LAB\lab1.xml --host 192.168.0.208 --unity_node MQ3-2
#python .\SimPublisher\sii\franka_quest_teleop.py --xml .\LAB\lab1_T_stack_vizCollisionSeparated_v3_groups_named.xml --host 192.168.0.208 --unity_node MQ3-2
#C:\Users\user\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\LAB\lab1.xml
#C:\Users\user\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\LAB\lab2_boxes_cups_vizCollisionSeparated_v3_groups_named.xml
#LAB\lab3_stick_maze_vizCollisionSeparated_v3_groups_named.xml



#python .\SimPublisher\sii\franka_quest_teleop.py --xml .\LAB\lab2_boxes_cups.xml --host 192.168.0.208 --unity_node MQ3-2
#python .\SimPublisher\sii\franka_quest_teleop.py --xml .\LAB\lab3_stick_maze_vizCollisionSeparated_v3_groups_named.xml --host 192.168.0.208 --unity_node MQ3-2
#python .\SimPublisher\sii\franka_quest_teleop.py --xml .\LAB\lab1_T_stack.xml --host 192.168.0.208 --unity_node MQ3-2

# .\LAB\lab2_boxes_cups.xml
# .\LAB\lab3_stick_maze_vizCollisionSeparated_v3_groups_named.xml 
# .\LAB\lab1_T_stack.xml \
    # 
    # # .\LAB\lab2.xml
  
#  & c:\Users\user\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\.venv\Scripts\python.exe .\SimPublisher\sii\simpub_sii_sensors-pz --xml .\LAB\lab1_T_stack.xml --bind_ip 0.0.0.0 --host_ip 192.168.0.208 --fps 30 --w 640 --h 480 --cams top --pc --pc_cams top

#  & c:\Users\user\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\.venv\Scripts\python.exe .\SimPublisher\sii\simpub_sii_sensors-pz --xml .\LAB\lab1_T_stack.xml --bind_ip 0.0.0.0 --host_ip 192.168.0.208 --fps 30 --w 640 --h 480 --cams top front left right wrist --pc --pc_cams top front left right wrist
