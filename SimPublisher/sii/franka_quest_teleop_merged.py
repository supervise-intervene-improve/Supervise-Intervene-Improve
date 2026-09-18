import argparse
import json
import math
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import zmq

import mujoco
import cv2

# --- Your existing imports (from franka_quest_teleop.py) ---
from simpub.xr_device.meta_quest3 import MetaQuest3
from simpub.sim.mj_publisher import MujocoPublisher


# ----------------------------
# Utilities: topic framing
# ----------------------------
def pack_topic_message(topic: str, payload: bytes) -> bytes:
    # IRXR NetMQ side typically uses "topic|payload" in a single frame
    return topic.encode("utf-8") + b"|" + payload


# ----------------------------
# RGBD blob format
# ----------------------------
def build_rgbd_blob(
    cam_name: str,
    width: int,
    height: int,
    fovy_deg: float,
    timestamp: float,
    rgb_u8: np.ndarray,     # (H,W,3) uint8 RGB
    depth_f32: np.ndarray,  # (H,W) float32 meters
    pc_f32: Optional[np.ndarray] = None,  # (N,7) float32
) -> bytes:
    """
    Blob layout:
        [u32 json_len little-endian]
        [json bytes]
        [rgb jpeg bytes]
        [depth float32 bytes: width*height*4]
        [pc float32 bytes: header contains pc_byte_len; optional]
    """
    assert rgb_u8.dtype == np.uint8 and rgb_u8.shape == (height, width, 3)
    assert depth_f32.dtype == np.float32 and depth_f32.shape == (height, width)

    # Encode RGB as JPEG (Unity-side is typically fine decoding jpg)
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    rgb_bytes = jpg.tobytes()

    depth_bytes = depth_f32.tobytes(order="C")

    pc_bytes = b""
    if pc_f32 is not None:
        pc_bytes = pc_f32.astype(np.float32, copy=False).tobytes(order="C")

    header = {
        "cam_name": cam_name,
        "width": int(width),
        "height": int(height),
        "fovy_deg": float(fovy_deg),
        "timestamp": float(timestamp),
        "rgb_format": "jpg",
        "rgb_byte_len": len(rgb_bytes),
        "depth_format": "f32",
        "depth_byte_len": len(depth_bytes),
        "has_pc": pc_f32 is not None,
        "pc_format": "f32x7",
        "pc_byte_len": len(pc_bytes),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<I", len(header_bytes))

    return prefix + header_bytes + rgb_bytes + depth_bytes + pc_bytes


# ----------------------------
# Point cloud creation
# ----------------------------
def depth_to_pointcloud_cam(
    depth: np.ndarray,        # (H,W) float32 meters
    rgb: np.ndarray,          # (H,W,3) uint8 RGB
    fovy_deg: float,
    stride: int,
    point_size: float = 0.01
) -> np.ndarray:
    """
    Returns (N,7) float32: x,y,z,r,g,b,size in CAMERA frame.
    """
    h, w = depth.shape
    fovy = math.radians(fovy_deg)
    fy = 0.5 * h / math.tan(0.5 * fovy)
    fx = fy  # assume square pixels
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    us = np.arange(0, w, stride, dtype=np.float32)
    vs = np.arange(0, h, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)

    z = depth[::stride, ::stride].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.01) & (z < 20.0)

    uu = uu[valid]
    vv = vv[valid]
    z = z[valid]

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    cols = rgb[::stride, ::stride, :].reshape(-1, 3)[valid.reshape(-1)]
    cols_f = (cols.astype(np.float32) / 255.0)

    size = np.full((z.shape[0], 1), point_size, dtype=np.float32)
    pc = np.concatenate(
        [
            x.reshape(-1, 1),
            y.reshape(-1, 1),
            z.reshape(-1, 1),
            cols_f,
            size
        ],
        axis=1
    ).astype(np.float32, copy=False)
    return pc


def transform_points(
    pc: np.ndarray,            # (N,7) float32, x,y,z,r,g,b,size
    R: np.ndarray,             # (3,3)
    t: np.ndarray,             # (3,)
) -> np.ndarray:
    pts = pc[:, 0:3]
    pts_w = (pts @ R.T) + t.reshape(1, 3)
    out = pc.copy()
    out[:, 0:3] = pts_w
    return out


def mujoco_to_unity_axes(pc: np.ndarray) -> np.ndarray:
    """
    Common mapping mentioned in our plan: (-y, z, x)
    Adjust here if your overlay is rotated/offset.
    """
    out = pc.copy()
    x = pc[:, 0].copy()
    y = pc[:, 1].copy()
    z = pc[:, 2].copy()
    out[:, 0] = -y
    out[:, 1] = z
    out[:, 2] = x
    return out


# ----------------------------
# XR node: discovery + GetNodeInfo + PUB
# ----------------------------
@dataclass
class XRNodeConfig:
    name: str
    bind_ip: str
    host_ip: str
    service_port: int
    topic_port: int
    mcast_group: str = "239.255.10.10"
    discovery_port: int = 7720


class XRNode:
    def __init__(self, cfg: XRNodeConfig, topic_list: List[str]):
        self.cfg = cfg
        self.node_id = str(uuid.uuid4())
        self.nodeInfoID = str(uuid.uuid4())
        self.topic_list = topic_list

        self._stop = threading.Event()

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://{cfg.bind_ip}:{cfg.topic_port}")

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.bind(f"tcp://{cfg.bind_ip}:{cfg.service_port}")

        self._t_disc = threading.Thread(target=self._discovery_loop, daemon=True)
        self._t_rep = threading.Thread(target=self._rep_loop, daemon=True)

    def start(self):
        self._t_disc.start()
        self._t_rep.start()

    def stop(self):
        self._stop.set()
        try:
            self._rep.close(0)
            self._pub.close(0)
        except Exception:
            pass

    def publish(self, topic: str, payload: bytes):
        self._pub.send(pack_topic_message(topic, payload))

    def _node_info_dict(self) -> Dict:
        return {
            "name": self.cfg.name,
            "type": "SimPub",
            "nodeInfoID": self.nodeInfoID,
            "servicePort": int(self.cfg.service_port),
            "topicPort": int(self.cfg.topic_port),
            "serviceList": [],
            "topicList": list(self.topic_list),
            "ip": self.cfg.host_ip,
        }

    def _discovery_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # On Windows, forcing NIC can fail; best-effort:
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.cfg.host_ip))
        except OSError:
            pass

        raw = (self.node_id + self.nodeInfoID + f"{self.cfg.service_port:04d}").encode("utf-8")

        while not self._stop.is_set():
            try:
                sock.sendto(raw, (self.cfg.mcast_group, self.cfg.discovery_port))
            except Exception:
                pass
            time.sleep(0.5)

    def _rep_loop(self):
        while not self._stop.is_set():
            try:
                msg = self._rep.recv()
            except Exception:
                break

            # Accept both "GetNodeInfo" and "GetNodeInfo|" forms
            if b"GetNodeInfo" in msg:
                payload = json.dumps(self._node_info_dict()).encode("utf-8")
                # IMPORTANT: reply must be pure JSON bytes (no "OK", no extra frames)
                try:
                    self._rep.send(payload)
                except Exception:
                    break
            else:
                try:
                    self._rep.send(b"{}")
                except Exception:
                    break


# ----------------------------
# MuJoCo rendering for cameras
# ----------------------------
class MujocoRgbdRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int):
        self.model = model
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, width, height)

    def render_rgbd(self, data: mujoco.MjData, cam_name: str) -> Tuple[np.ndarray, np.ndarray, float, int]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model")

        self.renderer.update_scene(data, camera=cam_name)
        rgb = self.renderer.render()  # RGB uint8 (H,W,3)
        depth = self.renderer.render(depth=True)  # float32 depth

        # Camera fovy from model.cam_fovy[cam_id]
        fovy_deg = float(self.model.cam_fovy[cam_id])

        return rgb, depth.astype(np.float32, copy=False), fovy_deg, cam_id


# ----------------------------
# Main: teleop + sensors in one process
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--host", required=True, help="PC host IP (same subnet as Quest), e.g. 192.168.0.208")
    ap.add_argument("--unity_node", default="MQ3-2")
    ap.add_argument("--bind_ip", default="0.0.0.0")

    ap.add_argument("--service_port", type=int, default=7740)
    ap.add_argument("--topic_port", type=int, default=7741)

    ap.add_argument("--sensors", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--cams", default="overhead,top,front,right,left")

    ap.add_argument("--pc", action="store_true")
    ap.add_argument("--pc_stride", type=int, default=4)
    ap.add_argument("--pc_unity_axes", action="store_true", help="Apply (-y,z,x) to pointcloud before publishing")

    args = ap.parse_args()

    # --- MuJoCo setup (from your teleop) ---
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    mj_pub = MujocoPublisher(
        mj_model=model,
        mj_data=data,
        host=args.host,
        visible_geoms_groups=[2],
        preferred_xr_name=args.unity_node,
    )
    try:
        mj_pub.start()
    except Exception:
        # if mj_pub.start() is not required in your version, ignore
        pass

    # --- VR controller client ---
    mq3 = MetaQuest3(args.unity_node)

    # --- Sensor node (discovery + GetNodeInfo + PUB topics) ---
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    sensor_topics = []
    if args.sensors:
        for c in cams:
            sensor_topics.append(f"SimPub/Sensors/{c}/rgbd")
        if args.pc:
            sensor_topics.append("PointCloud")
            sensor_topics.append("SimPub/PointCloud")

    node = None
    renderer = None
    if args.sensors:
        cfg = XRNodeConfig(
            name="SimPub",
            bind_ip=args.bind_ip,
            host_ip=args.host,
            service_port=args.service_port,
            topic_port=args.topic_port,
        )
        node = XRNode(cfg, topic_list=sensor_topics)
        node.start()
        renderer = MujocoRgbdRenderer(model, args.w, args.h)

        print(f"[Merged] Sensor node up: REP tcp://{args.bind_ip}:{args.service_port} PUB tcp://{args.bind_ip}:{args.topic_port}")
        print(f"[Merged] Topics: {sensor_topics}")

    # --- Timing (teleop loop) ---
    # Keep your previous dt logic simple and stable:
    sim_hz = 200.0
    dt = 1.0 / sim_hz

    next_sensor_t = time.time()
    sensor_period = 1.0 / max(args.fps, 1e-3)

    try:
        while True:
            t0 = time.time()

            # Step sim
            mujoco.mj_step(model, data)

            # Controller → robot (keep exactly as in your teleop file)
            try:
                lhand, rhand, lbtn, rbtn = mq3.get_controller_data()
                # TODO: apply to robot controls here if you already have it wired in your local merged file
                # This merged template focuses on sensors publishing + sim stepping correctness.
            except Exception:
                pass

            # Publish sensors
            if args.sensors and node is not None and renderer is not None:
                now = time.time()
                if now >= next_sensor_t:
                    timestamp = time.time()

                    for cam_name in cams:
                        rgb, depth, fovy_deg, cam_id = renderer.render_rgbd(data, cam_name)

                        pc = None
                        if args.pc:
                            pc = depth_to_pointcloud_cam(depth, rgb, fovy_deg, stride=max(1, args.pc_stride))

                            # camera->world using MuJoCo cam pose
                            # data.cam_xmat: 9 values row-major
                            R = data.cam_xmat[cam_id].reshape(3, 3)
                            t = data.cam_xpos[cam_id].reshape(3,)
                            pc = transform_points(pc, R=R, t=t)

                            if args.pc_unity_axes:
                                pc = mujoco_to_unity_axes(pc)

                            # Publish pointcloud on both names (so whichever exists on Quest side will hit)
                            pc_bytes = pc.astype(np.float32, copy=False).tobytes(order="C")
                            node.publish("PointCloud", pc_bytes)
                            node.publish("SimPub/PointCloud", pc_bytes)

                        blob = build_rgbd_blob(
                            cam_name=cam_name,
                            width=args.w,
                            height=args.h,
                            fovy_deg=fovy_deg,
                            timestamp=timestamp,
                            rgb_u8=rgb,
                            depth_f32=depth,
                            pc_f32=pc,
                        )
                        node.publish(f"SimPub/Sensors/{cam_name}/rgbd", blob)

                    next_sensor_t = now + sensor_period

            # Keep real-time-ish
            elapsed = time.time() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
        try:
            mj_pub.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
