import argparse
import json
import socket
import time
import uuid
from dataclasses import dataclass, asdict

import zmq
import numpy as np
import cv2

# ----------------------------
# MUST MATCH Unity MsgUtils.SEPARATOR
# Open MsgUtils.cs and copy the separator exactly.
# Common examples are "|" or "||" or "<SEP>" etc.
SEPARATOR = "|"
# ----------------------------


# ---------- IRXR NodeInfo schema (must match Unity NodeInfo/NodeAddress) ----------
@dataclass
class NodeAddress:
    ip: str
    port: int = 0

@dataclass
class NodeInfo:
    name: str
    nodeID: str
    addr: NodeAddress
    type: str
    servicePort: int
    topicPort: int
    serviceList: list
    topicList: list


def combine_header_with_message(topic: str, payload_bytes: bytes) -> bytes:
    """
    Mirrors Unity MsgUtils.CombineHeaderWithMessage(topic, bytes).
    Unity does: [topic + SEPARATOR + payload] then SplitByte() on receiver.
    """
    head = (topic + SEPARATOR).encode("utf-8")
    return head + payload_bytes


def get_local_ip_for_target(target_ip: str) -> str:
    """
    Mimics NetworkUtils.GetLocalIPsInSameSubnet behavior in a practical way.
    If target is localhost -> localhost.
    Otherwise let OS choose route and read the chosen interface IP.
    """
    if target_ip == "127.0.0.1":
        return "127.0.0.1"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_ip, 1))
        return s.getsockname()[0]
    finally:
        s.close()


def udp_broadcast_discovery(nodeinfo: NodeInfo, discovery_port: int, interval_sec: float = 0.2):
    """
    Unity IRXRNetManager listens on UDP DISCOVERY (7720) and expects messages starting with "SimPub".
    In your C#:
        if (!message.StartsWith("SimPub")) continue;
        split = message.Split(MsgUtils.SEPARATOR, 2);
        NodeInfo info = Deserialize(split[1])
    So we send: "SimPub" + SEPARATOR + json(nodeinfo)
    """
    msg = "SimPub" + SEPARATOR + json.dumps(asdict(nodeinfo))
    data = msg.encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return sock, data, interval_sec, discovery_port


def encode_rgb_jpeg(rgb_uint8: np.ndarray, quality: int = 80) -> bytes:
    # rgb_uint8 expected RGB, convert to BGR for OpenCV jpeg
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def depth_to_pointcloud(depth_m: np.ndarray, fovy_deg: float, stride: int = 4) -> np.ndarray:
    """
    depth_m: float32 [H,W] meters
    returns float32 [N,3] in camera coords
    """
    H, W = depth_m.shape
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xx, yy = np.meshgrid(xs, ys)
    z = depth_m[yy, xx]

    valid = np.isfinite(z) & (z > 0)
    xx = xx[valid].astype(np.float32)
    yy = yy[valid].astype(np.float32)
    z  = z[valid].astype(np.float32)

    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy

    return np.stack([x, y, z], axis=1).astype(np.float32)


# ----------------------------
# YOU plug your MuJoCo renderer here:
# Must return per-camera: (rgb_uint8_HWC, depth_float32_HW, fovy_deg)
# ----------------------------
def render_all_cameras_stub(cams, width, height):
    """
    Replace this with your working mjr_readPixels + GLFWContext route.
    For now it produces dummy frames so IRXR pipeline can be validated.
    """
    out = {}
    for cam in cams:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(rgb, cam, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth = np.ones((height, width), dtype=np.float32) * 1.0  # 1 meter everywhere
        fovy = 45.0
        out[cam] = (rgb, depth, fovy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_ip", default="127.0.0.1", help="IP Unity should connect to (master node IP)")
    ap.add_argument("--discovery_port", type=int, default=7720)
    ap.add_argument("--service_port", type=int, default=7730)
    ap.add_argument("--topic_port", type=int, default=7731)

    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--pc", action="store_true")
    ap.add_argument("--pc_stride", type=int, default=4)

    ap.add_argument("--cams", nargs="*", default=["front", "right", "left", "top", "overhead"])
    ap.add_argument("--topic_prefix", default="SimPub/Sensors", help="Base topic prefix")
    args = ap.parse_args()

    # Build NodeInfo that Unity expects to discover
    node_id = str(uuid.uuid4())
    local_ip = get_local_ip_for_target(args.master_ip)

    nodeinfo = NodeInfo(
        name="SimPub",
        nodeID=node_id,
        addr=NodeAddress(ip=local_ip, port=0),
        type="SimPub",
        servicePort=args.service_port,
        topicPort=args.topic_port,
        serviceList=[
            "RegisterNode",   # we’ll answer minimal services
            "NodeOffline"
        ],
        topicList=[],
    )

    # ZMQ setup
    ctx = zmq.Context.instance()

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.topic_port}")

    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://*:{args.service_port}")

    # UDP discovery broadcaster
    disc_sock, disc_data, disc_dt, disc_port = udp_broadcast_discovery(nodeinfo, args.discovery_port, interval_sec=0.2)

    print(f"[IRXR SimPub] local_ip={local_ip}")
    print(f"[IRXR SimPub] UDP discovery : *:{args.discovery_port}")
    print(f"[IRXR SimPub] ZMQ REP(service): *:{args.service_port}")
    print(f"[IRXR SimPub] ZMQ PUB(topic)  : *:{args.topic_port}")
    print(f"[IRXR SimPub] cams={args.cams} fps={args.fps} pc={'ON' if args.pc else 'OFF'} stride={args.pc_stride}")
    print(f"[IRXR SimPub] IMPORTANT: SEPARATOR='{SEPARATOR}' must match Unity MsgUtils.SEPARATOR")

    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)

    dt = 1.0 / float(args.fps)
    t_last_disc = 0.0
    t_last_frame = 0.0

    # Predeclare topics (so Unity can see them if it uses topicList)
    # If your Unity relies on masterInfo.topicList, publish them in discovery too.
    # Many setups don’t require it because Subscribe("") is used.
    for cam in args.cams:
        nodeinfo.topicList.append(f"{args.topic_prefix}/{cam}/rgbd")

    while True:
        now = time.time()

        # 1) UDP discovery broadcast
        if now - t_last_disc >= disc_dt:
            try:
                disc_sock.sendto(disc_data, ("255.255.255.255", disc_port))
            except Exception:
                pass
            t_last_disc = now

        # 2) Service handling (RegisterNode / NodeOffline minimal)
        socks = dict(poller.poll(timeout=0))
        if rep in socks and socks[rep] == zmq.POLLIN:
            req = rep.recv()
            # Unity sends: $"{service}{SEPARATOR}{request}" (string)
            # Here we just always say SUCCESS to keep node alive.
            # You can expand later to track nodes, heartbeat, etc.
            rep.send(b"SUCCESS")

        # 3) Publish camera frames at FPS
        if now - t_last_frame >= dt:
            frames = render_all_cameras_stub(args.cams, args.w, args.h)  # REPLACE with your real renderer

            for cam, (rgb, depth, fovy_deg) in frames.items():
                rgb_jpg = encode_rgb_jpeg(rgb, quality=80)
                if not rgb_jpg:
                    continue

                depth = depth.astype(np.float32)
                depth_bytes = depth.tobytes()

                pc = np.zeros((0, 3), dtype=np.float32)
                if args.pc:
                    pc = depth_to_pointcloud(depth, fovy_deg, stride=args.pc_stride)

                header = {
                    "camera": cam,
                    "w": args.w,
                    "h": args.h,
                    "fovy_deg": float(fovy_deg),
                    "pc_points": int(pc.shape[0]),
                    "rgb_size": len(rgb_jpg),
                    "depth_size": len(depth_bytes),
                }

                payload = {
                    "header": header,
                    "rgb_jpeg": None,   # binary sent separately below
                    "depth_f32": None,  # binary sent separately below
                    "pc_f32": None,     # binary sent separately below
                }

                # Publish as ONE topic message containing:
                # JSON header + raw blocks (so Unity can decode deterministically)
                # We pack as: [json_len(uint32)][json][rgb][depth][pc(optional)]
                json_bytes = json.dumps(payload).encode("utf-8")
                json_len = np.uint32(len(json_bytes)).tobytes()

                blob = bytearray()
                blob += json_len
                blob += json_bytes
                blob += rgb_jpg
                blob += depth_bytes
                if args.pc:
                    blob += pc.tobytes()

                topic = f"{args.topic_prefix}/{cam}/rgbd"
                pub.send(combine_header_with_message(topic, bytes(blob)))

            t_last_frame = now

        time.sleep(0.001)


if __name__ == "__main__":
    main()
