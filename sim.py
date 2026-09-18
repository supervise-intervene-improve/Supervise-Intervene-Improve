import mujoco
import mujoco.viewer
import numpy as np
import zmq
import json
import time

# -------- ZMQ RECEIVE ROBOT JOINTS (q, dq) --------
context = zmq.Context()
pull_socket = context.socket(zmq.PULL)
pull_socket.bind("tcp://*:5001")   # robot → sim

# -------- ZMQ SEND COMMANDS TO ROBOT --------
push_socket = context.socket(zmq.PUSH)
push_socket.bind("tcp://*:5002")   # sim → robot

# -------- Load MuJoCo Scene --------
MODEL_PATH = "franka_emika_panda/mjx_panda.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# ----------- Run Simulation -----------
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Simulation running...")

    while viewer.is_running():
        try:
            msg = pull_socket.recv(flags=zmq.NOBLOCK)
            r = json.loads(msg.decode())
            data.qpos[:] = r["q"]
            data.qvel[:] = r["dq"]
        except zmq.Again:
            pass

        # small safe motion
        q_des = (data.qpos + 0.02*np.sin(time.time())).tolist()

        # send to robot
        cmd = {"q_des": q_des}
        push_socket.send(json.dumps(cmd).encode())

        mujoco.mj_step(model, data)
        viewer.sync()