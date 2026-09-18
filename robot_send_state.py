import zmq
import json
import time

def get_robot_q():
    return [0,0,0,0,0,0,0]

def get_robot_dq():
    return [0,0,0,0,0,0,0]

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://YOUR_PC_IP:5001")    

print("Sending robot state...")

while True:
    q  = get_robot_q()
    dq = get_robot_dq()

    msg = {"q": q, "dq": dq}
    socket.send(json.dumps(msg).encode())

    time.sleep(0.01)
