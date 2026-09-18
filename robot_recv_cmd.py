import zmq
import json

def set_robot_command(q_des):

    print("Robot command:", q_des)

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://YOUR_PC_IP:5002")     

print("Receiving commands from simulation...")

while True:
    msg = socket.recv()
    data = json.loads(msg.decode())
    set_robot_command(data["q_des"])

