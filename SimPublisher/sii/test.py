# demos/xr/get_input_data.py
from simpub.core.node_manager import init_xr_node_manager
from simpub.xr_device.meta_quest3 import MetaQuest3
import time

PC_IP = "192.168.0.208"
UNITY_NODE_NAME = "MQ3-2"  # <-- must match dashboard

net_manager = init_xr_node_manager(PC_IP)
net_manager.start_discover_node_loop()

mq3 = MetaQuest3(UNITY_NODE_NAME)

while True:
    data = mq3.get_controller_data()
    if data:
        print(data["right"].keys())
        print("right pos:", data["right"]["pos"], "rot:", data["right"]["rot"])
        print("trigger:", data["right"].get("index_trigger"), "grip:", data["right"].get("hand_trigger"))
    time.sleep(0.05)
