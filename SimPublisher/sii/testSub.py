import zmq, struct, json

TOPIC_PORT = 7741   # whatever your sensor topicPort is
HOST = "127.0.0.1"  # or your LAN IP if testing from another machine

ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.connect(f"tcp://{HOST}:{TOPIC_PORT}")
sub.setsockopt(zmq.SUBSCRIBE, b"SimPub/Sensors/")  # subscribe to all sensors topics

while True:
    msg = sub.recv()
    topic, blob = msg.split(b"|", 1)
    topic = topic.decode("utf-8")

    (json_len,) = struct.unpack("<I", blob[:4])
    json_bytes = blob[4:4+json_len]
    meta = json.loads(json_bytes.decode("utf-8"))
    hdr = meta["header"]

    print("TOPIC:", topic)
    print("HDR:", hdr)
    print("blob bytes:", len(blob))
    print("-"*60)
#C:\Users\user\Desktop\Franka-Robot-Real-to-Sim-Teleoperation-VR-Pointcloud-Simulation\SimPublisher\sii\testSub.py

# # python .\SimPublisher\sii\testSub.py