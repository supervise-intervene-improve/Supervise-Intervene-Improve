import zmq, struct, json
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect("tcp://192.168.0.208:7741")
s.setsockopt(zmq.SUBSCRIBE, b"SimPub/Sensors")

topic = s.recv_string()
blob  = s.recv()
print("TOPIC:", topic, "BLOB:", len(blob))

json_len = struct.unpack("<I", blob[:4])[0]
hdr = json.loads(blob[4:4+json_len])
print("HDR:", hdr)
print("jpg magic:", blob[4+json_len:4+json_len+2])

# python .\SimPublisher\sii\tiny_sub.py