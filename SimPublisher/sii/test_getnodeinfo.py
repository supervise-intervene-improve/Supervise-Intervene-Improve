import zmq
import time

ctx = zmq.Context.instance()
req = ctx.socket(zmq.REQ)
req.RCVTIMEO = 1000
req.SNDTIMEO = 1000
req.connect("tcp://192.168.0.208:7740")

try:
	req.send(b"GetNodeInfo")
	print(req.recv().decode("utf-8"))
except zmq.error.Again:
	print("Error: Connection timeout. Ensure SimPublisher is running on 192.168.0.208:7740")
# To run this test, ensure that the SimPublisher is running and listening on the specified IP and port.
# python .\SimPublisher\sii\test_getnodeinfo.py
#python .\SimPublisher\sii\testgetnodeinfo.py