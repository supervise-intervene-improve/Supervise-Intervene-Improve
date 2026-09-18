# import zmq
# import threading
# import multiprocessing
# import time


# ZMQ_PORT = 35555
# _context = zmq.Context()
# _socket = _context.socket(zmq.PUSH)
# _socket.setsockopt(zmq.SNDHWM, 1)
# _socket.bind(f"tcp://*:{ZMQ_PORT}")

# _msg_queue = multiprocessing.Queue()


# def start_sender_thread():

#     # print("in start_sender_thread")
#     def sender():
#         frame_count = 0
#         last_fps_time = time.time()
#         while True:
#             try:
#                 #print("get from queue")
#                 msg = _msg_queue.get()
#                 # print("sending")
#                 _socket.send(msg, zmq.NOBLOCK)
#             except Exception as e:
#                 pass
            
#             # ---- FPS Tracking ----
#             frame_count += 1
#             now = time.time()
#             if now - last_fps_time >= 1.0:  # jede Sekunde ausgeben
#                 print(f"[FPS sending] {frame_count} fps")
#                 frame_count = 0  # reset counters
#                 last_fps_time = now


#     thread = threading.Thread(target=sender, daemon=True)
#     thread.start()

# def stop_sender_thread():
#     # print("in stop_sender_thread")
#     _socket.close()
#     _context.term()

# def send(msg: bytes):
#     # print("putting msg in queue")
#     _msg_queue.put(msg)
import zmq
import multiprocessing
import time

ZMQ_PORT = 35555
ADDRESS = "10.42.0.27"  # Server-Adresse (localhost, wenn auf demselben Rechner)
_msg_queue = multiprocessing.Queue()

# --- Sender-Prozess ---
def sender_process(msg_queue: multiprocessing.Queue):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{ADDRESS}:{ZMQ_PORT}")
    socket.setsockopt(zmq.SNDHWM, 1)

    frame_count = 0
    last_fps_time = time.time()

    while True:
        try:
            msg = msg_queue.get()   # blockierend
            socket.send(msg, zmq.NOBLOCK)
            # print("Message sent")
        except Exception:
            pass

        # FPS Tracking
        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            print(f"[FPS sending] {frame_count} fps")
            frame_count = 0
            last_fps_time = now


# --- Prozesssteuerung ---
_sender_process = None

def start_sender_process():
    global _sender_process
    _sender_process = multiprocessing.Process(target=sender_process, args=(_msg_queue,), daemon=True)
    _sender_process.start()

def stop_sender_process():
    global _sender_process
    if _sender_process is not None:
        _sender_process.terminate()
        _sender_process.join()
        _sender_process = None

def send(msg: bytes):
    _msg_queue.put(msg)


# --- Beispielnutzung ---
if __name__ == "__main__":
    start_sender_process()

    for i in range(100):
        send(f"Message {i}".encode())
        time.sleep(0.01)

    time.sleep(2)
    stop_sender_process()
