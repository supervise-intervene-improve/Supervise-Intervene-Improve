import threading


class CommandState:
    def __init__(self):
        self.lock = threading.Lock()
        self.start_requested = False
        self.stop_requested = False
        self.discard_requested = False
        self.quit_requested = False

    def request_start(self):
        with self.lock:
            self.start_requested = True

    def request_stop(self):
        with self.lock:
            self.stop_requested = True

    def request_discard(self):
        with self.lock:
            self.discard_requested = True

    def request_quit(self):
        with self.lock:
            self.quit_requested = True

    def consume_start(self):
        with self.lock:
            v = self.start_requested
            self.start_requested = False
            return v

    def consume_stop(self):
        with self.lock:
            v = self.stop_requested
            self.stop_requested = False
            return v

    def consume_discard(self):
        with self.lock:
            v = self.discard_requested
            self.discard_requested = False
            return v

    def is_quit_requested(self):
        with self.lock:
            return self.quit_requested