from collections import namedtuple
import pickle
import zmq

DEFAULT_ROBOT_PORT = 6000

class AbstractGello():

    def __init__(self, name: str, host: str = "127.0.0.1", port: int = DEFAULT_ROBOT_PORT):
        self.name = name
        self.host = host
        self.port = port
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

    def connect(self) -> bool:
        print("Connecting to {}: ".format(self.name))
        try:
            self._socket.connect(f"tcp://{self.host}:{self.port}")
            return True
        except Exception as e:
            print("Failed with exception: ", e)
            return False

    def close(self):
        self._socket.disconnect(f"tcp://{self.host}:{self.port}")
        return True

    def okay(self):
        return True
    
    def apply_commands(self):
        # At the moment Gello can't be controled
        return 0

    def reset(self):
        # At the moment Gello can't be controled
        return 0

class GelloArm(AbstractGello):

    def get_state(self):
            if self._socket.closed:
                raise Exception(f"Not connected to {self.name}")
            
            request = {"method": "get_joint_state"}
            send_message = pickle.dumps(request)
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())

            return result

class GelloGripper(AbstractGello):

    def get_sensors(self):
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        request = {"method": "get_gripper_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())

        return result
