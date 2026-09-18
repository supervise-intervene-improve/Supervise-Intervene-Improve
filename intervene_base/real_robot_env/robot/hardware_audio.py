# import pyaudio
import wave
import numpy as np

from multiprocessing import Process, Event, Queue
import array #traceback
import time

from real_robot_env.robot.hardware_devices import ContinuousDevice

# class AudioInput():
#     def __init__(self, frames_per_buffer: int = 0, channels: int = 2, rate: int = 44100, audio_format: int = pyaudio.paInt16): #65536
#         self.frames_per_buffer=frames_per_buffer
#         self.channels=channels
#         self.rate=rate
#         self.audio_format=audio_format
    
#     def __str__(self) -> str:
#         return f"[AudioInput with frames_per_buffer={self.frames_per_buffer}, channels={self.channels}, rate={self.rate}, audio_format={self.audio_format}]"

# class AudioInterface(ContinuousDevice):
#     """
#     This class can be considered a wrapper class for recording audio through audio interfaces.
#     To get the audio interfaces, it is good practice to use `AudioInterface.get_specific_devices(iface_name)` instead of `AudioInterface.get_devices()`.

#     This class inherits its functions from `real_robot_env.robot.hardware_devices.ContinuousDevice`.
#     """

#     def __init__(self, device_id: str, name = None, audio_input = AudioInput()) -> None:
#         super().__init__(device_id, name if name else f"AudioInterface_{device_id}")
#         self.format = '.wav'
#         self.audio_input = audio_input

#         self.device = None
#         self.timestamp = None
    
#     def _setup_connect(self):
#         self.device = pyaudio.PyAudio()

#     def start_recording(self) -> bool:
#         if self.device is None: return False
#         self.frames = []

#         self.stream = self.device.open(
#             format=self.audio_input.audio_format,
#             channels=self.audio_input.channels,
#             rate=self.audio_input.rate,
#             frames_per_buffer=self.audio_input.frames_per_buffer,
#             input=True,
#             input_device_index=int(self.device_id),
#             stream_callback=self.__frame_callback
#         )

#         self.stream.start_stream()
#         return True

#     def stop_recording(self) -> bool:
#         self.stream.stop_stream()
#         self.stream.close()
#         return True

#     def store_recording(self, directory, filename = None, _ = None) -> bool:
#         filename = filename if filename else f"{self.name}_recording"
#         print(f"Attempting to store {len(self.frames)} frames for {self.name}")
#         wf = wave.open(str(directory / filename) + self.format, 'wb')
#         wf.setnchannels(self.audio_input.channels)
#         wf.setsampwidth(pyaudio.get_sample_size(self.audio_input.audio_format))
#         wf.setframerate(self.audio_input.rate)
#         wf.writeframes(b''.join(self.frames))
#         wf.close()
#         print(f"Frames stored for {self.name}")
#         return True

#     def delete_recording(self) -> bool:
#         self.frames = []
#         return True

#     def close(self) -> bool:
#         self.device.terminate()
#         self.device = None
#         return True

#     def __frame_callback(self, in_data, frame_count, time_info, _):
#         self.frames.append(in_data)
#         print("Appending frame..")
#         if not self.timestamp: self.timestamp = time_info
#         return np.zeros(frame_count * self.audio_input.channels).tostring(), pyaudio.paContinue



#     @staticmethod
#     def print_found_devices():
#         "Prints all audio devices connected to the PC and info about them on the console."
#         p = pyaudio.PyAudio()
#         for idx in range(0, p.get_device_count()):
#             print(p.get_device_info_by_index(idx))

#     @staticmethod
#     def get_specific_devices(iface_name: str, amount: int = -1) -> list['AudioInterface']:
#         """
#         Finds and returns a list of audio interfaces with a specific name.

#         Parameters:
#         ----------
#         - `iface_name` (str): Name of audio interfaces.
#         - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        
#         Returns:
#         --------
#         - `audio_iface` (list[AudioInterface]): List of found audio interfaces. If no devices are found, `[]` is returned.
#         """
#         getter = AudioInterface.__construct_getter(lambda device_info : device_info['name'].startswith(iface_name))
#         return getter(amount)

#     @staticmethod
#     def get_devices(amount: int = -1, **kwargs) -> list['AudioInterface']:
#         """
#         Finds and returns specific amount of instances of this class. Suitable devices are determined by `input_channels > 0`.

#         Better practice: Use `AudioInterface.get_specific_devices(iface_name)` instead!

#         Parameters:
#         ----------
#         - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
#         - `**kwargs`: Arbitrary keyword arguments.
        
#         Returns:
#         --------
#         - `devices` (list[AudioInterface]): List of found devices. If no devices are found, `[]` is returned.
#         """
#         super(AudioInterface,AudioInterface).get_devices(amount, type="AudioInterface", **kwargs)
#         getter = AudioInterface.__construct_getter(lambda device_info : int(device_info['maxInputChannels']) > 0 and device_info['name'] != 'default')
#         return getter(amount)

#     @staticmethod
#     def __construct_getter(condition):

#         def get_cond_devices(amount: int = -1):
#             p = pyaudio.PyAudio()
#             audio_ifaces = []
#             counter = 0
#             for idx in range(0, p.get_device_count()):
#                 if amount != -1 and counter >= amount: break
#                 device_info = p.get_device_info_by_index(idx)
#                 if condition(device_info):
#                     audio_input = AudioInput(channels=int(device_info['maxInputChannels']), rate=int(device_info['defaultSampleRate']))
#                     audio_ifaces.append(AudioInterface(f'{idx}', name=device_info['name'], audio_input=audio_input))
#             return audio_ifaces
        
#         return get_cond_devices
    
#     def __str__(self) -> str:
#         return f"AudioInterface: {self.name}"

# TODO make this class into the only AudioInterface class!
class AsyncAudioInterface(ContinuousDevice):
    """
    Audio Interface, that runs on a separate process (how it always should be as opposed to being on a separate thread).

    This is implemented because the audio recordings glitch out due to the buffer overloading, when the audio recording happens on the same process as the main data collection loop.
    """

    def __init__(self, device_id: str, name = None, audio_input = AudioInput()) -> None:
        super().__init__(device_id, name if name else f"AudioInterface_{device_id}")
        
        self.device = None
        self.timestamp = None
        self._stop = Event()
        self._is_stopped = Event()
        self.storage_queue = None
        self.format = '.wav'
        self.audio_input = audio_input
    
    def _setup_connect(self):
        self.device = pyaudio.PyAudio()
    
    def close(self):
        self._process.join()
        self.device.terminate()
        self.device = None
        return True

    def start_recording(self) -> bool:
        self.storage_queue = Queue()

        self._process = Process(
            target=self._continuous_capture, 
            args=(
                self._stop,
                self._is_stopped,
                self.device,
                self.audio_input,
                self.device_id,
                self.storage_queue
            ), 
            daemon=True
        )
        self._stop.clear()
        self._is_stopped.clear()
        self.timestamp = time.time()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        print("> stop the recording.")
        self._stop.set()
        self._is_stopped.wait()
        return True

    def store_recording(self, directory, filename = None, _ = None):
        # convert queue to list
        frames = self.storage_queue.get()
        self.storage_queue.close()
        
        # store list
        filename = filename if filename else f"{self.name}_recording"
        print(f"Attempting to store {len(frames)} frames for {self.name}")
        wf = wave.open(str(directory / filename) + self.format, 'wb')
        wf.setnchannels(self.audio_input.channels)
        wf.setsampwidth(pyaudio.get_sample_size(self.audio_input.audio_format))
        wf.setframerate(self.audio_input.rate)
        frames_bytes = frames.tobytes()
        print(type(frames_bytes))
        wf.writeframes(frames_bytes)
        wf.close()
        print(f"Frames stored for {self.name}")
        return True

    def delete_recording(self):
        self.storage_queue.close()
        return True

    @staticmethod
    def _continuous_capture(stop_event, is_stopped_event, device: pyaudio.PyAudio, audio_input: AudioInput, device_id, storage_queue: Queue) -> None:
        is_stopped_event.clear()
        stream = device.open(format=audio_input.audio_format,
                channels=audio_input.channels,
                rate=audio_input.rate,
                input=True,
                frames_per_buffer=audio_input.frames_per_buffer,
                input_device_index=int(device_id)
        )
        frames = array.array('h')
        while not stop_event.is_set():
            print("Appending frame..")
            # try:
            data = stream.read(int(audio_input.frames_per_buffer/8), exception_on_overflow=False)
            frames.frombytes(data)
            # except OSError:
            #     print("OSError occurred")
            #     traceback.print_exc() 
            #     break
        stream.close()
        storage_queue.put(frames, block=False)
        is_stopped_event.set()
        #print("Stream closed")

    @staticmethod
    def print_found_devices():
        "Prints all audio devices connected to the PC and info about them on the console."
        p = pyaudio.PyAudio()
        for idx in range(0, p.get_device_count()):
            print(p.get_device_info_by_index(idx))

    @staticmethod
    def get_specific_devices(iface_name: str, amount: int = -1) -> list['AsyncAudioInterface']:
        """
        Finds and returns a list of audio interfaces with a specific name.

        Parameters:
        ----------
        - `iface_name` (str): Name of audio interfaces.
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        
        Returns:
        --------
        - `audio_iface` (list[AsyncAudioInterface]): List of found audio interfaces. If no devices are found, `[]` is returned.
        """
        getter = AsyncAudioInterface.__construct_getter(lambda device_info : device_info['name'].startswith(iface_name))
        return getter(amount)

    @staticmethod
    def get_devices(amount: int = -1, **kwargs) -> list['AsyncAudioInterface']:
        """
        Finds and returns specific amount of instances of this class. Suitable devices are determined by `input_channels > 0`.

        Better practice: Use `AsyncAudioInterface.get_specific_devices(iface_name)` instead!

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list[AsyncAudioInterface]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(AsyncAudioInterface,AsyncAudioInterface).get_devices(amount, type="AsyncAudioInterface", **kwargs)
        getter = AsyncAudioInterface.__construct_getter(lambda device_info : int(device_info['maxInputChannels']) > 0 and device_info['name'] != 'default')
        return getter(amount)
    
    @staticmethod
    def __construct_getter(condition):

        def get_cond_devices(amount: int = -1):
            p = pyaudio.PyAudio()
            audio_ifaces = []
            counter = 0
            for idx in range(0, p.get_device_count()):
                if amount != -1 and counter >= amount: break
                device_info = p.get_device_info_by_index(idx)
                if condition(device_info):
                    audio_input = AudioInput(channels=int(device_info['maxInputChannels']), rate=int(device_info['defaultSampleRate']))
                    audio_ifaces.append(AsyncAudioInterface(device_id=f'{idx}', name=device_info['name'], audio_input=audio_input))
                    counter += 1
            return audio_ifaces
        
        return get_cond_devices

    def __str__(self) -> str:
        return f"AsyncAudioInterface: {self.name}"
    

class AsyncAudioInterface2(ContinuousDevice):
    """
    Audio Interface, that runs on a separate process (how it always should be as opposed to being on a separate thread).

    This is implemented because the audio recordings glitch out due to the buffer overloading, when the audio recording happens on the same process as the main data collection loop.
    """

    def __init__(self, device_id: str, name = None, audio_input = AudioInput()) -> None:
        super().__init__(device_id, name if name else f"AudioInterface_{device_id}")
        
        self.timestamp = None
        self._stop = Event()
        self._is_stopped = Event()
        self.storage_queue = None
        self.format = '.wav'
        self.audio_input = audio_input
    
    def _setup_connect(self):
        pass
    
    def close(self):
        self._process.join()
        return True

    def start_recording(self) -> bool:
        self.storage_queue = Queue()

        self._process = Process(
            target=self._continuous_capture, 
            args=(
                self._stop,
                self._is_stopped,
                self.audio_input,
                self.device_id,
                self.storage_queue
            ), 
            daemon=True
        )
        self._stop.clear()
        self._is_stopped.clear()
        self.timestamp = time.time()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        print("> stop the recording.")
        self._stop.set()
        self._is_stopped.wait()
        return True

    def store_recording(self, directory, filename = None, _ = None):
        # convert queue to list
        frames = self.storage_queue.get()
        self.storage_queue.close()
        
        # store list
        filename = filename if filename else f"{self.name}_recording"
        print(f"Attempting to store {len(frames)} frames for {self.name}")
        wf = wave.open(str(directory / filename) + self.format, 'wb')
        wf.setnchannels(self.audio_input.channels)
        wf.setsampwidth(pyaudio.get_sample_size(self.audio_input.audio_format))
        wf.setframerate(self.audio_input.rate)
        frames_bytes = frames.tobytes()
        wf.writeframes(frames_bytes)
        wf.close()
        print(f"Frames stored for {self.name}")
        return True

    def delete_recording(self):
        self.storage_queue.close()
        return True

    @staticmethod
    def _continuous_capture(stop_event, is_stopped_event, audio_input: AudioInput, device_id, storage_queue: Queue) -> None:
        is_stopped_event.clear()
        frames = array.array('h')

        def callback(in_data, frame_count, time_info, status):
            frames.frombytes(in_data)
            #print("Appending frame..")
            next_state = pyaudio.paContinue if not stop_event.is_set() else pyaudio.paAbort
            return np.zeros(frame_count * audio_input.channels).tostring(), next_state

        device = pyaudio.PyAudio()

        stream = device.open(format=audio_input.audio_format,
            channels=audio_input.channels,
            rate=audio_input.rate,
            input=True,
            frames_per_buffer=audio_input.frames_per_buffer,
            input_device_index=int(device_id),
            stream_callback=callback
        )
        while stream.is_active():
            time.sleep(0.1)

        stream.close()
        device.terminate()
        storage_queue.put(frames, block=False)
        is_stopped_event.set()

    @staticmethod
    def print_found_devices():
        "Prints all audio devices connected to the PC and info about them on the console."
        p = pyaudio.PyAudio()
        for idx in range(0, p.get_device_count()):
            print(p.get_device_info_by_index(idx))

    @staticmethod
    def get_specific_devices(iface_name: str, amount: int = -1) -> list['AsyncAudioInterface2']:
        """
        Finds and returns a list of audio interfaces with a specific name.

        Parameters:
        ----------
        - `iface_name` (str): Name of audio interfaces.
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        
        Returns:
        --------
        - `audio_iface` (list[AsyncAudioInterface]): List of found audio interfaces. If no devices are found, `[]` is returned.
        """
        getter = AsyncAudioInterface2.__construct_getter(lambda device_info : device_info['name'].startswith(iface_name))
        return getter(amount)

    @staticmethod
    def get_devices(amount: int = -1, **kwargs) -> list['AsyncAudioInterface']:
        """
        Finds and returns specific amount of instances of this class. Suitable devices are determined by `input_channels > 0`.

        Better practice: Use `AsyncAudioInterface.get_specific_devices(iface_name)` instead!

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list[AsyncAudioInterface]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(AsyncAudioInterface2,AsyncAudioInterface2).get_devices(amount, type="AsyncAudioInterface", **kwargs)
        getter = AsyncAudioInterface2.__construct_getter(lambda device_info : int(device_info['maxInputChannels']) > 0 and device_info['name'] != 'default')
        return getter(amount)
    
    @staticmethod
    def __construct_getter(condition):

        def get_cond_devices(amount: int = -1):
            p = pyaudio.PyAudio()
            audio_ifaces = []
            counter = 0
            for idx in range(0, p.get_device_count()):
                if amount != -1 and counter >= amount: break
                device_info = p.get_device_info_by_index(idx)
                if condition(device_info):
                    audio_input = AudioInput(channels=int(device_info['maxInputChannels']), rate=int(device_info['defaultSampleRate']))
                    audio_ifaces.append(AsyncAudioInterface2(device_id=f'{idx}', name=device_info['name'], audio_input=audio_input))
                    counter += 1
            return audio_ifaces
        
        return get_cond_devices

    def __str__(self) -> str:
        return f"AsyncAudioInterface: {self.name}"