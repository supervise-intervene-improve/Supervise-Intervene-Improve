import cv2
import numpy as np
from abc import abstractmethod
from bisect import bisect_right
from datetime import datetime
import time
from typing import Any, Optional, TypeVar, Generic, List, Type
from multiprocessing import Process, Event
from multiprocessing.managers import BaseManager
from pathlib import Path
from time import sleep
from tempfile import TemporaryDirectory
import shutil
from multiprocessing import Process

# async cam imports
from bisect import bisect_right
from datetime import datetime
from typing import Any, Optional, TypeVar, Generic, List, Type
from multiprocessing import Event
from multiprocessing.managers import BaseManager
from time import sleep
from tempfile import TemporaryDirectory
import shutil

# I installed this through:
#  pip install --trusted-host pypi.python.org moviepy
#  pip install imageio-ffmpeg
# from moviepy.editor import VideoFileClip

from real_robot_env.robot.hardware_devices import DiscreteDevice, ContinuousDevice

from zmq_server import send
import zmq
import struct

class DiscreteCamera(DiscreteDevice):
    """
    This class acts as a generalization for cameras, whose recording is captured frame by frame. It implements the method `cam.store_last_frame(dir, title)`.

    Additionally, this class inherits from `DiscreteDevice`, so its functionality is also included.
    """

    """
    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512, start_frame_latency: int = 0) -> None:
        super().__init__(device_id, name if name else f"discrete_cam_{device_id}", start_frame_latency)
        self.format = '.png'
        self.height, self.width = height, width
    """
    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512, start_frame_latency: int = 0, streaming = False, save=True) -> None:
        super().__init__(device_id, name if name else f"discrete_cam_{device_id}", start_frame_latency)
        self.format = '.png'
        self.height, self.width = height, width       
        self.frame_count = 0
        self.last_time = time.time()
        self.name = name
        print("init", streaming)
        self.streaming = streaming
        self.save = save
        self.timedebugging = True


    @abstractmethod
    def _get_sensors(self) -> dict[str, Any]:
        """
        Prompts the camera to output a single frame. Is overwritten by subclass.
        Output should have the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd' [opt]: depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': timestamp, 'rgb': rgb_vals, ...}`.
            
        """
        pass

    def _get_intrinsics(self) -> dict[str, Any]:
        """
        Use to get the depth camera intrinsics for the data streaming.
        """
        pass

    def get_format(self) -> str:
        return self.format

    def store_last_frame(self, directory: Path, filename: str):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.format` (default: ".png").

        Parameters:
        ----------
        - `directory` (Path): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        """
        start = time.time()
        sensor_data = self._get_sensors()
        end = time.time()

        # if self.timedebugging:
        #     print(f"[DEBUG] [{self.device_id}] Sensor data received in {end - start:.4f} seconds.")

        # Erst prüfen, ob sensor_data korrekt ist
        if not sensor_data or "rgb" not in sensor_data or "d" not in sensor_data: # or "time" not in sensor_data:
            print(f"[WARN] invalid Sensordata receivec – Camera: {self.name}, sensor_data: {sensor_data}")
            return
        
        # Sensor-Daten
        img = sensor_data["rgb"]
        depth = sensor_data["d"]
        # timestamp = sensor_data["time"]

        # Debug-Ausgabe
        # print(f"[DEBUG] Kamera: {self.name} – depth-Type: {type(depth)}, depth-Inhalt: {depth}")

        # Wenn depth ein Tuple ist, entpacke es
        if isinstance(depth, tuple):
            # print(f"[WARN] depth ist ein Tuple, wird entpackt – Kamera: {self.name}")
            depth = depth[1]

        # Prüfe, ob img und depth NumPy-Arrays sind
        if not isinstance(img, np.ndarray):
            print(f"[ERROR] Ungültiges RGB-Bildformat: {type(img)}, Kamera: {self.name}")
            return
        if not isinstance(depth, np.ndarray):
            print(f"[ERROR] Ungültiges Tiefenbildformat: {type(depth)}, Kamera: {self.name}")
            return

        
        # self.timestamps.append(timestamp)


        # Bilder vorbereiten
        resized_rgb = cv2.resize(img, (self.width, self.height))
        resized_depth = cv2.resize(depth, (self.width, self.height))
        resized_depth = np.nan_to_num(resized_depth, nan=0, posinf=0, neginf=0)
        resized_depth = resized_depth.astype(np.uint16)  # wichtig für korrekte Depth-Speicherung

        # RGB konvertieren (RGB → BGR für OpenCV) nur bei RealSense
        if self.__class__.__name__ == "RealSense":
            cvt_rgb = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        else:
            cvt_rgb = resized_rgb
        # Dateinamen (z. B. "2025-06-09T12:00:00.123456")
        # dt_object = datetime.fromtimestamp(timestamp)
        # filename = dt_object.isoformat()

        if self.save:
            filename = filename if filename else f"{self.name}_{int(time.time()*1000)}" # fallback
            # Bilder speichern
            cv2.imwrite(str(directory / f"{filename}_rgb") + self.format, cvt_rgb)
            cv2.imwrite(str(directory / f"{filename}_depth") + self.format, resized_depth)

            # DEBUG
            # Depth speichern nur wenn PNG
            if self.format.lower() == ".png":
                cv2.imwrite(str(directory / f"{filename}_depth") + self.format, resized_depth)
            else:
                print(f"[WARN] Depth nicht gespeichert – {self.format} unterstützt kein 16-bit.")

        if self.streaming:
            self.stream_data(img, depth)


    def stream_data(self, rgb, d):
        if self.__class__.__name__ == "Azure":
            rgb = self.rgb_to_depth_resolution(rgb, d)
        
        rgb_bytes = np.array(rgb, dtype=np.uint8).tobytes()
        d = np.nan_to_num(d, nan=0, posinf=0, neginf=0)
        depth_bytes = np.array(d, dtype=np.uint16).tobytes()

        _getrinsics = self._get_intrinsics()

        name = self.name.encode('utf-8')
        h, w = rgb.shape[:2]
        packet = (
            struct.pack('<I', len(name)) + name +
            struct.pack('<II', h, w) +          # nur Höhe und Breite
            struct.pack('<I', len(rgb_bytes)) + rgb_bytes +
            struct.pack('<I', len(depth_bytes)) + depth_bytes +
            struct.pack('<4f', _getrinsics['fx'], _getrinsics['fy'], _getrinsics['cx'], _getrinsics['cy'])
        )

        send(packet)

        # === FPS-Messung ===
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time

        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            print(f"[{self.device_id}] FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_time = now




    @staticmethod
    @abstractmethod
    def get_devices(amount: int, height: int = 512, width: int = 512, type="discrete", **kwargs) -> list['DiscreteCamera']:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` may return all instances (not always).
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(f"Looking for {'up to ' + str(amount) if amount != -1 else 'all'} {type} cameras to capture {height}x{width} frames.")



class ContinuousCamera(ContinuousDevice):

    def __init__(self, device_id: str, name: Optional[str] = None, height: int = 512, width: int = 512, default_fps: float = 20, cut_ending=True, **kwargs) -> None:
        super().__init__(device_id, name if name else f"continuous_cam_{device_id}")
        self.format = '.mp4'
        self.height, self.width = height, width
        self.latency = 0.0 # in s
        self.default_fps = default_fps
        self.cut_ending = cut_ending

        self.frame_extraction_processes = []

    def get_format(self) -> str:
        return self.format

    def start_recording(self) -> bool:
        self.recording_start = time.time() + self.latency # timestamp, where recording actually started
        return True

    def stop_recording(self) -> bool:
        self.recording_stop = time.time() # timestamp, where recording SHOULD end
        return True
    
    def store_recording(self, directory, filename = None, timestamps = None):
        filename = filename if filename else f"{self.name}_recording"
        video_file = str(directory / filename) + self.format
        self._store_video(video_file)

        if timestamps:
            duration = timestamps[-1] - timestamps[0]
            print(f"duration: {duration}, timestamps length: {len(timestamps)} => fps: {len(timestamps)/duration}")
            self.__extract_frames_at_timestamps(video_file, directory, timestamps, self.recording_start)
        else:
            self.__extract_frames(video_file, directory, self.default_fps, self.recording_start, self.recording_stop, self.cut_ending)
        return True


    @abstractmethod
    def _store_video(self, video_file: str):
        pass


    def __extract_frames_at_timestamps(self, video_file, directory, timestamps, recording_start):

        def convert_to_images():
            vidcap = cv2.VideoCapture(video_file)
            idx = 0
            for timestamp in timestamps:
                ms_time = max((timestamp - recording_start) * 1000, 0) # relative video position in ms
                vidcap.set(cv2.CAP_PROP_POS_MSEC,ms_time)
                success,image = vidcap.read()
                if success:
                    resized_img = cv2.resize(image, (self.width, self.height))
                    cv2.imwrite(str(directory / f"{idx}.png"), resized_img)       # save frame as png file
                    idx += 1
        
        process = Process(target=convert_to_images)
        process.start()
        self.frame_extraction_processes.append(process)


    # def __extract_frames(self, video_file, directory, fps, recording_start, recording_stop, cut_ending=True):
    
    #     def convert_to_images():
    #         if cut_ending:
    #             duration = recording_stop - recording_start # in seconds
    #             clip = VideoFileClip(video_file).subclip(0, duration)
    #         else:
    #             clip = VideoFileClip(video_file)
    #         clip.write_images_sequence(str(directory / '%d.png'), fps=fps, logger=None) # logger='bar'
        
    #     process = Process(target=convert_to_images)
    #     process.start()
    #     self.frame_extraction_processes.append(process)

    
    def close(self) -> bool:
        for process in self.frame_extraction_processes:
            process.join()
        return True


    @staticmethod
    @abstractmethod
    def get_devices(amount: int, height: int = 512, width: int = 512, type="continuous", **kwargs) -> list['DiscreteCamera']:
        """
        Finds and returns specific amount of instances of this class. Is overwritten by subclass.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` may return all instances (not always).
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list): List of found devices. If no devices are found, `[]` is returned.
        """
        print(f"Looking for {'up to ' + str(amount) if amount != -1 else 'all'} {type} cameras to capture {height}x{width} frames.")


# author of class: TimWindecker
T = TypeVar("T", bound=DiscreteCamera)
class AsynchronousCamera(ContinuousCamera, Generic[T]):
    """
    This class is a wrapper for a DiscreteCamera to act as a ContinuousCamera by running it in a separate process.
    """

    def __init__(self, camera_class: Type[T], capture_interval=0, **kwargs):
        super().__init__(
            **kwargs
        )

        # Create event to signal that the process should stop
        self._stop = Event()

        # Define custom manager to share the camera object between processes
        class CameraManager(BaseManager):
            pass
        V = TypeVar("V")
        class Container(Generic[V]):
            def __init__(self, value: V):
                self._value = value
            def get_value(self) -> V:
                return self._value
        CameraManager.register("Camera", camera_class, exposed=("_setup_connect", "_failed_connect", "close", "store_last_frame", "get_format"))
        CameraManager.register("CaptureInterval", Container[int])
        CameraManager.register("TempDirectoryPath", Container[str])

        # Start manager and create shared objects
        self._manager = CameraManager()
        self._manager.start()
        self._proxy_camera = self._manager.Camera(**kwargs)
        self._proxy_capture_interval = self._manager.CaptureInterval(capture_interval)
        self._temp_dir = TemporaryDirectory(prefix='camera_tmp_')
        self._proxy_temp_dir_path = self._manager.TempDirectoryPath(self._temp_dir.name)
        assert(Path(self._proxy_temp_dir_path.get_value()).exists())

    def _setup_connect(self):
        self._proxy_camera._setup_connect()

    def _failed_connect(self):
        self._proxy_camera._failed_connect()

    def close(self):
        super().close()
        self._proxy_camera.close()

    @staticmethod
    def get_devices(**kwargs) -> list['T']:
        return T.get_devices(**kwargs)

    def _store_video(self, video_file: str):
        raise NotImplementedError

    def start_recording(self) -> bool:
        super().start_recording()

        # Clear temp folder
        self._clear_temp()

        # Start process
        self._process = Process(
            target=self._continuous_capture, 
            args=(
                self._stop,
                self._proxy_camera,
                self._proxy_capture_interval, 
                self._proxy_temp_dir_path
            ), 
            daemon=True
        )
        self._stop.clear()
        self._process.start()

        return True

    def stop_recording(self) -> bool:
        super().stop_recording()
        self._stop.set()
        self._process.join()
        return True

    def store_recording(self, directory: Path, filename: str = None, timestamps: List[float] = None):
        """
        Stores the last frame received by camera (only the RGB data) as a `self.format` (default: ".png").

        Parameters:
        ----------
        - `directory` (Path): Directory, where last frame should be stored.
        - `filename` (str): Title of the frame.
        - `timestamps` (List[float]): Timesamps of the required images as list of seconds since the Unix epoch. 
                                      If the exact timestamp is not available the closest image before is selected.
        """

        if timestamps is None:
            # If no timestamps are given, copy all images
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            image_format = self._proxy_camera.get_format()
            for image_path in temp_dir.glob(f"*{image_format}"):
                destination_path = directory / f"{image_path.name}"
                shutil.copy(image_path, destination_path)
        else:
            """
            # Extract map from timestamps to image paths
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            image_format = self._proxy_camera.get_format()
            timestamp_path_map = []
            for image_path in temp_dir.glob(f"*{image_format}"):
                try:
                    timestamp_str = image_path.stem
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamp_unix = timestamp.timestamp()
                    timestamp_path_map.append((timestamp_unix, image_path))
                except ValueError:
                    print(f"Skipping file with invalid timestamp: {image_path.name}")

            # Sort map by timestamps
            timestamp_path_map.sort(key=lambda t: t[0])

            # Map desired timestamps to paths
            sorted_timetamps = [t for t, _ in timestamp_path_map]
            desired_indices = [bisect_right(sorted_timetamps, t) - 1 for t in timestamps] # Find index of element <= t
            desired_paths  = [timestamp_path_map[i][1] for i in desired_indices]

            # Copy images
            for i, path in enumerate(desired_paths):
                destination_path = directory / f"{i}{image_format}"
                shutil.copy(path, destination_path)

        # Clear temp folder
        self._clear_temp()
        """
                    
            temp_dir = Path(self._proxy_temp_dir_path.get_value())
            timestamp_path_map = []

            # Durchsuche alle *_rgb.png-Dateien (depth folgt aus demselben Timestamp)
            for image_path in temp_dir.glob("*_rgb.png"):
                try:
                    # Extrahiere Timestamp vor dem Suffix
                    timestamp_str = image_path.stem.replace("_rgb", "")
                    timestamp = datetime.fromisoformat(timestamp_str)
                    timestamp_unix = timestamp.timestamp()
                    timestamp_path_map.append((timestamp_unix, timestamp_str))
                except ValueError:
                    print(f"Skipping invalid filename: {image_path.name}")

            # Sortieren
            timestamp_path_map.sort(key=lambda t: t[0])
            sorted_timestamps = [t[0] for t in timestamp_path_map]

            # Für jeden gewünschten Zeitpunkt das passende Bild suchen
            desired_indices = [bisect_right(sorted_timestamps, t) - 1 for t in timestamps]
            
            # print(f"[DEBUG] timestamp_path_map len: {len(timestamp_path_map)}")
            # print(f"[DEBUG] desired_indices: {desired_indices}")

            selected_timestamp_strs = [timestamp_path_map[i][1] for i in desired_indices]

            # Zielordner erstellen
            rgb_dir = directory / "rgb"
            depth_dir = directory / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # Bilder kopieren
            for i, ts_str in enumerate(selected_timestamp_strs):
                rgb_path = temp_dir / f"{ts_str}_rgb.png"
                depth_path = temp_dir / f"{ts_str}_depth.png"

                if rgb_path.exists():
                    shutil.copy(rgb_path, rgb_dir / f"{i}.png")
                else:
                    print(f"Fehlt: {rgb_path.name}")

                if depth_path.exists():
                    shutil.copy(depth_path, depth_dir / f"{i}.png")
                else:
                    print(f"Fehlt: {depth_path.name}")

        # Temp-Ordner leeren
        self._clear_temp()

    def delete_recording(self):
        self._clear_temp()

    def _clear_temp(self):
        for file in Path(self._proxy_temp_dir_path.get_value()).rglob("*"):
            file.unlink()

    def _continuous_capture(self, stop_event, camera, capture_interval, dir_path) -> None:
        
        directory = Path(dir_path.get_value())

        while not stop_event.is_set():

            # Capture and store
            filename = datetime.now().isoformat()
            camera.store_last_frame(directory, filename)

            # Sleep    
            sleep(capture_interval.get_value())