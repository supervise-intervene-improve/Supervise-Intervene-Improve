import logging
import time
from typing import Sequence, Union, Optional
import pyzed.sl as sl
import cv2
import numpy as np

from real_robot_env.robot.hardware_cameras import DiscreteCamera

log = logging.getLogger(__name__)

class Zed(DiscreteCamera):  # TODO: import specs directly!
    """
    This class can be considered a wrapper class for ZED cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_cameras.DiscreteCamera`.
    """

    def __init__(
        self,
        device_id: Union[int, str],
        name: Optional[str] = None,
        resolution: str = "VGA",
        depth_mode: str = "NEURAL",
        fps=30,
        height=640,
        width=480,
        extrinsics: Optional[Sequence[Sequence[float]]] = None,
        warm_start=30,
        start_frame_latency=0,
        streaming=False,
        save=True
    ):
        super().__init__(
            device_id=str(device_id),
            name=name if name else f"Zed_{device_id}",
            height=480, width=640,
            start_frame_latency=start_frame_latency,
            streaming=streaming,
            save=save
        )

        if resolution == "VGA":
            self.resolution = sl.RESOLUTION.VGA
        elif resolution == "HD720":
            self.resolution = sl.RESOLUTION.HD720
        elif resolution == "HD1080":
            self.resolution = sl.RESOLUTION.HD1080
        elif resolution == "HD2K":
            self.resolution = sl.RESOLUTION.HD2K
        else:
            self.resolution = sl.RESOLUTION.VGA
            log.warning(f"Resolution {resolution} not detected. Set to default: VGA")

        if depth_mode == "QUALITY":
            self.depth_mode = sl.DEPTH_MODE.QUALITY
        elif depth_mode == "ULTRA":
            self.depth_mode = sl.DEPTH_MODE.ULTRA
        elif depth_mode == "NONE":
            self.depth_mode = sl.DEPTH_MODE.NONE
        elif depth_mode == "NEURAL":
            self.depth_mode = sl.DEPTH_MODE.NEURAL
        elif depth_mode == "PERFORMANCE":
            self.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        else:
            self.depth_mode = sl.DEPTH_MODE.QUALITY
            log.warning(
                f"Depth_Mode {depth_mode} not detected. Set to default: QUALITY"
            )

        self.fps = fps
        self._extrinsics = (
            np.asarray(extrinsics).reshape(4, 4) if extrinsics is not None else None
        )
        self.warm_start = warm_start
        self.pipe = None

    def connect(self) -> bool:
        """
        Connects to this instance.

        Returns:
        --------
        - `success` (bool): Indicates a successful connection.
        """
        log.info(f"Connecting to Zed {self.name}...")
        try:
            self._setup_connect()
            log.info(f"Connection to Zed {self.name} successful.")
            return True

        except Exception as e:
            log.exception(f"Connection to Zed {self.name} failed.")

        log.info(f"Resetting Zed {self.name}...")
        devices = sl.Camera.get_device_list()
        for device in devices:
            # if device.get_camera_information().serial_number == self.device_id:
            if str(device.serial_number) == str(self.device_id):
                # device.zed.close()  # TODO: Is there a better reset command? close() might be unnecessary, because the connection never estabishes!
                pass
        log.info(f"Retrying connection to Zed {self.name}...")
        self._setup_connect()
        log.info(f"Connection to Zed {self.name} successful.")
        return True

    def _setup_connect(self):
        print("Here - Zed")
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.resolution
        init_params.camera_fps = self.fps
        init_params.depth_mode = self.depth_mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Error {}, exit program".format(err))
            exit()

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.depth = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            for _ in range(self.warm_start):
                self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)

    def get_intrinsics_dict(self):
        intrinsics = (
            self.zed.get_camera_information().camera_configuration.calibration_parameters
        )

        cx = intrinsics.left_cam.cx
        cy = intrinsics.left_cam.cy
        fx = intrinsics.left_cam.fx
        fy = intrinsics.left_cam.fy
        width = intrinsics.left_cam.image_size.width
        height = intrinsics.left_cam.image_size.height
        distortion = intrinsics.left_cam.disto
        baseline = intrinsics.stereo_transform.get_translation().get()[0] / 1000

        return {
            "cx": cx,
            "cy": cy,
            "fx": fx,
            "fy": fy,
            "width": width,
            "height": height,
            "distortion": distortion,
            "baseline": baseline,
        }
    
    def _get_intrinsics(self):
        return self.get_intrinsics_dict()

    @property
    def extrinsics(self) -> Optional[np.ndarray]:
        return self._extrinsics

    def __get_frames(self):
        if not self.zed:
            raise Exception(f"Not connected to {self.name}")

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            # self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            # self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            # self.timestamp = time.time()

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'left': rgb_vals, 'right': rgb_vals, 'depth': depth_vals}`
        The depth values are in millimeters.
        The RGB values are in BGRA format.

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'left': uint8, 'right': uint8, 'depth': Any }`.
        """
        if not self.zed:
            raise Exception(f"Not connected to {self.name}")

        self.__get_frames()

        image_left_np = self.image_left.get_data()  # BGRA
        image_left_np = cv2.cvtColor(image_left_np, cv2.COLOR_BGRA2BGR)  # RGB
        
        # image_right_np = self.image_right.get_data()  # BGRA
        # image_right_np = cv2.cvtColor(image_right_np, cv2.COLOR_BGRA2BGR)  # RGB
        
        depth_np = self.depth.get_data()
        # point_cloud_np = point_cloud.get_data()

        return {
            #"time": self.timestamp,
            "rgb": image_left_np,
            "d": depth_np,
        }  # , "point_cloud": point_cloud_np}

    def close(self):
        if self.zed:
            self.zed.close()

    @staticmethod
    def get_devices(amount=-1, resolution="VGA", **kwargs) -> list["Zed"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `resolution` (String): Resolution of the camera. Can be "VGA", "HD720", "HD1080", or "HD2K".
        - `fps` (int): Frames per second. Default is 30. 15, 30, 60 and 100 are supported.
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(Zed, Zed).get_devices(amount, resolution="VGA", type="ZED")
        devices = sl.Camera.get_device_list()
        amount = amount if amount != -1 else len(devices)

        cameras = [
            Zed(
                device.serial_number,
                resolution=resolution,
                **kwargs,
            )
            for device in devices[:amount]
        ]
        return cameras

    @staticmethod
    def get_device(serial_number: int, **kwargs) -> "Zed":
        devices = sl.Camera.get_device_list()
        for device in devices:
            # if device.get_camera_information().serial_number == str(serial_number):
            if str(device.serial_number) == str(serial_number):
                return Zed(
                    # device.get_camera_information().serial_number,
                    device.serial_number,
                    **kwargs,
                )
        raise ValueError(f"RealSense with serial number {serial_number} not found.")


if __name__ == "__main__":
    devices = sl.Camera.get_device_list()
    for device in devices:
        print(device.serial_number)
    print("ZED camera test")

    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NEURAL

    zed = sl.Camera()
    err = zed.open(init)
    print("ZED open result:", err)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    print("✅ Kamera läuft")

    runtime = sl.RuntimeParameters()
    frame_count = 0
    second_start = time.time()

    # Grab frames for 10 seconds and print FPS every second
    end_time = time.time() + 10
    image_left = sl.Mat()
    depth = sl.Mat()
    while time.time() < end_time:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            # self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            image_left_np = image_left.get_data()
            depth_np = depth.get_data()

            frame_count += 1

        # Check if one second has passed
        if time.time() - second_start >= 1.0:
            print(f"FPS last second: {frame_count}")
            frame_count = 0
            second_start = time.time()

    zed.close()
    print("✅ Test finished")
