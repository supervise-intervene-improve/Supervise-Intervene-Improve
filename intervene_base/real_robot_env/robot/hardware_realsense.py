# Original Author: Marcel Ruehle
import time
import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict

from real_robot_env.robot.hardware_cameras import DiscreteCamera


import logging
log = logging.getLogger(__name__)


class RealSense(DiscreteCamera):
    """
    Wrapper that implements boilerplate code for RealSense cameras.
    The recording fails, when using different dimensions than 480x640, so these are now hardcoded to capture the frames.

    Warning: This script is a bit buggy. Sometimes, this error appears (sometimes it doesn't):

    ``File "/home/user/audio-pipeline/real_robot/real_robot_env/robot/hardware_realsense.py", line 48, in __get_frames`` \n
      ``return self.align.process(frameset)``
    ``RuntimeError: Error occured during execution of the processing block! See the log for more info``
    """
    RECORDING_HEIGHT = 480 # 1080
    RECORDING_WIDTH  = 640 # 1920

    def __init__(self, device_id, name=None, height=480, width=640, fps=30, warm_start=30, start_frame_latency = 0, streaming = False, save=True):
        super().__init__(device_id, name if name else f"RealSense_{device_id}", height, width, start_frame_latency, streaming=streaming, save=save)
        self.fps = fps
        self.warm_start=warm_start
        self.pipe = None
    
    def _setup_connect(self):
        print("Here Realsense")
        self.pipe= rs.pipeline()
        config = rs.config()
        print("Here 1")

        config.enable_device(self.device_id)
        config.enable_stream(rs.stream.depth, self.RECORDING_WIDTH, self.RECORDING_HEIGHT, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.RECORDING_WIDTH, self.RECORDING_HEIGHT, rs.format.rgb8, self.fps)
        self.profile = self.pipe.start(config)
        self.align = rs.align(rs.stream.color)
        print("Here 2")

        """
        # Change Filter
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 5)  # Size of Kernel
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.7)
        self.spatial.set_option(rs.option.filter_smooth_delta, 40)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.7)
        self.temporal.set_option(rs.option.filter_smooth_delta, 40)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)
        """

        for _ in range(self.warm_start):
            self.__get_frames()

    def get_intrinsics_dict(self):
        stream = self.profile.get_streams()[1]
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        param_dict = dict([(p, getattr(intrinsics, p)) for p in dir(intrinsics) if not p.startswith('__')])
        param_dict['model'] = param_dict['model'].name
        return param_dict

    def __get_frames(self):
        if self.pipe is None:
            raise RuntimeError('Please connect first.')
        frameset = self.pipe.wait_for_frames()
        return self.align.process(frameset)
    
    # def __get_frames(self):
    #     if self.pipe is None:
    #         raise RuntimeError('Please connect first.')

    #     try:
    #         frameset = self.pipe.wait_for_frames(timeout_ms=5000)
    #         return self.align.process(frameset)
    #     except RuntimeError as e:
    #         print(f"⚠️ RealSense warning: {e}")
    #         print("🔁 Restarting the pipeline...")
    #         try:
    #             self.pipe.stop()
    #             time.sleep(1)
    #             self.pipe.start(self.config)
    #         except Exception as restart_err:
    #             print(f"❌ Failed to restart pipeline: {restart_err}")
    #         return None  # signal failure upward

    def get_rgbd(self):
        """
        returns color image as np.ndarray [h, w, 3] with RGB[0-255] and depth as np.ndarray [h, w] in millimeters
        """
        frameset = self.__get_frames()
        if frameset is None:
            return None, None

        rgb = np.empty([self.RECORDING_HEIGHT, self.RECORDING_WIDTH, 3], dtype=np.uint16)
        d = np.empty([self.RECORDING_HEIGHT, self.RECORDING_WIDTH], dtype=np.uint16)

        color_frame = frameset.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())

        depth_frame = frameset.get_depth_frame()

        # change Filter
        """
        depth_frame_f = self.spatial.process(depth_frame)
        depth_frame_f = self.temporal.process(depth_frame)
        depth_frame_f = self.hole_filling.process(depth_frame)
        """
        d = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units() * 1000  # in millimeters

        return rgb, d

    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd': depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': NDArray[uint16], 'd': NDArray[uint16]}`.
        """
        # get all data from all topics
        rgb, d = self.get_rgbd()
        timestamp = time.time()
        return {'time': timestamp, 'rgb': rgb, 'd': d}

    def close(self):
        self.pipe.stop()
        return True
    
    @staticmethod
    def get_devices(amount=-1, height: int = 480, width: int = 640, **kwargs) -> list['RealSense']:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `480`
        - `width` (int): Pixel-width of captured frames. Default: `640`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(RealSense,RealSense).get_devices(amount, height=height, width=width, type="RealSense", **kwargs)
        cam_list = rs.context().query_devices()
        cams = []
        counter = 0
        for device in cam_list:
            if amount != -1 and counter >= amount: break
            cam = RealSense(device.get_info(rs.camera_info.serial_number), height=height, width=width)
            cams.append(cam)
            counter += 1
        return cams
    
    def _get_intrinsics(self):
        """
        Prompts the device to output the intrinsics of the camera.
        Output has the following format: `{'width': int, 'height': int, 'fx': float, 'fy': float, 'cx': float, 'cy': float}`

        Returns:
        -------
        - `intrinsics` (dict): Intrinsics of the camera in the format `{'width': int, 'height': int, 'fx': float, 'fy': float, 'cx': float, 'cy': float}`.
        """
        # Hol dir die Tiefenkamera-Parameter
        depth_stream = self.profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        depth_intrinsics = OrderedDict([
            ('width', intrinsics.width),
            ('height', intrinsics.height),
            ('fx', intrinsics.fx),
            ('fy', intrinsics.fy),
            ('cx', intrinsics.ppx),
            ('cy', intrinsics.ppy)
        ])
        return depth_intrinsics


if __name__ == "__main__":

    # import pyrealsense2 as rs
    ctx = rs.context()
    print("Connected devices:")
    for dev in ctx.query_devices():
        print("-", dev.get_info(rs.camera_info.serial_number))



    cam = RealSense("944622073668")
    cam.connect()
    print(cam.get_intrinsics_dict())

    for i in range(100):
        rgb, d = cam.get_rgbd()
        print(i,rgb.shape)