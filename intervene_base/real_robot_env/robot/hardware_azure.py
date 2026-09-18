import cv2
# import pykinect_azure as pykinect
import logging
from typing import Optional
import time
import numpy as np
from collections import OrderedDict
# import pyk4a

from real_robot_env.robot.hardware_cameras import DiscreteCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Azure(DiscreteCamera):
    """
    This class can be considered a wrapper class for Azure cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_cameras.DiscreteCamera`.
    """

    def __init__(self, device_id, name = None, height = 512, width = 512, start_frame_latency = 0, streaming = False):
        super().__init__(device_id, name if name else f"Azure_{device_id}", height, width, start_frame_latency, streaming=streaming)

        self.__set_device_configuration() # sets self.device_config
        self.device = None

    def _setup_connect(self):
        
        pykinect.initialize_libraries()
        print("Init libs")
        self.device = pykinect.start_device(device_index=self.device_id, config=self.device_config)
        print("start device")

    def _failed_connect(self):
        self.device = None


    def _get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': Any}`.
        """
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        
        success = False
        while not success: 
            capture = self.device.update()
            success, image = capture.get_color_image()
            timestamp = time.time()
            depth_image = capture.get_depth_image()

        return {'time': timestamp, 'rgb': image, 'd': depth_image }

    def close(self):
        self.device.close()
        self.device = None

    def __set_device_configuration(self):
        self.device_config = pykinect.default_configuration
        self.device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
        # self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_OFF
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
        self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        self.device_config.synchronized_images_only = True

    @staticmethod
    def get_devices(amount, height: int = 512, width: int = 512, **kwargs) -> list['Azure']:
        """
        Returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Amount of instances to be created.
        - `height` (int): Pixel-height of captured frames. Default: `512`
        - `width` (int): Pixel-width of captured frames. Default: `512`
        - `**kwargs`: Arbitrary keyword arguments.
        
        Returns:
        --------
        - `devices` (list[Azure]): List of created instances.
        """
        super(Azure,Azure).get_devices(amount, height=height, width=width, type="Azure", **kwargs)
        cams = []
        for i in range(amount):
            cam = Azure(device_id=i, height=height, width=width)
            cams.append(cam)
        return cams

    def get_point_cloud(self):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        success = False
        while not success:
            capture = self.device.update()
            success, pc = capture.get_transformed_pointcloud()

        return pc

    # see https://github.com/ibaiGorordo/pyKinectAzure/blob/fdfd70ee0fc4287e0750d6b99a658210cf53cf7c/pykinect_azure/k4a/calibration.py#L6
    def get_intrinsics(self, sensor_type: str = "color"):
        if not self.device:
            raise Exception(f"Not connected to {self.name}")
        if sensor_type == "color":
            type = pykinect.K4A_CALIBRATION_TYPE_COLOR
        else:
            type = pykinect.K4A_CALIBRATION_TYPE_DEPTH

        return self.device.get_calibration().get_matrix(type)


    def _get_intrinsics(self):

        depth_mode = self.device_config.depth_mode
        color_resolution = self.device_config.color_resolution

        calibration = self.device.get_calibration(depth_mode, color_resolution)

        depth_params = calibration.depth_params
        fx_d = depth_params.fx
        fy_d = depth_params.fy
        cx_d = depth_params.cx
        cy_d = depth_params.cy
        # Für Tiefenkamera

        depth_dict = {
            'fx': fx_d,
            'fy': fy_d,
            'cx': cx_d,
            'cy': cy_d,
            # falls nötig, Verzerrung hier auch extrahieren
        }
        return depth_dict


    def rgb_to_depth_resolution(self, color_image, depth_image):
        # print(dir(self.device))
        # transform = self.device.get_transform()
        # transformed_color = transform.color_image_to_depth_camera(color_image, depth_image)

        # calibration = self.device.get_calibration(self.device_config.depth_mode, self.device_config.color_resolution)
        
        # transformed_color_image = calibration.convert_color_2d_to_depth_2d(depth_image, color_image)     
        depth_mode = self.device_config.depth_mode
        color_resolution = self.device_config.color_resolution

        calibration = self.device.get_calibration(depth_mode, color_resolution)
        # print(dir(calibration))

        # depth_height, depth_width = depth_image.shape[:2]
        # color_height, color_width = color_image.shape[:2]

        # transformed_color_image = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)

        # for y in range(depth_height):
        #     for x in range(depth_width):
        #         # Transformiere die Depth-Koordinate in Color-Koordinate
        #         color_point = calibration.convert_depth_2d_to_color_2d((x, y), depth_image[y, x])
        #         cx, cy = int(color_point[0]), int(color_point[1])
        #         if 0 <= cx < color_width and 0 <= cy < color_height:
        #             transformed_color_image[y, x] = color_image[cy, cx]

        # depth_height, depth_width = depth_image.shape[:2]
        # color_height, color_width = color_image.shape[:2]

        # # Ausgabe-Array in Depth-Auflösung (RGB)
        # transformed_color = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)

        # for y in range(depth_height):
        #     for x in range(depth_width):
        #         depth_value = depth_image[y, x]
        #         # Falls Tiefenwert 0 ist (kein Tiefenwert), überspringen
        #         if depth_value == 0:
        #             continue

        #         # Konvertiere Depth 2D-Koordinate zu Color 2D-Koordinate
        #         color_coord = calibration.convert_depth_2d_to_color_2d((x, y), depth_value)
        #         cx, cy = int(round(color_coord[0])), int(round(color_coord[1]))

        #         if 0 <= cx < color_width and 0 <= cy < color_height:
        #             transformed_color[y, x] = color_image[cy, cx]
        
        
        # depth_height, depth_width = depth_image.shape[:2]
        # transformed_color = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
        
        # color_height, color_width = color_image.shape[:2]
        #     # Angenommen, du hast eine capture-Instanz:
        # capture = self.device.get_capture()
        # depth_image = pykinect.k4a.capture_get_depth_image(capture)

        # # hole das k4a_image für depth:
        # print(dir(capture))
        # depth_image = capture.depth

        # # hole das k4a_image für color:    
        # # Für jedes Color-Pixel konvertiere zu Depth-Pixel
        # for y_c in range(color_height):
        #     for x_c in range(color_width):
                
        #         depth_coord = calibration.convert_color_2d_to_depth_2d(pykinect.k4a_float2_t((x_c, y_c)), depth_image)
        #         x_d, y_d = int(round(depth_coord[0])), int(round(depth_coord[1]))
        #         if 0 <= x_d < depth_width and 0 <= y_d < depth_height:
        #             transformed_color[y_d, x_d] = color_image[y_c, x_c]

        # return transformed_color
        depth_height, depth_width = depth_image.shape[:2]
        # Resize
        resized_bgra = cv2.resize(color_image, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
        # BGRA -> RGB (entfernt Alpha-Kanal und Reihenfolge BGR -> RGB)
        resized_rgb = cv2.cvtColor(resized_bgra, cv2.COLOR_BGRA2RGB)

        return resized_rgb
        # return transformed_color_image



if __name__ == "__main__":
    rs = Azure(device_id=1)
    rs.connect()

    for i in range(50):
        img = rs._get_sensors()
        if img['rgb'] is not None:
            print("Received image{} of size:".format(i), img['rgb'].shape, flush=True)
            cv2.imshow("rgb", img['rgb'])
            cv2.waitKey(1)

        if img['rgb'] is None:
            print(img)

        time.sleep(0.1)

    rs.close()