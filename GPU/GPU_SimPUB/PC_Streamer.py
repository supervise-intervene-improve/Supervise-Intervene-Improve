import numpy as np
import open3d as o3d
import os
import time
import Source as s
import Action as a
import os
import yaml
from contextlib import contextmanager
import argparse

# ArgumentParser erstellen
parser = argparse.ArgumentParser(description="Parameter für das Programm.")

# Argumente definieren
parser.add_argument(
    "--quality",
    type=str,
    choices=["HIGH", "MEDIUM", "LOW"],
    default="HIGH",
    help='Qualität: "HIGH", "MEDIUM" oder "LOW"'
)
parser.add_argument(
    "--use_segmentation",
    type=lambda x: (str(x).lower() == 'true'),  # True/False Argument
    default=True,
    help="Segmentation verwenden: True oder False"
)
parser.add_argument(
    "--max_points",
    type=int,
    default=45000,
    help="Maximale Punktanzahl"
)
args = parser.parse_args()



@contextmanager
def track_time(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # print(f"[{label}] {elapsed:.4f} s")

def build_tensor_pc(posM, colM=None, device="CPU:0"):
    """
    Create o3d.t.geometry.PointCloud with private storage.
    Accepts NumPy or o3d.core.Tensor. Never aliases shared memory.
    """
    dev = o3d.core.Device(device)

    if isinstance(posM, np.ndarray):
        pos = o3d.core.Tensor(posM.copy(), dtype=o3d.core.Dtype.Float32, device=dev)
    elif isinstance(posM, o3d.core.Tensor):
        pos = posM.to(dev, copy=True)  # force private buffer
    else:
        raise TypeError("posM must be np.ndarray or o3d.core.Tensor")

    pc = o3d.t.geometry.PointCloud(dev)
    pc.point["positions"] = pos

    if colM is not None:
        if isinstance(colM, np.ndarray):
            col = o3d.core.Tensor(colM.copy(), dtype=o3d.core.Dtype.Float32, device=dev)
        elif isinstance(colM, o3d.core.Tensor):
            col = colM.to(dev, copy=True)
        else:
            raise TypeError("colM must be np.ndarray or o3d.core.Tensor")
        pc.point["colors"] = col

    return pc

def crop_tensor_pc(pc, lo_xyz, hi_xyz):
    # Crop with AABB (fast)
    aabb = o3d.t.geometry.AxisAlignedBoundingBox(
        o3d.core.Tensor(lo_xyz, dtype=o3d.core.Dtype.Float32, device=pc.device),
        o3d.core.Tensor(hi_xyz, dtype=o3d.core.Dtype.Float32, device=pc.device),
    )
    return pc.crop(aabb)

def voxel_downsample_tensor(pc, voxel_size):
    """Voxel grid downsample on tensor PC; returns a new PC."""
    return pc.voxel_down_sample(voxel_size)

def save_pointcloud(pcd_tensor, filename, output_folder):
    # Punkte extrahieren
    points = pcd_tensor.point.positions.cpu().numpy()
    
    # Farben extrahieren, falls vorhanden
    colors = None
    if "colors" in pcd_tensor.point:
        colors = pcd_tensor.point.colors.cpu().numpy()
    
    # Legacy PointCloud erstellen
    pcd_legacy = o3d.geometry.PointCloud()
    pcd_legacy.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd_legacy.colors = o3d.utility.Vector3dVector(colors)
    
    # Speichern
    path = os.path.join(output_folder, filename)
    o3d.io.write_point_cloud(path, pcd_legacy)
    print(f"Saved {path}")

def load_extrinsics_yml(file_path, inverse=False):
    """
    Loads the extrinsic transformation matrices and initial EE pose from a YML file.
    Optionally returns their inverses.

    :param file_path: Path to the YML file.
    :param inverse: If True, return inverses of the transforms.
    :return: Dictionary of transformation matrices (4x4 np.arrays), initial EE pose.
    """
    with open(file_path, "r") as file:
        transform = yaml.load(file, Loader=yaml.SafeLoader)

    # Convert all transforms to numpy arrays and optionally invert
    transforms_np = {}
    for k, v in transform.items():
        mat = np.array(v, dtype=float)
        transforms_np[k] = np.linalg.inv(mat) if inverse else mat

    print(f"✅ transform{' (inverses)' if inverse else ''} loaded from {file_path}")
    return transforms_np



WRIST_ALGORITHM_CHOOSE = "LM_extrinsics.yml" # 0_extrinsics.yml , 1_extrinsics.yml , 2_extrinsics.yml , LM_extrinsics.yml
ALGORITHM_CHOOSE = "1_extrinsics.yml" # 0_extrinsics.yml , 1_extrinsics.yml , 2_extrinsics.yml , static_LM_extrinsics.yml

CAM_MAIN_NAME = "RealSense" # "RealSense" , "Azure"
MAIN_PATH = "/home/user/Documents/paper_anonymous_anonymous/calibration/" + CAM_MAIN_NAME + "/extrinsic/"

INTERVAL_NAME = "2025_10_14-18_22_12" # "2025_05_14-09_18_38" , "2025_05_13-11_09_12" , "2025_05_13-11_39_27" , "combined"

WANT_T0_USE_MANUEL_INTRINSIC = False # 2025_05_14-09_18_38

TRANSFORM_STATIC_GLOBAL = MAIN_PATH + INTERVAL_NAME + "/"
TRANSFORM_WRIST_GLOBAL = MAIN_PATH + INTERVAL_NAME + "/" + "wrist_to_main_" + WRIST_ALGORITHM_CHOOSE

T_x2b = load_extrinsics_yml(TRANSFORM_WRIST_GLOBAL)

T_s2s_left_to_main = load_extrinsics_yml(TRANSFORM_STATIC_GLOBAL + "left_to_main_" + ALGORITHM_CHOOSE)
T_s2s_upper_to_main = load_extrinsics_yml(TRANSFORM_STATIC_GLOBAL + "upper_to_main_" + ALGORITHM_CHOOSE)

T = np.eye(4)
T[:3, 3] = [-0.5, 0.0, 0.0]


a.start_discovery_server()
sending = a.ZMQPublishPointCloudAction()


QUALITY = args.quality
USE_SEGMENTATION = args.use_segmentation # For Segmentation the YOLO server must running (see documentation)
MAX_POINTS = args.max_points

upper_id = "007522061936"
left_id  = "243322073029"
right_id = "944622073668"

if QUALITY == "HIGH":
    width = 1280
    height = 720
    fps = 30
elif QUALITY == "MEDIUM":
    width = 848
    height = 480
    fps = 60
elif QUALITY == "LOW":
    width = 640
    height = 480
    fps = 60

upper = s.RealSenseCamera(id=upper_id, org_width=width, org_height=height, res_width=width, res_height=height, fps=fps, use_imgs=False, use_segmentation=USE_SEGMENTATION, use_pc_creation=True)
left  = s.RealSenseCamera(id=left_id, org_width=width, org_height=height, res_width=width, res_height=height, fps=fps, use_imgs=False, use_segmentation=USE_SEGMENTATION, use_pc_creation=True)
right = s.RealSenseCamera(id=right_id, org_width=width, org_height=height, res_width=width, res_height=height, fps=fps, use_imgs=False, use_segmentation=USE_SEGMENTATION, use_pc_creation=True)

try:
    while True:
        start_total = time.time()
        dev = "CUDA:0"  # oder "CPU:0"
        pcd_upper = None
        pcd_right = None
        pcd_left = None


        with track_time("------ READING ------ "):
            while True:
                if pcd_upper is None:
                    pcd_upper  = upper.get_pcd(dev)
                if pcd_left is None:
                    pcd_left = left.get_pcd(dev)
                if pcd_right is None:
                    pcd_right = right.get_pcd(dev)
                if pcd_upper is not None and pcd_left is not None and pcd_right is not None:
                    break
                time.sleep(0.005)


        with track_time("------ TRANSFORMATION ------ "):
            pcd_upper.transform(T_s2s_upper_to_main["T_follower_to_main"])
            pcd_upper.transform(T_x2b["T_static_to_base"])

            pcd_left.transform(T_s2s_left_to_main["T_follower_to_main"])
            pcd_left.transform(T_x2b["T_static_to_base"])

            pcd_right.transform(T_x2b["T_static_to_base"])


        with track_time("------ MERGE ------ "):

            posU = pcd_upper.point["positions"]
            posL = pcd_left.point["positions"]
            posR = pcd_right.point["positions"]

            posM = o3d.core.concatenate((posU, posL, posR), axis=0)

            if "colors" in pcd_upper.point and "colors" in pcd_left.point and "colors" in pcd_right.point:
                colU = pcd_upper.point["colors"]
                colL = pcd_left.point["colors"]
                colR = pcd_right.point["colors"]
                
                colM = o3d.core.concatenate((colU, colL, colR), axis=0)
            else:
                colM = None


        with track_time("------ CREATE NEW PCD ------ "):

            pcd_merged = o3d.t.geometry.PointCloud(posM.device)
            pcd_merged.point["positions"] = posM
            if colM is not None:
                pcd_merged.point["colors"] = colM


        with track_time("------ APPLY BBOX ------ "):

            if isinstance(pcd_merged, o3d.t.geometry.PointCloud):
                pcd_merged.transform(o3d.core.Tensor(T, o3d.core.Dtype.Float32, device=pcd_merged.device))
                pcd_cropped = crop_tensor_pc(
                    pcd_merged, lo_xyz=(-0.4, -0.4, 0.0175), hi_xyz=(0.275, 0.4, 1.0)
                )


        with track_time("------ VOXEL DOWN SAMPLE ------ "):
            if QUALITY == "HIGH":
                voxel = 0.00175
            elif QUALITY == "MEDIUM":
                voxel = 0.0015
            elif QUALITY == "LOW":
                voxel = 0.001

            if voxel == 0.001:
                pcd_downsampled = pcd_cropped
            else:
                pcd_downsampled = voxel_downsample_tensor(pcd_cropped, voxel)


        with track_time("------ EUQLIZE PCD ------ "):
            # pcd_downsampled may be tensor *or* legacy depending on your path
            if isinstance(pcd_downsampled, o3d.t.geometry.PointCloud):
                # already tensor → keep it
                pcd_merged = pcd_downsampled
            else:
                # legacy → make sure it's on CPU before from_legacy
                try:
                    # legacy CUDA -> CPU (no-op if already CPU in many builds)
                    pcd_downsampled = pcd_downsampled.cpu()
                except AttributeError:
                    pass  # some builds lack .cpu(); it's fine if already on CPU

                # convert legacy(CPU) -> tensor (place result on same device as posM)
                dev = posM.device if isinstance(posM, o3d.core.Tensor) else o3d.core.Device("CPU:0")
                pcd_merged = o3d.t.geometry.PointCloud.from_legacy(
                    pcd_downsampled, device=dev
                )


        with track_time("------ FILTER ------ "):
            
            if QUALITY == "HIGH":
                pcd_merged, _ = pcd_merged.remove_radius_outliers(nb_points=10, search_radius=voxel*1.75)
            
            elif QUALITY == "MEDIUM":
                pcd_merged, _ = pcd_merged.remove_radius_outliers(nb_points=10, search_radius=voxel*2.25)

                pcd_merged, _ = pcd_merged.remove_statistical_outliers(
                    nb_neighbors=20,
                    std_ratio=0.5
                )

            elif QUALITY == "LOW":
                pcd_merged, _ = pcd_merged.remove_radius_outliers(nb_points=10, search_radius=voxel*2.5)

                pcd_merged, _ = pcd_merged.remove_statistical_outliers(
                    nb_neighbors=20,
                    std_ratio=0.5
                )


        with track_time("------ MOVE TO CPU ------ "):

            xyz = pcd_merged.point["positions"].cpu().numpy()
            rgb = pcd_merged.point["colors"].cpu().numpy()


        with track_time("------ CORRECTION ------ "):

            if QUALITY == "HIGH":
                scale = np.array([1.45, 1.425, 1.55], dtype=xyz.dtype)
            elif QUALITY == "MEDIUM":
                scale = np.array([1.1, 1.1, 1.3], dtype=xyz.dtype)
            elif QUALITY == "LOW":
                scale = np.array([1.4, 1.4, 1.5], dtype=xyz.dtype)
            xyz *= scale
            xyz[:, 0] *= -1
            rgb = rgb[:, ::-1]  # swap BGR → RGB


        with track_time("------ SENDING ------ "):
            num_points = xyz.shape[0]

            if num_points > MAX_POINTS:
                idx = np.random.choice(num_points, MAX_POINTS, replace=False)
                xyz_sampled = xyz[idx]
                rgb_sampled = rgb[idx]
            else:
                xyz_sampled = xyz
                rgb_sampled = rgb

            sending.set_pointcloud(xyz_sampled, rgb_sampled)
            sending.execute(num_points=xyz_sampled.shape[0])

        end_total =  time.time()

except KeyboardInterrupt:
    pass