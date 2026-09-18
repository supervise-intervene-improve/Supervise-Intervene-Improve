# Environment Setup

The system uses **two Python environments**. Do not merge them.

| Environment | Used by | Python | Main contents |
|---|---|---|---|
| Policy / robot (conda env `polymetis`) | `intervene_base/utils/run_main_policy.sh` | 3.10 | lerobot (ACT), PyTorch (CPU), MuJoCo, Polymetis, FACTR service |
| VR streaming (`./.venv`) | `run_multi_window_robot.sh` | 3.13 | MuJoCo runtimes, ZMQ streaming, CUDA Open3D point clouds |

Tested on Ubuntu 22.04 (x86_64) with an NVIDIA GPU.

## Files in this folder

| File | Content |
|---|---|
| `conda_policy_polymetis_linux64_explicit.txt` | Exact conda package list of the policy env (best Linux → Linux reproduction) |
| `requirements_policy_polymetis_py310_portable.txt` | Portable pip pins for the policy env |
| `requirements_policy_polymetis_py310_lock.txt` | Raw `pip freeze` of the policy env |
| `requirements_vr_py313_portable.txt` | Portable pip pins for the VR env |
| `requirements_vr_py313_lock.txt` | Raw `pip freeze` of the VR env |
| `requirements_*_cuda_python_wheels.txt` | CUDA-related wheels found in each env |
| `system_cuda_nvidia_dpkg_snapshot.txt`, `system_cuda_paths_and_versions.txt` | Driver / CUDA toolkit snapshot of the test machine |
| `polymetis_working_machine_capture.txt`, `polymetis_reference/` | Polymetis install capture |

## System packages

```bash
sudo apt install -y build-essential cmake ninja-build git git-lfs pkg-config \
  libgl1 libglib2.0-0 libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6 \
  libegl1 libglfw3 libusb-1.0-0 libasound2-dev portaudio19-dev adb net-tools iproute2
```

For the GPU point-cloud backend you need a working NVIDIA driver (`nvidia-smi`) and a
CUDA toolkit compatible with your Open3D build (the test machine used CUDA 12.8).

## 1. Policy / robot environment

```bash
conda create -n polymetis --file env/conda_policy_polymetis_linux64_explicit.txt
conda activate polymetis
# or, if the explicit list does not resolve on your machine:
#   conda create -n polymetis python=3.10 -y && conda activate polymetis
python -m pip install -r env/requirements_policy_polymetis_py310_portable.txt
python -m pip install -e SimPublisher
```

**Polymetis.** You only need Polymetis for the KT condition (real Franka). The
system was run with a Polymetis fork installed as an editable `polymetis==0.2`:

```bash
python -m pip install -e /path/to/polymetis/polymetis
python -c "import polymetis, torchcontrol; print(polymetis.__file__)"
```

`run_main_policy.sh` also picks up a checkout at `<repo>/src/polymetis/polymetis/python`,
or `POLYMETIS_PYTHON_DIR=<dir>`. The VR launcher loads Polymetis' native libraries
(`libtorchscript_pinocchio.so`, `libtorchrot.so`) from `POLYMETIS_LIB_DIR`, which
defaults to `~/miniforge3/envs/polymetis/lib` or `~/miniconda3/envs/polymetis/lib`.

Sanity check: `START_VR=0 MAX_STEPS=1 bash intervene_base/utils/run_main_policy.sh`.

## 2. VR streaming environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r env/requirements_vr_py313_portable.txt
python -m pip install -e SimPublisher
```

**Open3D with CUDA.** The GPU point-cloud backend needs Open3D 0.19.0 built with
`BUILD_CUDA_MODULE=ON` for the venv's Python ABI (cp313). Install the resulting
wheel into the venv, or point `OPEN3D_PYTHON_PACKAGE` at `<open3d_build>/lib/python_package`.
Without CUDA Open3D (`pip install open3d==0.19.0`), `run_multi_window_robot.sh` selects
the CPU `legacy` backend automatically.

```bash
python -c "import open3d as o3d; print(o3d.__version__, o3d.core.cuda.is_available())"
python -c "import mujoco, glfw, zmq; print('ok')"
adb devices   # the Quest should be listed
```

PyTorch in both environments is the CPU build. CUDA is used only for point-cloud
generation.

## 3. Hardware-specific configuration

- **Robot IPs / ports:** see `ROBOTS` in `intervene_base/record.py` and choose the entry
  with `ROBOT_KEY` / `INTERVENE_ROBOT_KEY`.
- **FACTR:** `intervene_base/factr/leader.yaml`. The launcher reads the rest and start poses
  from `factr/validation/rest_pose.json` and `factr/validation/init_pose.json`
  (override them with `FACTR_REST_POSE` / `FACTR_INIT_POSE`). These files are specific
  to one leader arm. Capture your own with `utils/save_factr_pose.sh`, or start from the
  poses used in the studies in `factr/validation/candidates/`. For smooth teleoperation, set the USB-serial latency timer to 1 ms:
  `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`.
- **Headset:** Quest and PC on the same Wi-Fi subnet. The headset finds the PC through a UDP
  beacon on port 8720.
