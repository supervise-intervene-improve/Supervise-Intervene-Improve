from runtime_impl import main


if __name__ == "__main__":
    main()


# ============================================================
# LAUNCH VARIABLES  (set these before running)
# ============================================================
#
# --- Network mode: pick ONE of the two HOST_IP blocks below ---
#
# Option A: WiFi  (Linux PC and Quest on same WiFi network)
#   $HOST_IP = "192.168.0.208"   # <- Linux PC's WiFi IP  (ip -4 addr show wlan0)
#
# Option B: USB-C tether  (no WiFi needed; Quest connected via USB-C cable)
#   Step 1 - on the Linux PC (one-time per boot, run as root or with sudo):
#     sudo ip link set usb0 up          # interface may be rndis0 / eth1 - check with: ip link show
#     sudo dhclient usb0                # get IP via DHCP from the Quest
#     ip -4 addr show usb0             # note the IP assigned to the PC  -> use it below
#   Step 2 - set HOST_IP to that IP:
#   $HOST_IP="192.168.0.38"           # <- PC's USB interface IP from step 1
#   Note: --bind_ip 0.0.0.0 already listens on all interfaces so no other change needed.
#   Note: Quest 3 must have developer mode ON and USB debugging allowed.
#
# Quick transport flags (runtime_impl.py):
#   --transport wifi
#   --transport usb --usb_auto_reverse
# Optional in USB mode:
#   --usb_adb_serial <adb_serial>
#   --usb_extra_reverse_ports 7721
#   --usb_publisher_ip 127.0.0.1   (default)
#
# Linux USB example (auto adb reverse for topic/service ports):
# .venv/bin/python ./SimPublisher/sii/integration_v1/intervention_vr_runtime.py \
#   --xml "$LAB_XML" --trajectory "$TRAJECTORY" --host "$HOST_IP" --unity_node "$UNITY_NODE" \
#   --transport usb --usb_auto_reverse --bind_ip 0.0.0.0

# $UNITY_NODE="MQ3-2"
# $ROBOT_KEY="p1"
#
# --- XML options (pick one) ---
# $LAB_XML=".\intervene_base\mujoco_scenes\working_scenes\with_soft_gripper\sii_scene_table_T_shape_multiwindow_fast.xml"
# $LAB_XML = ".\LAB\lab2_boxes_cups_merged.xml"
# $LAB_XML = ".\LAB\lab3_stick_maze_merged.xml"
#
# --- Trajectory (match the lab XML you chose) ---
# $TRAJECTORY = ".\intervene_base\teleop_logs\p1_traj_1772714528.npz"


# --- Sim-only (no real robot) ---

# .\.venv\Scripts\python.exe .\SimPublisher\sii\integration_v1\intervention_vr_runtime.py `
#   --xml "$LAB_XML" `
#   --trajectory "$TRAJECTORY" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --rgb_cams wrist `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 50000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --log_every 30 `
#   --show_mujoco_window `
#   --mujoco_window_cam wrist `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 `
#   --start_paused


# --- With real robot mirror + replanning (append to the command above) ---
# IMPORTANT: always include --start_paused with --mirror_robot.
# The robot will NOT move until you press A (or Space) to resume replay.
#
# Robot communication runs on a BACKGROUND THREAD (RobotThread) so that
# blocking network calls to the Franka arm never stall sensor publishing
# (wrist camera, point clouds). The thread is created automatically when
# --mirror_robot is set. No extra flags needed.
#
#   --mirror_robot `
#   --robot_key "$ROBOT_KEY" `
#   --control_hz 120 `
#   --start_paused

# Robot keys:  p1=172.16.1.1  p2=172.16.2.2  p3/p4=192.0.2.153


# Controls (keyboard or VR controller):
#   A / Space      = pause / resume  (mirror starts on resume if --mirror_robot)
#   B / Backspace  = reset to start frame
#   X / Enter      = (paused)    enter replan-hol 
#                    (hold-1st)  confirm hold  ("grab the robot now")
#                    (hold-2nd)  start human-guidance recording
#                    (recording) finish, stitch NPZ, reload trajectory
#   Y / Esc        = cancel replan at any stage  /  pause if running
#
# Replan is TWO X presses when robot is connected (safety):
#   1st X = confirms hold, operator grabs the robot
#   2nd X = switches to HUMAN_CONTROL and starts recording
# This prevents the robot from jumping when mode changes.


# MuJoCo window:
#   --show_mujoco_window       enables the GLFW window (also needed for keyboard input)
#   --mujoco_window_cam top    pick which camera to display (default: first in --cams)
#   The window shows ONE stable camera. Without --mujoco_window_cam it uses the
#   first camera in --cams (typically "wrist").


# Optional first-frame Quest point-cloud visibility dump:
#   --pc_debug_visibility --pc_debug_visibility_cams top right left


# --- Variant: no table clipping (use when table plane detection is unreliable) ---

# .\.venv\Scripts\python.exe .\SimPublisher\sii\integration_v1\intervention_vr_runtime.py `
#   --xml "$LAB_XML" `
#   --trajectory "$TRAJECTORY" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --rgb_cams wrist `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 50000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --log_every 30 `
#   --show_mujoco_window `
#   --mujoco_window_cam wrist `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 `
#   --start_paused


# --- Variant: lower table clearance + lab2 home pose fix ---
# Use --robot_start_q to override the initial robot pose (lab2 trajectory starts from a
# different arm configuration; these values match the lab1 frame-0 home position).

# .\.venv\Scripts\python.exe .\SimPublisher\sii\integration_v1\intervention_vr_runtime.py `
#   --xml "$LAB_XML" `
#   --trajectory "$TRAJECTORY" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --rgb_cams wrist `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 50000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.010 --pc_top_table_clearance 0.030 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --log_every 30 `
#   --show_mujoco_window `
#   --mujoco_window_cam wrist `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 `
#   --robot_start_q -0.119 0.203 -0.052 -2.062 0.024 1.992 -0.879 `
#   --start_paused


# ============================================================
# THREADING ARCHITECTURE (for reference)
# ============================================================
#
# When --mirror_robot is set, a background thread ("RobotComm") handles
# all Franka arm communication. This is critical for Linux operation where
# network calls to the robot can block for 50-500ms+, which would otherwise
# starve the wrist camera and point cloud publishing to the Quest headset.
#
# Main thread (main loop @ control_hz):
#   - MuJoCo physics stepping
#   - Trajectory replay / frame advancing
#   - Sensor publishing (cameras, point clouds) via ZMQ
#   - MuJoCo window rendering (GLFW)
#   - VR controller / keyboard input polling
#
# Robot thread (RobotThread, daemon):
#   - Processes command queue: mirror_command, hold_pose (fire-and-forget)
#   - Synchronous transitions: start_human_guidance, ensure_mode (with Event)
#   - Continuous state reading when recording (get_state() from main thread)
#
# Data flow:
#   Main -> Robot:  queue.Queue (commands)
#   Robot -> Main:  threading.Lock protected dict (latest robot state)
#
# In sim-only mode (no --mirror_robot), no thread is created. robot_thread=None.




# .\.venv\Scripts\python.exe .\SimPublisher\sii\integration_v1\intervention_vr_runtime.py `
#   --xml "$LAB_XML" `
#   --trajectory "$TRAJECTORY" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --rgb_cams wrist `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 50000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --log_every 30 `
#   --show_mujoco_window `
#   --mujoco_window_cam wrist `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 `
#   --mirror_robot `
#   --robot_key "$ROBOT_KEY" `
#   --control_hz 120 `
#   --start_paused


# LAB_XML="intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml"
# TRAJECTORY="intervene_base/teleop_logs/p1_traj_1772714528.npz"
# HOST_IP="192.168.0.38"
# UNITY_NODE="MQ3-2"
# ROBOT_KEY="p1"

# .venv/bin/python ./SimPublisher/sii/integration_v1/intervention_vr_runtime.py \
#   --xml "$LAB_XML" \
#   --trajectory "$TRAJECTORY" \
#   --host "$HOST_IP" \
#   --unity_node "$UNITY_NODE" \
#   --bind_ip 0.0.0.0 \
#   --visible_geoms_groups 2 \
#   --fps 30 --w 960 --h 720 \
#   --cams wrist top right left \
#   --rgb_cams wrist \
#   --pc --pc_cams top right left \
#   --pc_sampling stable_random \
#   --pc_stride 8 --pc_max_points 50000 \
#   --pc_min_depth 0.08 --pc_max_depth 2.8 \
#   --pc_intrinsics_mode mujoco \
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 \
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 \
#   --cam_extrinsics_profile lab_standard \
#   --pc_no_anchor_auto_translate \
#   --log_every 30 \
#   --show_mujoco_window \
#   --object_control \
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 \
#   --mirror_robot \
#   --robot_key "$ROBOT_KEY" \
#   --control_hz 120 \
#   --start_paused


# LAB_XML="intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml"
# TRAJECTORY="intervene_base/teleop_logs/p1_traj_1772714528.npz"
# HOST_IP="192.168.0.38"
# UNITY_NODE="MQ3-2"
# ROBOT_KEY="p1"

# .venv/bin/python ./SimPublisher/sii/integration_v1/intervention_vr_runtime.py \
#   --xml "$LAB_XML" \
#   --trajectory "$TRAJECTORY" \
#   --host "$HOST_IP" \
#   --unity_node "$UNITY_NODE" \
#   --bind_ip 0.0.0.0 \
#   --visible_geoms_groups 2 \
#   --fps 30 --w 960 --h 720 \
#   --cams wrist top right left \
#   --rgb_cams wrist \
#   --pc --pc_cams top right left \
#   --pc_sampling stable_random \
#   --pc_stride 8 --pc_max_points 50000 \
#   --pc_min_depth 0.08 --pc_max_depth 2.8 \
#   --pc_intrinsics_mode mujoco \
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 \
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 \
#   --cam_extrinsics_profile lab_standard \
#   --pc_no_anchor_auto_translate \
#   --log_every 30 \
#   --show_mujoco_window \
#   --object_control \
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 \
#   --mirror_robot \
#   --transport usb \
#   --usb_auto_reverse \
#   --usb_extra_reverse_ports 7721 \
#   --robot_key "$ROBOT_KEY" \
#   --control_hz 120 \
#   --start_paused
