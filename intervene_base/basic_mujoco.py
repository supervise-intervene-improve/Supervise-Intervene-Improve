import time
import mujoco
import mujoco.viewer
from simpub.sim.mj_publisher import MujocoPublisher

# XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/generated_panda/soft_hand.xml"
# XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/generated_panda/sii_panda_model_soft_gripper.xml"
XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
publisher = MujocoPublisher(model, data, host="192.168.0.38", visible_geoms_groups=[0, 2])


mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)