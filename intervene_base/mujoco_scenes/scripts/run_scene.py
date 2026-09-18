# run_scene.py
import mujoco
import mujoco.viewer

def main():
    xml_path = "sii_scene_table_cups.xml"
    # xml_path = "sii_scene_table_T_shape.xml"
    # xml_path = "mjx_panda.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()