import os

# ========= CONFIG =========

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_PATH = os.path.join(REPO_ROOT, "mujoco_scenes", "assets", "wire_base")

MODEL_NAME = "wire_base_via_stl"
BODY_NAME = "object"
VISUAL_OBJ = "Basic_Model_Body.obj"
COLLISION_PREFIX = "Basic_Model_Body_collision_"
NUM_COLLISION = 4

# If your OBJ files are in millimeters, keep this True.
# It converts scales like 1.0 -> 0.001 for MuJoCo meters.
USE_MM = True
MM_TO_M = 0.001

BASE_SCALE = "1.0 1.0 1.0"
BASE_POS = "0.6 -0.2 0.22"
BASE_HAS_FREE_JOINT = True

OUTPUT_FILE = os.path.join(
    REPO_ROOT,
    "mujoco_scenes",
    "generated_wire_game",
    "generated_base_via_stl.xml",
)


# ========= HELPERS =========

def apply_unit_scale(scale_str):
    sx, sy, sz = map(float, scale_str.split())

    if USE_MM:
        sx *= MM_TO_M
        sy *= MM_TO_M
        sz *= MM_TO_M

    return f"{sx:g} {sy:g} {sz:g}"


def validate_assets():
    needed = [VISUAL_OBJ]
    needed.extend(f"{COLLISION_PREFIX}{i}.obj" for i in range(NUM_COLLISION))

    missing = [
        os.path.join(BASE_PATH, filename)
        for filename in needed
        if not os.path.exists(os.path.join(BASE_PATH, filename))
    ]

    if missing:
        raise FileNotFoundError("Missing wire base mesh files:\n" + "\n".join(missing))


# ========= GENERATION =========

def generate_defaults():
    return """  <default>

    <default class="visual_wire_base">
      <geom group="2" type="mesh" contype="0" conaffinity="0"/>
    </default>

    <default class="collision_wire_base">
      <geom group="3" type="mesh" rgba="0 0 0 0"
        contype="1"
        conaffinity="1"
        condim="1"
        friction="0 0 0"
        solimp="0.95 0.99 0.001 0.5 2"
        solref="0.01 1"
        margin="0"
        gap="0"/>
    </default>

  </default>"""


def generate_assets():
    scale = apply_unit_scale(BASE_SCALE)

    lines = []
    lines.append("  <asset>")
    lines.append('    <material name="wire_base_orange" rgba="1 0.356 0.133 1"/>')
    lines.append(
        f'    <mesh name="wire_base_visual" '
        f'file="{os.path.join(BASE_PATH, VISUAL_OBJ)}" '
        f'scale="{scale}"/>'
    )

    for i in range(NUM_COLLISION):
        lines.append(
            f'    <mesh name="wire_base_col_{i}" '
            f'file="{os.path.join(BASE_PATH, f"{COLLISION_PREFIX}{i}.obj")}" '
            f'scale="{scale}"/>'
        )

    lines.append("  </asset>")
    return "\n".join(lines)


def generate_worldbody():
    lines = []
    lines.append("  <worldbody>")
    lines.append(f'    <body name="{BODY_NAME}" pos="{BASE_POS}">')

    if BASE_HAS_FREE_JOINT:
        lines.append('      <joint name="wire_base_free" type="free"/>')

    lines.append(
        '      <geom name="wire_base_visual" '
        'mesh="wire_base_visual" '
        'class="visual_wire_base" '
        'material="wire_base_orange"/>'
    )

    for i in range(NUM_COLLISION):
        lines.append(
            f'      <geom name="wire_base_col_{i}" '
            f'mesh="wire_base_col_{i}" '
            f'class="collision_wire_base"/>'
        )

    lines.append("    </body>")
    lines.append("  </worldbody>")
    return "\n".join(lines)


def generate_xml():
    xml = []
    xml.append(f'<mujoco model="{MODEL_NAME}">')
    xml.append("")
    xml.append(generate_defaults())
    xml.append("")
    xml.append(generate_assets())
    xml.append("")
    xml.append(generate_worldbody())
    xml.append("</mujoco>")

    return "\n".join(xml)


# ========= RUN =========

if __name__ == "__main__":
    validate_assets()
    xml_content = generate_xml()

    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"MJCF file generated: {OUTPUT_FILE}")
