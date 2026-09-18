import os

# ========= CONFIG =========

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASE_PATH = os.path.join(REPO_ROOT, "mujoco_scenes", "assets", "spoon")

MODEL_NAME = "spoon_via_stl"
BODY_NAME = "spoon1"
VISUAL_OBJ = "Spoon_wider.obj"
COLLISION_PREFIX = "Spoon_wider_collision_"
NUM_COLLISION = 17

# If your OBJ files are in millimeters, keep this True.
# It converts scales like 1.0 -> 0.001 for MuJoCo meters.
USE_MM = True
MM_TO_M = 0.001

SPOON_SCALE = "1.0 1.0 1.0"
SPOON_POS = "0.4 -0.1 0.3"
SPOON_EULER = "1.5708 0 1.5708"

OUTPUT_FILE = os.path.join(
    REPO_ROOT,
    "mujoco_scenes",
    "generated_wire_game",
    "generated_spoon_via_stl.xml",
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
        raise FileNotFoundError("Missing spoon mesh files:\n" + "\n".join(missing))


# ========= GENERATION =========

def generate_defaults():
    return """  <default>

    <default class="visual_spoon">
      <geom group="2" type="mesh" contype="0" conaffinity="0"/>
    </default>

    <default class="collision_spoon">
      <geom group="3" type="mesh" rgba="0 0 1 0.25"
        contype="1"
        conaffinity="1"
        condim="3"
        friction="1.2 0.015 0.0015"
        solimp="0.97 0.995 0.001 0.5 2"
        solref="0.008 1"/>
    </default>

  </default>"""


def generate_assets():
    scale = apply_unit_scale(SPOON_SCALE)

    lines = []
    lines.append("  <asset>")
    lines.append(
        '    <material name="spoon_green" rgba="0.223 0.392 0.278 1"/>'
    )
    lines.append(
        f'    <mesh name="{BODY_NAME}_visual" '
        f'file="{os.path.join(BASE_PATH, VISUAL_OBJ)}" '
        f'scale="{scale}"/>'
    )

    for i in range(NUM_COLLISION):
        lines.append(
            f'    <mesh name="{BODY_NAME}_col_{i}" '
            f'file="{os.path.join(BASE_PATH, f"{COLLISION_PREFIX}{i}.obj")}" '
            f'scale="{scale}"/>'
        )

    lines.append("  </asset>")
    return "\n".join(lines)


def generate_worldbody():
    lines = []
    lines.append("  <worldbody>")
    lines.append(
        f'    <body name="{BODY_NAME}" pos="{SPOON_POS}" euler="{SPOON_EULER}">'
    )
    lines.append(f'      <joint name="{BODY_NAME}_free" type="free"/>')
    lines.append(
        f'      <geom name="{BODY_NAME}_visual" '
        f'mesh="{BODY_NAME}_visual" '
        f'class="visual_spoon" '
        f'material="spoon_green"/>'
    )

    for i in range(NUM_COLLISION):
        lines.append(
            f'      <geom name="{BODY_NAME}_col_{i}" '
            f'mesh="{BODY_NAME}_col_{i}" '
            f'class="collision_spoon"/>'
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
