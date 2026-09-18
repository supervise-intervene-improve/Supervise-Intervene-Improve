import os

# ========= CONFIG =========

BASE_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/assets/cup"

OBJ_NAME = "the_mug.obj"
COLLISION_PREFIX = "the_mug_collision_"
NUM_COLLISION = 13

# If your OBJ files are in millimeters, keep this True.
# It converts scales like 1.0 -> 0.001 for MuJoCo meters.
USE_MM = True
MM_TO_M = 0.001

VERY_SMALL_CUP_SCALE = "0.75 0.75 1.0"
SMALL_CUP_SCALE = "0.95 0.95 1.0"
MIDDLE_CUP_SCALE = "1.0 1.0 1.0"
BIG_CUP_SCALE = "1.3 1.30 1.0"

# Keep visual mesh scales unchanged. This multiplier is applied only to
# collision meshes, making contacts a little more forgiving.
#
# Z stays at 1.0 ON PURPOSE. The cups OOD pipeline encodes its signal in cup mesh
# scale[2] and requires a single Z per cup across all 14 of its meshes -- see
# `ood_scene_variants._cup_scale_z`, `builder._assert_descriptor_matches_base`, and
# `builder.apply_descriptor_to_spec`, which writes one Z to every cup mesh. A Z
# multiplier != 1.0 makes every corpus scene refuse itself (cups OOD silently degrades
# to an in-distribution draw) and would be flattened back to 1.0 in any compiled
# variant. It also sinks the collision hull ~2.4 mm below the visual base (96.9 mm
# tall mug), so the cups would visibly float.
COLLISION_SCALE_MULTIPLIER = "1.03 1.03 1.0"

# Contact friction on the collision meshes: "sliding torsional rolling".
#
# Raised 2026-08-13 from "0.04 0.005 0.0001" because cups slipped. MuJoCo takes the
# ELEMENT-WISE MAX of the two contacting geoms' friction, so the old 0.04 was invisible
# against the tabletop (1.0) and against the soft finger pads (3.0) -- the one place it
# actually governed was cup-on-cup, which is exactly the stacking the cups task scores,
# and 0.04 there is effectively frictionless. Sliding is set above the tabletop's 1.0 so
# it is the value that governs rather than being masked by it.
#
# Raised to 4.0 on 2026-08-14 because the cup slides down in the gripper. The same MAX
# rule sets a hard floor here: the soft finger pads are 3.0, so ANY cup value <= 3.0 is
# completely masked at the grasp and changes nothing. 1.5 was in that dead band -- the
# pad-vs-cup coefficient was 3.0 regardless. 4.0 is therefore the smallest value that
# actually raises the grasp friction at all, and it makes the cup the governing geom
# against the tabletop and against other cups too.
#
# Measured caveat, so nobody re-derives it: replaying the recorded cups demo showed no
# significant change from 1.5 -> 4.0 -> 6.0 (penetration and lift height identical to 2
# decimal places), because THAT grasp is geometric -- the pads wrap the cup wall rather
# than relying on friction. If raising this does not stop the sliding on hardware, the
# next lever is the pads' own 3.0 in generated_panda/sii_panda_model_soft_gripper.xml,
# which would also affect the T-shape task, or the gripper's commanded width / grasp force.
COLLISION_FRICTION = "0.2 0.05 0.005"


# Better cup positions, like your target output
CUPS = {
    "cup1": {
        "scale": VERY_SMALL_CUP_SCALE,
        "pos": "0.75 -0.12 0.225",
        "material": "cup_red",
    },
    "cup2": {
        "scale": SMALL_CUP_SCALE,
        "pos": "0.45 -0.12 0.225",
        "material": "cup_green",
    },
    "cup3": {
        "scale": MIDDLE_CUP_SCALE,
        "pos": "0.75 0.12 0.225",
        "material": "cup_blue",
    },
    "cup4": {
        "scale": BIG_CUP_SCALE,
        "pos": "0.45 0.12 0.225",
        "material": "cup_yellow",
    },
}

OUTPUT_FILE = (
    "/path/to/Intervention_IL_AR/intervene_base/"
    "mujoco_scenes/generated_cups/generated_cups_via_stl"
)


# ========= HELPERS =========

def apply_unit_scale(scale_str):
    sx, sy, sz = map(float, scale_str.split())

    if USE_MM:
        sx *= MM_TO_M
        sy *= MM_TO_M
        sz *= MM_TO_M

    return f"{sx:g} {sy:g} {sz:g}"


def apply_collision_scale(scale_str):
    sx, sy, sz = map(float, scale_str.split())
    mx, my, mz = map(float, COLLISION_SCALE_MULTIPLIER.split())
    return apply_unit_scale(f"{sx * mx} {sy * my} {sz * mz}")


# ========= GENERATION =========

def generate_defaults():
    return f"""  <default>

    <default class="visual_cup">
      <geom group="2" type="mesh" contype="0" conaffinity="0"/>
    </default>

    <default class="collision_cup">
      <geom group="3" type="mesh" rgba="1 1 1 0.0"
        contype="1"
        conaffinity="1"
        condim="4"
        friction="{COLLISION_FRICTION}"
        solimp="0.85 0.95 0.005 0.5 2"
        solref="0.012 3.0"/>
    </default>

  </default>"""


def generate_assets():
    lines = []
    lines.append("  <asset>")

    # Materials
    lines.append('    <material name="cup_red"    rgba="0.85 0.20 0.20 1"/>')
    lines.append('    <material name="cup_green"  rgba="0.20 0.75 0.30 1"/>')
    lines.append('    <material name="cup_blue"   rgba="0.20 0.45 0.90 1"/>')
    lines.append('    <material name="cup_yellow" rgba="0.95 0.80 0.20 1"/>')

    for cup_name, cfg in CUPS.items():
        visual_scale = apply_unit_scale(cfg["scale"])
        collision_scale = apply_collision_scale(cfg["scale"])

        # Visual mesh
        lines.append(
            f'    <mesh name="{cup_name}_visual" '
            f'file="{BASE_PATH}/{OBJ_NAME}" '
            f'scale="{visual_scale}"/>'
        )

        # Collision meshes
        for i in range(NUM_COLLISION):
            lines.append(
                f'    <mesh name="{cup_name}_col_{i}" '
                f'file="{BASE_PATH}/{COLLISION_PREFIX}{i}.obj" '
                f'scale="{collision_scale}"/>'
            )

        # Empty line between cups, only for readability
        lines.append("")

    # Remove last empty line
    if lines[-1] == "":
        lines.pop()

    lines.append("  </asset>")
    return "\n".join(lines)


def generate_body(cup_name, cfg):
    pos = cfg["pos"]
    material = cfg["material"]

    lines = []
    lines.append(f'    <body name="{cup_name}" pos="{pos}">')

    # Free joint
    lines.append(f'      <joint name="{cup_name}_free" type="free" damping="0.01"/>')

    # Visual mesh
    lines.append(
        f'      <geom mesh="{cup_name}_visual" '
        f'class="visual_cup" '
        f'material="{material}"/>'
    )

    # Collision meshes
    for i in range(NUM_COLLISION):
        lines.append(
            f'      <geom mesh="{cup_name}_col_{i}" class="collision_cup"/>'
        )

    lines.append("    </body>")
    return "\n".join(lines)


def generate_worldbody():
    lines = []
    lines.append("  <worldbody>")

    for cup_name, cfg in CUPS.items():
        lines.append(generate_body(cup_name, cfg))

    lines.append("  </worldbody>")
    return "\n".join(lines)


def generate_xml():
    xml = []
    xml.append('<mujoco model="multi_cups">')
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
    xml_content = generate_xml()

    output_file = OUTPUT_FILE + f"_{len(CUPS)}.xml"

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(xml_content)

    print(f"✅ MJCF file generated: {output_file}")
