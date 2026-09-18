from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class BoxHalfSize:
    x: float
    y: float
    z: float


@dataclass
class TShapeSpec:
    name: str
    body_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Half-sizes
    stem_size: BoxHalfSize = BoxHalfSize(0.025, 0.0225, 0.0625)
    bar_size: BoxHalfSize = BoxHalfSize(0.075, 0.0225, 0.025)

    # RGBA
    rgba: Tuple[float, float, float, float] = (0.65, 0.25, 0.45, 1.0)

    # Optional friction: (sliding, torsional, rolling)
    friction: Optional[Tuple[float, float, float]] = None

    # Dynamic or fixed
    free_joint: bool = True


def fmt_vec(values) -> str:
    return " ".join(f"{v:g}" for v in values)


def geom_xml(
    name: str,
    size: BoxHalfSize,
    pos: Tuple[float, float, float],
    rgba: Tuple[float, float, float, float],
    friction: Optional[Tuple[float, float, float]] = None,
) -> str:
    attrs = [
        f'name="{name}"',
        'type="box"',
        f'size="{fmt_vec((size.x, size.y, size.z))}"',
        f'pos="{fmt_vec(pos)}"',
        f'rgba="{fmt_vec(rgba)}"',
        'contype="1"',
        'conaffinity="1"',
    ]

    if friction is not None:
        attrs.append(f'friction="{fmt_vec(friction)}"')

    return f'    <geom {" ".join(attrs)}/>'


def tshape_body_xml(spec: TShapeSpec) -> str:
    # non-intersecting placement
    stem_center_z = spec.stem_size.z
    bar_center_z = 2.0 * spec.stem_size.z + spec.bar_size.z

    lines = [f'<body name="{spec.name}" pos="{fmt_vec(spec.body_pos)}">']

    if spec.free_joint:
        lines.append(f'    <joint name="{spec.name}_free" type="free"/>')

    lines.append(
        geom_xml(
            name=f"{spec.name}_stem",
            size=spec.stem_size,
            pos=(0.0, 0.0, stem_center_z),
            rgba=spec.rgba,
            friction=spec.friction,
        )
    )

    lines.append(
        geom_xml(
            name=f"{spec.name}_bar",
            size=spec.bar_size,
            pos=(0.0, 0.0, bar_center_z),
            rgba=spec.rgba,
            friction=spec.friction,
        )
    )

    lines.append("</body>")
    return "\n".join(lines)


def indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def tshape_worldbody_file(spec: TShapeSpec) -> str:
    body_xml = tshape_body_xml(spec)
    return f"""<mujoco model="{spec.name}">
  <option gravity="0 0 0"/>
  <worldbody>
{indent_block(body_xml, 4)}
  </worldbody>
</mujoco>
"""


def save_tshape_files(spec: TShapeSpec, output_dir: str = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    snippet_path = out / f"{spec.name}_body.xml"
    full_path = out / f"{spec.name}.xml"

    snippet_path.write_text(tshape_body_xml(spec) + "\n", encoding="utf-8")
    full_path.write_text(tshape_worldbody_file(spec), encoding="utf-8")

    print(f"Saved snippet: {snippet_path}")
    print(f"Saved full MJCF: {full_path}")


if __name__ == "__main__":

    stem_size = BoxHalfSize(0.025, 0.0225, 0.0625)
    bar_size  = BoxHalfSize(0.075, 0.0225, 0.025)


    t1 = TShapeSpec(
        name="T1",
        body_pos = (0.55, -0.12, 0.225),
        stem_size = stem_size,
        bar_size  = bar_size,  
        rgba=(0, 1, 0, 1.0),
        friction=None,
        free_joint=True,
    )

    t2 = TShapeSpec(
        name="T2",
        body_pos  =(0.72, 0.12, 0.225),
        stem_size = stem_size,
        bar_size  = bar_size,
        rgba=(1, 0, 0, 1.0),
        friction=None,
        free_joint=True,
    )

    save_tshape_files(t1, output_dir="generated_tshapes")
    save_tshape_files(t2, output_dir="generated_tshapes")