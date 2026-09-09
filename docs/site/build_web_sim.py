"""Build the browser-compatible MuJoCo scene without changing desktop assets."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = ROOT / "two_segment_tdcr_opencr" / "two_segment_tdcr.xml"
SOURCE_LUNG = ROOT / "meshes" / "part" / "bronchus.stl"
OUTPUT_DIR = Path(__file__).resolve().parent / "sim"


def build_model() -> Path:
    tree = ET.parse(SOURCE_MODEL)
    root = tree.getroot()
    root.set("model", "bronchoscope_web_tendon_simulation")

    option = root.find("option")
    if option is None:
        raise RuntimeError("Source model has no MuJoCo option element")
    option.set("timestep", "0.001")
    option.set("iterations", "45")
    option.set("tolerance", "1e-7")

    worldbody = root.find("worldbody")
    robot = worldbody.find("body[@name='tdcr_base']") if worldbody is not None else None
    if robot is None:
        raise RuntimeError("Source model has no tdcr_base body")

    # Match the current desktop model's centered entrance pose. The slider then
    # advances the complete robot in the local +X direction.
    robot.set("pos", "0.1660102 0.00293 1.09464")
    insertion = ET.Element(
        "joint",
        {
            "name": "web_insertion",
            "type": "slide",
            "axis": "1 0 0",
            "range": "-0.015 0.22",
            "damping": "2",
            "armature": "0.02",
        },
    )
    robot.insert(0, insertion)
    robot.insert(
        1,
        ET.Element(
            "geom",
            {
                "name": "web_sheath",
                "type": "cylinder",
                "fromto": "-0.14 0 0 0 0 0",
                "size": "0.00215",
                "mass": "0",
                "rgba": "0.035 0.11 0.17 1",
                "contype": "0",
                "conaffinity": "0",
            },
        ),
    )

    actuators = root.find("actuator")
    if actuators is None:
        raise RuntimeError("Source model has no actuator element")
    actuators.append(
        ET.Element(
            "position",
            {
                "name": "web_insertion_actuator",
                "joint": "web_insertion",
                "kp": "900",
                "kv": "70",
                "ctrllimited": "true",
                "ctrlrange": "-0.015 0.22",
                "forcelimited": "true",
                "forcerange": "-250 250",
            },
        )
    )

    key = root.find("keyframe/key")
    if key is not None:
        controls = key.get("ctrl", "").split()
        controls.append("0")
        key.set("ctrl", " ".join(controls))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "bronchoscope_web.xml"
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    model = build_model()
    shutil.copy2(SOURCE_LUNG, OUTPUT_DIR / "bronchus.stl")
    print(f"Built {model.relative_to(ROOT)}")
    print(f"Copied {(OUTPUT_DIR / 'bronchus.stl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
