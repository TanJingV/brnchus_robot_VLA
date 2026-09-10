"""Stage the complete desktop MuJoCo scene for browser execution."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "mujoco_desktop_system" / "meshes"
SOURCE_MODEL = SOURCE_DIR / "cable_robot_bronch_final_seg2.xml"
OUTPUT_DIR = Path(__file__).resolve().parent / "sim"
OUTPUT_MODEL = OUTPUT_DIR / "bronchoscope_web.xml"


def referenced_meshes(model_path: Path) -> list[Path]:
    root = ET.parse(model_path).getroot()
    paths = []
    for element in root.findall("./asset/mesh"):
        file_name = element.get("file")
        if file_name:
            paths.append(Path(file_name))
    for element in root.iter("flexcomp"):
        file_name = element.get("file")
        if file_name:
            paths.append(Path(file_name))
    return list(dict.fromkeys(paths))


def build_model() -> Path:
    tree = ET.parse(SOURCE_MODEL)
    root = tree.getroot()
    root.set("model", "bronchoscope_full_web_simulation")

    # The browser runs MuJoCo and rendering on one UI thread. A 2 ms implicit
    # step keeps the flexible chain stable while making the live demo run at
    # real-time speed instead of inheriting the desktop model's 0.2 ms budget.
    option = root.find("option")
    if option is None:
        raise RuntimeError("The source model has no MuJoCo option element")
    option.set("timestep", "0.002")
    option.set("iterations", "20")
    option.set("tolerance", "1e-6")

    # The CDN WebAssembly build does not statically register MuJoCo's optional
    # elasticity plugin. Keep the complete passive joint chain and its native
    # stiffness/damping, but remove only the unavailable plugin declarations.
    extension = root.find("extension")
    if extension is not None:
        root.remove(extension)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "plugin":
                parent.remove(child)

    # Three.js renders the translucent airway directly from LUNG_URL. Remove
    # the duplicate visual mesh from the MuJoCo package while retaining the
    # independent high-resolution collision mesh used by the physics engine.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("name") == "bronchial_visual":
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") == "visual_mesh":
                asset.remove(mesh)

    # Use a thick, over-damped shell. The contact time constant is comfortably
    # above the browser timestep, so impact energy is absorbed without chatter.
    airway = root.find(".//flexcomp[@name='bronchial_wall_nonconvex']")
    if airway is None:
        raise RuntimeError("The bronchial collision flex is missing from the source model")
    airway.set("radius", "0.0015")
    airway_contact = airway.find("contact")
    if airway_contact is None:
        raise RuntimeError("The bronchial collision flex has no contact configuration")
    airway_contact.set("priority", "20")
    airway_contact.set("solref", "0.008 3")
    airway_contact.set("solimp", "0.92 0.995 0.001")
    airway_contact.set("margin", "0.00035")

    insertion_actuator = root.find(".//actuator/general[@name='act_slid_M']")
    if insertion_actuator is None:
        raise RuntimeError("The insertion actuator is missing from the source model")
    insertion_actuator.set("gainprm", "25000")
    insertion_actuator.set("biasprm", "0 -25000 -220")
    insertion_actuator.set("forcerange", "-500 500")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_MODEL, encoding="utf-8", xml_declaration=True)

    for relative_path in referenced_meshes(OUTPUT_MODEL):
        source = SOURCE_DIR / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"Missing MuJoCo asset: {source}")
        destination = OUTPUT_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    return OUTPUT_MODEL


def main() -> None:
    model = build_model()
    assets = referenced_meshes(model)
    total_bytes = model.stat().st_size + sum(
        (OUTPUT_DIR / asset).stat().st_size for asset in assets
    )
    print(f"Built {model.relative_to(ROOT)}")
    print(f"Staged {len(assets)} referenced meshes ({total_bytes:,} bytes total)")


if __name__ == "__main__":
    main()
