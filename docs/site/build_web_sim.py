"""Stage the complete desktop MuJoCo scene for browser execution."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "meshes"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_MODEL, encoding="utf-8", xml_declaration=True)

    for relative_path in referenced_meshes(SOURCE_MODEL):
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
