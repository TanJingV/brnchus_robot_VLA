#!/usr/bin/env python3
"""Replace the legacy two active tip sections in the full bronchoscope MJCF.

The passive cable, insertion mechanism, bronchial model, cameras and external
scene are retained. The generated two-section TDCR is mounted as a free-rooted
assembly and rigidly welded to ``cable_stiffB_last``, matching the legacy
passive-to-active connection strategy while replacing only the active tip.
"""

from __future__ import annotations

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

from generate_model import (
    DEFAULT_CONFIG,
    _fmt,
    _indent,
    build_model,
    load_config,
    radial_position,
    wire_layout,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_TARGET = PROJECT_ROOT / "meshes" / "cable_robot_bronch_final_seg2.xml"
DEFAULT_ACTIVE_ORIGIN = "0.0488 0 1.094"
DEFAULT_ROBOT_ROOT_POSITION = "-0.38279 0.00293 1.00064"
BRONCHIAL_FLEX_NAME = "bronchial_wall_nonconvex"
BRONCHIAL_COLLISION_GROUP = 4
SMOOTH_COLLISION_PREFIX = "smooth_collision_"
BRONCHIAL_WALL_RADIUS = 0.001
SMOOTH_CONTACT_SOLREF = "0.002 1"
SMOOTH_CONTACT_SOLIMP = "0.95 0.99 0.001"
SMOOTH_CONTACT_MARGIN = "0.00015"
PASSIVE_FOLLOWER_JOINT_COUNT = 6
PASSIVE_FOLLOWER_STIFFNESS = (8.0, 6.0, 4.5, 3.0, 2.0, 1.0)
PASSIVE_FOLLOWER_DAMPING = (1.6, 1.4, 1.2, 1.0, 0.8, 0.6)
PASSIVE_FOLLOWER_MAX_BEND_DEG = (6.0, 8.0, 10.0, 12.0, 15.0, 18.0)


def named_child(parent: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    return next(
        (child for child in parent.findall(tag) if child.get("name") == name), None
    )


def named_descendant(parent: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    return next(
        (child for child in parent.iter(tag) if child.get("name") == name), None
    )


def find_parent(root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
    return next((parent for parent in root.iter() if target in list(parent)), None)


def remove_named_children(
    parent: ET.Element, tag: str, names: Iterable[str]
) -> None:
    names = set(names)
    for child in list(parent):
        if child.tag == tag and child.get("name") in names:
            parent.remove(child)


def ensure_section(root: ET.Element, tag: str) -> ET.Element:
    section = root.find(tag)
    if section is None:
        section = ET.SubElement(root, tag)
    return section


def configure_continuum_scene(root: ET.Element) -> None:
    """Apply the floor and backdrop used by the original continuum demo.

    Only rendering attributes are changed here.  The floor contact settings
    are retained, while its visibility group is set to 0 because MuJoCo hides
    groups 3--5 by default.
    """
    visual = ensure_section(root, "visual")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("shadowsize", "4096")

    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0.15 0.25 0.35 1")

    visual_map = visual.find("map")
    if visual_map is None:
        visual_map = ET.SubElement(visual, "map")
    visual_map.set("stiffness", "700")
    visual_map.set("shadowscale", "0.5")
    visual_map.set("fogstart", "1")
    visual_map.set("fogend", "15")
    visual_map.set("zfar", "40")
    visual_map.set("haze", "1")

    asset = ensure_section(root, "asset")
    skybox = next(
        (texture for texture in asset.findall("texture") if texture.get("type") == "skybox"),
        None,
    )
    if skybox is None:
        skybox = ET.Element("texture")
        asset.insert(0, skybox)
    skybox.attrib.update(
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "512",
        }
    )

    floor_texture = named_child(asset, "texture", "plane")
    if floor_texture is None:
        floor_texture = ET.SubElement(asset, "texture", name="plane")
    floor_texture.attrib.update(
        {
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.15 0.2",
            "width": "512",
            "height": "512",
            "mark": "cross",
            "markrgb": "0.8 0.8 0.8",
        }
    )

    floor_material = named_child(asset, "material", "plane")
    if floor_material is None:
        floor_material = ET.SubElement(asset, "material", name="plane")
    floor_material.attrib.update(
        {
            "texture": "plane",
            "reflectance": "0.3",
            "texrepeat": "10 10",
            "texuniform": "true",
        }
    )

    worldbody = ensure_section(root, "worldbody")
    floor = named_child(worldbody, "geom", "floor")
    if floor is None:
        floor = ET.Element(
            "geom",
            name="floor",
            type="plane",
            size="0 0 0.25",
            condim="3",
            group="0",
        )
        worldbody.insert(0, floor)
    floor.set("material", "plane")
    floor.set("group", "0")


def configure_nonconvex_bronchial_wall(root: ET.Element) -> None:
    """Replace thousands of convex parts with one rigid non-convex flex.

    A regular MuJoCo mesh geom collides through its convex hull, so the intact
    bronchus STL cannot be used as a hollow airway that way. A rigid 2-D flex
    retains the original triangle surface and therefore provides native
    non-convex collision without creating any deformable degrees of freedom.
    """
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("Model must contain asset and worldbody sections.")

    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").startswith("part_"):
            asset.remove(mesh)

    pipe = named_descendant(worldbody, "body", "pipe")
    if pipe is None:
        raise RuntimeError("Could not find the bronchial pipe body.")

    for parent in pipe.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh", "").startswith("part_"):
                parent.remove(child)
            elif (
                child.tag == "flexcomp"
                and child.get("name") == BRONCHIAL_FLEX_NAME
            ):
                parent.remove(child)

    visual_geom = next(
        (geom for geom in pipe.iter("geom") if geom.get("mesh") == "visual_mesh"),
        None,
    )
    if visual_geom is None:
        raise RuntimeError("Could not find the bronchial visual mesh geom.")
    visual_geom.set("name", "bronchial_visual")
    visual_geom.set("contype", "0")
    visual_geom.set("conaffinity", "0")
    visual_geom.set("group", "1")

    flex_attributes = {
        "name": BRONCHIAL_FLEX_NAME,
        "type": "mesh",
        "file": "part/bronchus_collision_solid_nonconvex.stl",
        "rigid": "true",
        "scale": "0.001 0.001 0.001",
        # MuJoCo sweeps this radius to both sides of the triangle surface. A
        # former 3 mm value therefore narrowed the lumen by 3 mm everywhere
        # and could virtually close small bifurcations even when the visual
        # airway was open. One millimetre supplies a finite shell without
        # over-constricting the lumen; the hard normal constraint, short time
        # step and rounded robot collision geoms provide the anti-tunnelling
        # behaviour instead of an unrealistically thick wall.
        "radius": _fmt(BRONCHIAL_WALL_RADIUS),
        # Group 4 is reserved exclusively for the bronchial collision surface.
        # It is hidden by default and can be toggled with the viewer's 4 key.
        "rgba": "0.05 0.85 0.18 0.42",
        "group": str(BRONCHIAL_COLLISION_GROUP),
    }
    for pose_attribute in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        if visual_geom.get(pose_attribute):
            flex_attributes[pose_attribute] = visual_geom.get(pose_attribute)
    bronchial_flex = ET.Element("flexcomp", flex_attributes)
    ET.SubElement(
        bronchial_flex,
        "contact",
        internal="false",
        selfcollide="none",
        contype="1",
        conaffinity="1",
        priority="10",
        # A lubricated bronchoscope/airway interface is dominated by the normal
        # reaction. Removing tangential constraint dimensions eliminates the
        # stick-slip chatter that made the light distal links jump.
        condim="1",
        friction="0 0 0",
        solref=SMOOTH_CONTACT_SOLREF,
        solimp=SMOOTH_CONTACT_SOLIMP,
        margin=SMOOTH_CONTACT_MARGIN,
    )
    pipe.insert(list(pipe).index(visual_geom) + 1, bronchial_flex)


def add_smooth_robot_collision_capsules(
    root: ET.Element, active_base: ET.Element
) -> None:
    """Replace flat-ended robot collision cylinders with rounded envelopes.

    The original cylinders remain visual and inertial geoms.  A massless,
    overlapping capsule is added solely for collision, so joint mechanics and
    the rendered 3.5 mm diameter are unchanged while the contact surface no
    longer exposes a rim at every discrete segment boundary.
    """
    collision_roots = (root, active_base)
    for collision_root in collision_roots:
        for parent in collision_root.iter():
            for child in list(parent):
                if (
                    child.tag == "geom"
                    and child.get("name", "").startswith(SMOOTH_COLLISION_PREFIX)
                ):
                    parent.remove(child)

    for collision_root in collision_roots:
        for parent in collision_root.iter():
            for geom in list(parent):
                if geom.tag != "geom" or geom.get("type") != "cylinder":
                    continue
                name = geom.get("name", "")
                if not name.startswith(
                    ("cable_stiffG", "section_1_element_", "section_2_element_")
                ):
                    continue
                is_passive = name.startswith("cable_stiffG")
                smooth = ET.Element(
                    "geom",
                    name=f"{SMOOTH_COLLISION_PREFIX}{name}",
                    type="capsule",
                    contype="1",
                    # The passive chain keeps its original 1/1 mask. Active
                    # links use 1/0 so neighbouring active bodies cannot
                    # self-collide through their deliberately overlapping
                    # rounded envelopes.
                    conaffinity="1" if is_passive else "0",
                    condim="1",
                    friction="0 0 0",
                    solref=SMOOTH_CONTACT_SOLREF,
                    solimp=SMOOTH_CONTACT_SOLIMP,
                    margin=SMOOTH_CONTACT_MARGIN,
                    mass="0",
                    group="5",
                    rgba="0.1 0.8 1 0",
                )
                for attribute in (
                    "size",
                    "pos",
                    "quat",
                    "euler",
                    "axisangle",
                    "xyaxes",
                    "zaxis",
                    "fromto",
                ):
                    if geom.get(attribute) is not None:
                        smooth.set(attribute, geom.get(attribute))

                insertion_index = list(parent).index(geom) + 1
                parent.insert(insertion_index, smooth)
                geom.set("contype", "0")
                geom.set("conaffinity", "0")


def add_section_interface_collision_fairing(active_base: ET.Element) -> None:
    """Use one smooth collision envelope across the two active sections."""
    def geom_length(geom: ET.Element) -> float:
        if geom.get("fromto"):
            values = [float(value) for value in geom.get("fromto").split()]
            return math.dist(values[:3], values[3:6])
        size = [float(value) for value in geom.get("size", "").split()]
        if len(size) >= 2:
            return 2.0 * size[1]
        raise RuntimeError(f"Cannot determine length of {geom.get('name')}.")

    interface = named_descendant(active_base, "body", "section_interface")
    if interface is None:
        raise RuntimeError("Generated TDCR is missing section_interface.")
    for old_fairing in list(interface.findall("geom")):
        if old_fairing.get("name") == "section_interface_collision_fairing":
            interface.remove(old_fairing)

    transition_names = (
        "section_1_element_11",
        "section_1_element_12",
        "section_2_element_0",
        "section_2_element_1",
    )
    transition_geoms = [
        named_descendant(active_base, "geom", name)
        for name in transition_names
    ]
    if any(geom is None for geom in transition_geoms):
        raise RuntimeError("Could not find active section-interface geoms.")
    rigid_collision = named_descendant(
        active_base, "geom", "inter_section_rigid_collision"
    )
    if rigid_collision is None:
        raise RuntimeError("Could not find inter-section rigid collision geom.")
    for original in transition_geoms:
        smooth = named_descendant(
            active_base,
            "geom",
            f"{SMOOTH_COLLISION_PREFIX}{original.get('name')}",
        )
        if smooth is not None:
            smooth.set("contype", "0")
            smooth.set("conaffinity", "0")
    rigid_collision.set("contype", "0")
    rigid_collision.set("conaffinity", "0")

    proximal_reach = sum(
        geom_length(geom)
        for geom in transition_geoms[:2]
    )
    distal_reach = sum(
        geom_length(geom)
        for geom in transition_geoms[2:]
    )
    rigid_length = geom_length(rigid_collision)
    radius = max(
        min(float(geom.get("size").split()[0]) for geom in transition_geoms)
        - 0.0002,
        0.0014,
    )
    fairing = ET.Element(
        "geom",
        name="section_interface_collision_fairing",
        type="capsule",
        fromto=(
            f"{_fmt(-proximal_reach)} 0 0 "
            f"{_fmt(rigid_length + distal_reach)} 0 0"
        ),
        size=_fmt(radius),
        contype="1",
        conaffinity="0",
        condim="1",
        friction="0 0 0",
        solref=SMOOTH_CONTACT_SOLREF,
        solimp=SMOOTH_CONTACT_SOLIMP,
        margin="0.0001",
        mass="0",
        group="5",
        rgba="0.2 0.9 0.6 0.10",
    )
    first_child_body = next(
        (index for index, child in enumerate(interface) if child.tag == "body"),
        len(interface),
    )
    interface.insert(first_child_body, fairing)


def configure_passive_follower_section(root: ET.Element) -> None:
    """Taper passive-tip stiffness so it follows active-section curvature."""
    passive_joints = [
        joint
        for joint in root.iter("joint")
        if joint.get("name", "").startswith("cable_stiffJ")
        and joint.get("type") == "ball"
    ]
    if len(passive_joints) < PASSIVE_FOLLOWER_JOINT_COUNT:
        raise RuntimeError("Could not find enough passive terminal ball joints.")
    terminal_joints = passive_joints[-PASSIVE_FOLLOWER_JOINT_COUNT:]
    for joint, stiffness, damping, bend_deg in zip(
        terminal_joints,
        PASSIVE_FOLLOWER_STIFFNESS,
        PASSIVE_FOLLOWER_DAMPING,
        PASSIVE_FOLLOWER_MAX_BEND_DEG,
    ):
        joint.set("stiffness", _fmt(stiffness))
        joint.set("damping", _fmt(damping))
        joint.set("limited", "true")
        joint.set("range", f"0 {_fmt(math.radians(bend_deg))}")


def remove_canonical_xml_noise(root: ET.Element) -> None:
    """Remove redundant writer output and empty sections from maintained MJCF."""
    asset = root.find("asset")
    if asset is not None:
        for texture in asset.findall("texture"):
            if texture.get("colorspace") == "auto":
                texture.attrib.pop("colorspace", None)
        for mesh in asset.findall("mesh"):
            # Every retained mesh has a standard file extension, so repeating
            # model/stl adds no information and breaks older MuJoCo 3.x builds.
            if mesh.get("content_type") == "model/stl":
                mesh.attrib.pop("content_type", None)
    for section_name in ("equality",):
        section = root.find(section_name)
        if section is not None and not list(section):
            root.remove(section)


def add_compatibility_sites(active_base: ET.Element, config: dict) -> None:
    """Keep body/site names consumed by the existing UI and navigation code."""
    angles = wire_layout(config)
    radius = config["tendons"]["radius_from_center_m"]
    site_radius = config["tendons"]["wire_diameter_m"] / 2.0
    joints = config["discretization"]["joints_per_section"]
    element_length = config["geometry"]["section_length_m"] / joints

    interface = named_descendant(active_base, "body", "section_interface")
    if interface is None:
        raise RuntimeError("Generated TDCR is missing section_interface.")
    interface.set("name", "seg2_body")

    # Legacy UI names s1/s3/s5 represent the three proximal wires.
    for site_name, wire in (("s1", 1), ("s3", 2), ("s5", 3)):
        ET.SubElement(
            interface,
            "site",
            name=site_name,
            pos=radial_position(radius, angles[wire]),
            size=_fmt(site_radius),
            rgba="1 0 0 1",
        )

    last_body = named_descendant(
        active_base, "body", f"section_2_joint_body_{joints - 1}"
    )
    if last_body is None:
        raise RuntimeError("Generated TDCR is missing its distal joint body.")
    distal_tip = named_descendant(active_base, "body", "distal_tip_rigid")
    if distal_tip is None:
        raise RuntimeError("Generated TDCR is missing distal_tip_rigid.")
    distal_tip_length = config["geometry"]["rigid_sections"]["distal_tip_length_m"]
    end_body = ET.SubElement(
        distal_tip,
        "body",
        name="end_6",
        pos=f"{_fmt(distal_tip_length)} 0 0",
    )
    ET.SubElement(
        end_body,
        "geom",
        name="end_6_mass_geom",
        type="sphere",
        size="0.00001",
        mass="0",
        rgba="0 0 0 0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        end_body,
        "camera",
        name="tip_camera",
        pos="0.004 0 0",
        euler="1.57079632679 -1.57079632679 0",
        fovy="70",
    )
    ET.SubElement(
        end_body,
        "site",
        name="seg2_S_last",
        pos="0 0 0",
        size="0.0001",
        rgba="0 0 0 0",
    )
    # Legacy UI names s2/s4/s6 represent the three distal wires.
    for site_name, wire in (("s2", 4), ("s4", 5), ("s6", 6)):
        ET.SubElement(
            end_body,
            "site",
            name=site_name,
            pos=radial_position(radius, angles[wire]),
            size=_fmt(site_radius),
            rgba="1 0.35 0 1",
        )


def convert_weld_to_tree_connection(target_path: Path) -> None:
    """Expand the passive composite and make the active base its rigid child.

    Equality welds are solver constraints and always have finite numerical
    compliance. Reparenting the active assembly under the passive terminal body
    removes all six relative DOFs, making the interface exactly rigid by model
    topology rather than approximately rigid through constraint forces.
    """
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "Exact passive/active attachment requires the MuJoCo Python package. "
            "Run this script with the project's MuJoCo Conda environment."
        ) from exc

    compile_path = target_path.with_name(f"{target_path.stem}_compile_tmp.xml")
    expanded_path = target_path.with_name(f"{target_path.stem}_expanded_tmp.xml")
    # Saving a compiled model expands high-level flexcomp declarations into
    # enormous vertex/element arrays. Compile a temporary copy without the
    # rigid bronchial flex, then restore its concise declaration afterward.
    source_parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    source_tree = ET.parse(target_path, parser=source_parser)
    source_root = source_tree.getroot()
    source_worldbody = source_root.find("worldbody")
    if source_worldbody is None:
        raise RuntimeError("Source model has no worldbody.")
    for parent in source_worldbody.iter():
        for child in list(parent):
            if (
                child.tag == "flexcomp"
                and child.get("name") == BRONCHIAL_FLEX_NAME
            ):
                parent.remove(child)
    _indent(source_root)
    source_tree.write(compile_path, encoding="utf-8", xml_declaration=True)
    expanded_path.touch()
    try:
        model = mujoco.MjModel.from_xml_path(str(compile_path))
        mujoco.mj_saveLastXML(str(expanded_path), model)

        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(expanded_path, parser=parser)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("Expanded model has no worldbody.")
        active_base = named_child(worldbody, "body", "active_tdcr_base")
        passive_last = named_descendant(
            worldbody, "body", "cable_stiffB_last"
        )
        passive_tip_site = named_descendant(
            passive_last, "site", "cable_stiffS_last"
        ) if passive_last is not None else None
        if active_base is None or passive_last is None or passive_tip_site is None:
            raise RuntimeError(
                "Could not find expanded active base and passive terminal body/site."
            )

        configure_nonconvex_bronchial_wall(root)

        worldbody.remove(active_base)
        for child in list(active_base):
            if child.tag in {"freejoint", "joint"} and child.get("name") == "active_tdcr_freejoint":
                active_base.remove(child)
        active_base.set("pos", passive_tip_site.get("pos", "0 0 0"))
        active_base.attrib.pop("quat", None)
        active_base.attrib.pop("euler", None)
        passive_last.append(active_base)

        # Keep the rendered splice continuous, but make its short transition
        # zone collision-free. The adjoining passive and active envelopes still
        # contact the airway, while the diameter change itself cannot hook it.
        passive_tip_geom = named_descendant(
            passive_last, "geom", "cable_stiffG29"
        )
        active_entry_geoms = [
            named_descendant(active_base, "geom", f"section_1_element_{index}")
            for index in (0, 1)
        ]
        if passive_tip_geom is None or any(
            geom is None for geom in active_entry_geoms
        ):
            raise RuntimeError("Could not find splice collision geoms.")
        active_radius = float(active_entry_geoms[0].get("size").split()[0])
        # The passive chain may be resized for this model.  Match its complete
        # outer envelope to the 3.5 mm active continuum so the splice has no
        # diameter shoulder in either travel direction.
        for passive_geom in worldbody.iter("geom"):
            if passive_geom.get("name", "").startswith("cable_stiffG"):
                size_parts = passive_geom.get("size", "").split()
                if size_parts:
                    size_parts[0] = _fmt(active_radius)
                    passive_geom.set("size", " ".join(size_parts))
        splice_collision_geoms = [passive_tip_geom, *active_entry_geoms]
        for original_geom in (passive_tip_geom, *active_entry_geoms):
            smooth_geom = named_descendant(
                root,
                "geom",
                f"{SMOOTH_COLLISION_PREFIX}{original_geom.get('name')}",
            )
            if smooth_geom is not None:
                splice_collision_geoms.append(smooth_geom)
        for geom in splice_collision_geoms:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

        for old_fairing in list(active_base.findall("geom")):
            if old_fairing.get("name") in {
                "splice_passive_collision",
                "splice_collision_fairing",
            }:
                active_base.remove(old_fairing)

        passive_length = float(active_base.get("pos", "0 0 0").split()[0])
        active_overlap = sum(
            2.0 * float(geom.get("size").split()[1])
            for geom in active_entry_geoms
        )
        transition_rear = min(0.0045, passive_length * 0.25)
        passive_end = min(0.0035, transition_rear * 0.80)
        passive_fairing = ET.Element(
            "geom",
            name="splice_passive_collision",
            type="capsule",
            fromto=(
                f"{_fmt(-passive_length)} 0 0 "
                f"{_fmt(-passive_end)} 0 0"
            ),
            size=_fmt(max(active_radius - 0.0001, active_radius * 0.90)),
            contype="0",
            conaffinity="0",
            condim="1",
            friction="0 0 0",
            solref=SMOOTH_CONTACT_SOLREF,
            solimp=SMOOTH_CONTACT_SOLIMP,
            margin="0.00015",
            mass="0",
            group="5",
            rgba="0.1 0.8 1 0.08",
        )
        splice_fairing = ET.Element(
            "geom",
            name="splice_collision_fairing",
            type="capsule",
            fromto=(
                f"{_fmt(-transition_rear)} 0 0 "
                f"{_fmt(active_overlap)} 0 0"
            ),
            size=_fmt(max(active_radius - 0.0003, active_radius * 0.80)),
            contype="0",
            conaffinity="0",
            condim="1",
            friction="0 0 0",
            solref=SMOOTH_CONTACT_SOLREF,
            solimp=SMOOTH_CONTACT_SOLIMP,
            margin="0.0001",
            mass="0",
            group="5",
            rgba="0.1 0.8 1 0.12",
        )
        first_child_body = next(
            (index for index, child in enumerate(active_base) if child.tag == "body"),
            len(active_base),
        )
        active_base.insert(first_child_body, passive_fairing)
        active_base.insert(first_child_body + 1, splice_fairing)
        configure_passive_follower_section(root)

        equality = root.find("equality")
        if equality is not None:
            remove_named_children(equality, "weld", ("active_base_weld",))
        contact = root.find("contact")
        if contact is not None:
            for exclude in list(contact.findall("exclude")):
                if {
                    exclude.get("body1"), exclude.get("body2")
                } == {"cable_stiffB_last", "active_tdcr_base"}:
                    contact.remove(exclude)

        remove_canonical_xml_noise(root)
        _indent(root)
        tree.write(target_path, encoding="utf-8", xml_declaration=True)
    finally:
        compile_path.unlink(missing_ok=True)
        expanded_path.unlink(missing_ok=True)

def integrate(config_path: Path, target_path: Path) -> None:
    config = load_config(config_path)
    generated_root, _ = build_model(config)
    generated_worldbody = generated_root.find("worldbody")
    if generated_worldbody is None:
        raise RuntimeError("Generated standalone model has no worldbody.")
    active_base = named_child(generated_worldbody, "body", "tdcr_base")
    if active_base is None:
        raise RuntimeError("Generated standalone model has no tdcr_base.")
    active_base = copy.deepcopy(active_base)

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(target_path, parser=parser)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Target bronchoscope model has no worldbody.")
    robot_root = named_child(worldbody, "body", "base_link")
    if robot_root is None:
        raise RuntimeError("Target bronchoscope model has no base_link body.")
    robot_root.set("pos", DEFAULT_ROBOT_ROOT_POSITION)
    configure_continuum_scene(root)

    # Preserve the old active-section world origin on first conversion and the
    # integrated origin on subsequent idempotent regenerations.
    origin = DEFAULT_ACTIVE_ORIGIN
    for body_name in ("active_tdcr_base", "seg1_body"):
        old_body = named_child(worldbody, "body", body_name)
        if old_body is not None and old_body.get("pos"):
            origin = old_body.get("pos")
            break

    # On repeat generation the exact model already has active_tdcr_base nested
    # under cable_stiffB_last. Remove it before inserting the newly generated
    # active assembly; the final conversion reparents the replacement again.
    existing_active = named_descendant(worldbody, "body", "active_tdcr_base")
    if existing_active is not None:
        existing_parent = find_parent(worldbody, existing_active)
        if existing_parent is not None:
            existing_parent.remove(existing_active)

    remove_named_children(
        worldbody,
        "body",
        ("seg1_body", "seg2_body", "slider", "active_tdcr_base"),
    )
    active_base.set("name", "active_tdcr_base")
    active_base.set("pos", origin)
    active_base.insert(
        0,
        ET.Element(
            "joint",
            name="active_tdcr_freejoint",
            type="free",
            damping="0.05",
            armature="0.0001",
        ),
    )

    # The full model has a broad global geom default. Override masks locally so
    # the active chain does not self-collide but still hits the bronchial wall.
    for geom in active_base.iter("geom"):
        if geom.get("contype") == "1":
            geom.set("conaffinity", "0")

    add_smooth_robot_collision_capsules(root, active_base)
    add_section_interface_collision_fairing(active_base)

    # Use the same smooth, normal-only response on both passive and active robot
    # geoms. If either side kept condim=3 MuJoCo would restore frictional
    # contact and the distal stick-slip oscillation would return. Runtime
    # contact regularization is kept outside MJCF so free bending retains the
    # calibrated damping and armature from config.json.
    contact_geoms = list(root.iter("geom")) + list(active_base.iter("geom"))
    for geom in contact_geoms:
        if geom.get("contype") == "1":
            geom.set("margin", SMOOTH_CONTACT_MARGIN)
            geom.set("condim", "1")
            geom.set("friction", "0 0 0")
            geom.set("solref", SMOOTH_CONTACT_SOLREF)
            geom.set("solimp", SMOOTH_CONTACT_SOLIMP)

    # The passive cable has a 1.8 mm radius. Make the active base connector's
    # rear face exactly coplanar with the passive terminal face and match that
    # radius, eliminating the visual/physical interface gap.
    base_disk = named_descendant(active_base, "geom", "base_disk")
    if base_disk is None:
        raise RuntimeError("Generated TDCR is missing base_disk.")
    base_disk.set("pos", "-0.00005 0 0")
    base_disk.set("size", "0.0018 0.00005")
    base_disk.set("mass", "0")

    add_compatibility_sites(active_base, config)
    front_table = named_child(worldbody, "body", "front_object_table")
    if front_table is None:
        worldbody.append(active_base)
    else:
        worldbody.insert(list(worldbody).index(front_table), active_base)

    # Replace legacy welds with one rigid passive-to-active base splice.
    equality = ensure_section(root, "equality")
    remove_named_children(
        equality,
        "weld",
        ("right_boundary1", "mid_joint", "right_boundary", "weld_end6", "active_base_weld"),
    )
    ET.SubElement(
        equality,
        "weld",
        name="active_base_weld",
        body1="cable_stiffB_last",
        body2="active_tdcr_base",
        solref="0.0005 1",
        solimp="0.999 0.9999 0.0001",
        torquescale="0.01",
    )

    contact = ensure_section(root, "contact")
    for exclude in list(contact.findall("exclude")):
        names = {exclude.get("body1"), exclude.get("body2")}
        if names & {
            "seg1_B_last",
            "seg2_B_last",
            "seg2_body",
            "slider",
            "active_tdcr_base",
            "cable_stiffB_last",
        }:
            contact.remove(exclude)
    ET.SubElement(
        contact,
        "exclude",
        body1="cable_stiffB_last",
        body2="active_tdcr_base",
    )

    generated_tendon = generated_root.find("tendon")
    if generated_tendon is None:
        raise RuntimeError("Generated TDCR has no tendon section.")
    tendon = ensure_section(root, "tendon")
    for child in list(tendon):
        if child.get("name", "").startswith(("wire_", "tendon")):
            tendon.remove(child)
    for wire, spatial in enumerate(generated_tendon.findall("spatial"), start=1):
        copied = copy.deepcopy(spatial)
        copied.set("name", f"tendon{wire}")
        tendon.append(copied)

    actuator = ensure_section(root, "actuator")
    # Keep wall loading bounded, then tune the position servo as a damped PD
    # loop.  The derivative term is essential here: a proportional-only slider
    # repeatedly overshoots both a free-space target and the airway wall.
    slider_joint = named_descendant(root, "joint", "slid_M")
    if slider_joint is None:
        raise RuntimeError("Integrated model is missing insertion joint slid_M.")
    slider_joint.set("armature", "0.1")
    slider_joint.set("damping", "5")
    slider_joint.set("frictionloss", "0.2")

    slider_actuator = named_descendant(root, "general", "act_slid_M")
    if slider_actuator is None:
        raise RuntimeError("Integrated model is missing actuator act_slid_M.")
    # High servo stiffness supplies useful force through a small, bounded
    # command lead. The force ceiling prevents destructive wall loading while
    # the UI gradually increases force only after measured motion stalls.
    slider_actuator.set("forcelimited", "true")
    slider_actuator.set("forcerange", "-250 250")
    slider_actuator.set("gainprm", "6000")
    slider_actuator.set("biasprm", "0 -6000 -180")

    active_actuator_names = {
        "up",
        "right",
        *(f"force_s{wire}" for wire in range(1, 7)),
        *(f"act_t{wire}" for wire in range(1, 7)),
        *(f"wire_{wire}" for wire in range(1, 7)),
    }
    # Canonical MJCF exported by MuJoCo rewrites shortcut actuators such as
    # <position> to <general>. Remove by name regardless of actuator tag so
    # regeneration remains idempotent for both compact and expanded models.
    for child in list(actuator):
        if child.get("name") in active_actuator_names:
            actuator.remove(child)
    generated_actuator = generated_root.find("actuator")
    if generated_actuator is None:
        raise RuntimeError("Generated TDCR has no actuator section.")
    for wire, position in enumerate(generated_actuator.findall("position"), start=1):
        copied = copy.deepcopy(position)
        copied.set("name", f"act_t{wire}")
        copied.set("tendon", f"tendon{wire}")
        actuator.append(copied)

    sensor = ensure_section(root, "sensor")
    generated_sensor_names = {
        "tip_position",
        "tip_orientation",
        *(f"wire_{wire}_force" for wire in range(1, 7)),
    }
    for child in list(sensor):
        if child.get("name") in generated_sensor_names:
            sensor.remove(child)
    generated_sensor = generated_root.find("sensor")
    if generated_sensor is not None:
        for child in generated_sensor:
            copied = copy.deepcopy(child)
            actuator_name = copied.get("actuator")
            if actuator_name and actuator_name.startswith("wire_"):
                copied.set("actuator", f"act_t{actuator_name.split('_')[1]}")
            sensor.append(copied)

    # The slider actuator precedes the six wire actuators in the merged model.
    # Recreate the home key with slider=0 and the generated pretension controls.
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)
    generated_key = generated_root.find("./keyframe/key")
    if generated_key is None:
        raise RuntimeError("Generated TDCR has no pretension keyframe.")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        ctrl=f"0 {generated_key.get('ctrl')}",
    )

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.0002")
    option.set("integrator", "implicitfast")
    option.set("iterations", "120")
    option.set("tolerance", "1e-9")
    option.set("impratio", "100")
    option.set("cone", "pyramidal")

    # Bronchial_model.xml contains its own legacy <option>. Keep the include
    # before this model's option so the active-model solver settings below win
    # after MuJoCo expands the include.
    bronchial_include = next(
        (
            child
            for child in root.findall("include")
            if child.get("file") == "Bronchial_model.xml"
        ),
        None,
    )
    if bronchial_include is not None:
        root.remove(bronchial_include)
        root.insert(list(root).index(option), bronchial_include)
    root.set("model", "bronchoscope_with_two_section_opencr_tdcr")

    _indent(root)
    tree.write(target_path, encoding="utf-8", xml_declaration=True)
    convert_weld_to_tree_connection(target_path)
    print(f"Integrated two-section TDCR into: {target_path}")
    print(
        "Exact tree connection: "
        "cable_stiffB_last/cable_stiffS_last -> active_tdcr_base"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    integrate(args.config.resolve(), args.target.resolve())


if __name__ == "__main__":
    main()
