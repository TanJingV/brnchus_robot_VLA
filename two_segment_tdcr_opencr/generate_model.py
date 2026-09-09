#!/usr/bin/env python3
"""Generate the two-section, six-wire TDCR MuJoCo model.

The construction follows opencr-mujoco's central modelling idea:
the continuum backbone is discretised into a serial rigid-body chain, paired
hinges represent biaxial bending, hinge stiffness is derived from beam theory,
and native MuJoCo spatial tendons are routed through guide sites on the chain.

Only the Python standard library is required to generate the MJCF file.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT = HERE / "two_segment_tdcr.xml"

PROXIMAL_WIRES = (1, 2, 3)
DISTAL_WIRES = (4, 5, 6)
WIRE_COLOURS = {
    1: "1.0 0.0 0.9 1",
    2: "1.0 0.0 0.9 1",
    3: "1.0 0.0 0.9 1",
    4: "0.95 0.35 0.05 1",
    5: "0.95 0.35 0.05 1",
    6: "0.95 0.35 0.05 1",
}


def _fmt(value: float) -> str:
    """Compact, deterministic floating-point formatting for MJCF attributes."""
    return f"{value:.12g}"


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(float(value)) for value in values)


def _indent(element: ET.Element, level: int = 0) -> None:
    whitespace = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = whitespace + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = whitespace
    if level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: Dict) -> None:
    geometry = config["geometry"]
    tendons = config["tendons"]
    discretization = config["discretization"]

    if geometry["number_of_sections"] != 2:
        raise ValueError("This generator intentionally models exactly two sections.")
    if discretization["joints_per_section"] < 2:
        raise ValueError("joints_per_section must be at least 2.")
    if tendons["routing_mode"] not in {"independent", "coupled"}:
        raise ValueError("routing_mode must be 'independent' or 'coupled'.")

    ro = geometry["outer_diameter_m"] / 2.0
    ri = geometry["working_channel_diameter_m"] / 2.0
    wire_radius = tendons["wire_diameter_m"] / 2.0
    tendon_radius = tendons["radius_from_center_m"]
    if not (ri + wire_radius <= tendon_radius <= ro - wire_radius):
        raise ValueError(
            "Tendon centre radius does not fit between the working channel and "
            "the outer diameter."
        )

    expected_wall = ro - ri
    if not math.isclose(
        geometry["wall_thickness_m"], expected_wall, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "wall_thickness_m must equal (outer_diameter-working_channel_diameter)/2."
        )

    rigid = geometry["rigid_sections"]
    rigid_lengths = (
        rigid["inter_section_length_m"],
        rigid["distal_tip_length_m"],
    )
    if any(length <= 0.0 for length in rigid_lengths):
        raise ValueError("All rigid-section lengths must be greater than zero.")
    if not math.isclose(
        sum(rigid_lengths), rigid["total_rigid_length_m"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Rigid-section component lengths must sum to total_rigid_length_m.")
    expected_total = geometry["total_active_length_m"] + rigid["total_rigid_length_m"]
    if not math.isclose(
        geometry["total_continuum_length_m"], expected_total, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("total_continuum_length_m must equal active plus rigid lengths.")


def beam_properties(config: Dict) -> Dict[str, float]:
    """Return section and per-joint properties using K = N*E*I/L."""
    geometry = config["geometry"]
    material = config["effective_material"]
    joints = config["discretization"]["joints_per_section"]
    length = geometry["section_length_m"]
    ro = geometry["outer_diameter_m"] / 2.0
    ri = geometry["working_channel_diameter_m"] / 2.0

    area = math.pi * (ro**2 - ri**2)
    second_moment = math.pi * (ro**4 - ri**4) / 4.0
    polar_moment = 2.0 * second_moment
    youngs_modulus = material["youngs_modulus_pa"]
    shear_modulus = youngs_modulus / (2.0 * (1.0 + material["poisson_ratio"]))
    section_mass = material["density_kg_m3"] * area * length

    return {
        "area_m2": area,
        "second_moment_m4": second_moment,
        "polar_moment_m4": polar_moment,
        "bending_ei_nm2": youngs_modulus * second_moment,
        "torsional_gj_nm2": shear_modulus * polar_moment,
        "joint_bending_stiffness_nm_per_rad": joints
        * youngs_modulus
        * second_moment
        / length,
        "section_mass_kg": section_mass,
        "element_length_m": length / joints,
        "element_mass_kg": section_mass / joints,
    }


def wire_layout(config: Dict) -> Dict[int, float]:
    tendons = config["tendons"]
    proximal = tendons["proximal_wire_angles_deg"]
    distal = tendons["distal_wire_angles_deg"]
    if len(proximal) != 3 or len(distal) != 3:
        raise ValueError("Each section must have exactly three tendon angles.")
    return {
        wire: math.radians(angle)
        for wire, angle in zip(range(1, 7), [*proximal, *distal])
    }


def radial_position(radius: float, angle_rad: float, x: float = 0.0) -> str:
    # Longitudinal axis is +X; theta=0 is +Y and theta=90 deg is +Z.
    return _vec((x, radius * math.cos(angle_rad), radius * math.sin(angle_rad)))


def add_wire_sites(
    parent: ET.Element,
    prefix: str,
    x: float,
    wires: Sequence[int],
    angles: Dict[int, float],
    tendon_radius: float,
    site_radius: float,
    site_paths: Dict[int, List[str]],
) -> None:
    for wire in wires:
        name = f"wire_{wire}_{prefix}"
        ET.SubElement(
            parent,
            "site",
            name=name,
            pos=radial_position(tendon_radius, angles[wire], x),
            size=_fmt(site_radius),
            rgba=WIRE_COLOURS[wire],
            group="3",
        )
        site_paths[wire].append(name)


def add_backbone_geom(
    parent: ET.Element,
    name: str,
    length: float,
    outer_radius: float,
    mass: float,
    rgba: str,
) -> None:
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="cylinder",
        fromto=f"0 0 0 {_fmt(length)} 0 0",
        size=_fmt(outer_radius),
        mass=_fmt(mass),
        rgba=rgba,
        contype="1",
        conaffinity="0",
        margin="0.00005",
        # Normal-only contact models the lubricated airway interface and avoids
        # tangential stick-slip exciting the light continuum links.
        condim="1",
        friction="0.3 0.001 0.0001",
        solref="0.005 2",
        solimp="0.9 0.95 0.001",
    )


def add_rigid_section_geoms(
    parent: ET.Element,
    name: str,
    length: float,
    outer_radius: float,
) -> None:
    """Add a visible rigid sleeve and a smooth collision envelope."""
    ET.SubElement(
        parent,
        "geom",
        name=name,
        type="cylinder",
        fromto=f"0 0 0 {_fmt(length)} 0 0",
        size=_fmt(outer_radius * 1.16),
        mass="0",
        rgba="0.22 0.25 0.30 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        parent,
        "geom",
        name=f"{name}_collision",
        type="capsule",
        fromto=f"0 0 0 {_fmt(length)} 0 0",
        size=_fmt(outer_radius),
        mass="0",
        rgba="0.22 0.25 0.30 0",
        contype="1",
        conaffinity="0",
        margin="0.00005",
        condim="1",
        friction="0.3 0.001 0.0001",
        solref="0.005 2",
        solimp="0.9 0.95 0.001",
        group="5",
    )


def add_joint_pair(
    body: ET.Element,
    section: int,
    index: int,
    stiffness: float,
    damping: float,
    armature: float,
    joint_range: float,
) -> None:
    common = {
        "type": "hinge",
        "pos": "0 0 0",
        "limited": "true",
        "range": _vec((-joint_range, joint_range)),
        "stiffness": _fmt(stiffness),
        "damping": _fmt(damping),
        "armature": _fmt(armature),
    }
    ET.SubElement(
        body,
        "joint",
        name=f"section_{section}_joint_{index}_y",
        axis="0 1 0",
        **common,
    )
    ET.SubElement(
        body,
        "joint",
        name=f"section_{section}_joint_{index}_z",
        axis="0 0 1",
        **common,
    )


def add_flexible_section(
    *,
    base: ET.Element,
    section: int,
    joints: int,
    element_length: float,
    element_mass: float,
    outer_radius: float,
    tendon_radius: float,
    site_radius: float,
    angles: Dict[int, float],
    site_paths: Dict[int, List[str]],
    guide_wires: Sequence[int],
    stiffness: float,
    damping: float,
    armature: float,
    joint_range: float,
    colour: str,
) -> ET.Element:
    """Add N midpoint joints; return the body holding the last half element."""
    parent = base
    for index in range(joints):
        body_offset = element_length / 2.0 if index == 0 else element_length
        body = ET.SubElement(
            parent,
            "body",
            name=f"section_{section}_joint_body_{index}",
            pos=_vec((body_offset, 0, 0)),
        )
        add_joint_pair(
            body, section, index, stiffness, damping, armature, joint_range
        )

        remaining_length = element_length if index < joints - 1 else element_length / 2.0
        remaining_mass = element_mass if index < joints - 1 else element_mass / 2.0
        add_backbone_geom(
            body,
            f"section_{section}_element_{index + 1}",
            remaining_length,
            outer_radius,
            remaining_mass,
            colour,
        )
        add_wire_sites(
            body,
            f"section_{section}_joint_{index}",
            0.0,
            guide_wires,
            angles,
            tendon_radius,
            site_radius,
            site_paths,
        )
        parent = body
    return parent


def named_descendant(parent: ET.Element, tag: str, name: str) -> ET.Element:
    """Return a generated descendant by name, failing loudly on drift."""
    for element in parent.iter(tag):
        if element.get("name") == name:
            return element
    raise RuntimeError(f"Generated model is missing {tag} {name}.")


def add_measurement_site(parent: ET.Element, index: int, local_x: float = 0.0) -> None:
    """Add a non-physical optical keypoint at an exact backbone arclength."""
    ET.SubElement(
        parent,
        "site",
        name=f"measurement_kp_{index}",
        pos=_vec((local_x, 0.0, 0.0)),
        size="0.00022",
        rgba="1 0.85 0.05 0.9",
        group="5",
    )


def build_model(config: Dict) -> Tuple[ET.Element, Dict[str, float]]:
    geometry = config["geometry"]
    discretization = config["discretization"]
    material = config["effective_material"]
    tendon_config = config["tendons"]
    simulation = config["simulation"]
    properties = beam_properties(config)
    angles = wire_layout(config)

    joints = discretization["joints_per_section"]
    element_length = properties["element_length_m"]
    element_mass = properties["element_mass_kg"]
    outer_radius = geometry["outer_diameter_m"] / 2.0
    tendon_radius = tendon_config["radius_from_center_m"]
    site_radius = tendon_config["wire_diameter_m"] / 2.0
    stiffness = properties["joint_bending_stiffness_nm_per_rad"]
    damping = material["joint_damping_nms_per_rad"]
    armature = material["joint_armature_kg_m2"]
    joint_range = discretization["joint_range_rad"]
    rigid = geometry["rigid_sections"]
    inter_section_length = rigid["inter_section_length_m"]
    distal_tip_length = rigid["distal_tip_length_m"]

    model = ET.Element("mujoco", model=config["model_name"])
    ET.SubElement(model, "compiler", angle="radian", autolimits="true", boundmass="1e-12")
    ET.SubElement(
        model,
        "option",
        timestep=_fmt(simulation["timestep_s"]),
        integrator=simulation["integrator"],
        gravity=_vec(simulation["gravity_m_per_s2"]),
        cone="pyramidal",
        impratio="100",
        iterations="80",
        tolerance="1e-7",
    )
    visual = ET.SubElement(model, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.25 0.25 0.25")
    ET.SubElement(visual, "global", azimuth="130", elevation="-20")

    asset = ET.SubElement(model, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.9 0.95 1", rgb2="0.25 0.35 0.5", width="256", height="256")

    worldbody = ET.SubElement(model, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0.04 0.08", dir="0 -0.4 -1", diffuse="0.9 0.9 0.9")
    base = ET.SubElement(worldbody, "body", name="tdcr_base", pos="0 0 0")
    ET.SubElement(base, "site", name="base_frame", pos="0 0 0", size="0.00012", rgba="0 0.8 1 1")
    add_measurement_site(base, 0)
    ET.SubElement(
        base,
        "geom",
        name="base_disk",
        type="cylinder",
        pos="-0.0002 0 0",
        euler="0 1.57079632679 0",
        size=_vec((outer_radius * 1.12, 0.0002)),
        mass="0",
        rgba="0.15 0.18 0.22 1",
        contype="0",
        conaffinity="0",
    )

    site_paths: Dict[int, List[str]] = {wire: [] for wire in range(1, 7)}
    add_wire_sites(
        base,
        "base",
        0.0,
        range(1, 7),
        angles,
        tendon_radius,
        site_radius,
        site_paths,
    )

    # Half element at the base, followed by N paired-hinge stations.
    add_backbone_geom(
        base,
        "section_1_element_0",
        element_length / 2.0,
        outer_radius,
        element_mass / 2.0,
        "0.25 0.55 0.9 0.72",
    )
    section_1_last = add_flexible_section(
        base=base,
        section=1,
        joints=joints,
        element_length=element_length,
        element_mass=element_mass,
        outer_radius=outer_radius,
        tendon_radius=tendon_radius,
        site_radius=site_radius,
        angles=angles,
        site_paths=site_paths,
        guide_wires=range(1, 7),
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        joint_range=joint_range,
        colour="0.25 0.55 0.9 0.72",
    )
    # Joint bodies lie at element midpoints (0.875, 2.625, ... mm).
    # A +0.875 mm local offset on bodies 3 and 7 therefore gives exact
    # arclengths 7 and 14 mm rather than merely selecting the nearest joint.
    add_measurement_site(
        named_descendant(base, "body", "section_1_joint_body_3"), 1, element_length / 2.0
    )
    add_measurement_site(
        named_descendant(base, "body", "section_1_joint_body_7"), 2, element_length / 2.0
    )

    interface = ET.SubElement(
        section_1_last,
        "body",
        name="section_interface",
        pos=_vec((element_length / 2.0, 0.0, 0.0)),
    )
    ET.SubElement(interface, "site", name="interface_center", pos="0 0 0", size="0.00015", rgba="0 1 0 1")
    add_measurement_site(interface, 3)
    add_rigid_section_geoms(
        interface,
        "inter_section_rigid",
        inter_section_length,
        outer_radius,
    )
    add_wire_sites(
        interface,
        "interface_inlet",
        0.0,
        range(1, 7),
        angles,
        tendon_radius,
        site_radius,
        site_paths,
    )
    # The far side of the rigid interface terminates proximal wires and starts
    # the distal routing, matching the physical spacer between active sections.
    add_wire_sites(
        interface,
        "interface",
        inter_section_length,
        range(1, 7),
        angles,
        tendon_radius,
        site_radius,
        site_paths,
    )

    section_2_mount = ET.SubElement(
        interface,
        "body",
        name="section_2_mount",
        pos=_vec((inter_section_length, 0.0, 0.0)),
    )
    add_backbone_geom(
        section_2_mount,
        "section_2_element_0",
        element_length / 2.0,
        outer_radius,
        element_mass / 2.0,
        "0.25 0.82 0.52 0.72",
    )
    section_2_last = add_flexible_section(
        base=section_2_mount,
        section=2,
        joints=joints,
        element_length=element_length,
        element_mass=element_mass,
        outer_radius=outer_radius,
        tendon_radius=tendon_radius,
        site_radius=site_radius,
        angles=angles,
        site_paths=site_paths,
        guide_wires=DISTAL_WIRES,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        joint_range=joint_range,
        colour="0.25 0.82 0.52 0.72",
    )
    add_measurement_site(
        named_descendant(section_2_mount, "body", "section_2_joint_body_3"), 4, element_length / 2.0
    )
    add_measurement_site(
        named_descendant(section_2_mount, "body", "section_2_joint_body_7"), 5, element_length / 2.0
    )
    tip_x = element_length / 2.0
    distal_tip = ET.SubElement(
        section_2_last,
        "body",
        name="distal_tip_rigid",
        pos=_vec((tip_x, 0.0, 0.0)),
    )
    add_rigid_section_geoms(
        distal_tip,
        "distal_tip_rigid",
        distal_tip_length,
        outer_radius,
    )
    add_measurement_site(distal_tip, 6)
    ET.SubElement(
        distal_tip,
        "site",
        name="tip_center",
        pos=_vec((distal_tip_length, 0, 0)),
        size="0.00018",
        rgba="1 0.2 0.1 1",
    )
    add_wire_sites(
        distal_tip,
        "tip",
        distal_tip_length,
        DISTAL_WIRES,
        angles,
        tendon_radius,
        site_radius,
        site_paths,
    )

    # Spatial tendon paths. In independent mode the distal effective tendon begins
    # at the inter-section disk; in coupled mode it passes eccentrically through
    # section 1 and therefore also loads that section.
    tendon_element = ET.SubElement(model, "tendon")
    routing_mode = tendon_config["routing_mode"]
    effective_paths: Dict[int, List[str]] = {}
    for wire in PROXIMAL_WIRES:
        # base -> all section-1 joint sites -> interface
        effective_paths[wire] = [
            name
            for name in site_paths[wire]
            if "section_2" not in name and "tip" not in name
        ]
    for wire in DISTAL_WIRES:
        if routing_mode == "independent":
            effective_paths[wire] = [
                name
                for name in site_paths[wire]
                if "interface" in name or "section_2" in name or "tip" in name
            ]
        else:
            effective_paths[wire] = site_paths[wire]

    for wire in range(1, 7):
        spatial = ET.SubElement(
            tendon_element,
            "spatial",
            name=f"wire_{wire}_tendon",
            width=_fmt(site_radius * 1.15),
            rgba=WIRE_COLOURS[wire],
        )
        for site_name in effective_paths[wire]:
            ET.SubElement(spatial, "site", site=site_name)

    actuator = ET.SubElement(model, "actuator")
    section_length = geometry["section_length_m"]
    kp = tendon_config["position_kp_n_per_m"]
    kv = tendon_config.get("position_kd_n_s_per_m", 0.0)
    pretension = tendon_config["pretension_n"]
    max_tension = tendon_config["maximum_tension_n"]
    max_angle = math.radians(tendon_config["maximum_commanded_bend_per_section_deg"])
    rest_lengths: Dict[int, float] = {}
    straight_controls: List[float] = []
    for wire in range(1, 7):
        proximal_routing_length = (
            section_length + inter_section_length
        )
        distal_routing_length = section_length + distal_tip_length
        rest_length = (
            proximal_routing_length
            if wire in PROXIMAL_WIRES
            else distal_routing_length
        )
        if wire in DISTAL_WIRES and routing_mode == "coupled":
            rest_length = proximal_routing_length + distal_routing_length
        rest_lengths[wire] = rest_length
        straight_control = rest_length - pretension / kp
        straight_controls.append(straight_control)
        excursion = tendon_radius * max_angle
        # Coupled distal wires traverse both sections. Their actuator must
        # retain one full section of travel after compensating a fully bent
        # proximal section.
        command_sections = (
            2.0 if wire in DISTAL_WIRES and routing_mode == "coupled" else 1.0
        )
        actuator_excursion = command_sections * excursion
        ET.SubElement(
            actuator,
            "position",
            name=f"wire_{wire}",
            tendon=f"wire_{wire}_tendon",
            kp=_fmt(kp),
            kv=_fmt(kv),
            ctrllimited="true",
            ctrlrange=_vec((
                straight_control - actuator_excursion,
                rest_length + actuator_excursion,
            )),
            forcelimited="true",
            forcerange=_vec((-max_tension, 0.0)),
        )

    sensor = ET.SubElement(model, "sensor")
    ET.SubElement(sensor, "framepos", name="tip_position", objtype="site", objname="tip_center")
    ET.SubElement(sensor, "framequat", name="tip_orientation", objtype="site", objname="tip_center")
    for wire in range(1, 7):
        ET.SubElement(sensor, "actuatorfrc", name=f"wire_{wire}_force", actuator=f"wire_{wire}")

    keyframe = ET.SubElement(model, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="straight_pretensioned",
        ctrl=_vec(straight_controls),
    )

    properties.update(
        {
            "straight_control_proximal_m": rest_lengths[1] - pretension / kp,
            "straight_control_distal_m": rest_lengths[4] - pretension / kp,
            "proximal_rest_length_m": rest_lengths[1],
            "distal_rest_length_m": rest_lengths[4],
            "total_rigid_length_m": rigid["total_rigid_length_m"],
            "total_continuum_length_m": geometry["total_continuum_length_m"],
        }
    )
    return model, properties


def generate(config_path: Path, output_path: Path) -> Dict[str, float]:
    config = load_config(config_path)
    model, properties = build_model(config)
    _indent(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(model).write(output_path, encoding="utf-8", xml_declaration=True)
    return properties


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    properties = generate(args.config.resolve(), args.output.resolve())
    print(f"Generated: {args.output.resolve()}")
    print(
        "Beam properties: "
        f"EI={properties['bending_ei_nm2']:.6g} N*m^2, "
        f"K_joint={properties['joint_bending_stiffness_nm_per_rad']:.6g} N*m/rad, "
        f"section_mass={properties['section_mass_kg'] * 1e3:.4g} g"
    )


if __name__ == "__main__":
    main()
