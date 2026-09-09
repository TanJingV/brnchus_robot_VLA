#!/usr/bin/env python3
"""Create a publication-ready schematic of the 54 mm MuJoCo TDCR model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CONFIG_PATH = PROJECT_ROOT / "two_segment_tdcr_opencr" / "config.json"
OUTPUT_STEM = HERE / "tdcr_mujoco_modeling_schematic"

WIDTH_MM = 183.0
HEIGHT_MM = 108.0
MM_PER_INCH = 25.4

COLORS = {
    "ink": "#252A31",
    "muted": "#68717D",
    "line": "#526170",
    "proximal": "#3775BA",
    "proximal_light": "#DCEAF7",
    "distal": "#42949E",
    "distal_light": "#DDF0F1",
    "rigid": "#4D5661",
    "rigid_light": "#AEB6BE",
    "wire_proximal": "#B54A87",
    "wire_distal": "#E07A3F",
    "joint_y": "#275DAD",
    "joint_z": "#D56A3A",
    "channel": "#F5F6F7",
    "panel_bg": "#FAFBFC",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def load_geometry() -> dict[str, float]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    geometry = config["geometry"]
    rigid = geometry["rigid_sections"]
    values = {
        "section": float(geometry["section_length_m"]) * 1000.0,
        "interface": float(rigid["inter_section_length_m"]) * 1000.0,
        "tip": float(rigid["distal_tip_length_m"]) * 1000.0,
        "total": float(geometry["total_continuum_length_m"]) * 1000.0,
        "outer_diameter": float(geometry["outer_diameter_m"]) * 1000.0,
        "channel_diameter": float(geometry["working_channel_diameter_m"]) * 1000.0,
        "tendon_radius": float(config["tendons"]["radius_from_center_m"]) * 1000.0,
        "wire_diameter": float(config["tendons"]["wire_diameter_m"]) * 1000.0,
        "joints": int(config["discretization"]["joints_per_section"]),
    }
    assert math.isclose(2 * values["section"] + values["interface"] + values["tip"], values["total"])
    assert math.isclose(values["total"], 54.0)
    return values


def panel_label(ax, label: str) -> None:
    ax.text(
        0.0,
        1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def dim_arrow(ax, x0, x1, y, label, color=None, text_offset=0.35, lw=0.85) -> None:
    color = color or COLORS["line"]
    ax.plot([x0, x0], [y - 0.22, y + 0.22], color=color, lw=lw, clip_on=False)
    ax.plot([x1, x1], [y - 0.22, y + 0.22], color=color, lw=lw, clip_on=False)
    arrow = patches.FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="<->",
        mutation_scale=6,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    ax.text(
        (x0 + x1) / 2,
        y + text_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=COLORS["ink"],
    )


def draw_active_section(ax, x0, length, y0, height, face, edge, joints) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            length,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.18",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            zorder=2,
        )
    )
    spacing = length / joints
    for index in range(joints + 1):
        x = x0 + index * spacing
        ax.plot([x, x], [y0 + 0.14, y0 + height - 0.14], color=edge, lw=0.48, alpha=0.82, zorder=3)
    ax.plot([x0, x0 + length], [y0 + height / 2] * 2, color=COLORS["ink"], lw=1.0, zorder=4)


def draw_rigid_section(ax, x0, length, y0, height, name) -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0 - 0.14),
            length,
            height + 0.28,
            boxstyle="round,pad=0.02,rounding_size=0.32",
            facecolor=COLORS["rigid"],
            edgecolor=COLORS["ink"],
            linewidth=0.95,
            zorder=5,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x0 + 0.15, y0 + height * 0.60),
            max(length - 0.3, 0.1),
            height * 0.14,
            facecolor=COLORS["rigid_light"],
            edgecolor="none",
            alpha=0.55,
            zorder=6,
        )
    )
    ax.text(x0 + length / 2, y0 + height / 2, name, ha="center", va="center", fontsize=5.8, color="white", zorder=7)


def draw_overall(ax, g) -> None:
    panel_label(ax, "a")
    ax.set_xlim(-7.5, 57.0)
    ax.set_ylim(-3.1, 8.0)
    ax.axis("off")

    y0, height = 0.0, 2.25
    section = g["section"]
    interface = g["interface"]
    tip = g["tip"]

    # The base is shown for orientation but excluded from the 54 mm dimension.
    ax.add_patch(
        patches.FancyBboxPatch(
            (-6.2, -0.25),
            6.2,
            height + 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.30",
            facecolor="#303740",
            edgecolor=COLORS["ink"],
            linewidth=0.95,
            zorder=4,
        )
    )
    ax.add_patch(patches.Rectangle((-5.8, 1.38), 5.2, 0.24, color="#838C96", alpha=0.62, zorder=5))
    ax.text(-3.1, 1.12, "Base", ha="center", va="center", color="white", fontsize=6.8, fontweight="bold", zorder=6)
    ax.text(-3.1, 0.50, "excluded", ha="center", va="center", color="#D9DDE1", fontsize=5.5, zorder=6)

    x_proximal = 0.0
    x_interface = x_proximal + section
    x_distal = x_interface + interface
    x_tip = x_distal + section
    x_end = x_tip + tip

    draw_active_section(ax, x_proximal, section, y0, height, COLORS["proximal_light"], COLORS["proximal"], g["joints"])
    draw_rigid_section(ax, x_interface, interface, y0, height, "Rigid")
    draw_active_section(ax, x_distal, section, y0, height, COLORS["distal_light"], COLORS["distal"], g["joints"])
    draw_rigid_section(ax, x_tip, tip, y0, height, "Rigid tip")

    # Representative tendon paths; the cross-section panel defines all six.
    for offset in (0.36, height - 0.36):
        ax.plot([0, x_interface + interface], [y0 + offset] * 2, color=COLORS["wire_proximal"], lw=1.0, zorder=8)
        ax.plot([0, x_end], [y0 + offset + (0.11 if offset < height / 2 else -0.11)] * 2, color=COLORS["wire_distal"], lw=0.9, zorder=8)

    ax.text(section / 2, y0 + height + 0.10, "Proximal active segment", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=COLORS["proximal"])
    ax.text(x_distal + section / 2, y0 + height + 0.10, "Distal active segment", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=COLORS["distal"])

    dim_arrow(ax, 0, section, 3.18, "21 mm")
    dim_arrow(ax, x_interface, x_distal, 3.18, "6 mm")
    dim_arrow(ax, x_distal, x_tip, 3.18, "21 mm")
    dim_arrow(ax, x_tip, x_end, 3.18, "6 mm")
    dim_arrow(ax, 0, x_end, 5.56, "Total continuum length = 54 mm", color=COLORS["ink"], text_offset=0.32, lw=1.0)

    ax.annotate(
        "Model longitudinal axis (+x)",
        xy=(54.0, -1.52),
        xytext=(35.0, -1.52),
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color=COLORS["muted"], shrinkA=0, shrinkB=0),
    )
    ax.plot([], [], color=COLORS["wire_proximal"], lw=1.2, label="Wires 1–3")
    ax.plot([], [], color=COLORS["wire_distal"], lw=1.2, label="Wires 4–6")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, -0.02), ncol=2, frameon=False, fontsize=5.8, handlelength=2.0, columnspacing=1.4)


def draw_joint_symbol(ax, x, y) -> None:
    ax.add_patch(patches.Circle((x, y), 0.10, facecolor="white", edgecolor=COLORS["ink"], lw=0.8, zorder=5))
    ax.add_patch(patches.Arc((x, y), 0.50, 0.50, theta1=35, theta2=290, color=COLORS["joint_y"], lw=0.9, zorder=6))
    ax.add_patch(patches.FancyArrowPatch((x - 0.19, y + 0.15), (x - 0.23, y + 0.08), arrowstyle="-|>", mutation_scale=5, color=COLORS["joint_y"], lw=0.7, zorder=7))
    ax.add_patch(patches.Arc((x, y), 0.72, 0.26, theta1=5, theta2=330, color=COLORS["joint_z"], lw=0.9, zorder=6))


def draw_discretization(ax, g) -> None:
    panel_label(ax, "b")
    ax.set_xlim(-0.45, 7.25)
    ax.set_ylim(-1.05, 3.25)
    ax.axis("off")
    ax.text(0.0, 2.82, "Discrete backbone model", ha="left", va="center", fontsize=7.2, fontweight="bold", color=COLORS["ink"])

    y = 1.18
    element = g["section"] / g["joints"]
    visual_length = 1.42
    for index in range(4):
        x0 = 0.25 + index * visual_length
        ax.add_patch(
            patches.FancyBboxPatch(
                (x0, y - 0.25),
                visual_length * 0.88,
                0.50,
                boxstyle="round,pad=0.01,rounding_size=0.16",
                facecolor=COLORS["proximal_light"],
                edgecolor=COLORS["proximal"],
                linewidth=0.9,
            )
        )
        ax.plot([x0 + 0.10, x0 + visual_length * 0.80], [y + 0.13] * 2, color=COLORS["wire_proximal"], lw=0.9)
        ax.plot([x0 + 0.10, x0 + visual_length * 0.80], [y - 0.13] * 2, color=COLORS["wire_distal"], lw=0.9)
        if index < 3:
            draw_joint_symbol(ax, x0 + visual_length * 0.94, y)

    dim_arrow(ax, 0.25, 0.25 + visual_length * 0.88, 0.22, f"Δs = {element:.2f} mm", text_offset=0.20)
    ax.annotate("paired hinge\n$(q_y, q_z)$", xy=(1.58, 1.18), xytext=(1.35, 2.18), ha="center", va="center", fontsize=6.3, color=COLORS["ink"], arrowprops=dict(arrowstyle="->", lw=0.75, color=COLORS["line"]))
    ax.text(4.85, 2.20, "12 stations per active segment", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
    ax.text(4.85, 1.82, "24 rotational DoF per segment", ha="center", va="center", fontsize=6.1, color=COLORS["muted"])
    ax.text(4.85, 0.36, r"$K_{\theta}=N EI/L$", ha="center", va="center", fontsize=7.2, color=COLORS["ink"])
    ax.text(4.85, -0.05, "native spatial-tendon routing", ha="center", va="center", fontsize=6.1, color=COLORS["muted"])


def draw_cross_section(ax, g) -> None:
    panel_label(ax, "c")
    ax.set_aspect("equal")
    ax.set_xlim(-2.75, 3.55)
    ax.set_ylim(-2.55, 2.80)
    ax.axis("off")
    ax.text(-2.55, 2.43, "Cross-section and tendon layout", ha="left", va="center", fontsize=7.2, fontweight="bold", color=COLORS["ink"])

    ro = g["outer_diameter"] / 2.0
    ri = g["channel_diameter"] / 2.0
    rt = g["tendon_radius"]
    rw = g["wire_diameter"] / 2.0

    ax.add_patch(patches.Circle((0, 0), ro, facecolor="#E7EAED", edgecolor=COLORS["ink"], lw=1.0))
    ax.add_patch(patches.Circle((0, 0), ri, facecolor=COLORS["channel"], edgecolor="#7F8993", lw=0.8))
    ax.text(0, 0, "Working\nchannel", ha="center", va="center", fontsize=5.7, color=COLORS["muted"])

    angles = [0, 120, 240, 60, 180, 300]
    for wire, angle in enumerate(angles, start=1):
        theta = math.radians(angle)
        x, y = rt * math.cos(theta), rt * math.sin(theta)
        color = COLORS["wire_proximal"] if wire <= 3 else COLORS["wire_distal"]
        ax.add_patch(patches.Circle((x, y), max(rw, 0.105), facecolor=color, edgecolor="white", lw=0.45, zorder=5))
        label_scale = 1.22
        ax.text(x * label_scale, y * label_scale, f"W{wire}", ha="center", va="center", fontsize=5.3, color=COLORS["ink"], fontweight="bold")

    # Dimension leaders are kept outside the working channel.
    ax.annotate("", xy=(-ro, -2.15), xytext=(ro, -2.15), arrowprops=dict(arrowstyle="<->", lw=0.8, color=COLORS["line"]))
    ax.plot([-ro, -ro], [-2.02, -2.27], color=COLORS["line"], lw=0.75)
    ax.plot([ro, ro], [-2.02, -2.27], color=COLORS["line"], lw=0.75)
    ax.text(0, -2.42, f"Outer diameter = {g['outer_diameter']:.1f} mm", ha="center", va="center", fontsize=6.0, color=COLORS["ink"])
    ax.annotate(
        f"Tendon radius\n{rt:.2f} mm",
        xy=(rt * math.cos(math.radians(60)), rt * math.sin(math.radians(60))),
        xytext=(2.68, 1.95),
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["ink"],
        arrowprops=dict(arrowstyle="->", lw=0.7, color=COLORS["line"]),
    )
    ax.text(2.76, -0.42, "W1–W3  proximal", color=COLORS["wire_proximal"], fontsize=5.8, ha="center")
    ax.text(2.76, -0.84, "W4–W6  distal", color=COLORS["wire_distal"], fontsize=5.8, ha="center")
    ax.text(2.76, -1.32, "60° alternating", color=COLORS["muted"], fontsize=5.6, ha="center")


def export_figure(fig) -> None:
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def verify_exports() -> None:
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = OUTPUT_STEM.with_suffix(suffix)
        if not path.exists() or path.stat().st_size < 1000:
            raise RuntimeError(f"Missing or empty export: {path}")
    svg_text = OUTPUT_STEM.with_suffix(".svg").read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text was converted to paths instead of remaining editable.")
    expected_px = (
        round(WIDTH_MM / MM_PER_INCH * 600),
        round(HEIGHT_MM / MM_PER_INCH * 600),
    )
    with Image.open(OUTPUT_STEM.with_suffix(".png")) as preview:
        if any(abs(actual - expected) > 1 for actual, expected in zip(preview.size, expected_px)):
            raise RuntimeError(f"Unexpected PNG dimensions: {preview.size}, expected {expected_px}")
    with Image.open(OUTPUT_STEM.with_suffix(".tiff")) as tiff:
        if any(abs(actual - expected) > 1 for actual, expected in zip(tiff.size, expected_px)):
            raise RuntimeError(f"Unexpected TIFF dimensions: {tiff.size}, expected {expected_px}")


def main() -> None:
    geometry = load_geometry()
    fig = plt.figure(
        figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.20, 1.0),
        width_ratios=(1.35, 1.0),
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.965,
        hspace=0.22,
        wspace=0.14,
    )
    draw_overall(fig.add_subplot(grid[0, :]), geometry)
    draw_discretization(fig.add_subplot(grid[1, 0]), geometry)
    draw_cross_section(fig.add_subplot(grid[1, 1]), geometry)
    export_figure(fig)
    plt.close(fig)
    verify_exports()
    print(f"Generated publication figure: {OUTPUT_STEM}")
    print(f"Final size: {WIDTH_MM:.0f} x {HEIGHT_MM:.0f} mm")
    print("Exports: editable SVG, PDF, 600 dpi PNG, 600 dpi LZW TIFF")


if __name__ == "__main__":
    main()
