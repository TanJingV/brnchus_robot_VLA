#!/usr/bin/env python3
"""Generate the local figures used by TDCR_model_principles.md."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
plt.rcParams.update(
    {
        "font.family": "Microsoft YaHei",
        "font.size": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
    }
)

NAVY = "#17365D"
BLUE = "#3182CE"
GREEN = "#28A76F"
MAGENTA = "#D81BCE"
ORANGE = "#E66B19"
GOLD = "#F4B400"
RED = "#D64545"
GRAY = "#5B6573"
LIGHT = "#EDF3F8"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arrow(ax, start, end, color=NAVY, lw=1.8, mutation=12, connectionstyle="arc3"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation,
            connectionstyle=connectionstyle,
        ),
    )


def system_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.set_xlim(-8, 112)
    ax.set_ylim(-19, 22)
    ax.axis("off")

    # Drive and insertion mechanism.
    ax.add_patch(patches.FancyBboxPatch((-5, -8), 17, 16, boxstyle="round,pad=0.8", fc="#D9DEE5", ec=GRAY, lw=1.5))
    ax.text(3.5, 2.5, "驱动/滑块", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(3.5, -2.5, "插入轴 $d$\n" + r"$0\!\sim\!577$ mm", ha="center", va="center", fontsize=9)
    arrow(ax, (12, 0), (18, 0), color=GRAY, lw=2.2)

    # Passive chain represented with 29 rounded modules.
    x0, x1 = 18, 58
    for i in range(29):
        x = x0 + (x1 - x0) * i / 29
        ax.add_patch(patches.FancyBboxPatch((x, -2.3), 1.25, 4.6, boxstyle="round,pad=0.1", fc="#2D5AC7", ec="white", lw=0.4))
    ax.text((x0+x1)/2, 8.2, "被动段：29 个球关节", ha="center", color="#2248A3", weight="bold")
    ax.text((x0+x1)/2, 5.3, "等效刚度 500 N·m/rad；阻尼 10 N·m·s/rad", ha="center", fontsize=8.8, color=GRAY)

    # Rigid splice.
    ax.add_patch(patches.FancyBboxPatch((57.2, -3.3), 5.0, 6.6, boxstyle="round,pad=0.3", fc=GOLD, ec="#B8860B", lw=1.4))
    ax.text(59.7, -7.6, "刚性拼接\n无相对自由度", ha="center", va="top", fontsize=9, color="#7A5700")

    # Two active sections.
    for a, b, color, label in [(62, 82, BLUE, "近端主动段\n21 mm"), (82, 102, GREEN, "远端主动段\n21 mm")]:
        ax.add_patch(patches.FancyBboxPatch((a, -3), b-a, 6, boxstyle="round,pad=0.3", fc=color, ec="white", lw=1.3, alpha=0.90))
        for j in range(1, 12):
            x = a + (b-a)*j/12
            ax.plot([x, x], [-2.8, 2.8], color="white", lw=0.55, alpha=0.8)
        ax.text((a+b)/2, 8.0, label, ha="center", color=color, weight="bold")
        ax.text((a+b)/2, 5.0, "12 站 × 双轴 hinge", ha="center", fontsize=8.6, color=GRAY)
    ax.add_patch(patches.Circle((102.8, 0), 2.2, fc="#FFD54A", ec="#A87400", lw=1.2))
    ax.text(106, -5.8, "末端/相机", ha="center", fontsize=9)

    # Tendon paths: distal pass through both active sections.
    for dy in (-1.15, 0, 1.15):
        ax.plot([12, 62, 82], [dy+12, dy+12, dy+12], color=MAGENTA, lw=1.8, alpha=0.85)
    ax.text(72, 15.8, "Wire 1–3：到段间盘终止", color=MAGENTA, ha="center", weight="bold")
    for dy in (-1.15, 0, 1.15):
        ax.plot([12, 62, 82, 103], [dy-12, dy-12, dy-12, dy-12], color=ORANGE, lw=1.8, alpha=0.9)
    ax.text(80, -17.0, "Wire 4–6：穿过近端并连接到底座", color=ORANGE, ha="center", weight="bold")

    ax.text(55, 20, "双主动段 Tendon-Driven Continuum 支气管机器人总体结构", ha="center", fontsize=15, weight="bold", color=NAVY)
    save(fig, "01_system_overview")


def wire_cross_section() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-2.35, 2.55)
    outer, inner, tendon_r = 1.75, 1.30, 1.54
    ax.add_patch(patches.Circle((0, 0), outer, fc="#DDE8F8", ec="black", lw=2.2))
    ax.add_patch(patches.Circle((0, 0), inner, fc="white", ec=BLUE, lw=1.8))
    ax.add_patch(patches.Circle((0, 0), tendon_r, fill=False, ec="#7892B2", ls="--", lw=1.1))

    angles = {1:0, 4:60, 2:120, 5:180, 3:240, 6:300}
    for w, deg in angles.items():
        a = math.radians(deg)
        y, z = tendon_r*math.cos(a), tendon_r*math.sin(a)
        color = MAGENTA if w <= 3 else ORANGE
        ax.plot([0, y], [0, z], color="#98A6B5", lw=0.8, ls=":" if w not in (1,4) else "--")
        ax.add_patch(patches.Circle((y, z), 0.13, fc=color, ec="white", lw=1.0, zorder=5))
        tx, tz = 1.23*y, 1.23*z
        ax.text(tx, tz, f"Wire {w}\n{deg}°", ha="center", va="center", color=color, weight="bold", fontsize=9.5)

    arc = patches.Arc((0,0), 1.05, 1.05, theta1=0, theta2=60, ec=GOLD, lw=2)
    ax.add_patch(arc)
    ax.text(0.60, 0.30, "60°", color="#9A6B00", fontsize=11, weight="bold")
    ax.annotate("", xy=(0.0, -2.05), xytext=(1.75, -2.05), arrowprops=dict(arrowstyle="<->", color=GRAY))
    ax.text(0.88, -2.25, "$R_o=1.75$ mm", ha="center", color=GRAY)
    ax.text(0, 0.05, "工作通道\n" + r"$\varnothing\,2.6$ mm", ha="center", va="center", color=NAVY, weight="bold")
    ax.text(0, 2.35, "六根驱动丝截面排布（纵向轴为 +X）", ha="center", fontsize=14, weight="bold", color=NAVY)
    ax.text(0, -2.05+0.45, "丝中心半径 $r_t=1.54$ mm；单组内相隔 120°", ha="center", fontsize=9.5, color=GRAY)
    save(fig, "02_wire_cross_section")


def discretization() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-4.1, 4.6)
    ax.axis("off")
    ax.text(6, 4.1, "连续梁到 MuJoCo 串联刚体链的离散化", ha="center", fontsize=15, weight="bold", color=NAVY)
    y = 1.1
    for i in range(13):
        left = i-0.48 if i not in (0,12) else i-0.24
        width = 0.96 if i not in (0,12) else 0.48
        ax.add_patch(patches.FancyBboxPatch((left, y-0.55), width, 1.1, boxstyle="round,pad=0.05", fc=BLUE, ec="white", lw=1))
    for i in range(12):
        x=i+0.5
        ax.add_patch(patches.Circle((x, y+0.74), 0.16, fc=GREEN, ec="white"))
        ax.add_patch(patches.Circle((x, y-0.74), 0.16, fc=ORANGE, ec="white"))
        if i in (0,5,11):
            ax.text(x, y+1.15, f"站 {i+1}", ha="center", fontsize=8.5, color=GRAY)
    ax.text(-0.55, y, "半单元", ha="right", va="center", fontsize=9, color=GRAY)
    ax.text(12.55, y, "半单元", ha="left", va="center", fontsize=9, color=GRAY)
    ax.text(6, -0.45, r"$N=12,\quad h=L/N=1.75\,\mathrm{mm}$", ha="center", fontsize=12, color=NAVY)
    ax.text(1.5, -1.45, "绿色：绕 Y 轴 hinge", color=GREEN, weight="bold")
    ax.text(4.5, -1.45, "橙色：绕 Z 轴 hinge", color=ORANGE, weight="bold")
    ax.text(7.7, -1.45, "每站允许任意弯曲方向", color=GRAY)
    ax.add_patch(patches.FancyBboxPatch((0.3,-3.75), 12.0, 1.45, boxstyle="round,pad=0.25", fc=LIGHT, ec="#B8C6D5"))
    ax.text(6.3, -2.72, "能量等效：", ha="center", va="center", color=NAVY, weight="bold")
    ax.text(6.3, -3.27, r"$U_c=\frac{1}{2}EI\kappa^2L=N\frac{1}{2}k_\theta(\kappa h)^2\;\Rightarrow\;k_\theta=\frac{EI}{h}=\frac{NEI}{L}$", ha="center", va="center", fontsize=13, color=NAVY)
    save(fig, "03_discretization")


def tendon_routing() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(-3, 48)
    ax.set_ylim(-10, 10)
    ax.axis("off")
    ax.text(22.5, 9.2, "近端绳与远端绳的耦合走线", ha="center", fontsize=15, weight="bold", color=NAVY)
    ax.add_patch(patches.FancyBboxPatch((0,-4), 21, 8, boxstyle="round,pad=0.4", fc="#CFE1F8", ec=BLUE, lw=1.5))
    ax.add_patch(patches.FancyBboxPatch((21,-4), 21, 8, boxstyle="round,pad=0.4", fc="#D5F2E5", ec=GREEN, lw=1.5))
    for x in np.linspace(0,42,25):
        ax.plot([x,x],[-3.8,3.8],color="white",lw=0.45)
    ax.plot([21,21],[-5,5],color="#7B8794",lw=2.2)
    ax.text(10.5,5.3,"近端段 21 mm",ha="center",color=BLUE,weight="bold")
    ax.text(31.5,5.3,"远端段 21 mm",ha="center",color=GREEN,weight="bold")
    ax.text(21,-5.8,"段间盘",ha="center",color=GRAY)

    for y in (2.5,0,-2.5):
        ax.plot([-1,21],[y,y],color=MAGENTA,lw=2.0)
        ax.add_patch(patches.Circle((21,y),0.16,fc=MAGENTA,ec="white"))
    ax.text(10.5,7.0,"Wire 1–3：底座 → 近端导向孔 → 段间盘",ha="center",color=MAGENTA,weight="bold")

    for y in (1.5,0,-1.5):
        yy=y-0.45
        ax.plot([-1,0,21,42],[yy,yy,yy,yy],color=ORANGE,lw=2.0,alpha=0.92)
        ax.add_patch(patches.Circle((42,yy),0.16,fc=ORANGE,ec="white"))
    ax.text(28.5,-7.4,"Wire 4–6：末端 → 远端段 → 近端段 → 底座",ha="center",color=ORANGE,weight="bold")
    arrow(ax,(45,0),(42.4,0),color=RED)
    ax.text(46,1.1,r"拉力 $T_i$",ha="center",color=RED,weight="bold")
    ax.text(1,-8.9,"耦合含义：近端弯曲会改变远端三根绳的总几何长度与力矩臂；它不是两套完全独立的执行器。",ha="left",color=NAVY,fontsize=10.5)
    save(fig, "04_tendon_routing")


def pcc_kinematics() -> None:
    fig, axes = plt.subplots(1,2,figsize=(12.5,5.4))
    ax=axes[0]
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-0.5,5.8); ax.set_ylim(-0.6,5.2)
    R=3.3; theta=1.15; t=np.linspace(0,theta,100)
    x=R*np.sin(t); y=R*(1-np.cos(t))
    ax.plot(x,y,color=BLUE,lw=5,solid_capstyle="round")
    ax.plot([0,R*np.sin(theta)],[R,R*(1-np.cos(theta))],ls="--",color=GRAY,lw=1)
    ax.plot([0,0],[0,R],ls=":",color=GRAY)
    ax.add_patch(patches.Arc((0,R),1.5,1.5,theta1=-90,theta2=-90+math.degrees(theta),ec=GOLD,lw=2))
    ax.text(0.9,R-0.45,r"$\theta$",color="#956700",fontsize=13)
    ax.text(1.1,1.0,r"$R_c=1/\kappa$",rotation=33,color=GRAY)
    arrow(ax,(0,0),(1.1,0),color=NAVY); ax.text(1.22,-0.15,"+X",color=NAVY)
    ax.scatter([x[-1]],[y[-1]],s=95,c=RED,zorder=5)
    ax.text(x[-1]+0.18,y[-1],r"末端 $p(\theta,\phi)$",va="center",color=RED)
    ax.text(2.5,4.85,"单段常曲率圆弧",ha="center",fontsize=13,weight="bold",color=NAVY)

    ax=axes[1]
    ax.axis("off"); ax.set_xlim(-0.5,10.5); ax.set_ylim(-0.5,8)
    ax.text(5,7.5,"二维弯曲向量与三维弯曲平面",ha="center",fontsize=13,weight="bold",color=NAVY)
    ax.add_patch(patches.Circle((3.0,3.6),2.0,fill=False,ec="#92A6BA",lw=1.5))
    arrow(ax,(3,3.6),(5.0,3.6),color=NAVY); ax.text(5.15,3.45,"$b_y$",color=NAVY)
    arrow(ax,(3,3.6),(3,5.65),color=NAVY); ax.text(2.75,5.8,"$b_z$",color=NAVY)
    phi=0.72; end=(3+1.65*math.cos(phi),3.6+1.65*math.sin(phi))
    arrow(ax,(3,3.6),end,color=RED,lw=2.5)
    ax.add_patch(patches.Arc((3,3.6),1.3,1.3,theta1=0,theta2=math.degrees(phi),ec=GOLD,lw=2))
    ax.text(3.7,3.88,r"$\phi$",color="#956700",fontsize=13)
    ax.text(3,1.05,r"$\mathbf{b}=[b_y,b_z]^T$",ha="center",fontsize=12,color=NAVY)
    ax.text(3,0.45,r"$\theta=\|\mathbf{b}\|,\quad\phi=\operatorname{atan2}(b_z,b_y)$",ha="center",fontsize=11.5,color=NAVY)
    ax.add_patch(patches.FancyBboxPatch((6.1,1.2),3.8,4.9,boxstyle="round,pad=0.3",fc=LIGHT,ec="#B7C6D6"))
    ax.text(8,5.55,"双段复合",ha="center",weight="bold",color=NAVY,fontsize=12)
    ax.text(8,4.65,r"$\mathbf{p}= [d,0,0]^T+\mathbf{p}_1+\mathbf{R}_1\mathbf{p}_2$",ha="center",fontsize=11.5)
    ax.text(8,3.75,r"$\mathbf{R}=\mathbf{R}_1\mathbf{R}_2$",ha="center",fontsize=12)
    ax.text(8,2.75,"状态：",ha="center",color=GRAY)
    ax.text(8,2.10,r"$\mathbf{q}_k=[d,b_{1y},b_{1z},b_{2y},b_{2z}]^T$",ha="center",fontsize=10.7,color=NAVY)
    save(fig, "05_pcc_kinematics")


def control_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2,7.0))
    ax.axis("off"); ax.set_xlim(0,13.2); ax.set_ylim(0,7)
    ax.text(6.6,6.65,"轨迹逆解、模型补偿与双闭环控制架构",ha="center",fontsize=15,weight="bold",color=NAVY)
    def box(x,y,w,h,text,fc=LIGHT,ec=NAVY,small=False):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.18",fc=fc,ec=ec,lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9.2 if small else 10.2,color=NAVY,weight="bold" if not small else None)
    box(0.25,3.6,1.45,1.0,"末端轨迹\n" + r"$\mathbf{p}_d(t)$",fc="#E8F1FB")
    box(2.05,3.45,1.75,1.3,"PCC 逆运动学\n最小二乘 + 暖启动")
    box(4.25,3.45,1.65,1.3,"三轮仿真标定\n仿射逆补偿",fc="#FFF5D9",ec="#B98700")
    box(6.35,3.45,1.55,1.3,"绳长映射\n6 tendon + $d$",fc="#FCE8FA",ec=MAGENTA)
    box(8.35,3.45,1.45,1.3,"绳长/插入\nPD 内环",fc="#E7F5ED",ec=GREEN)
    box(10.25,3.35,1.65,1.5,"MuJoCo\n机器人 + 肺壁\n动力学",fc="#EAF0F6")
    box(5.8,1.1,2.5,1.15,"阻尼雅可比外环\n" + r"$\Delta q=KJ_\lambda^\dagger e$",fc="#FFECEC",ec=RED)
    box(10.25,0.95,1.65,1.15,"末端测量\n" + r"$\mathbf{p}_m$",fc="#F2F4F7",ec=GRAY)
    for a,b in [((1.7,4.1),(2.05,4.1)),((3.8,4.1),(4.25,4.1)),((5.9,4.1),(6.35,4.1)),((7.9,4.1),(8.35,4.1)),((9.8,4.1),(10.25,4.1))]: arrow(ax,a,b)
    arrow(ax,(11.1,3.35),(11.1,2.1),color=GRAY)
    arrow(ax,(10.25,1.5),(8.3,1.5),color=RED)
    arrow(ax,(5.8,1.65),(1.0,1.65),color=RED,connectionstyle="arc3,rad=-0.05")
    arrow(ax,(1.0,1.65),(1.0,3.6),color=RED)
    arrow(ax,(7.05,2.25),(7.05,3.45),color=RED)
    ax.text(3.35,1.93,r"位置误差 $\mathbf{e}=\mathbf{p}_d-\mathbf{p}_m$",ha="center",color=RED,fontsize=9.5)
    ax.text(6.6,5.45,"前馈链：目标轨迹 → 7 个控制量",ha="center",color=NAVY,weight="bold")
    ax.text(6.6,0.35,"外环只修正运动学命令；原有末端刚度、阻尼和 tendon PD 不被替换。",ha="center",color=GRAY,fontsize=10)
    save(fig, "06_control_architecture")


def collision_model() -> None:
    fig, axes=plt.subplots(1,3,figsize=(13.2,4.8))
    titles=["① 可视网格","② 非凸碰撞壁","③ 光滑机器人包络"]
    for ax,title in zip(axes,titles): ax.axis("off"); ax.set_aspect("equal"); ax.set_title(title,color=NAVY,weight="bold",fontsize=12)
    # Branching airway sketch.
    for ax in axes[:2]:
        for off,lw,c in [(0,18,"#D76A6A"),(0,12,"white")]:
            ax.plot([0,0,1.2],[2.8,1.1,0],lw=lw,color=c,solid_capstyle="round")
            ax.plot([0,-1.5,-2.2],[1.15,0.3,-0.4],lw=lw,color=c,solid_capstyle="round")
            ax.plot([0,1.55,2.25],[1.15,0.35,-0.4],lw=lw,color=c,solid_capstyle="round")
        ax.set_xlim(-3,3);ax.set_ylim(-1.2,3.4)
    axes[0].text(0,-0.95,"151,060 三角形 STL\n仅负责显示",ha="center",color=GRAY)
    # Triangle overlay on collision view.
    rng=np.random.default_rng(3)
    for _ in range(55):
        x,y=rng.uniform(-2.2,2.2),rng.uniform(-0.4,2.5)
        axes[1].plot([x,x+0.18,x-0.12,x],[y,y+0.13,y+0.18,y],color="#43A96B",lw=0.45,alpha=0.65)
    axes[1].text(0,-0.95,"20,000 三角形 rigid flex\n双侧扫掠半径 1.0 mm",ha="center",color=GRAY)

    ax=axes[2]; ax.set_xlim(-3,3);ax.set_ylim(-1.2,3.4)
    # Airway wall and capsule chain.
    ax.plot([-2.6,-1.4,0,1.4,2.6],[-0.1,0.35,0.5,0.2,-0.3],lw=16,color="#D76A6A",solid_capstyle="round",alpha=.55)
    ax.plot([-2.6,-1.4,0,1.4,2.6],[-0.1,0.35,0.5,0.2,-0.3],lw=10,color="white",solid_capstyle="round")
    pts=np.array([[-2.2,0.0],[-1.55,0.28],[-.85,.43],[-.15,.49],[.55,.42],[1.2,.23]])
    ax.plot(pts[:,0],pts[:,1],color=BLUE,lw=8,solid_capstyle="round")
    ax.scatter(pts[:,0],pts[:,1],s=92,c=BLUE,edgecolors="white",zorder=4)
    arrow(ax,(.6,.43),(.8,1.05),color=RED)
    ax.text(1.25,1.17,"法向接触力",color=RED,ha="center",weight="bold")
    ax.text(0,-0.95,"重叠 capsule，半径 1.75 mm\ncondim=1，切向摩擦为 0",ha="center",color=GRAY)
    fig.suptitle("肺壁限制工作空间的非凸接触建模",fontsize=15,weight="bold",color=NAVY,y=1.03)
    save(fig,"07_collision_model")


def guide_force_diagram() -> None:
    fig, ax=plt.subplots(figsize=(10.8,5.4))
    ax.axis("off");ax.set_xlim(-1,10);ax.set_ylim(-3.5,4.2)
    ax.text(4.5,3.75,"导向点上的 tendon 力与离散关节广义力",ha="center",fontsize=15,weight="bold",color=NAVY)
    centers=[0.6,3.0,5.4,7.8]
    for x in centers:
        ax.add_patch(patches.FancyBboxPatch((x-0.8,-.45),1.6,.9,boxstyle="round,pad=.08",fc=BLUE,ec="white"))
        ax.add_patch(patches.Circle((x+.82,0),.17,fc=ORANGE,ec="white",zorder=5))
    for a,b in zip(centers[:-1],centers[1:]):
        ax.plot([a+.82,b+.82], [0,0.45 if b==5.4 else 0],color=ORANGE,lw=2.5)
    arrow(ax,(5.4+.82,.45),(4.7,1.35),color=RED)
    arrow(ax,(5.4+.82,.45),(7.05,.12),color=RED)
    ax.text(4.55,1.55,r"$T\,\hat{t}^{-}$",color=RED,ha="center")
    ax.text(7.25,.2,r"$T\,\hat{t}^{+}$",color=RED,ha="center")
    ax.text(6.25,1.1,"导向孔合力\n" + r"$\mathbf{f}_k=T(\hat{t}^+-\hat{t}^-)$",ha="center",color=NAVY,weight="bold")
    ax.add_patch(patches.Arc((3.82,0),1.2,1.2,theta1=90,theta2=260,ec=GREEN,lw=2.2))
    ax.text(3.0,-1.35,"弹性铰链",ha="center",color=GREEN,weight="bold")
    ax.text(3.0,-1.95,r"$\tau_s=-k_\theta q-c_\theta\dot q$",ha="center",fontsize=12,color=NAVY)
    ax.add_patch(patches.FancyBboxPatch((0.0,-3.1),8.7,.75,boxstyle="round,pad=.2",fc=LIGHT,ec="#B7C6D6"))
    ax.text(4.35,-2.72,r"虚功关系：$\delta W=-\mathbf{T}^T\delta\mathbf{l}=-\mathbf{T}^T\mathbf{J}_l\delta\mathbf{q}$，因此 $\tau_t=-\mathbf{J}_l^T\mathbf{T}$",ha="center",va="center",fontsize=11.5,color=NAVY)
    save(fig,"08_tendon_forces")


def _read_result(path: Path):
    rows=list(csv.DictReader(path.open(encoding="utf-8")))
    time=np.array([float(r["time_s"]) for r in rows])
    target=np.array([[float(r[f"target_{a}_m"]) for a in "xyz"] for r in rows])*1000
    actual=np.array([[float(r[f"actual_{a}_m"]) for a in "xyz"] for r in rows])*1000
    return time,target,actual


def tracking_results() -> bool:
    before=ROOT/"two_segment_tdcr_opencr"/"ik_trajectory_before_improvement.csv"
    # Use the locked validation artifact for reproducible documentation.
    # The default live-results file can be overwritten when a viewer closes.
    after=ROOT/"two_segment_tdcr_opencr"/"ik_feedback_iter3.csv"
    if not before.exists() or not after.exists():
        print("Skipping tracking-results figure: optional IK CSV data is absent.")
        return False
    tb,pb,ab=_read_result(before); ta,pa,aa=_read_result(after)
    eb=np.linalg.norm(ab-pb,axis=1); ea=np.linalg.norm(aa-pa,axis=1)
    fig=plt.figure(figsize=(13.2,8.2))
    gs=fig.add_gridspec(2,3,height_ratios=[1.1,.9])
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2]); ax4=fig.add_subplot(gs[1,:2]); ax5=fig.add_subplot(gs[1,2])
    for ax,title,target,actual in [(ax1,"修改前：Y–Z 投影",pb,ab),(ax2,"修改后：Y–Z 投影",pa,aa)]:
        ax.plot(target[:,1],target[:,2],"o-",ms=3.2,lw=1.4,color=BLUE,label="目标")
        ax.plot(actual[:,1],actual[:,2],"o-",ms=3.2,lw=1.4,color=GREEN,label="实际")
        ax.set_xlabel("Y / mm");ax.set_ylabel("Z / mm");ax.grid(alpha=.22);ax.set_aspect("equal",adjustable="datalim");ax.set_title(title,color=NAVY,weight="bold");ax.legend(frameon=False,fontsize=8)
    ax3.bar(["修改前","修改后"],[math.sqrt(np.mean(eb**2)),math.sqrt(np.mean(ea**2))],color=["#AAB4C0",GREEN],width=.58)
    ax3.set_ylabel("轨迹 RMSE / mm");ax3.set_title("总体误差",color=NAVY,weight="bold");ax3.grid(axis="y",alpha=.2)
    for i,v in enumerate([math.sqrt(np.mean(eb**2)),math.sqrt(np.mean(ea**2))]): ax3.text(i,v+.15,f"{v:.2f}",ha="center",weight="bold")
    ax4.plot(tb,eb,color="#8A949E",lw=2,label="修改前")
    ax4.plot(ta,ea,color=GREEN,lw=2,label="修改后")
    ax4.fill_between(ta,0,ea,color=GREEN,alpha=.12)
    ax4.set_xlabel("时间 / s");ax4.set_ylabel("末端位置误差 / mm");ax4.set_title("逐采样点误差",color=NAVY,weight="bold");ax4.grid(alpha=.22);ax4.legend(frameon=False)
    conv=[3.5019,2.6679,2.0775,1.7998]
    ax5.plot(range(4),conv,"o-",color=ORANGE,lw=2.2,ms=7)
    ax5.set_xticks(range(4),["试验1","试验2","试验3","正式追踪"],rotation=18)
    ax5.set_ylabel("RMSE / mm");ax5.set_title("三轮标定收敛",color=NAVY,weight="bold");ax5.grid(alpha=.22)
    for i,v in enumerate(conv): ax5.text(i,v+.09,f"{v:.2f}",ha="center",fontsize=8.5)
    fig.suptitle("逆运动学与控制改进前后轨迹误差",fontsize=15,weight="bold",color=NAVY,y=.99)
    fig.tight_layout(rect=(0,0,1,.96))
    save(fig,"09_tracking_results")
    return True


if __name__ == "__main__":
    system_overview()
    wire_cross_section()
    discretization()
    tendon_routing()
    pcc_kinematics()
    control_architecture()
    collision_model()
    guide_force_diagram()
    tracking_results()
    print(f"Generated figures in {ASSETS}")
