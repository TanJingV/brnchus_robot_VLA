# 双段 Tendon-Driven Continuum MuJoCo 模型

这个目录是一个独立、可重新生成的双段末端模型。建模方法沿用
[ContinuumRoboticsLab/opencr-mujoco](https://github.com/ContinuumRoboticsLab/opencr-mujoco)
的核心思路：将连续骨架离散为串联刚体单元；每个离散站用两条正交转动
关节表达双向弯曲；由梁理论计算关节刚度；使用 MuJoCo 原生
`spatial tendon` 沿各离散站的 guide site 走线。

## 已落实的几何与排线

| 项目 | 模型值 |
|---|---:|
| 主动段数 | 2 |
| 每段主动长度 | 21 mm |
| 总主动长度 | 42 mm |
| 刚性连接段 | 6 mm段间连接 + 6 mm远端头 = 12 mm |
| 连续体总长（不含基座） | 54 mm |
| 外径 | 3.5 mm |
| 工作通道直径 | 2.6 mm |
| 环形壁厚 | 0.45 mm |
| 驱动丝直径 | 0.15 mm |
| 驱动丝中心半径 | 1.54 mm（按排布图中的标注解释） |
| 每段离散弯曲站 | 12，每站 2 个正交 hinge |

纵向为模型的 `+X` 轴。截面极角定义为：`0° = +Y`，`90° = +Z`。
完整支气管仿真中用于送进的长导管是独立的插入载体，不计入上述54 mm末端连续体组件。

| Wire | 极角 | 颜色 | 控制段 |
|---:|---:|---|---|
| 1 | 0° | 洋红 | 近端 |
| 2 | 120° | 洋红 | 近端 |
| 3 | 240° | 洋红 | 近端 |
| 4 | 60° | 橙色 | 远端 |
| 5 | 180° | 橙色 | 远端 |
| 6 | 300° | 橙色 | 远端 |

因此六根丝在截面上每隔 60° 交替排列，而每个主动段自己的三根丝仍相隔
120°。

## 路由方式

默认 `config.json` 使用 `routing_mode: "coupled"`，对应实物连续走线：

- Wire 1–3：从基座穿过近端主动段，走到段间刚性连接远侧并终止；
- Wire 4–6：从54 mm末端出发，沿1.54 mm偏心导向位置依次贯穿远端刚性头、
  远端主动段、段间刚性连接和近端主动段，最终连接到基座；
  其直线基准长度为54 mm。

因此 Wire 4–6 主要用于远端控制，但它们在近端段内也是偏心走线，会同时向
近端段施加弯矩。这属于真实的段间耦合，而不是程序错误。若仅用于算法对比、
希望完全隔离两个段，可临时改为 `routing_mode: "independent"` 后重新生成。

## 肺部非凸碰撞

完整模型不再导入 1912 个 `prec_part_*.stl` 凸分块，也没有把完整肺部 STL
直接作为普通 mesh geom 使用。普通 MuJoCo mesh geom 只按凸包碰撞，会封闭支气管
腔道。现在使用一个 `rigid="true"` 的二维 `flexcomp` 表示非凸三角面碰撞：

- `bronchus.stl`：151060 个三角面，只负责高精度可视化；
- `bronchus_collision_solid_nonconvex.stl`：20000 个三角面的高精度非凸碰撞面，采样表面误差 P99 为 0.239 mm；
- 可视网格和碰撞网格使用完全相同的位姿与毫米到米缩放；
- 碰撞面固定在 `pipe` body 上，不增加自由度，也不会发生形变；
- 碰撞厚度半径为 3.0 mm（总厚度 6.0 mm），接触 margin 为 0.3 mm；
- 碰撞面独占 Flex Group 4；在 MuJoCo Viewer 中按数字键 `4` 显示或隐藏。
- 肺壁接触采用法向硬约束，避免切向粘滑导致末端抖动；窗口每次只批量推进 20 个物理步，避免接触时阻塞界面。
- 插入轴使用 `Kp=2000 N/m`、`Kd=80 N·s/m` 的快速阻尼位置伺服，最大推拉力为 `±5000 N`；交互窗口不修改碰撞后的位姿或速度，阻挡、滑移和回弹均由 MuJoCo 接触动力学产生。
- 主动连续体已恢复原始快速力学参数：关节阻尼 `0.006 N·m·s/rad`、armature `1e-10 kg·m²`、钢丝位置增益 `Kp=10000 N/m`。自由空间弯曲不再为碰撞稳定性而增大关节阻尼或惯量。
- `contact_stabilization.py` 只在主动末端与肺壁产生真实接触时临时启用 `1e-5 kg·m²` 数值 armature，并在离开接触后恢复 XML 原值；它不锁止、不重置 `qpos/qvel`，也不修改弯曲刚度、关节阻尼或钢丝控制目标。
- 被动段外径统一调整为与主动段相同的 3.5 mm。连接处三个平端圆柱仅保留显示和质量；碰撞改由半径 1.75 mm 的 `splice_collision_fairing` 圆头 capsule 跨缝覆盖，并延伸到主动段前两个单元，消除直径台阶和连接圆环卡在分叉口的问题。

如需从高清 STL 重新生成碰撞面：

```powershell
& 'D:\anaconda3\envs\Bronchoscope\python.exe' `
  .\two_segment_tdcr_opencr\optimize_bronchial_mesh.py
```

脚本会输出双向采样表面误差，防止简化后的三角面跨越支气管分支。

## 文件

- `config.json`：尺寸、离散数、有效材料、腱参数和仿真设置；
- `generate_model.py`：参数化 MJCF 生成器，仅依赖 Python 标准库；
- `two_segment_tdcr.xml`：生成后的 MuJoCo 模型；
- `control_demo.py`：六路腱长控制示例，输入每段弯曲角和弯曲方向；
- `validate_model.py`：XML 编译、数值有限性和分段选择性检查。
- `integrate_bronchoscope.py`：将该双段模型替换进完整支气管镜 MJCF，并把
  `active_tdcr_base` 焊接到被动段末端 `cable_stiffB_last`；
- `validate_integrated_model.py`：检查完整模型、被动—主动拼接、相机、六腱
  长度和短时数值稳定性。
- `optimize_bronchial_mesh.py`：从高清肺部 STL 生成轻量非凸碰撞面并报告误差。
- `contact_stabilization.py`：交互程序和导航环境共用的接触态数值正则器，保证自由弯曲快速、肺壁接触稳定。
- `passive_joint_control.py`：运行时缩放 29 个被动球关节的等效刚度与阻尼；主 PyQt 控制页提供刚度滑块、阻尼倍率和恢复默认按钮。

## 生成与运行

在当前电脑已有的 MuJoCo Conda 环境中：

```powershell
& 'D:\anaconda3\envs\mujoco\python.exe' .\two_segment_tdcr_opencr\generate_model.py
& 'D:\anaconda3\envs\mujoco\python.exe' .\two_segment_tdcr_opencr\validate_model.py
& 'D:\anaconda3\envs\mujoco\python.exe' .\two_segment_tdcr_opencr\control_demo.py `
  --proximal-bend 45 --proximal-direction 0 `
  --distal-bend 45 --distal-direction 90 `
  --duration 3600 --render-fps 60
```

Viewer 默认每帧先批量执行约 1/60 秒对应的物理步，再刷新画面。模型仍使用
`0.0002 s` 物理步长，并按小批次推进物理计算，避免肺部接触时阻塞界面；因此显示时间会与
仿真时间基本同步。`--render-fps` 只改变显示刷新频率，不改变腱执行器动力学。

无窗口运行并记录轨迹：

```powershell
& 'D:\anaconda3\envs\mujoco\python.exe' .\two_segment_tdcr_opencr\control_demo.py `
  --headless --duration 2 --csv .\two_segment_tdcr_opencr\trajectory.csv
```

重新生成完整支气管镜模型：

```powershell
& 'D:\anaconda3\envs\mujoco\python.exe' `
  .\two_segment_tdcr_opencr\integrate_bronchoscope.py
& 'D:\anaconda3\envs\mujoco\python.exe' `
  .\two_segment_tdcr_opencr\validate_integrated_model.py
```

集成脚本只替换原来的 `seg1_body`、`seg2_body` 和末端 `slider` 主动结构；
插入滑台、650 mm 被动段、支气管模型和外部视角均保留。新的主动段底座与
被动段末端端面共面，接口圆盘半径匹配被动段的 1.8 mm。集成时先展开被动
`composite`，再把 `active_tdcr_base` 直接设为 `cable_stiffB_last` 在末端 site
处的子节点；接口没有 free joint 或 equality weld，全部 6 个相对自由度从
模型拓扑中被移除，因此不会发生相对运动。末端继续提供 `end_6` 和
`tip_camera`。

常曲率控制映射为

```text
l_i = L - F_pre/kp - r * theta * cos(alpha_i - phi)
```

其中 `theta` 是该段目标弯曲角，`phi` 是弯曲方向，`alpha_i` 是对应 Wire 的
截面极角。控制量仍是 `wire_1` 到 `wire_6` 六个独立 MuJoCo actuator。

## 需要实验标定的参数

图片没有给出激光切槽形状、槽距、残余梁宽、TPU 材料参数、驱动丝摩擦和
实测力—形状曲线，因此不能从 3.5/2.6 mm 直径直接得到真实等效弯曲刚度。
配置中的 `youngs_modulus_pa = 20 MPa` 是把“切槽 NiTi 骨架 + TPU 包覆”整体
均匀化后的初始值，不是块状 NiTi 的弹性模量。它的用途是让模型可运行并
保留正确的梁理论缩放关系。正式用于控制或论文前，应使用实物的拉丝位移、
拉力和中心线形状数据拟合以下量：

1. 每段等效 `EI` 或 `youngs_modulus_pa`；
2. `joint_damping_nms_per_rad`；
3. 六根丝各自的 `position_kp_n_per_m`、预紧力和零长偏置；
4. 若采用 coupled 路由，还需识别段间摩擦、回差和角度偏置。

总弯曲范围 `>190°` 被解释为两段总能力。当前控制范围允许每段最多命令
160°；在无接触验证中，两段局部弯曲角之和超过 190°。进入支气管接触环境
后，最终角度仍由有效刚度、最大拉力和接触共同决定。
