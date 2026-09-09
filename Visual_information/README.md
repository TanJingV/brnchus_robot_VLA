# Visual_information

本目录集中保存项目内所有相机采集、图像分割、七标志点识别、三维重建及视觉模型资源。

## 目录

- `d435_tdcr_capture/`：D435、侧相机、内窥镜、EM 与电机的同步采集。
- `seven_marker_3d_fusion/`：黑色连续体分割、七点追踪、深度融合与三维曲线重建。
- `seven_marker_yolo_pose/`：七关键点姿态模型的数据集定义与标注说明。
- `legacy_segmentation/`：老探针界面继续使用的 UNet 网络定义。
- `models/`：SAM2.1、CoTracker、YOLO Pose 与老探针 UNet 权重。
- `outputs/`：视觉重建输出；属于生成数据，不提交到 Git。
- `tests/`：视觉采集、融合和工作站界面的回归测试。
- `docs/`、`assets/`：实验文档和七色环打印模板。

## 启动

在项目根目录执行：

```powershell
python -m Visual_information.d435_tdcr_capture
python -m Visual_information.seven_marker_3d_fusion.ui
```

也可以继续双击项目根目录的 `run_d435_capture.bat` 和
`run_seven_marker_workstation.bat`。原始采集数据仍保存在根目录
`d435_sessions/`，整理过程不会移动或删除实验数据。

