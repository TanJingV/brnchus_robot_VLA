# TDCR 七关键点 YOLO Pose 数据集

这里保存针对当前真实连续体的专用学习数据。通用 YOLO Pose 权重只认识人体关键点，不能直接识别 K0–K6。

## 操作

1. 运行项目工作站，在“末端目标初始化与跟踪”中点击“打开7点标注器”。
2. 打开采集得到的 `rgb.mp4`，每隔约 20–60 帧选一张有代表性的画面。
3. 必须从连接器一侧开始，依次点击 `K0 → K1 → … → K6` 的色带中心，然后保存。
4. 覆盖直线、两个方向弯曲、不同插入深度、肺模型遮挡、亮暗曝光和运动模糊。
5. 至少准备 80 张训练图和 20 张验证图；实际建议 300–600 张。
6. 返回主界面点击“训练YOLO Pose”。训练完成的权重会自动复制到：

   `Visual_information/models/tdcr_yolo_pose/best.pt`

重新加载 session 后，RGB顶部状态必须显示 `YOLO-POSE-CUDA ... LEARNED`。如果显示 `RULE-FALLBACK`，说明专用模型尚未加载，结果仍是旧规则法，不能当成YOLO识别结果。

## 训练命令

```powershell
D:\anaconda3\envs\Bronchoscope\python.exe -m Visual_information.seven_marker_3d_fusion.train_pose
```

训练前会检查数据量；数据不足会拒绝启动，避免生成看似可用但严重过拟合的模型。
