# Neural segmentation models

`sam2.1-hiera-tiny/` contains the official Meta `facebook/sam2.1-hiera-tiny`
checkpoint used by `seven_marker_3d_fusion.neural_body_segmenter`.

- Source: https://huggingface.co/facebook/sam2.1-hiera-tiny
- Paper: *SAM 2: Segment Anything in Images and Videos*
- License declared by the model repository: Apache-2.0
- Runtime: PyTorch CUDA; the current project defaults to an NVIDIA GPU.

The directory is ignored by Git because `model.safetensors` is approximately
156 MB. To restore it, run from the project environment:

```powershell
$env:HF_HUB_DISABLE_XET="1"
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-tiny', local_dir='Visual_information/models/sam2.1-hiera-tiny', allow_patterns=['config.json','model.safetensors','preprocessor_config.json','processor_config.json','video_preprocessor_config.json'])"
```

Run that command from the repository root. The JSON configuration files are
versioned; downloaded checkpoint binaries are excluded.

## Other checkpoints

- `legacy_unet/1205weights_49.pth`: existing project segmentation weights,
  relocated from `window/1205weights_49.pth` without changing their contents.
- `cotracker3/scaled_offline.pth`: obtain the official checkpoint using the
  [CoTracker3 setup guide](../seven_marker_3d_fusion/COTRACKER3_GUIDE.md).
- `tdcr_yolo_pose/yolo11n-pose.pt`: optional Ultralytics initialization weights;
  see the [pose dataset guide](../seven_marker_yolo_pose/README.md).

Training runs and local captures are not included in the public source release.
