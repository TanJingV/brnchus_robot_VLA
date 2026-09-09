"""Train and validate the project-specific seven-keypoint YOLO Pose model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


VISUAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VISUAL_ROOT.parent
DATASET = VISUAL_ROOT / "seven_marker_yolo_pose" / "tdcr_pose.yaml"
OUTPUT = VISUAL_ROOT / "models" / "tdcr_yolo_pose"


def _prepare_environment() -> None:
    config = VISUAL_ROOT / ".ultralytics"
    config.mkdir(parents=True, exist_ok=True)
    (VISUAL_ROOT / ".hf").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    os.environ.setdefault("XDG_CONFIG_HOME", str(config))
    os.environ.setdefault("HF_HOME", str(VISUAL_ROOT / ".hf"))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _runtime_dataset_yaml() -> Path:
    """Write an absolute dataset descriptor independent of Ultralytics settings."""
    path = DATASET.parent.resolve().as_posix()
    runtime = DATASET.parent / ".tdcr_pose_runtime.yaml"
    runtime.write_text(
        f"path: {path}\n"
        "train: images/train\nval: images/val\ntest: images/val\n"
        "kpt_shape: [7, 3]\nflip_idx: [0, 1, 2, 3, 4, 5, 6]\n"
        "names:\n  0: TDCR_terminal\n"
        "kpt_names:\n  0: [K0, K1, K2, K3, K4, K5, K6]\n",
        encoding="utf-8",
    )
    return runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default=str(OUTPUT / "yolo11n-pose.pt"))
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    train_labels = list((DATASET.parent / "labels" / "train").glob("*.txt"))
    val_labels = list((DATASET.parent / "labels" / "val").glob("*.txt"))
    if len(train_labels) < 80 or len(val_labels) < 20:
        raise SystemExit(
            f"标注不足：train={len(train_labels)}, val={len(val_labels)}。"
            "至少需要80张训练帧和20张验证帧；建议覆盖不同弯曲、插入、曝光和遮挡，共300–600张。"
        )
    _prepare_environment()
    from ultralytics import YOLO

    dataset_yaml = _runtime_dataset_yaml()
    model = YOLO(args.model)
    result = model.train(
        data=str(dataset_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=4, cache="disk", amp=True, close_mosaic=15,
        degrees=12.0, translate=0.12, scale=0.30, perspective=0.0005,
        fliplr=0.0, flipud=0.0, hsv_h=0.035, hsv_s=0.55, hsv_v=0.35,
        project=str(OUTPUT / "runs"), name="train", exist_ok=True,
        plots=True, patience=45,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练结束但未找到权重：{best}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, OUTPUT / "best.pt")
    model = YOLO(str(OUTPUT / "best.pt"))
    metrics = model.val(data=str(dataset_yaml), imgsz=args.imgsz, device=args.device, plots=True)
    print(f"best.pt: {OUTPUT / 'best.pt'}")
    print(f"box mAP50-95={metrics.box.map:.4f}; pose mAP50-95={metrics.pose.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
