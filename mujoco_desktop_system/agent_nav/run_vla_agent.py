import argparse
import os

import cv2
import numpy as np
import torch

from agent_nav.bronchoscope_nav_env import BronchoscopeNavEnv, NavEnvConfig, load_target_points
from agent_nav.vla_model import build_vla

LOCKED_XML_REL = os.path.join("meshes", "cable_robot_bronch_final_seg2.xml")


def resolve_locked_xml() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.abspath(os.path.join(root, LOCKED_XML_REL))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到锁定仿真环境文件: {xml_path}")
    return xml_path


def load_vla(path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    clip_model_name = str(payload.get("clip_model_name", "ViT-H-14"))
    clip_pretrained = str(payload.get("clip_pretrained", "laion2b_s32b_b79k"))
    model = build_vla(
        state_dim=int(payload["state_dim"]),
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        train_clip=bool(payload.get("train_clip", False)),
        state_width=int(payload.get("state_width", 256)),
        fusion_width=int(payload.get("fusion_width", 1024)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def parse_args():
    parser = argparse.ArgumentParser(description="部署多模态导航策略")
    parser.add_argument("--model", default=os.path.join("agent_nav", "models", "vla_policy.pt"))
    parser.add_argument("--targets", default=os.path.join("agent_nav", "target_points.json"))
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--camera", default="tip_camera")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_vla(args.model, device)
    image_size = 224

    targets = load_target_points(args.targets)
    idx = int(np.clip(args.target_index, 0, len(targets) - 1))
    fixed_target = targets[idx]

    cfg = NavEnvConfig(
        xml_path=resolve_locked_xml(),
        target_points=[fixed_target],
        max_episode_steps=args.max_steps,
        render_camera=args.camera,
    )
    env = BronchoscopeNavEnv(config=cfg, render_mode="rgb_array", fixed_target=fixed_target)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep + 3000)
        target = info["target"]
        instruction = f"navigate bronchoscope tip to target {target[0]:.3f} {target[1]:.3f} {target[2]:.3f}"

        total_reward = 0.0
        while True:
            frame = env.render()
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)

            with torch.no_grad():
                image_t = torch.from_numpy(frame_u8).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                state_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                action = model(image_t, state_t, [instruction]).squeeze(0).cpu().numpy().astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if args.show:
                show_img = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    show_img,
                    f"dist={info['distance_to_target']:.4f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("Multimodal Policy Deployment", show_img)
                if cv2.waitKey(1) == 27:
                    terminated = True
                    truncated = True

            if terminated or truncated:
                print(
                    f"[Episode {ep + 1}] reward={total_reward:.3f}, "
                    f"success={info['is_success']}, final_dist={info['distance_to_target']:.5f}"
                )
                break

    env.close()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

