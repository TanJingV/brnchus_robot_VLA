import argparse
import os
from typing import List

import cv2
import numpy as np
import torch

from agent_nav.bronchoscope_nav_env import BronchoscopeNavEnv, NavEnvConfig, load_target_points
from agent_nav.expert_policy import heuristic_expert_action
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
    parser = argparse.ArgumentParser(description="DAgger 回灌数据采集")
    parser.add_argument("--model", default=os.path.join("agent_nav", "models", "vla_policy.pt"))
    parser.add_argument("--targets", default=os.path.join("agent_nav", "target_points.json"))
    parser.add_argument("--out", default=os.path.join("agent_nav", "data", "dagger_dataset.npz"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--camera", default="tip_camera")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_vla(args.model, device)
    image_size = 224

    cfg = NavEnvConfig(
        xml_path=resolve_locked_xml(),
        target_points=load_target_points(args.targets),
        max_episode_steps=args.max_steps,
        render_camera=args.camera,
    )
    env = BronchoscopeNavEnv(config=cfg, render_mode="rgb_array")

    images: List[np.ndarray] = []
    states: List[np.ndarray] = []
    texts: List[str] = []
    actions: List[np.ndarray] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep + 1000)
        target = info["target"]
        instruction = f"navigate bronchoscope tip to target {target[0]:.3f} {target[1]:.3f} {target[2]:.3f}"

        for _ in range(args.max_steps):
            frame = env.render()
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)

            with torch.no_grad():
                image_t = torch.from_numpy(frame_u8).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                state_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
                pred_action = model(image_t, state_t, [instruction]).squeeze(0).cpu().numpy().astype(np.float32)

            expert_action = heuristic_expert_action(obs)
            action = np.clip(0.9 * pred_action + 0.1 * expert_action, -1.0, 1.0)

            images.append(frame_u8)
            states.append(obs.astype(np.float32))
            texts.append(instruction)
            actions.append(expert_action.astype(np.float32))

            obs, reward, terminated, truncated, info = env.step(action)

            if args.show:
                bgr = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"ep={ep} dist={info['distance_to_target']:.4f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("DAgger Collection", bgr)
                if cv2.waitKey(1) == 27:
                    args.show = False
                    cv2.destroyAllWindows()

            if terminated or truncated:
                break

        print(f"[DAgger] episode={ep + 1}/{args.episodes}, final_dist={info['distance_to_target']:.5f}")

    env.close()
    cv2.destroyAllWindows()

    np.savez_compressed(
        args.out,
        images=np.stack(images, axis=0),
        states=np.stack(states, axis=0),
        texts=np.asarray(texts, dtype="<U256"),
        actions=np.stack(actions, axis=0),
    )
    print(f"[Done] DAgger 数据保存: {args.out}, samples={len(actions)}")


if __name__ == "__main__":
    main()

