import argparse
import os
from typing import List

import cv2
import numpy as np

from agent_nav.bronchoscope_nav_env import BronchoscopeNavEnv, NavEnvConfig, load_target_points
from agent_nav.expert_policy import heuristic_expert_action

LOCKED_XML_REL = os.path.join("meshes", "cable_robot_bronch_final_seg2.xml")


def resolve_locked_xml() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.abspath(os.path.join(root, LOCKED_XML_REL))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到锁定仿真环境文件: {xml_path}")
    return xml_path


def parse_args():
    parser = argparse.ArgumentParser(description="采集 VLA 训练数据（图像+状态+文本+动作）")
    parser.add_argument("--targets", default=os.path.join("agent_nav", "target_points.json"))
    parser.add_argument("--out", default=os.path.join("agent_nav", "data", "demo_dataset.npz"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--camera", default="tip_camera")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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
        obs, info = env.reset(seed=ep)
        target = info["target"]
        instruction = f"navigate bronchoscope tip to target {target[0]:.3f} {target[1]:.3f} {target[2]:.3f}"

        for _ in range(args.max_steps):
            frame = env.render()
            if frame is None:
                break
            frame = cv2.resize(frame, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            action = heuristic_expert_action(obs)
            # 少量扰动，提升鲁棒性
            action = np.clip(action + np.random.normal(0.0, 0.05, size=(3,)).astype(np.float32), -1.0, 1.0)

            images.append(frame)
            states.append(obs.astype(np.float32))
            texts.append(instruction)
            actions.append(action.astype(np.float32))

            obs, reward, terminated, truncated, info = env.step(action)

            if args.show:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"ep={ep} dist={info['distance_to_target']:.4f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("VLA Data Collection", bgr)
                if cv2.waitKey(1) == 27:
                    args.show = False
                    cv2.destroyAllWindows()

            if terminated or truncated:
                break

        print(f"[Collect] episode={ep + 1}/{args.episodes}, final_dist={info['distance_to_target']:.5f}")

    env.close()
    cv2.destroyAllWindows()

    np.savez_compressed(
        args.out,
        images=np.stack(images, axis=0),
        states=np.stack(states, axis=0),
        texts=np.asarray(texts, dtype="<U256"),
        actions=np.stack(actions, axis=0),
    )
    print(f"[Done] 数据集保存: {args.out}, samples={len(actions)}")


if __name__ == "__main__":
    main()

