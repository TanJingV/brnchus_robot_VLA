import argparse
import os

import cv2
import numpy as np
from stable_baselines3 import PPO

from agent_nav.bronchoscope_nav_env import BronchoscopeNavEnv, NavEnvConfig, load_target_points

LOCKED_XML_REL = os.path.join("meshes", "cable_robot_bronch_final_seg2.xml")


def resolve_locked_xml() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.abspath(os.path.join(root, LOCKED_XML_REL))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到锁定仿真环境文件: {xml_path}")
    return xml_path


def parse_args():
    parser = argparse.ArgumentParser(description="部署已训练的支气管机器人导航策略")
    parser.add_argument("--model", default=os.path.join("agent_nav", "models", "ppo_bronch_nav.zip"))
    parser.add_argument("--targets", default=os.path.join("agent_nav", "target_points.json"))
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--show", action="store_true", help="显示渲染窗口")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = resolve_locked_xml()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"未找到模型: {args.model}")

    targets = load_target_points(args.targets)
    idx = int(np.clip(args.target_index, 0, len(targets) - 1))
    fixed_target = targets[idx]

    env_cfg = NavEnvConfig(
        xml_path=xml_path,
        target_points=[fixed_target],
        max_episode_steps=args.max_steps,
    )
    env = BronchoscopeNavEnv(config=env_cfg, render_mode="rgb_array", fixed_target=fixed_target)
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        success = False

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if args.show:
                frame = env.render()
                if frame is not None:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    tip = obs[0:3]
                    target = obs[6:9]
                    dist = info["distance_to_target"]
                    cv2.putText(
                        bgr,
                        f"Tip: [{tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}]",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        bgr,
                        f"Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}] Dist={dist:.4f}m",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                    )
                    cv2.imshow("Bronchoscope Nav Deployment", bgr)
                    if cv2.waitKey(1) == 27:
                        terminated = True
                        truncated = True

            if terminated or truncated:
                success = bool(info.get("is_success", False))
                break

        print(
            f"[Episode {ep + 1}] reward={ep_reward:.3f}, success={success}, "
            f"final_dist={info['distance_to_target']:.5f} m"
        )

    env.close()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

