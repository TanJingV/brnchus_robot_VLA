import argparse
import importlib.util
import json
import os

import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from agent_nav.bronchoscope_nav_env import BronchoscopeNavEnv, NavEnvConfig, load_target_points

LOCKED_XML_REL = os.path.join("meshes", "cable_robot_bronch_final_seg2.xml")


def resolve_locked_xml() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    xml_path = os.path.abspath(os.path.join(root, LOCKED_XML_REL))
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到锁定仿真环境文件: {xml_path}")
    return xml_path


def make_env(xml_path: str, target_file: str, max_steps: int, render_mode=None, render_camera="tip_camera"):
    def _factory():
        cfg = NavEnvConfig(
            xml_path=xml_path,
            target_points=load_target_points(target_file),
            max_episode_steps=max_steps,
            render_camera=render_camera,
        )
        return Monitor(BronchoscopeNavEnv(config=cfg, render_mode=render_mode))

    return _factory


class PreviewCallback(BaseCallback):
    def __init__(self, preview_freq: int = 10, window_name: str = "Training Preview"):
        super().__init__()
        self.preview_freq = max(1, int(preview_freq))
        self.window_name = window_name
        self.disabled = False

    def _on_training_start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _on_step(self) -> bool:
        if self.disabled:
            return True
        if self.n_calls % self.preview_freq != 0:
            return True
        try:
            frame = self.training_env.envs[0].render()
            if frame is None:
                return True
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr,
                f"training_step={self.num_timesteps}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow(self.window_name, bgr)
            if cv2.waitKey(1) == 27:
                self.disabled = True
        except cv2.error as e:
            print(f"[Warn] 训练预览失败，已自动关闭预览: {e}")
            self.disabled = True
        return True

    def _on_training_end(self) -> None:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="训练支气管机器人 MuJoCo 自主导航策略（PPO）")
    parser.add_argument("--targets", default=os.path.join("agent_nav", "target_points.json"))
    parser.add_argument("--timesteps", type=int, default=400000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--model-out", default=os.path.join("agent_nav", "models", "ppo_bronch_nav"))
    parser.add_argument("--log-dir", default=os.path.join("agent_nav", "logs"))
    parser.add_argument("--preview", action="store_true", help="训练时实时可视化（Esc关闭预览）")
    parser.add_argument("--preview-freq", type=int, default=10, help="每N步刷新一次预览")
    parser.add_argument("--preview-camera", default="tip_camera", help="预览相机名，如 tip_camera/agentview/topview")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = resolve_locked_xml()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    render_mode = "rgb_array" if args.preview else None
    train_env = DummyVecEnv(
        [make_env(xml_path, args.targets, args.max_steps, render_mode=render_mode, render_camera=args.preview_camera)]
    )
    eval_env = DummyVecEnv([make_env(xml_path, args.targets, args.max_steps)])
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    tb_log_dir = args.log_dir if tensorboard_available else None
    if not tensorboard_available:
        print("[Warn] tensorboard 未安装，已自动关闭 TensorBoard 日志。")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=tb_log_dir,
        policy_kwargs={"net_arch": [256, 256, 128]},
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]
    if args.preview:
        callbacks.append(PreviewCallback(preview_freq=args.preview_freq))
    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks))
    model.save(args.model_out)

    metadata = {
        "xml_path": xml_path,
        "targets_file": args.targets,
        "timesteps": args.timesteps,
        "algorithm": "PPO",
    }
    with open(args.model_out + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Done] 模型已保存: {args.model_out}.zip")
    print(f"[Done] 元数据已保存: {args.model_out}_meta.json")


if __name__ == "__main__":
    main()

