import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tkinter import filedialog, messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parent.parent
LOCKED_XML = ROOT_DIR / "meshes" / "cable_robot_bronch_final_seg2.xml"


@dataclass
class RunSpec:
    title: str
    command: List[str]
    cwd: Path = ROOT_DIR


class ControlCenter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("支气管机器人导航控制中心")
        self.geometry("1100x780")

        self.proc: Optional[subprocess.Popen] = None
        self.proc_name: Optional[str] = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self._build_vars()
        self._build_ui()
        self._poll_logs()

    def _build_vars(self) -> None:
        self.targets_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "target_points.json"))
        self.ppo_model_out_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "models" / "ppo_bronch_nav"))
        self.ppo_timesteps_var = tk.StringVar(value="400000")
        self.ppo_max_steps_var = tk.StringVar(value="400")
        self.ppo_preview_var = tk.BooleanVar(value=True)
        self.ppo_preview_freq_var = tk.StringVar(value="10")
        self.ppo_preview_camera_var = tk.StringVar(value="tip_camera")
        self.ppo_run_model_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "models" / "ppo_bronch_nav.zip"))
        self.ppo_run_target_var = tk.StringVar(value="0")
        self.ppo_run_episodes_var = tk.StringVar(value="3")
        self.ppo_run_show_var = tk.BooleanVar(value=True)

        self.vla_data_out_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "data" / "demo_dataset.npz"))
        self.vla_collect_episodes_var = tk.StringVar(value="40")
        self.vla_collect_steps_var = tk.StringVar(value="400")
        self.vla_collect_camera_var = tk.StringVar(value="tip_camera")
        self.vla_collect_show_var = tk.BooleanVar(value=True)

        self.vla_data_pattern_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "data" / "*.npz"))
        self.vla_epochs_var = tk.StringVar(value="20")
        self.vla_batch_size_var = tk.StringVar(value="32")
        self.vla_lr_var = tk.StringVar(value="0.0001")
        self.vla_model_out_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "models" / "vla_policy.pt"))
        self.vla_clip_model_var = tk.StringVar(value="ViT-H-14")
        self.vla_clip_pretrained_var = tk.StringVar(value="laion2b_s32b_b79k")
        self.vla_train_clip_var = tk.BooleanVar(value=False)
        self.vla_state_width_var = tk.StringVar(value="256")
        self.vla_fusion_width_var = tk.StringVar(value="1024")

        self.vla_dagger_model_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "models" / "vla_policy.pt"))
        self.vla_dagger_out_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "data" / "dagger_dataset.npz"))
        self.vla_dagger_episodes_var = tk.StringVar(value="20")
        self.vla_dagger_steps_var = tk.StringVar(value="400")
        self.vla_dagger_camera_var = tk.StringVar(value="tip_camera")
        self.vla_dagger_show_var = tk.BooleanVar(value=True)

        self.vla_deploy_model_var = tk.StringVar(value=str(ROOT_DIR / "agent_nav" / "models" / "vla_policy.pt"))
        self.vla_deploy_target_var = tk.StringVar(value="0")
        self.vla_deploy_episodes_var = tk.StringVar(value="3")
        self.vla_deploy_steps_var = tk.StringVar(value="500")
        self.vla_deploy_camera_var = tk.StringVar(value="tip_camera")
        self.vla_deploy_show_var = tk.BooleanVar(value=True)

        self.manual_control_path_var = tk.StringVar(value=str(ROOT_DIR / "meshes" / "run_physics_model_control.py"))
        self.status_var = tk.StringVar(value="就绪")

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x")

        ttk.Label(top, textvariable=self.status_var, font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        ttk.Button(top, text="停止当前任务", command=self.stop_process).pack(side="right")

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True, pady=(10, 10))

        notebook.add(self._build_ppo_tab(notebook), text="PPO导航")
        notebook.add(self._build_vla_tab(notebook), text="VLA流程")
        notebook.add(self._build_manual_tab(notebook), text="手动控制")

        log_frame = ttk.LabelFrame(main, text="运行日志")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=12)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

    def _build_path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, browse: str = "file") -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 6))
        if browse == "file":
            ttk.Button(row, text="选择", command=lambda: self._browse_file(variable)).pack(side="left")
        elif browse == "dir":
            ttk.Button(row, text="选择", command=lambda: self._browse_dir(variable)).pack(side="left")

    def _build_ppo_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        self._build_path_row(frame, "目标点文件", self.targets_var)
        self._build_path_row(frame, "训练模型输出", self.ppo_model_out_var)

        opts = ttk.Frame(frame)
        opts.pack(fill="x", pady=4)
        self._add_labeled_entry(opts, "Timesteps", self.ppo_timesteps_var)
        self._add_labeled_entry(opts, "Max Steps", self.ppo_max_steps_var)
        self._add_labeled_entry(opts, "Preview Freq", self.ppo_preview_freq_var)
        self._add_labeled_entry(opts, "Preview Cam", self.ppo_preview_camera_var)

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=8)
        ttk.Checkbutton(btns, text="训练预览", variable=self.ppo_preview_var).pack(side="left")
        ttk.Checkbutton(btns, text="推理显示", variable=self.ppo_run_show_var).pack(side="left", padx=8)
        ttk.Button(btns, text="启动 PPO 训练", command=self.start_ppo_train).pack(side="left", padx=6)
        ttk.Button(btns, text="启动 PPO 推理", command=self.start_ppo_run).pack(side="left", padx=6)
        return frame

    def _build_vla_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        collect = ttk.LabelFrame(frame, text="示教采集")
        collect.pack(fill="x", pady=6)
        self._build_path_row(collect, "数据输出", self.vla_data_out_var)
        self._build_path_row(collect, "目标点文件", self.targets_var)
        row = ttk.Frame(collect)
        row.pack(fill="x", pady=4)
        self._add_labeled_entry(row, "Episodes", self.vla_collect_episodes_var)
        self._add_labeled_entry(row, "Max Steps", self.vla_collect_steps_var)
        self._add_labeled_entry(row, "Camera", self.vla_collect_camera_var)
        ttk.Checkbutton(row, text="显示窗口", variable=self.vla_collect_show_var).pack(side="left", padx=8)
        ttk.Button(collect, text="启动采集", command=self.start_vla_collect).pack(anchor="w", pady=(4, 0))

        train = ttk.LabelFrame(frame, text="VLA SFT / 后训练")
        train.pack(fill="x", pady=6)
        self._build_path_row(train, "数据模式", self.vla_data_pattern_var)
        self._build_path_row(train, "模型输出", self.vla_model_out_var)
        row2 = ttk.Frame(train)
        row2.pack(fill="x", pady=4)
        self._add_labeled_entry(row2, "Epochs", self.vla_epochs_var)
        self._add_labeled_entry(row2, "Batch", self.vla_batch_size_var)
        self._add_labeled_entry(row2, "LR", self.vla_lr_var)
        self._add_labeled_entry(row2, "State W", self.vla_state_width_var)
        self._add_labeled_entry(row2, "Fusion W", self.vla_fusion_width_var)
        ttk.Checkbutton(row2, text="微调CLIP", variable=self.vla_train_clip_var).pack(side="left", padx=8)
        clip_row = ttk.Frame(train)
        clip_row.pack(fill="x", pady=4)
        self._add_labeled_entry(clip_row, "CLIP Model", self.vla_clip_model_var, width=16)
        self._add_labeled_entry(clip_row, "Pretrained", self.vla_clip_pretrained_var, width=16)
        ttk.Button(train, text="启动 VLA 训练", command=self.start_vla_train).pack(anchor="w", pady=(4, 0))

        dagger = ttk.LabelFrame(frame, text="DAgger 回灌")
        dagger.pack(fill="x", pady=6)
        self._build_path_row(dagger, "模型路径", self.vla_dagger_model_var)
        self._build_path_row(dagger, "数据输出", self.vla_dagger_out_var)
        row3 = ttk.Frame(dagger)
        row3.pack(fill="x", pady=4)
        self._add_labeled_entry(row3, "Episodes", self.vla_dagger_episodes_var)
        self._add_labeled_entry(row3, "Max Steps", self.vla_dagger_steps_var)
        self._add_labeled_entry(row3, "Camera", self.vla_dagger_camera_var)
        ttk.Checkbutton(row3, text="显示窗口", variable=self.vla_dagger_show_var).pack(side="left", padx=8)
        ttk.Button(dagger, text="启动 DAgger", command=self.start_dagger_collect).pack(anchor="w", pady=(4, 0))

        deploy = ttk.LabelFrame(frame, text="VLA 部署")
        deploy.pack(fill="x", pady=6)
        self._build_path_row(deploy, "模型路径", self.vla_deploy_model_var)
        row4 = ttk.Frame(deploy)
        row4.pack(fill="x", pady=4)
        self._add_labeled_entry(row4, "Target", self.vla_deploy_target_var)
        self._add_labeled_entry(row4, "Episodes", self.vla_deploy_episodes_var)
        self._add_labeled_entry(row4, "Max Steps", self.vla_deploy_steps_var)
        self._add_labeled_entry(row4, "Camera", self.vla_deploy_camera_var)
        ttk.Checkbutton(row4, text="显示窗口", variable=self.vla_deploy_show_var).pack(side="left", padx=8)
        ttk.Button(deploy, text="启动 VLA 推理", command=self.start_vla_deploy).pack(anchor="w", pady=(4, 0))
        return frame

    def _build_manual_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        self._build_path_row(frame, "控制脚本", self.manual_control_path_var)
        ttk.Button(frame, text="启动手动控制 UI", command=self.start_manual_control).pack(anchor="w", pady=6)
        note = (
            "说明：\n"
            "1. 所有训练/采集/推理仍然基于锁定环境 meshes\\cable_robot_bronch_final_seg2.xml。\n"
            "2. UI 只负责启动现有脚本，不会改动原始仿真文件。\n"
            "3. 运行时如果窗口不显示，优先检查本机是否已有图形界面和 MuJoCo/OpenCV 显示权限。"
        )
        note_frame = ttk.LabelFrame(frame, text="提示")
        note_frame.pack(fill="both", expand=True, pady=(10, 0))
        hint = tk.Text(note_frame, height=7, wrap="word")
        hint.pack(fill="both", expand=True)
        hint.insert("1.0", note)
        hint.configure(state="disabled")
        return frame

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, width: int = 10) -> None:
        box = ttk.Frame(parent)
        box.pack(side="left", padx=4)
        ttk.Label(box, text=label).pack(side="left")
        ttk.Entry(box, textvariable=variable, width=width).pack(side="left", padx=(4, 0))

    def _browse_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(initialdir=str(ROOT_DIR))
        if path:
            variable.set(path)

    def _browse_dir(self, variable: tk.StringVar) -> None:
        path = filedialog.askdirectory(initialdir=str(ROOT_DIR))
        if path:
            variable.set(path)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _poll_logs(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            name = self.proc_name or "任务"
            self._append_log(f"\n[{name}] 结束，退出码 {code}\n")
            self.status_var.set(f"{name} 已结束 (code={code})")
            self.proc = None
            self.proc_name = None

        self.after(120, self._poll_logs)

    def _ensure_idle(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("任务运行中", "当前已有任务在运行，请先停止它，或者等待其结束。")
            return False
        return True

    def _start_run(self, spec: RunSpec) -> None:
        if not self._ensure_idle():
            return
        if not spec.cwd.exists():
            messagebox.showerror("路径错误", f"工作目录不存在: {spec.cwd}")
            return

        self.log_text.delete("1.0", "end")
        self.status_var.set(f"启动中: {spec.title}")
        self._append_log(f"[Command] {' '.join(spec.command)}\n")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            spec.command,
            cwd=str(spec.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self.proc_name = spec.title

        def reader() -> None:
            assert self.proc is not None
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)

        threading.Thread(target=reader, daemon=True).start()
        self.status_var.set(f"运行中: {spec.title}")

    def stop_process(self) -> None:
        if self.proc is None:
            self.status_var.set("当前没有运行中的任务")
            return
        if self.proc.poll() is not None:
            self.status_var.set("当前任务已结束")
            return
        name = self.proc_name or "任务"
        self._append_log(f"\n[{name}] 正在停止...\n")
        self.proc.terminate()
        self.status_var.set(f"正在停止: {name}")
        self.after(1500, self._force_kill_if_needed)

    def _force_kill_if_needed(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()

    def _python_cmd(self, *args: str) -> List[str]:
        return [sys.executable, "-u", *args]

    def start_ppo_train(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.train_nav_agent",
            "--targets",
            self.targets_var.get(),
            "--timesteps",
            self.ppo_timesteps_var.get(),
            "--max-steps",
            self.ppo_max_steps_var.get(),
            "--model-out",
            self.ppo_model_out_var.get(),
        )
        if self.ppo_preview_var.get():
            cmd += ["--preview", "--preview-freq", self.ppo_preview_freq_var.get(), "--preview-camera", self.ppo_preview_camera_var.get()]
        self._start_run(RunSpec("PPO 训练", cmd))

    def start_ppo_run(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.run_nav_agent",
            "--model",
            self.ppo_run_model_var.get(),
            "--target-index",
            self.ppo_run_target_var.get(),
            "--episodes",
            self.ppo_run_episodes_var.get(),
        )
        if self.ppo_run_show_var.get():
            cmd.append("--show")
        self._start_run(RunSpec("PPO 推理", cmd))

    def start_vla_collect(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.collect_vla_data",
            "--targets",
            self.targets_var.get(),
            "--out",
            self.vla_data_out_var.get(),
            "--episodes",
            self.vla_collect_episodes_var.get(),
            "--max-steps",
            self.vla_collect_steps_var.get(),
            "--camera",
            self.vla_collect_camera_var.get(),
        )
        if self.vla_collect_show_var.get():
            cmd.append("--show")
        self._start_run(RunSpec("VLA 采集", cmd))

    def start_vla_train(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.train_vla_sft",
            "--data-pattern",
            self.vla_data_pattern_var.get(),
            "--epochs",
            self.vla_epochs_var.get(),
            "--batch-size",
            self.vla_batch_size_var.get(),
            "--lr",
            self.vla_lr_var.get(),
            "--model-out",
            self.vla_model_out_var.get(),
            "--clip-model",
            self.vla_clip_model_var.get(),
            "--clip-pretrained",
            self.vla_clip_pretrained_var.get(),
            "--state-width",
            self.vla_state_width_var.get(),
            "--fusion-width",
            self.vla_fusion_width_var.get(),
        )
        if self.vla_train_clip_var.get():
            cmd.append("--train-clip")
        self._start_run(RunSpec("VLA 训练", cmd))

    def start_dagger_collect(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.dagger_collect",
            "--model",
            self.vla_dagger_model_var.get(),
            "--targets",
            self.targets_var.get(),
            "--out",
            self.vla_dagger_out_var.get(),
            "--episodes",
            self.vla_dagger_episodes_var.get(),
            "--max-steps",
            self.vla_dagger_steps_var.get(),
            "--camera",
            self.vla_dagger_camera_var.get(),
        )
        if self.vla_dagger_show_var.get():
            cmd.append("--show")
        self._start_run(RunSpec("DAgger 采集", cmd))

    def start_vla_deploy(self) -> None:
        cmd = self._python_cmd(
            "-m",
            "agent_nav.run_vla_agent",
            "--model",
            self.vla_deploy_model_var.get(),
            "--targets",
            self.targets_var.get(),
            "--target-index",
            self.vla_deploy_target_var.get(),
            "--episodes",
            self.vla_deploy_episodes_var.get(),
            "--max-steps",
            self.vla_deploy_steps_var.get(),
            "--camera",
            self.vla_deploy_camera_var.get(),
        )
        if self.vla_deploy_show_var.get():
            cmd.append("--show")
        self._start_run(RunSpec("VLA 推理", cmd))

    def start_manual_control(self) -> None:
        path = Path(self.manual_control_path_var.get())
        if not path.exists():
            messagebox.showerror("路径错误", f"控制脚本不存在: {path}")
            return
        cmd = self._python_cmd(str(path))
        self._start_run(RunSpec("手动控制", cmd, cwd=ROOT_DIR))


def main() -> None:
    if not LOCKED_XML.exists():
        print(f"未找到锁定环境: {LOCKED_XML}")
        return
    app = ControlCenter()
    app.mainloop()


if __name__ == "__main__":
    main()

