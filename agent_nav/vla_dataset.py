import glob
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class VLADataset(Dataset):
    def __init__(self, files: List[str]):
        if not files:
            raise ValueError("未找到数据文件")

        all_images = []
        all_states = []
        all_texts = []
        all_actions = []

        for f in files:
            data = np.load(f, allow_pickle=False)
            all_images.append(data["images"])
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            if "texts" in data:
                all_texts.append(data["texts"].astype(str))
            elif "tokens" in data:
                # 向后兼容旧数据：没有文本时使用固定任务描述
                n = data["tokens"].shape[0]
                all_texts.append(np.array(["navigate bronchoscope tip to target"] * n, dtype="<U128"))
            else:
                raise ValueError(f"{f} 中缺少 texts/tokens 字段")

        self.images = np.concatenate(all_images, axis=0).astype(np.uint8)
        self.states = np.concatenate(all_states, axis=0).astype(np.float32)
        self.texts = np.concatenate(all_texts, axis=0).astype(str)
        self.actions = np.concatenate(all_actions, axis=0).astype(np.float32)

    @classmethod
    def from_pattern(cls, pattern: str):
        files = sorted(glob.glob(pattern))
        return cls(files)

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).float() / 255.0
        image = image.permute(2, 0, 1).contiguous()
        state = torch.from_numpy(self.states[idx])
        text = str(self.texts[idx])
        action = torch.from_numpy(self.actions[idx])
        return image, state, text, action

