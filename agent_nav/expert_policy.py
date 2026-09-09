from typing import Sequence

import numpy as np


def heuristic_expert_action(obs: Sequence[float]) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    tip = obs[0:3]
    target = obs[6:9]
    delta = target - tip

    bend_x = np.clip(delta[1] * 80.0, -1.0, 1.0)
    bend_y = np.clip(delta[2] * 80.0, -1.0, 1.0)
    insertion = np.clip(delta[0] * 20.0, -1.0, 1.0)
    return np.array([bend_x, bend_y, insertion], dtype=np.float32)

