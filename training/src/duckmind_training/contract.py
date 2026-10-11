"""The learner sees body feedback, never the teacher's command vector."""

from dataclasses import dataclass

import numpy as np
from duckmind_policy.schema import JOINT_NAMES

FPS = 50
IMAGE_SIZE = 256
STATE_NAMES = (*JOINT_NAMES, "gyro_x", "gyro_y", "gyro_z", "gravity_x", "gravity_y", "gravity_z")
IMAGE_KEY = "observation.images.head"
MICRODUCK_RL_REVISION = "8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec"


def wire_positions(positions: np.ndarray) -> np.ndarray:
    """Upstream alpha has 14 actuators; robotd's 15-joint contract includes the mouth."""
    if positions.shape != (14,):
        raise ValueError("expected 14 upstream joint positions")
    return np.insert(positions, JOINT_NAMES.index("mouth"), 0.0).astype(np.float32)


def state_vector(positions: np.ndarray, gyro: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    return np.concatenate([wire_positions(positions), gyro, gravity]).astype(np.float32)


@dataclass(frozen=True)
class Task:
    name: str
    texts: tuple[str, ...]
    vx: float = 0.0
    head_yaw: float = 0.0


# This catalog drives the expert during collection only. It is NOT an inference dispatcher.
TASKS = (
    Task("stand", ("保持站立", "站稳别动", "稳稳地站着")),
    # Smaller commands can leave this teacher standing; outcome checks still gate each rollout.
    Task("forward", ("向前走", "往前走一点", "往前迈步"), vx=0.4),
    Task("backward", ("向后走", "往后退一点", "往后退步"), vx=-0.4),
    Task("look_left", ("向左看", "把头转向左边", "看看左边"), head_yaw=0.3),
    Task("look_right", ("向右看", "把头转向右边", "看看右边"), head_yaw=-0.3),
)


def split_for(repetition: int, repetitions: int) -> str:
    """Hold out whole physical rollouts, not adjacent frames or duplicate paraphrases."""
    if repetitions < 2:
        raise ValueError("at least two repetitions per task are required for a split")
    return "eval" if repetition >= repetitions - max(1, repetitions // 5) else "train"


def episode_success(task: Task, displacement: float, head_yaw: float, stable: bool) -> bool:
    if not stable:
        return False
    if task.vx:
        return displacement * (1 if task.vx > 0 else -1) > 0.05
    if task.head_yaw:
        return abs(head_yaw - task.head_yaw) < 0.12
    return abs(displacement) < 0.08


def validate_rollouts(episodes: list[dict], repetitions: int) -> None:
    """Reject a still-running collection or missing successful task/split coverage."""
    families = {e["task_family"] for e in episodes}
    if not families or len(episodes) != repetitions * len(families):
        raise ValueError("collection is incomplete; wait for the collector to finish")
    for split in ("train", "eval"):
        accepted = {e["task_family"] for e in episodes if e["accepted"] and e["split"] == split}
        if accepted != families:
            raise ValueError(f"missing successful tasks in split: {split}")
