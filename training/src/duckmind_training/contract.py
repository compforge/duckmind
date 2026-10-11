"""Body feedback and demonstration semantics; teacher commands never enter learner state."""

from dataclasses import dataclass, replace

import numpy as np
from duckmind_policy.schema import JOINT_NAMES

FPS = 50
IMAGE_SIZE = 256
# Keep positions/IMU first; appended joint velocities distinguish identical poses in motion.
STATE_NAMES = (
    *JOINT_NAMES,
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    *(f"{name}_velocity" for name in JOINT_NAMES),
)
IMAGE_KEY = "observation.images.head"
MICRODUCK_RL_REVISION = "8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec"


def wire_positions(positions: np.ndarray) -> np.ndarray:
    """Insert the alpha body's absent mouth in robotd joint order (also for velocities)."""
    if positions.shape != (14,):
        raise ValueError("expected 14 upstream joints")
    return np.insert(positions, JOINT_NAMES.index("mouth"), 0.0).astype(np.float32)


def state_vector(
    positions: np.ndarray, gyro: np.ndarray, gravity: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    """All joint arrays use the 15-joint wire order, with a zero mouth in simulation."""
    return np.concatenate([positions, gyro, gravity, velocities]).astype(np.float32)


@dataclass(frozen=True)
class Task:
    name: str
    texts: tuple[str, ...]
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    initial_twist: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def sample(self, seed: int) -> "Task":
        # Text describes direction, not an exact speed/angle; record the sampled command.
        scale = float(np.random.default_rng(seed).uniform(1.0, 1.05))
        return replace(
            self,
            twist=tuple(v * scale for v in self.twist),
            head=tuple(v * scale for v in self.head),
        )


# Collection catalog only. Learned inference never dispatches on these names or texts.
TASKS = (
    Task("stand", ("保持站立", "站稳别动", "稳稳地站着")),
    Task("forward", ("向前走", "往前走一点", "往前迈步"), twist=(0.4, 0, 0)),
    Task("backward", ("向后走", "往后退一点", "往后退步"), twist=(-0.4, 0, 0)),
    Task("strafe_left", ("向左侧移", "往左边横着走", "向左挪动"), twist=(0, 0.3, 0)),
    Task("strafe_right", ("向右侧移", "往右边横着走", "向右挪动"), twist=(0, -0.3, 0)),
    Task("turn_left", ("向左转身", "原地左转", "身体转向左边"), twist=(0, 0, 1.5)),
    Task("turn_right", ("向右转身", "原地右转", "身体转向右边"), twist=(0, 0, -1.5)),
    Task("look_left", ("向左看", "把头转向左边", "看看左边"), head=(0, 0, 0.3, 0)),
    Task("look_right", ("向右看", "把头转向右边", "看看右边"), head=(0, 0, -0.3, 0)),
    Task("look_up", ("抬头", "往上看", "把头抬起来"), head=(0, -0.4, 0, 0)),
    Task("look_down", ("低头", "往下看", "把头低下来"), head=(0, 0.25, 0, 0)),
    Task("stop_forward", ("停下来站稳", "停止走动", "别走了，站好"), initial_twist=(0.4, 0, 0)),
    Task("stop_backward", ("停下来站稳", "停止走动", "别走了，站好"), initial_twist=(-0.4, 0, 0)),
)


def split_for(repetition: int, repetitions: int) -> str:
    if repetitions < 2:
        raise ValueError("at least two repetitions per task are required for a split")
    return "eval" if repetition >= repetitions - max(1, repetitions // 5) else "train"


def episode_success(task: Task, metrics: dict[str, float | bool], stable: bool) -> bool:
    """Check physical outcomes, not whether a command was accepted."""
    if not stable:
        return False
    for command, key, minimum in zip(
        task.twist, ("forward_m", "left_m", "yaw_rad"), (0.05, 0.05, 0.2), strict=True
    ):
        if command and float(metrics[key]) * np.sign(command) <= minimum:
            return False
    if task.twist[1]:
        # Sideways displacement with a large heading change is not a clean strafe demonstration.
        return abs(float(metrics["yaw_rad"])) < 0.5 and abs(float(metrics["forward_m"])) < abs(
            float(metrics["left_m"])
        )
    if any(task.twist):
        return True
    for command, key in zip(
        task.head,
        ("neck_pitch_delta", "head_pitch_delta", "head_yaw_delta", "head_roll_delta"),
        strict=True,
    ):
        if command and abs(float(metrics[key]) - command) >= 0.15:
            return False
    if any(task.initial_twist):
        # A small terminal speed alone can hide oscillation: measure the whole final second.
        return float(metrics["tail_speed_mps"]) < 0.06 and abs(float(metrics["forward_m"])) < 0.3
    return bool(np.hypot(float(metrics["forward_m"]), float(metrics["left_m"])) < 0.08)


def validate_rollouts(episodes: list[dict], repetitions: int, families: list[str]) -> None:
    """A declared task list makes a truncated collection distinguishable from a small one."""
    expected = {(f"{i:04d}-{family}", family) for i in range(repetitions) for family in families}
    actual = {(e["id"], e["task_family"]) for e in episodes}
    if not families or actual != expected or len(episodes) != len(expected):
        raise ValueError("collection is incomplete; wait for the collector to finish")
    for split in ("train", "eval"):
        accepted = {e["task_family"] for e in episodes if e["accepted"] and e["split"] == split}
        if accepted != set(families):
            raise ValueError(f"missing successful tasks in split: {split}")
