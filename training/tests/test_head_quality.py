"""Wrong-direction and transient poses cannot earn stationary task success."""

import json

from duckmind_training.contract import TASKS
from duckmind_training.correction_quality import stationary_head_outcome

TASK = {t.name: t for t in TASKS}


def trajectory(yaw: float = 0.0, pitch: float = 0.0) -> list[dict]:
    return [
        dict(
            stable=True,
            forward_m=0.0,
            left_m=0.0,
            yaw_rad=0.0,
            neck_pitch_delta=0.0,
            head_pitch_delta=pitch,
            head_yaw_delta=yaw,
            head_roll_delta=0.0,
        )
        for _ in range(201)
    ]


def test_tasks_require_distinct_head_pose() -> None:
    for name, yaw, pitch in [
        ("stand", 0, 0),
        ("look_left", 0.3, 0),
        ("look_right", -0.3, 0),
        ("look_up", 0, -0.4),
        ("look_down", 0, 0.25),
    ]:
        metrics = trajectory(yaw, pitch)
        for other in ("stand", "look_left", "look_right", "look_up", "look_down"):
            result = stationary_head_outcome(TASK[other], metrics, 200)
            assert result["accepted"] == (name == other)
            json.dumps(result)  # Run reports must also accept NumPy-derived checks.


def test_tail_error_cannot_cancel_or_hide_another_joint() -> None:
    metrics = trajectory()
    for i, row in enumerate(metrics):
        row["head_yaw_delta"] = 0.3 if i % 2 else -0.3
    assert not stationary_head_outcome(TASK["stand"], metrics, 200)["accepted"]
    assert not stationary_head_outcome(TASK["look_left"], trajectory(0.3, 0.25), 200)["accepted"]


def test_fall_excursion_or_incomplete_rollout_cannot_pass() -> None:
    for key, value in [("stable", False), ("forward_m", 0.1), ("yaw_rad", 0.6)]:
        metrics = trajectory(0.3)
        metrics[20][key] = value
        assert not stationary_head_outcome(TASK["look_left"], metrics, 200)["accepted"]
    assert not stationary_head_outcome(TASK["look_left"], trajectory(0.3)[:50], 200)["accepted"]


def test_teacher_neck_compensation_is_not_an_unrequested_zero_target() -> None:
    metrics = trajectory(pitch=-0.28)
    for row in metrics:
        row["neck_pitch_delta"] = -0.23
    assert stationary_head_outcome(TASK["look_up"], metrics, 200)["accepted"]
    assert not stationary_head_outcome(TASK["look_down"], metrics, 200)["accepted"]
