"""Motion rubrics reject wrong direction, transient motion, and endpoint cancellation."""

import json

from duckmind_training.contract import TASKS
from duckmind_training.correction_quality import locomotion_outcome

TASKS_BY_NAME = {task.name: task for task in TASKS}


def motion(forward: float = 0, yaw: float = 0, speed: float = 0) -> list[dict]:
    return [
        dict(
            stable=True,
            forward_m=forward * i / 200,
            left_m=0,
            yaw_rad=yaw * i / 200,
            tail_speed_mps=speed,
            gravity_z=-1.0,
            trunk_z_m=0.12,
        )
        for i in range(201)
    ]


def test_backward_requires_signed_progress_and_a_stable_tail() -> None:
    task = TASKS_BY_NAME["backward"]
    good = motion(forward=-0.4)
    assert locomotion_outcome(task, good, 200)["accepted"]
    assert not locomotion_outcome(task, motion(forward=0.4), 200)["accepted"]
    for index, row in enumerate(good):
        row["left_m"] = 0.4 * index / 200
    assert not locomotion_outcome(task, good, 200)["accepted"]


def test_turns_require_continuing_direction_and_limited_drift() -> None:
    left, right = TASKS_BY_NAME["turn_left"], TASKS_BY_NAME["turn_right"]
    good = motion(yaw=4.0)  # accumulated yaw must not wrap at pi
    result = locomotion_outcome(left, good, 200)
    assert result["accepted"]
    json.dumps(result)
    assert not locomotion_outcome(right, good, 200)["accepted"]
    for row in good[150:]:
        row["yaw_rad"] = 3.0
    assert not locomotion_outcome(left, good, 200)["accepted"]
    good = motion(yaw=4.0)
    good[40]["left_m"] = 0.25
    assert not locomotion_outcome(left, good, 200)["accepted"]


def test_stop_speed_cannot_be_replaced_by_net_displacement() -> None:
    task = TASKS_BY_NAME["stop_forward"]
    assert locomotion_outcome(task, motion(speed=0.01), 200)["accepted"]
    assert not locomotion_outcome(task, motion(speed=0.1), 200)["accepted"]
    rows = motion(speed=0.01)
    rows[50]["forward_m"] = 0.31
    assert not locomotion_outcome(task, rows, 200)["accepted"]


def test_incomplete_and_fallen_motion_cannot_pass() -> None:
    for name, rows in [
        ("backward", motion(forward=-0.4)),
        ("turn_left", motion(yaw=4.0)),
        ("stop_backward", motion(speed=0.01)),
    ]:
        task = TASKS_BY_NAME[name]
        assert not locomotion_outcome(task, rows[:60], 200)["accepted"]
        rows[30]["stable"] = False
        assert not locomotion_outcome(task, rows, 200)["accepted"]
