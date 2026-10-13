"""Outcome gates for expert correction windows, in the original task heading."""

import numpy as np

from duckmind_training.contract import FPS, Task, episode_success


def recovery_outcome(metrics: list[dict], expected_steps: int) -> dict:
    """Metrics include the handover state followed by every post-action physical state.

    Displacement before handover earns no credit. A final second of forward progress prevents
    a fall, lateral shuffle or brief push followed by a stop from counting as recovered walking.
    These conservative flat-floor dataset gates are not hardware safety guarantees.
    """
    start, end = metrics[0], metrics[-1]
    steps = len(metrics) - 1
    forward = float(end["forward_m"] - start["forward_m"])
    left = float(end["left_m"] - start["left_m"])
    tail = metrics[-FPS - 1 :]
    tail_forward = float(tail[-1]["forward_m"] - tail[0]["forward_m"])
    checks = {
        "complete": steps == expected_steps and steps >= 2 * FPS,
        "stable": all(m["stable"] for m in metrics),
        "forward_progress": forward > 0.05,
        "limited_lateral_drift": abs(left) < 0.02 + 0.5 * max(forward, 0.0),
        "original_heading": all(abs(m["yaw_rad"]) < 0.5 for m in metrics),
        "tail_forward_progress": tail_forward > 0.03,
        "tail_upright": bool(
            np.mean([m["gravity_z"] for m in tail]) < -0.9
            and np.mean([m["trunk_z_m"] for m in tail]) > 0.1
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "steps": steps,
        "forward_m": forward,
        "left_m": left,
        "tail_forward_m": tail_forward,
        "terminal": end,
    }


def stationary_head_outcome(task: Task, metrics: list[dict], expected_steps: int) -> dict:
    """Require a stationary body and sustained head pitch/yaw task tracking.

    spec: compute error before temporal averaging, and never average across joints.
    Opposite head commands or oscillations must not count as standing at neutral.
    Uncommanded neck/roll can compensate for balance; report them, not as zero targets.
    link: docs/project/vla-head-actions-2026-10-13.md
    """
    keys = ("neck_pitch_delta", "head_pitch_delta", "head_yaw_delta", "head_roll_delta")
    tail = np.array([[m[k] for k in keys] for m in metrics[-FPS:]])
    mean = tail.mean(0)
    errors = np.abs(tail - np.array(task.head)).mean(0)
    required_axes = sorted({1, 2} | {i for i, value in enumerate(task.head) if value})
    checks = {
        "complete": len(metrics) - 1 == expected_steps and expected_steps >= 2 * FPS,
        "stable": all(m["stable"] for m in metrics),
        "stationary": bool(max(np.hypot(m["forward_m"], m["left_m"]) for m in metrics) < 0.08),
        "heading": all(abs(m["yaw_rad"]) < 0.5 for m in metrics),
        "head_tracking": bool((errors[required_axes] < 0.15).all()),
        "direction": all(mean[i] * np.sign(v) > 0.1 for i, v in enumerate(task.head) if v),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "required_axes": [keys[i] for i in required_axes],
        "tail_head_mean_rad": mean.tolist(),
        "tail_head_error_rad": errors.tolist(),
        "legacy_success": episode_success(task, metrics[-1], checks["stable"]),
    }


def locomotion_outcome(task: Task, metrics: list[dict], expected_steps: int) -> dict:
    """Sustained backward/turn/stop tasks; stop requires a moving-start experiment witness.

    spec: use accumulated yaw and whole-path drift, not wrapped endpoint heading or net speed.
    case: opposite turns, sideways walking, oscillation and a late fall cannot pass.
    link: docs/project/vla-motion-curriculum-2026-10-13.md
    """
    if task.name == "backward":
        # The original forward rubric applies in the commanded backward direction.
        mirrored = [{**row, "forward_m": -row["forward_m"]} for row in metrics]
        return recovery_outcome(mirrored, expected_steps)
    if task.name not in ("turn_left", "turn_right", "stop_forward", "stop_backward"):
        raise ValueError(f"unsupported locomotion task: {task.name}")
    tail = metrics[-FPS - 1 :]
    drift = max(float(np.hypot(m["forward_m"], m["left_m"])) for m in metrics)
    checks = {
        "complete": len(metrics) - 1 == expected_steps and expected_steps >= 2 * FPS,
        "stable": all(m["stable"] for m in metrics),
        "tail_upright": bool(
            np.mean([m["gravity_z"] for m in tail]) < -0.9
            and np.mean([m["trunk_z_m"] for m in tail]) > 0.1
        ),
    }
    yaw = float(metrics[-1]["yaw_rad"] - metrics[0]["yaw_rad"])
    tail_yaw = float(tail[-1]["yaw_rad"] - tail[0]["yaw_rad"])
    if task.twist[2]:
        sign = np.sign(task.twist[2])
        checks.update(
            turn_direction=bool(sign * yaw > 0.2),
            tail_turn=bool(sign * tail_yaw > 0.15),
            limited_translation=drift < 0.20,
        )
    else:
        checks.update(
            stopped=float(metrics[-1]["tail_speed_mps"]) < 0.06,
            limited_travel=drift < 0.3,
            original_heading=all(abs(m["yaw_rad"]) < 0.5 for m in metrics),
        )
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "steps": len(metrics) - 1,
        "max_displacement_m": drift,
        "yaw_rad": yaw,
        "tail_yaw_rad": tail_yaw,
        "tail_speed_mps": float(metrics[-1]["tail_speed_mps"]),
        "terminal": metrics[-1],
    }
