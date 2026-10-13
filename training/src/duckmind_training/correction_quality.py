"""Outcome gates for expert correction windows, in the original task heading."""

import numpy as np

from duckmind_training.contract import FPS


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
