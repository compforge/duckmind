"""Implement this protocol with rules, a local model, or a remote model adapter."""

from typing import Protocol

from duckmind_policy.schema import ActionChunk, PredictRequest


class UnsupportedTask(ValueError):
    """A backend cannot interpret the requested task."""


class Policy(Protocol):
    def infer(self, request: PredictRequest) -> ActionChunk:
        """Predict targets; all temporal context is explicit in request.history."""
        ...


class FakePolicy:
    """Deterministic head-only demo, not a learned policy or a balance controller."""

    def infer(self, request: PredictRequest) -> ActionChunk:
        targets = {
            "向左看": 0.15,
            "往左看": 0.15,
            "look left": 0.15,
            "向右看": -0.15,
            "往右看": -0.15,
            "look right": -0.15,
            "保持姿势": None,
            "hold": None,
        }
        task = request.task.strip().lower()
        if task not in targets:
            raise UnsupportedTask("FakePolicy supports: 向左看 / 向右看 / 保持姿势")
        initial = request.observation.positions
        goal = targets[task]
        positions = []
        for i in range(50):
            frame = initial.copy()
            if goal is not None:
                # Absolute goal avoids accumulating a new relative turn on every replan.
                delta = max(-0.5 * (i + 1) * 0.02, min(goal - initial[7], 0.5 * (i + 1) * 0.02))
                frame[7] += delta
            positions.append(frame)
        return ActionChunk(observation_t_ns=request.observation.t_ns, positions=positions)
