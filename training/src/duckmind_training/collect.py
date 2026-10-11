"""Collect successful teacher rollouts with synchronized pre-action observations."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from duckmind_training.contract import (
    FPS,
    MICRODUCK_RL_REVISION,
    STATE_NAMES,
    TASKS,
    episode_success,
    split_for,
    validate_rollouts,
    wire_positions,
)
from duckmind_training.simulation import Simulation

LOG = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect(
    upstream: Path,
    teacher: Path,
    output: Path,
    repetitions: int,
    seconds: float,
    seed: int,
    task_names: list[str] | None = None,
) -> None:
    split_for(0, repetitions)
    if seconds < 2:
        raise ValueError("episodes need at least two seconds for outcome checks")
    tasks = [task for task in TASKS if task_names is None or task.name in task_names]
    if not tasks or (task_names is not None and set(task_names) != {t.name for t in tasks}):
        raise ValueError("unknown or empty task selection")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "fps": FPS,
        "task_families": [task.name for task in tasks],
        "state_names": STATE_NAMES,
        "action_space": "microduck-joints-v1/absolute_joint_position_rad",
        "reference_upstream_revision": MICRODUCK_RL_REVISION,
        "upstream_script_sha256": sha256(upstream / "scripts/infer_policy.py"),
        "teacher_sha256": sha256(teacher),
        "seed": seed,
        "seconds": seconds,
        "repetitions": repetitions,
        "episodes": [],
    }
    simulation = Simulation(upstream, teacher)
    accepted = 0
    try:
        for repetition in range(repetitions):
            split = split_for(repetition, repetitions)
            for template in tasks:
                task = template.sample(seed + repetition)
                # Same initial seed across tasks makes the instruction necessary at task onset.
                episode_seed = seed + repetition
                simulation.prepare(task, episode_seed)
                simulation.command(task)
                text_index = repetition % (len(task.texts) - 1) if split == "train" else -1
                text = task.texts[text_index]
                states, images, actions, previous_targets = [], [], [], []
                stable = bool(simulation.metrics()["stable"])
                for _ in range(round(seconds * FPS)):
                    state, image = simulation.observe()
                    previous_targets.append(wire_positions(simulation.bam.q_target.copy()))
                    target = simulation.teacher_step()
                    states.append(state)
                    images.append(image)
                    actions.append(wire_positions(target))
                    stable = stable and bool(simulation.metrics()["stable"])
                    if not stable:
                        break
                metrics = simulation.metrics()
                success = episode_success(task, metrics, stable)
                episode_id = f"{repetition:04d}-{task.name}"
                record = {
                    "id": episode_id,
                    "seed": episode_seed,
                    "task": text,
                    "task_family": task.name,
                    "split": split,
                    "accepted": success,
                    "frames": len(states),
                    "metrics": metrics,
                    "teacher_command": asdict(task),
                }
                if success:
                    # Raw, replayable evidence precedes format conversion; no failed trajectory
                    # is silently presented as a successful language-conditioned demonstration.
                    np.savez_compressed(
                        output / f"{episode_id}.npz",
                        state=np.stack(states),
                        image=np.stack(images),
                        action=np.stack(actions),
                        previous_target=np.stack(previous_targets),
                    )
                    accepted += 1
                manifest["episodes"].append(record)
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False)
                )
                LOG.info(
                    "episode=%s split=%s accepted=%s metrics=%s",
                    episode_id,
                    split,
                    success,
                    metrics,
                )
    finally:
        simulation.close()
    LOG.info("accepted %s/%s episodes at %s", accepted, repetitions * len(tasks), output)
    validate_rollouts(manifest["episodes"], repetitions, manifest["task_families"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", choices=[t.name for t in TASKS])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    collect(
        args.upstream,
        args.teacher,
        args.output,
        args.repetitions,
        args.seconds,
        args.seed,
        args.tasks,
    )
