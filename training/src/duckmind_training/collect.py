"""Collect successful teacher rollouts with synchronized pre-action observations."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from duckmind_training.contract import (
    FPS,
    MICRODUCK_RL_REVISION,
    STATE_NAMES,
    TASKS,
    episode_success,
    split_for,
    wire_positions,
)
from duckmind_training.simulation import Simulation

LOG = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect(
    upstream: Path, teacher: Path, output: Path, repetitions: int, seconds: float, seed: int
) -> None:
    split_for(0, repetitions)
    if seconds < 2:
        raise ValueError("episodes need at least two seconds for outcome checks")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "fps": FPS,
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
            for task in TASKS:
                # Same initial seed across tasks makes the instruction necessary at task onset.
                episode_seed = seed + repetition
                simulation.reset(episode_seed)
                simulation.command(task)
                text_index = repetition % (len(task.texts) - 1) if split == "train" else -1
                text = task.texts[text_index]
                states, images, actions = [], [], []
                stable = bool(simulation.metrics()["stable"])
                for _ in range(round(seconds * FPS)):
                    state, image = simulation.observe()
                    target = simulation.teacher_step()
                    states.append(state)
                    images.append(image)
                    actions.append(wire_positions(target))
                    stable = stable and bool(simulation.metrics()["stable"])
                    if not stable:
                        break
                metrics = simulation.metrics()
                success = episode_success(
                    task, float(metrics["displacement_m"]), float(metrics["head_yaw_rad"]), stable
                )
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
                    "teacher_command": {"vx": task.vx, "head_yaw": task.head_yaw},
                }
                if success:
                    # Raw, replayable evidence precedes format conversion; no failed trajectory
                    # is silently presented as a successful language-conditioned demonstration.
                    np.savez_compressed(
                        output / f"{episode_id}.npz",
                        state=np.stack(states),
                        image=np.stack(images),
                        action=np.stack(actions),
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
    LOG.info("accepted %s/%s episodes at %s", accepted, repetitions * len(TASKS), output)
    for split in ("train", "eval"):
        for task in TASKS:
            if not any(
                e["accepted"] and e["split"] == split and e["task_family"] == task.name
                for e in manifest["episodes"]
            ):
                raise RuntimeError(
                    f"no successful {split} demonstration for {task.name}; inspect manifest"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    collect(args.upstream, args.teacher, args.output, args.repetitions, args.seconds, args.seed)
