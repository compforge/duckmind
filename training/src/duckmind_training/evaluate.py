"""Report held-out prediction error separately from delay-free physics rollouts."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from duckmind_training.contract import FPS, STATE_NAMES, TASKS, episode_success
from duckmind_training.learned import SmolPolicy
from duckmind_training.simulation import Simulation


def evaluate(
    checkpoint: Path,
    raw: Path,
    output: Path,
    upstream: Path | None,
    teacher: Path | None,
    execute_steps: list[int],
    seeds: list[int],
    max_held_out_chunks: int,
    language_split: str = "held-out",
) -> None:
    if not seeds or not execute_steps or any(n < 1 or n > 50 for n in execute_steps):
        raise ValueError("provide seeds and execution horizons within 1..50 steps")
    if language_split not in ("trained", "held-out"):
        raise ValueError("language_split must be trained or held-out")
    torch.manual_seed(42)
    model = SmolPolicy(checkpoint)
    manifest = json.loads((raw / "manifest.json").read_text())
    if manifest["state_names"] != list(STATE_NAMES):
        raise ValueError("raw state layout differs from the model adapter")
    held_out = [e for e in manifest["episodes"] if e["split"] == "eval" and e["accepted"]]
    errors, times = [], []
    sample = None
    for episode in held_out:
        with np.load(raw / f"{episode['id']}.npz") as frames:
            starts = list(range(0, len(frames["state"]) - 49, 50))
            per_episode = max(1, max_held_out_chunks // len(held_out))
            for start in starts[:per_episode]:
                before = time.perf_counter()
                prediction = model.predict(
                    episode["task"], frames["state"][start], frames["image"][start]
                )
                times.append(time.perf_counter() - before)
                truth = frames["action"][start : start + len(prediction)]
                errors.append(float(np.mean(np.square(prediction - truth))))
                if sample is None:
                    sample = {
                        "task": episode["task"],
                        "shape": list(prediction.shape),
                        "first_target": prediction[0].tolist(),
                    }
    if not errors:
        raise ValueError("no complete held-out action chunks")
    report = {
        "checkpoint": str(checkpoint),
        "held_out_chunks": len(errors),
        "action_mse_rad2": float(np.mean(errors)),
        "inference_seconds_median": float(np.median(times)),
        "sample": sample,
        "physics": {"status": "not_run"},
        "physics_language_split": language_split,
    }
    if upstream is not None and teacher is not None:
        simulation = Simulation(upstream, teacher)
        trials = []
        try:
            for horizon in execute_steps:
                for seed in seeds:
                    for template in TASKS:
                        if template.name not in manifest["task_families"]:
                            continue
                        task = template.sample(seed)
                        # Hold physical starts/noise fixed when comparing familiar wording
                        # with a reserved paraphrase; action MSE above stays held out.
                        text = task.texts[0] if language_split == "trained" else task.texts[-1]
                        # The teacher only prepares the initial state; recorded steps use VLA.
                        simulation.prepare(task, seed)
                        stable = bool(simulation.metrics()["stable"])
                        predictions = []
                        torch.manual_seed(seed)
                        for tick in range(round(manifest["seconds"] * FPS)):
                            if tick % horizon == 0:
                                state, image = simulation.observe()
                                predictions = model.predict(text, state, image)
                            action = predictions[tick % horizon]
                            if not np.isfinite(action).all() or np.any(np.abs(action) > np.pi):
                                stable = False
                                break
                            simulation.apply(action)
                            stable = stable and bool(simulation.metrics()["stable"])
                            if not stable:
                                break
                        metrics = simulation.metrics()
                        trials.append(
                            {
                                "task_family": task.name,
                                "task": text,
                                "seed": seed,
                                "execute_steps": horizon,
                                "ticks": tick + 1,
                                "metrics": metrics,
                                "success": episode_success(task, metrics, stable),
                            }
                        )
                        logging.info("physics trial=%s", trials[-1])
                        # Preserve completed trials if a later trial fails or is interrupted.
                        report["physics"] = {"status": "running", "trials": trials}
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        finally:
            simulation.close()
        # Synchronous stepping pauses physics during inference. It is not a real-time runner test.
        report["physics"] = {
            "status": "completed",
            "mode": "synchronous_delay_free",
            "trials": trials,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logging.info("evaluation saved to %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--execute-steps", nargs="+", type=int, default=[10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[10042])
    parser.add_argument("--max-held-out-chunks", type=int, default=40)
    parser.add_argument("--language-split", choices=("trained", "held-out"), default="held-out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate(
        args.checkpoint,
        args.raw,
        args.output,
        args.upstream,
        args.teacher,
        args.execute_steps,
        args.seeds,
        args.max_held_out_chunks,
        args.language_split,
    )
