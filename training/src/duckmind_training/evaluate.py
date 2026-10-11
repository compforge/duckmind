"""Report held-out prediction error separately from delay-free physics rollouts."""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from duckmind_training.contract import FPS, TASKS, episode_success
from duckmind_training.learned import SmolPolicy
from duckmind_training.simulation import Simulation


def evaluate(
    checkpoint: Path, raw: Path, output: Path, upstream: Path | None, teacher: Path | None
) -> None:
    torch.manual_seed(42)
    model = SmolPolicy(checkpoint)
    manifest = json.loads((raw / "manifest.json").read_text())
    held_out = [e for e in manifest["episodes"] if e["split"] == "eval" and e["accepted"]]
    errors, times = [], []
    sample = None
    for episode in held_out:
        with np.load(raw / f"{episode['id']}.npz") as frames:
            for start in range(0, len(frames["state"]) - 49, 50):
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
    }
    if upstream is not None and teacher is not None:
        simulation = Simulation(upstream, teacher)
        trials = []
        try:
            for task in TASKS:
                # Expert is used only for initial settling; every recorded step uses the VLA.
                simulation.reset(10042)
                stable = True
                predictions = []
                for tick in range(round(manifest["seconds"] * FPS)):
                    if tick % 10 == 0:
                        state, image = simulation.observe()
                        predictions = model.predict(task.texts[-1], state, image).tolist()
                    action = np.asarray(predictions[tick % 10])
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
                        "task": task.texts[-1],
                        "ticks": tick + 1,
                        "metrics": metrics,
                        "success": episode_success(
                            task,
                            float(metrics["displacement_m"]),
                            float(metrics["head_yaw_rad"]),
                            stable,
                        ),
                    }
                )
                logging.info("physics trial=%s", trials[-1])
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    evaluate(args.checkpoint, args.raw, args.output, args.upstream, args.teacher)
