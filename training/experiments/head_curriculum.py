"""Bounded stationary-head curriculum with walking replay and matched text trials.

Run prepare first; inspect its teacher/data audit before train-evaluate. Native LeRobot
owns the optimizer and checkpoint format. This driver owns the experiment and evidence.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from duckmind_policy.schema import JOINT_NAMES
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from duckmind_training.collect import sha256
from duckmind_training.contract import (
    IMAGE_KEY,
    PREVIOUS_TARGET_STATE_NAMES,
    TASKS,
    Task,
    policy_state,
)
from duckmind_training.correction_quality import recovery_outcome, stationary_head_outcome
from duckmind_training.corrections import Trace, expert_rollout
from duckmind_training.dataset import precise_numeric_stats
from duckmind_training.learned import SmolPolicy
from duckmind_training.simulation import Simulation

NAMES = ("stand", "look_left", "look_right", "look_up", "look_down", "forward")
SELECTED = {task.name: task for task in TASKS if task.name in NAMES}
UNSEEN = dict(
    zip(
        NAMES,
        (
            "站在原地保持不动",
            "脑袋朝左边转一点",
            "脑袋朝右边转一点",
            "抬起脑袋看看上面",
            "垂下脑袋看看下面",
            "朝着正前方迈步前行",
        ),
        strict=True,
    )
)
HEAD_KEYS = ("neck_pitch_delta", "head_pitch_delta", "head_yaw_delta", "head_roll_delta")
TRAIN_SEEDS = tuple(range(600, 610))
EVAL_SEEDS = (620, 621, 622)
H, E = 10, 5
BLACK = np.zeros((256, 256, 3), dtype=np.uint8)


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def note(root: Path, message: str) -> None:
    print(message, flush=True)
    with (root / "run.md").open("a") as stream:
        stream.write(message + "\n")


def dataset(root: Path, *, chunks: bool = False) -> LeRobotDataset:
    return LeRobotDataset(
        "duckmind/" + root.name,
        root=root,
        video_backend="pyav",
        delta_timestamps={"action": [i / 50 for i in range(H)]} if chunks else None,
    )


def outcome(task: Task, metrics: list[dict], steps: int, gaze: np.ndarray) -> dict:
    if task.name == "forward":
        result = recovery_outcome(metrics, steps)
        result["quality_4s"] = recovery_outcome(metrics[:201], 200)
        return result
    result = stationary_head_outcome(task, metrics, steps)
    # Optical-axis motion prevents compensating joints from faking a look direction.
    shift = gaze[-50:] - gaze[0]
    shift = np.arctan2(np.sin(shift), np.cos(shift))
    expected = (task.head[2], -task.head[1])  # azimuth left+, elevation up+
    gaze_checks = [
        bool(np.mean(shift[:, i]) * np.sign(v) > 0.1)
        if v
        else bool(np.mean(np.abs(shift[:, i])) < 0.15)
        for i, v in enumerate(expected)
    ]
    result["gaze"] = {
        "tail_shift_azimuth_elevation": shift.mean(0).tolist(),
        "axis_pass": gaze_checks,
    }
    result["accepted"] = result["accepted"] and all(gaze_checks)
    return result


def initial_witness(root: Path, sim: Simulation, seed: int) -> None:
    folder = root / "initial"
    folder.mkdir(exist_ok=True)
    state, _ = sim.observe()
    values = {
        "state": state,
        "previous_target": sim.previous_target(),
        "physics": sim.integration_state(),
    }
    path = folder / f"{seed}.npz"
    if path.exists():
        with np.load(path) as saved:
            for key, value in values.items():
                np.testing.assert_array_equal(value, saved[key])
    else:
        np.savez_compressed(path, **values)


def processor_audit(checkpoint: Path, data: Path) -> dict:
    from safetensors.torch import load_file

    stats = dataset(data).meta.stats
    config = json.loads((checkpoint / "train_config.json").read_text())
    assert config["dataset"]["image_transforms"]["enable"] is False
    observed = {}
    for filename, registry, keys in (
        ("policy_preprocessor.json", "normalizer_processor", ("observation.state", "action")),
        ("policy_postprocessor.json", "unnormalizer_processor", ("action",)),
    ):
        steps = json.loads((checkpoint / filename).read_text())["steps"]
        matches = [s for s in steps if s["registry_name"] == registry]
        assert len(matches) == 1
        step = matches[0]
        assert step["config"]["norm_map"] == {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
        values = load_file(checkpoint / step["state_file"])
        for key in keys:
            for stat in ("mean", "std"):
                name = f"{key}.{stat}"
                actual = values[name].cpu()
                torch.testing.assert_close(
                    actual, torch.tensor(stats[key][stat], dtype=actual.dtype), rtol=0, atol=0
                )
                observed[f"{registry}:{name}"] = actual.tolist()
    return observed


def prepare(args: argparse.Namespace) -> None:
    import av
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    root = args.root
    root.mkdir(exist_ok=False, parents=True)
    assert shutil.disk_usage(root).free > 20 * 2**30
    write(
        root / "config.json",
        {
            **{k: str(v) for k, v in vars(args).items()},
            "tasks": [asdict(SELECTED[n]) for n in NAMES],
            "unseen": UNSEEN,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "budget_updates": 2002,
            "formal_updates": 2000,
            "optimizer_seed": 42,
            "batch_size": 8,
            "horizon": H,
            "execute_steps": E,
            "fixed_image": True,
            "physics_paused": True,
            "source": sha256(Path(__file__)),
        },
    )
    note(
        root,
        "# v18 head curriculum\n"
        "SHOULD: same initial physics/state/prior across tasks; text is the only task cue.\n"
        "SHOULD: teacher and null/wrong-goal scale before training; preserve every failure.\n"
        "SHOULD: training-only float64 stats, black images, state51, H10/E5; "
        "2 smoke +2000 formal updates maximum.\n",
    )
    raw = root / "teacher"
    raw.mkdir()
    report = {
        "stage": "started",
        "episodes": [],
        "evaluation": [],
        "teacher_sha256": sha256(args.teacher),
        "upstream_sha256": sha256(args.upstream / "scripts/infer_policy.py"),
    }
    numeric = []
    source = dataset(args.replay)
    # Existing forward data is replay, not new independent trajectories; preserve its text.
    for ep in range(source.num_episodes):
        rows = source.hf_dataset.filter(lambda row, ep=ep: row["episode_index"] == ep)
        numeric.append(
            {
                "observation.state": np.asarray(rows["observation.state"]),
                "action": np.asarray(rows["action"]),
                "texts": [str(source.meta.tasks.iloc[int(i)].name) for i in rows["task_index"]],
                "family": "forward",
            }
        )
    assert sum(len(e["action"]) for e in numeric) == 10600
    assert source.num_episodes == 30 and source.meta.fps == 50
    assert tuple(source.features["observation.state"]["names"]) == PREVIOUS_TARGET_STATE_NAMES
    assert tuple(source.features["action"]["names"]) == JOINT_NAMES
    sim = Simulation(args.upstream, args.teacher)
    try:
        for split, seeds in [("train", TRAIN_SEEDS), ("eval", EVAL_SEEDS)]:
            for seed in seeds:
                for name in NAMES if split == "eval" else NAMES[:-1]:
                    task = SELECTED[name] if split == "eval" else SELECTED[name].sample(seed)
                    steps = 600 if name == "forward" else 200
                    sim.prepare(task, seed)
                    initial_witness(root, sim, seed)
                    sim.command(task)
                    trace = expert_rollout(sim, steps)
                    identity = f"{split}-{seed}-{name}"
                    trace.save(raw / f"{identity}.npz")
                    gaze = trace_gaze(sim, trace.physics)
                    quality = outcome(task, trace.metrics, steps, gaze)
                    row = {
                        "id": identity,
                        "seed": seed,
                        "name": name,
                        "split": split,
                        "steps": len(trace.actions),
                        "command": asdict(task),
                        "quality": quality,
                    }
                    # Judge each outcome against all goals to expose wrong-goal shortcuts.
                    row["goal_scores"] = {
                        n: outcome(SELECTED[n], trace.metrics, steps, gaze)["accepted"]
                        for n in NAMES[:-1]
                    }
                    report["episodes" if split == "train" else "evaluation"].append(row)
                    write(raw / "report.json", report)
                    note(
                        root,
                        f"Teacher {identity}: ticks={len(trace.actions)} "
                        f"accepted={quality['accepted']}",
                    )
                    if split == "train" and quality["accepted"]:
                        states = policy_state(
                            np.asarray(trace.states),
                            np.asarray(trace.previous_targets),
                            include_previous_target=True,
                        )
                        actions = np.asarray(trace.actions)
                        np.testing.assert_array_equal(states[1:, 36:], actions[:-1])
                        numeric.append(
                            {
                                "observation.state": states,
                                "action": actions,
                                "texts": [task.texts[(seed - TRAIN_SEEDS[0]) % 2]] * len(actions),
                                "family": name,
                            }
                        )
    finally:
        sim.close()
    for name in NAMES[:-1]:
        assert (
            sum(r["quality"]["accepted"] and r["name"] == name for r in report["episodes"]) >= 8
        ), name
        assert all(r["quality"]["accepted"] for r in report["evaluation"] if r["name"] == name), (
            name
        )
    # Standing should never score as any of the four head commands.
    assert all(
        not r["goal_scores"][n]
        for r in report["evaluation"]
        if r["name"] == "stand"
        for n in NAMES[1:5]
    )
    report["stage"] = "complete"
    write(raw / "report.json", report)
    data_path = root / "dataset"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (51,),
            "names": list(PREVIOUS_TARGET_STATE_NAMES),
        },
        "action": {"dtype": "float32", "shape": (15,), "names": list(JOINT_NAMES)},
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        "duckmind/v18-heads",
        fps=50,
        features=features,
        root=data_path,
        robot_type="microduck",
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast", crf=23),
    )
    try:
        for episode in numeric:
            for state, action, text in zip(
                episode["observation.state"], episode["action"], episode["texts"], strict=True
            ):
                assert np.isfinite(state).all() and np.isfinite(action).all()
                ds.add_frame(
                    {
                        "observation.state": state,
                        "action": action,
                        IMAGE_KEY: BLACK.copy(),
                        "task": text,
                    }
                )
            ds.save_episode()
    finally:
        ds.finalize()
    precise_numeric_stats(
        data_path, ds, [{k: e[k] for k in ("observation.state", "action")} for e in numeric]
    )
    loaded = dataset(data_path, chunks=True)
    for key in ("observation.state", "action"):
        values = np.concatenate([e[key] for e in numeric])
        np.testing.assert_array_equal(np.asarray(loaded.hf_dataset[key]), values)
        for stat, expected in [
            ("mean", values.astype(float).mean(0)),
            ("std", values.astype(float).std(0)),
        ]:
            np.testing.assert_allclose(loaded.meta.stats[key][stat], expected, rtol=1e-6, atol=1e-8)
    actual_texts = [
        str(loaded.meta.tasks.iloc[int(i)].name) for i in loaded.hf_dataset["task_index"]
    ]
    assert actual_texts == [text for e in numeric for text in e["texts"]]
    assert not set(UNSEEN.values()).intersection(actual_texts)
    offset = 0
    for e in numeric:
        length = len(e["action"])
        item = loaded[offset + length - 1]
        np.testing.assert_array_equal(item["action_is_pad"].numpy(), [False] + [True] * 9)
        np.testing.assert_array_equal(
            item["action"].numpy(), np.repeat(e["action"][-1:], 10, axis=0)
        )
        rows = loaded.hf_dataset.select(range(offset, offset + length))
        np.testing.assert_allclose(np.asarray(rows["timestamp"]), np.arange(length) / 50, atol=1e-6)
        offset += length
    decoded = 0
    for path in data_path.rglob("*.mp4"):
        with av.open(str(path)) as stream:
            for frame in stream.decode(video=0):
                assert not frame.to_ndarray(format="rgb24").any()
                decoded += 1
    assert decoded == offset
    zero = root / "zero-checkpoint"
    shutil.copytree(args.initial, zero)
    cfg = SmolVLAConfig.from_pretrained(str(args.initial))
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(args.initial),
        dataset_stats=loaded.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
                "stats": loaded.meta.stats,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
                "stats": loaded.meta.stats,
            }
        },
    )
    pre.save_pretrained(zero)
    post.save_pretrained(zero)
    assert sha256(zero / "model.safetensors") == sha256(args.initial / "model.safetensors")
    write(
        root / "data-audit.json",
        {
            "episodes": [
                {"family": e["family"], "frames": len(e["action"]), "text": e["texts"][0]}
                for e in numeric
            ],
            "frames": offset,
            "black_frames": decoded,
            "zero_processors": processor_audit(zero, data_path),
            "initial_sha256": sha256(args.initial / "model.safetensors"),
            "source_sha256": {str(p): sha256(p) for p in args.replay.rglob("*") if p.is_file()},
            "dataset_sha256": {str(p): sha256(p) for p in data_path.rglob("*") if p.is_file()},
        },
    )
    write(
        root / "image-contract.json",
        {"shape": [256, 256, 3], "dtype": "uint8", "value": 0, "train_and_infer": True},
    )
    note(
        root,
        f"Prepared {len(numeric)} episodes/{offset} frames. "
        "Eval bodies excluded from training. No optimizer updates yet.",
    )


def evaluate(
    args: argparse.Namespace, checkpoint: Path, phase: str, *, smoke: bool = False
) -> None:
    root = args.root
    out = root / phase
    out.mkdir(exist_ok=False)
    reference = args.replay if phase == "baseline" else root / "dataset"
    report = {
        "phase": phase,
        "stage": "started",
        "trials": [],
        "physics_paused": True,
        "model_sha256": sha256(checkpoint / "model.safetensors"),
        "processors": processor_audit(checkpoint, reference),
    }
    model = SmolPolicy(checkpoint)
    # Train-set diagnostics only: compare identical states/text/noise with persistence.
    ds = dataset(root / "dataset", chunks=True)
    episodes = json.loads((root / "data-audit.json").read_text())["episodes"]
    samples, seen, offset = [], set(), 0
    for episode in episodes:
        if episode["family"] not in seen:
            samples.extend([(episode["family"], offset), (episode["family"], offset + 20)])
            seen.add(episode["family"])
        offset += episode["frames"]
    predictions, targets, persistence, inputs = [], [], [], []
    for _family, index in samples:
        item = ds[index]
        state = item["observation.state"].numpy()
        assert not item[IMAGE_KEY].any()
        torch.manual_seed(18000000 + index)
        prediction = model.predict(item["task"], state[:36], BLACK, state[36:])
        predictions.append(prediction)
        targets.append(item["action"].numpy())
        persistence.append(np.repeat(state[None, 36:], H, axis=0))
        inputs.append(state)
    np.savez_compressed(
        out / "fit.npz",
        prediction=predictions,
        target=targets,
        persistence=persistence,
        state=inputs,
        indices=[i for _, i in samples],
    )
    report["fit_train_only"] = [
        {
            "task": family,
            "index": index,
            "model_mse_rad2": float(np.mean((prediction - target) ** 2)),
            "persistence_mse_rad2": float(np.mean((prior - target) ** 2)),
        }
        for (family, index), prediction, target, prior in zip(
            samples, predictions, targets, persistence, strict=True
        )
    ]
    sim = Simulation(args.upstream, args.teacher)
    try:
        for seed in EVAL_SEEDS[:1] if smoke else EVAL_SEEDS:
            for name in NAMES:
                task = SELECTED[name]
                for language, text in (
                    [("trained", task.texts[0])]
                    if smoke
                    else [("trained", task.texts[0]), ("unseen", UNSEEN[name])]
                ):
                    sim.prepare(task, seed)
                    initial_witness(root, sim, seed)
                    # Do not set teacher command: learned text is the sole task input.
                    trace = Trace()
                    trace.mark(sim)
                    saved = {
                        "tick": [],
                        "state": [],
                        "previous_target": [],
                        "image": [],
                        "chunk": [],
                        "latency_s": [],
                    }
                    identity = f"{seed}-{name}-{language}"
                    steps = 10 if smoke else 600 if name == "forward" else 200
                    reason = "completed"
                    try:
                        for tick in range(steps):
                            if not sim.metrics()["stable"]:
                                reason = "unstable"
                                break
                            if tick % E == 0:
                                state, _ = sim.observe()
                                prior = sim.previous_target()
                                torch.manual_seed(seed * 10000 + tick)
                                start = time.perf_counter()
                                chunk = model.predict(text, state, BLACK, prior)
                                elapsed = time.perf_counter() - start
                                for k, value in [
                                    ("tick", tick),
                                    ("state", state),
                                    ("previous_target", prior),
                                    ("image", BLACK.copy()),
                                    ("chunk", chunk),
                                    ("latency_s", elapsed),
                                ]:
                                    saved[k].append(value)
                                assert chunk.shape == (H, 15) and np.isfinite(chunk).all()
                            action = chunk[tick % E]
                            if np.any(np.abs(action) > np.pi):
                                reason = "invalid_target"
                                break
                            trace.step(sim, action)
                    finally:
                        trace.save(out / f"{identity}.npz")
                        np.savez_compressed(
                            out / f"{identity}-inputs.npz",
                            **{k: np.asarray(v) for k, v in saved.items()},
                        )
                    gaze = trace_gaze(sim, trace.physics)
                    quality = outcome(task, trace.metrics, steps, gaze)
                    if not trace.metrics[-1]["stable"]:
                        reason = "unstable"
                    row = {
                        "id": identity,
                        "body": seed,
                        "task": name,
                        "language": language,
                        "text": text,
                        "ticks": len(trace.actions),
                        "reason": reason,
                        "quality": quality,
                        "stable": len(trace.actions) == steps
                        and all(m["stable"] for m in trace.metrics),
                        "goal_scores": {
                            n: outcome(SELECTED[n], trace.metrics, steps, gaze)["accepted"]
                            for n in NAMES[:-1]
                        },
                        "latency_median_s": float(np.median(saved["latency_s"])),
                    }
                    report["trials"].append(row)
                    write(out / "report.json", report)
                    note(
                        root,
                        f"{phase} {identity}: ticks={row['ticks']} "
                        f"accepted={quality['accepted']} stable={row['stable']}",
                    )
        if smoke:
            assert len(report["trials"]) == 6
            assert all(r["ticks"] == 10 and r["reason"] == "completed" for r in report["trials"]), (
                report["trials"]
            )
        report["stage"] = "complete"
        write(out / "report.json", report)
    finally:
        sim.close()
        del model
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    root = args.root
    assert not (root / "run.pid").exists(), "run already started; inspect before any recovery"
    (root / "run.pid").write_text(str(os.getpid()))
    audit = json.loads((root / "data-audit.json").read_text())
    files = [
        Path(__file__),
        *Path("src/duckmind_training").glob("*.py"),
        *Path("../policy/src/duckmind_policy").glob("*.py"),
        Path("requirements.txt"),
    ]
    witness = {str(p.resolve()): sha256(p) for p in files if p.is_file()}
    write(root / "run-source-sha256.json", witness)
    assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
    assert sha256(args.initial / "model.safetensors") == audit["initial_sha256"]

    def stage(name: str) -> None:
        (root / "run.stage").write_text(name)
        note(root, f"## {name}")

    def train(steps: int, label: str) -> Path:
        command = [
            sys.executable,
            "-c",
            "from duckmind_training.train import main; main()",
            "--dataset",
            str(root / "dataset"),
            "--model",
            str(args.initial),
            "--output",
            str(root / label),
            "--steps",
            str(steps),
            "--batch-size",
            "8",
            "--chunk-size",
            str(H),
        ]
        note(root, f"Native trainer argv: {command!r}")
        subprocess.run(command, check=True)
        checkpoint = root / label / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        processor_audit(checkpoint, root / "dataset")
        return checkpoint

    try:
        stage("smoke-train")
        smoke = train(2, "model-smoke")
        stage("smoke-evaluate")
        evaluate(args, smoke, "smoke", smoke=True)
        stage("evaluate-baseline")
        evaluate(args, args.initial, "baseline")
        stage("evaluate-zero")
        evaluate(args, root / "zero-checkpoint", "zero-eval")
        stage("train")
        final = train(2000, "model")
        stage("evaluate-final")
        evaluate(args, final, "final")
        stage("verify")
        assert all(sha256(Path(p)) == value for p, value in witness.items())
        assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
        assert sha256(args.initial / "model.safetensors") == audit["initial_sha256"]
        stage("complete")
    except Exception as error:
        note(root, f"Failed: {error!r}")
        (root / "run.exit-code").write_text("1\n")
        raise
    (root / "run.exit-code").write_text("0\n")


def trace_gaze(sim: Simulation, physics: list | np.ndarray) -> np.ndarray:
    """Recompute corrected camera optical axis in the trunk frame, without mutating the body."""
    import mujoco

    data = mujoco.MjData(sim.model)
    body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert body >= 0
    angles = []
    for state in physics:
        mujoco.mj_setState(sim.model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(sim.model, data)
        ray = -data.cam_xmat[sim.camera.camera].reshape(3, 3)[:, 2]
        relative = data.xmat[body].reshape(3, 3).T @ ray
        angles.append(
            [
                float(np.arctan2(relative[1], relative[0])),
                float(np.arctan2(relative[2], np.hypot(*relative[:2]))),
            ]
        )
    return np.asarray(angles)


def calibrate_gaze(args: argparse.Namespace) -> None:
    """Use saved physics to calibrate task meaning before looking at student scores."""
    sim = Simulation(args.upstream, args.teacher)
    rows = []
    try:
        for path in sorted((args.root / "teacher").glob("eval-*.npz")):
            with np.load(path) as saved:
                angles = trace_gaze(sim, saved["integration_state"])
            rows.append(
                {
                    "id": path.stem,
                    "initial_azimuth_elevation": angles[0].tolist(),
                    "tail_mean_azimuth_elevation": angles[-50:].mean(0).tolist(),
                }
            )
        write(args.root / "gaze-calibration.json", rows)
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train-evaluate", "calibrate-gaze"))
    for name in ("root", "replay", "initial", "upstream", "teacher"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "train-evaluate": run, "calibrate-gaze": calibrate_gaze}[args.stage](args)


if __name__ == "__main__":
    main()
