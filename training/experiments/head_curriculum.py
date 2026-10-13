"""Bounded language-action curricula with replay and matched text trials.

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
from dataclasses import asdict, dataclass, replace
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
from duckmind_training.correction_quality import (
    locomotion_outcome,
    recovery_outcome,
    stationary_head_outcome,
)
from duckmind_training.corrections import Trace, expert_rollout
from duckmind_training.dataset import precise_numeric_stats
from duckmind_training.learned import SmolPolicy
from duckmind_training.simulation import PreparationError, Simulation


@dataclass(frozen=True)
class Curriculum:
    name: str
    new_tasks: tuple[str, ...]
    tasks: tuple[str, ...]
    train_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    unseen: dict[str, str]
    formal_steps: int
    replay_episodes: int
    replay_frames: int


HEAD_TASKS = ("stand", "look_left", "look_right", "look_up", "look_down", "forward")
HEAD_UNSEEN = dict(
    zip(
        HEAD_TASKS,
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
MOTION_TASKS = ("backward", "turn_left", "turn_right", "stop_forward", "stop_backward")
COURSES = {
    "heads": Curriculum(
        "v18-heads",
        HEAD_TASKS[:-1],
        HEAD_TASKS,
        tuple(range(600, 610)),
        (620, 621, 622),
        (620, 621, 622),
        HEAD_UNSEEN,
        2000,
        30,
        10600,
    ),
    "motion": Curriculum(
        "v19-motion",
        MOTION_TASKS,
        HEAD_TASKS + MOTION_TASKS + ("continue_forward", "continue_backward"),
        tuple(range(700, 710)),
        (720, 721, 722),
        (730, 731, 732),
        {
            **HEAD_UNSEEN,
            "backward": "朝身后退着走",
            "turn_left": "让身体朝左边转过去",
            "turn_right": "让身体朝右边转过去",
            "stop_forward": "结束走动，停在这里",
            "stop_backward": "结束走动，停在这里",
            "continue_forward": "朝着正前方迈步前行",
            "continue_backward": "朝身后退着走",
        },
        4000,
        80,
        20600,
    ),
}
SELECTED = {task.name: task for task in TASKS}
for direction in ("forward", "backward"):
    moving = SELECTED[direction]
    SELECTED["continue_" + direction] = replace(
        moving, name="continue_" + direction, initial_twist=moving.twist
    )
HEAD_KEYS = ("neck_pitch_delta", "head_pitch_delta", "head_yaw_delta", "head_roll_delta")
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
    if task.name in ("forward", "continue_forward"):
        result = recovery_outcome(metrics, steps)
        result["quality_4s"] = recovery_outcome(metrics[:201], 200)
        return result
    if task.name == "continue_backward":
        return locomotion_outcome(replace(task, name="backward"), metrics, steps)
    if task.name in MOTION_TASKS:
        return locomotion_outcome(task, metrics, steps)
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


def initial_witness(root: Path, sim: Simulation, seed: int, task: Task, preparation: dict) -> None:
    folder = root / "initial"
    folder.mkdir(exist_ok=True)
    state, _ = sim.observe()
    values = {
        "state": state,
        "previous_target": sim.previous_target(),
        "physics": sim.integration_state(),
    }
    # Stop tasks must share the same moving reset across texts/models, not a standing reset.
    group = "-moving-" + str(task.initial_twist[0]) if any(task.initial_twist) else ""
    path = folder / f"{seed}{group}.npz"
    prep_path = path.with_suffix(".json")
    if prep_path.exists():
        assert json.loads(prep_path.read_text()) == preparation
    else:
        write(prep_path, preparation)
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


def source_witness() -> dict[str, str]:
    training = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        *training.joinpath("experiments").glob("run-*-curriculum.sh"),
        *training.joinpath("src/duckmind_training").glob("*.py"),
        *training.parent.joinpath("policy/src/duckmind_policy").glob("*.py"),
        training / "requirements.txt",
    ]
    return {str(p.resolve()): sha256(p) for p in files if p.is_file()}


def prepare(args: argparse.Namespace) -> None:
    course = args.course
    root = args.root
    root.mkdir(exist_ok=False, parents=True)
    assert shutil.disk_usage(root).free > 20 * 2**30
    write(
        root / "config.json",
        {
            **{k: str(v) for k, v in vars(args).items()},
            "curriculum": asdict(course),
            "tasks": [asdict(SELECTED[n]) for n in course.tasks],
            "unseen": course.unseen,
            "train_seeds": course.train_seeds,
            "calibration_seeds": course.calibration_seeds,
            "eval_seeds": course.eval_seeds,
            "budget_updates": course.formal_steps + 2,
            "formal_updates": course.formal_steps,
            "optimizer_seed": 42,
            "batch_size": 8,
            "horizon": H,
            "execute_steps": E,
            "fixed_image": True,
            "physics_paused": True,
            "source": sha256(Path(__file__)),
            "source_files": source_witness(),
        },
    )
    note(
        root,
        f"# {course.name} curriculum\n"
        "SHOULD: same initial physics/state/prior within each start mode; text is the task cue.\n"
        "SHOULD: teacher and null/wrong-goal scale before training; preserve every failure.\n"
        "SHOULD: training-only float64 stats, black images, state51, H10/E5; "
        f"2 smoke +{course.formal_steps} formal updates maximum.\n",
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
    # Replay retains episode labels as well as every numerical frame and language string.
    replay_audit = args.replay.parent / "data-audit.json"
    families = (
        [e["family"] for e in json.loads(replay_audit.read_text())["episodes"]]
        if course.name == "v19-motion"
        else ["forward"] * source.num_episodes
    )
    assert len(families) == source.num_episodes
    for ep in range(source.num_episodes):
        rows = source.hf_dataset.filter(lambda row, ep=ep: row["episode_index"] == ep)
        if course.name == "v19-motion":
            declared = json.loads(replay_audit.read_text())["episodes"][ep]
            assert declared["frames"] == len(rows)
            assert declared["text"] == str(source.meta.tasks.iloc[int(rows[0]["task_index"])].name)
        numeric.append(
            {
                "observation.state": np.asarray(rows["observation.state"]),
                "action": np.asarray(rows["action"]),
                "texts": [str(source.meta.tasks.iloc[int(i)].name) for i in rows["task_index"]],
                "family": families[ep],
            }
        )
    assert sum(len(e["action"]) for e in numeric) == course.replay_frames
    assert source.num_episodes == course.replay_episodes and source.meta.fps == 50
    assert tuple(source.features["observation.state"]["names"]) == PREVIOUS_TARGET_STATE_NAMES
    assert tuple(source.features["action"]["names"]) == JOINT_NAMES
    sim = Simulation(args.upstream, args.teacher)
    try:
        for split, seeds in [
            ("train", course.train_seeds),
            ("calibration", course.calibration_seeds),
        ]:
            for seed in seeds:
                for name in course.tasks if split == "calibration" else course.new_tasks:
                    task = SELECTED[name] if split == "calibration" else SELECTED[name].sample(seed)
                    steps = 600 if name == "forward" else 200
                    identity = f"{split}-{seed}-{name}"
                    try:
                        preparation = sim.prepare(task, seed)
                    except PreparationError as error:
                        write(raw / f"{identity}-preparation-failed.json", error.evidence)
                        row = {
                            "id": identity,
                            "seed": seed,
                            "name": name,
                            "split": split,
                            "steps": 0,
                            "command": asdict(task),
                            "quality": {"accepted": False, "reason": "invalid_initial_state"},
                            "preparation_error": str(error),
                        }
                        report["episodes" if split == "train" else "evaluation"].append(row)
                        write(raw / "report.json", report)
                        note(root, f"Teacher {identity}: {error}")
                        if split != "train":
                            raise
                        continue
                    initial_witness(root, sim, seed, task, preparation)
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
                        for n in course.tasks
                        if n != "forward" and SELECTED[n].initial_twist == task.initial_twist
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
                                "texts": [task.texts[(seed - course.train_seeds[0]) % 2]]
                                * len(actions),
                                "family": name,
                            }
                        )
        if course.name == "v19-motion":
            controls = []
            for seed in course.calibration_seeds:
                for name in ("stop_forward", "stop_backward"):
                    task = SELECTED[name]
                    for mode in ("continue", "hold_target"):
                        preparation = sim.prepare(task, seed)
                        initial_witness(root, sim, seed, task, preparation)
                        if mode == "continue":
                            sim.command(replace(task, twist=task.initial_twist))
                            trace = expert_rollout(sim, 200)
                        else:
                            trace = Trace()
                            trace.mark(sim)
                            target = sim.previous_target().copy()
                            for _ in range(200):
                                if not sim.metrics()["stable"]:
                                    break
                                trace.step(sim, target)
                        identity = f"control-{seed}-{name}-{mode}"
                        trace.save(raw / f"{identity}.npz")
                        quality = outcome(task, trace.metrics, 200, trace_gaze(sim, trace.physics))
                        controls.append({"id": identity, "mode": mode, "quality": quality})
                        write(raw / "stop-controls.json", controls)
                        note(root, f"Stop control {identity}: {quality!r}")
            assert all(
                not row["quality"]["accepted"] for row in controls if row["mode"] == "continue"
            )
    finally:
        sim.close()
    required_calibration = course.new_tasks + (
        ("continue_forward", "continue_backward") if course.name == "v19-motion" else ()
    )
    for name in course.new_tasks:
        assert (
            sum(r["quality"]["accepted"] and r["name"] == name for r in report["episodes"]) >= 8
        ), name
    for name in required_calibration:
        rows = [r for r in report["evaluation"] if r["name"] == name]
        assert len(rows) == len(course.calibration_seeds) and all(
            r["quality"]["accepted"] for r in rows
        ), name
    # Standing should never score as any of the four head commands.
    assert all(
        not r["goal_scores"][n]
        for r in report["evaluation"]
        if r["name"] == "stand"
        for n in HEAD_TASKS[1:5]
    )
    report["stage"] = "complete"
    write(raw / "report.json", report)
    prepare_data(args, numeric)


def prepare_data(args: argparse.Namespace, numeric: list[dict]) -> None:
    import av
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    root, course = args.root, args.course
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
        "duckmind/" + course.name,
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
    assert not set(course.unseen.values()).intersection(actual_texts)
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
        "Split provenance remains in the experiment manifest. No optimizer updates yet.",
    )


def evaluate(
    args: argparse.Namespace, checkpoint: Path, phase: str, *, smoke: bool = False
) -> None:
    course = args.course
    root = args.root
    out = root / phase
    out.mkdir(exist_ok=False)
    reference = args.replay if phase == "baseline" else root / "dataset"
    report = {
        "phase": phase,
        "stage": "started",
        "trials": [],
        "invalid_starts": [],
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
        for seed in course.eval_seeds[:1] if smoke else course.eval_seeds:
            for name in course.tasks:
                task = SELECTED[name]
                for language, text in (
                    [("trained", task.texts[0])]
                    if smoke
                    else [("trained", task.texts[0]), ("unseen", course.unseen[name])]
                ):
                    identity = f"{seed}-{name}-{language}"
                    try:
                        preparation = sim.prepare(task, seed)
                    except PreparationError as error:
                        write(out / f"{identity}-preparation-failed.json", error.evidence)
                        report["invalid_starts"].append(
                            {
                                "id": identity,
                                "body": seed,
                                "task": name,
                                "language": language,
                                "text": text,
                                "reason": "invalid_initial_state",
                                "error": str(error),
                                "student_predictions": 0,
                            }
                        )
                        write(out / "report.json", report)
                        note(root, f"{phase} {identity}: invalid_initial_state: {error}")
                        if smoke:
                            raise
                        continue
                    initial_witness(root, sim, seed, task, preparation)
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
                            for n in course.tasks
                            if n != "forward" and SELECTED[n].initial_twist == task.initial_twist
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
        report["scheduled_cases"] = len(course.tasks) * (1 if smoke else len(course.eval_seeds) * 2)
        report["eligible_cases"] = len(report["trials"])
        assert report["scheduled_cases"] == len(report["trials"]) + len(report["invalid_starts"])
        if smoke:
            assert len(report["trials"]) == len(course.tasks)
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
    prepared = json.loads((root / "config.json").read_text())
    assert prepared["curriculum"] == json.loads(json.dumps(asdict(args.course))), (
        "prepared curriculum differs"
    )
    assert prepared["source"] == sha256(Path(__file__)), "source changed since prepare"
    assert prepared["source_files"] == source_witness(), "runtime source changed since prepare"
    for key in ("replay", "initial", "upstream", "teacher"):
        assert prepared[key] == str(getattr(args, key)), f"prepared {key} differs"
    assert not (root / "run.pid").exists(), "run already started; inspect before any recovery"
    (root / "run.pid").write_text(str(os.getpid()))
    audit = json.loads((root / "data-audit.json").read_text())
    witness = source_witness()
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
        final = train(args.course.formal_steps, "model")
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
        paths = sorted((args.root / "teacher").glob("calibration-*.npz"))
        if not paths:
            paths = sorted((args.root / "teacher").glob("eval-*.npz"))
        if not paths:
            raise ValueError("no saved calibration trajectories")
        for path in paths:
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


def check_starts(args: argparse.Namespace) -> None:
    """Teacher-only preflight; no student predictions or optimizer updates."""
    path = args.root.parent / "start-check.json"
    assert not path.exists()
    course = args.course
    report = {"stage": "started", "source_files": source_witness(), "cases": []}
    sim = Simulation(args.upstream, args.teacher)
    try:
        for split, seeds in (
            ("train", course.train_seeds),
            ("calibration", course.calibration_seeds),
            ("evaluation", course.eval_seeds),
        ):
            for seed in seeds:
                for name in ("stop_forward", "stop_backward"):
                    try:
                        evidence = sim.prepare(SELECTED[name], seed)
                        error = None
                    except PreparationError as failure:
                        evidence, error = failure.evidence, str(failure)
                    report["cases"].append(
                        {
                            "split": split,
                            "seed": seed,
                            "task": name,
                            "error": error,
                            "preparation": evidence,
                        }
                    )
                    write(path, report)
                    print(f"Start {split}/{seed}/{name}: {error or 'valid'}", flush=True)
        report["stage"] = "complete"
        write(path, report)
    finally:
        sim.close()


def diagnose_backward(args: argparse.Namespace) -> None:
    """Separate on-expert imitation, text sensitivity and closed-loop execution.

    spec: weights, normalization, body state and flow noise stay fixed across texts.
    Task-name lookup supplies diagnostic labels only; the policy receives plain text.
    link: docs/project/vla-backward-diagnosis-2026-10-13.md
    """
    root, source = args.root, args.replay.parent
    root.mkdir(parents=True, exist_ok=False)
    (root / "run.pid").write_text(str(os.getpid()))
    (root / "dataset").symlink_to(args.replay)
    (root / "data-audit.json").symlink_to(source / "data-audit.json")
    witness = source_witness()
    checkpoint_sha = sha256(args.initial / "model.safetensors")
    texts = (*SELECTED["backward"].texts, args.course.unseen["backward"], "保持站立", "向前走")
    episodes = [f"train-{seed}-backward" for seed in (700, 701, 702)] + [
        f"calibration-{seed}-backward" for seed in (720, 721, 722)
    ]
    ticks = (0, 20, 60, 100, 140, 180)
    report = {
        "stage": "started",
        "optimizer_updates": 0,
        "argv": sys.argv,
        "paths": {
            k: str(getattr(args, k).resolve())
            for k in ("root", "replay", "initial", "upstream", "teacher")
        },
        "text_roles": [
            "trained",
            "trained",
            "never-trained-catalog",
            "v19-held-out",
            "wrong-goal-stand",
            "wrong-goal-forward",
        ],
        "model_sha256": checkpoint_sha,
        "source_files": witness,
        "processors": processor_audit(args.initial, args.replay),
        "texts": texts,
        "episodes": episodes,
        "ticks": ticks,
        "rows": [],
        "artifact_sha256": {},
    }
    note(
        root,
        "# Backward diagnosis\nSHOULD: 0 updates; identical state/prior/noise across texts.\n"
        "SHOULD: 6 episodes × 6 ticks × 6 texts = 216 paired predictions.\n"
        "Training/calibration/development states only; no new holdout claim.",
    )
    model = SmolPolicy(args.initial)
    predictions, targets, states, priors = [], [], [], []
    try:
        (root / "run.stage").write_text("offline")
        for episode_index, identity in enumerate(episodes):
            path = source / "teacher" / f"{identity}.npz"
            report["artifact_sha256"][str(path)] = sha256(path)
            with np.load(path) as raw:
                np.testing.assert_array_equal(raw["previous_target"][1:], raw["action"][:-1])
                for tick in ticks:
                    state, prior = raw["state"][tick], raw["previous_target"][tick]
                    target = raw["action"][tick : tick + H]
                    assert target.shape == (H, 15)
                    noise = 20000000 + episode_index * 1000 + tick
                    # Capture native processor outputs for one matched input per text.
                    if episode_index == 0 and tick == 0:
                        batches = {}
                        for text_index, text in enumerate(texts):
                            batch = model.preprocess(
                                {
                                    "observation.state": torch.from_numpy(
                                        policy_state(state, prior, include_previous_target=True)
                                    ),
                                    IMAGE_KEY: torch.from_numpy(BLACK.copy())
                                    .permute(2, 0, 1)
                                    .float(),
                                    "task": text,
                                }
                            )
                            for key, value in batch.items():
                                if isinstance(value, torch.Tensor):
                                    batches[f"{text_index}:{key}"] = value.cpu().numpy()
                        np.savez_compressed(root / "native-inputs.npz", **batches)
                    for text_index, text in enumerate(texts):
                        torch.manual_seed(noise)
                        chunk = model.predict(text, state, BLACK, prior)
                        if not predictions:
                            torch.manual_seed(noise)
                            repeated = model.predict(text, state, BLACK, prior)
                            np.testing.assert_array_equal(chunk, repeated)
                            report["same_input_smoke"] = "exact"
                        predictions.append(chunk)
                        targets.append(target)
                        states.append(state)
                        priors.append(prior)
                        error = (chunk - target) ** 2
                        persistence = np.repeat(prior[None], H, axis=0)
                        row = {
                            "episode": identity,
                            "tick": tick,
                            "text_index": text_index,
                            "noise_seed": noise,
                            "first_mse": float(error[0].mean()),
                            "executed5_mse": float(error[:E].mean()),
                            "chunk_mse": float(error.mean()),
                            "per_joint_mse": error.mean(0).tolist(),
                            "persistence_mse": float(((persistence - target) ** 2).mean()),
                        }
                        report["rows"].append(row)
            note(root, f"Offline {identity}: {len(report['rows'])}/216 predictions saved.")
            write(root / "report.json", report)
        np.savez_compressed(
            root / "offline.npz",
            prediction=predictions,
            target=targets,
            state=states,
            previous_target=priors,
        )
        assert len(predictions) == 216
        del model
        torch.cuda.empty_cache()
        (root / "run.stage").write_text("closed-loop")
        # Reuse the native evaluation/Trace path; no second simulator/controller implementation.
        args.course = replace(args.course, tasks=("backward",), eval_seeds=(700, 730, 731))
        evaluate(args, args.initial, "reproduce")
        comparisons = []
        for seed in (730, 731):
            for language in ("trained", "unseen"):
                identity = f"{seed}-backward-{language}"
                for suffix in (".npz", "-inputs.npz"):
                    old, new = (
                        source / "final" / (identity + suffix),
                        root / "reproduce" / (identity + suffix),
                    )
                    report["artifact_sha256"][str(old)] = sha256(old)
                    with np.load(old) as a, np.load(new) as b:
                        # Wall time/rendered evidence frames may vary; policy input is always black.
                        keys = (
                            ("tick", "state", "previous_target", "image", "chunk")
                            if suffix.startswith("-")
                            else (
                                "state",
                                "action",
                                "previous_target",
                                "integration_state",
                                "actuator_target",
                            )
                        )
                        comparisons.append(
                            {
                                "id": identity,
                                "file": suffix,
                                "exact": {
                                    key: bool(np.array_equal(a[key], b[key])) for key in keys
                                },
                            }
                        )
        report["reproduction"] = comparisons
        assert sha256(args.initial / "model.safetensors") == checkpoint_sha
        assert source_witness() == witness
        assert all(sha256(Path(p)) == digest for p, digest in report["artifact_sha256"].items())
        report["stage"] = "complete"
        exact = all(all(c["exact"].values()) for c in comparisons)
        note(
            root,
            "Complete: 0 updates; 216 paired predictions +1 same-input smoke; "
            f"6 closed-loop cases. Reproduction exact: {exact}",
        )
        (root / "run.stage").write_text("complete")
        (root / "run.exit-code").write_text("0")
    except BaseException:
        report["stage"] = "failed"
        (root / "run.exit-code").write_text("1")
        raise
    finally:
        write(root / "report.json", report)


def prepare_backward_corrections(args: argparse.Namespace) -> None:
    """Collect teacher recovery labels from exactly replayed student prefixes."""
    root = args.root
    if args.diagnosis_root is None:
        raise ValueError("--diagnosis-root is required for backward corrections")
    root.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(root).free > 20 * 2**30
    (root / "prepare.pid").write_text(str(os.getpid()))
    source = dataset(args.replay)
    original = json.loads((args.replay.parent / "data-audit.json").read_text())
    numeric = []
    for index, ep in enumerate(original["episodes"]):
        rows = source.hf_dataset.filter(lambda row, index=index: row["episode_index"] == index)
        assert len(rows) == ep["frames"]
        numeric.append(
            {
                "observation.state": np.asarray(rows["observation.state"]),
                "action": np.asarray(rows["action"]),
                "texts": [str(source.meta.tasks.iloc[int(i)].name) for i in rows["task_index"]],
                "family": ep["family"],
            }
        )
    assert len(numeric) == 129 and sum(len(e["action"]) for e in numeric) == 30400
    control_episodes = [i for i, e in enumerate(numeric) if e["family"] == "backward"]
    assert len(control_episodes) == 10
    raw = root / "teacher"
    raw.mkdir()
    report = {"stage": "started", "candidates": [], "controls": [], "source_sha256": {}}
    task = SELECTED["backward"]
    sim = Simulation(args.upstream, args.teacher)
    note(
        root,
        "# Backward focused correction pair\n"
        "SHOULD: exact student-prefix replay; teacher-only labels after handover.\n"
        "SHOULD: retain all rejected windows; unchanged strict backward gate.\n"
        "SHOULD: 2+1000 updates per arm, maximum2004, shared union normalization.\n"
        "This is a backward-only diagnostic; old-task regression is required.",
    )
    try:
        for seed in (700, 730, 740, 741, 742):
            sim.prepare(task, seed)
            sim.command(task)
            trace = expert_rollout(sim, 200)
            trace.save(raw / f"control-{seed}.npz")
            quality = locomotion_outcome(task, trace.metrics, 200)
            report["controls"].append({"seed": seed, "quality": quality})
            write(raw / "report.json", report)
            assert quality["accepted"], (seed, quality)
        for seed in (700, 730):
            path = args.diagnosis_root / "reproduce" / f"{seed}-backward-trained.npz"
            report["source_sha256"][str(path)] = sha256(path)
            with np.load(path) as saved:
                # First prove the entire recorded failure is reproducible before selecting prefixes.
                sim.prepare(task, seed)
                np.testing.assert_array_equal(
                    sim.integration_state(), saved["integration_state"][0]
                )
                for tick, action in enumerate(saved["action"]):
                    sim.apply(action)
                    np.testing.assert_array_equal(
                        sim.integration_state(), saved["integration_state"][tick + 1]
                    )
                assert not sim.metrics()["stable"]
                for prefix in (20, 50, 80):
                    sim.prepare(task, seed)
                    for action in saved["action"][:prefix]:
                        sim.apply(action)
                    before = sim.integration_state().copy()
                    np.testing.assert_array_equal(before, saved["integration_state"][prefix])
                    np.testing.assert_array_equal(
                        sim.previous_target(), saved["action"][prefix - 1]
                    )
                    assert sim.metrics()["stable"]
                    sim.resume_teacher(task)
                    np.testing.assert_array_equal(before, sim.integration_state())
                    trace = expert_rollout(sim, 200)
                    identity = f"recovery-{seed}-{prefix}"
                    trace.save(raw / f"{identity}.npz")
                    quality = locomotion_outcome(task, trace.metrics, 200)
                    row = {"id": identity, "seed": seed, "prefix": prefix, "quality": quality}
                    report["candidates"].append(row)
                    write(raw / "report.json", report)
                    note(root, f"{identity}: ticks={len(trace.actions)} quality={quality!r}")
                    if quality["accepted"]:
                        states = policy_state(
                            np.asarray(trace.states),
                            np.asarray(trace.previous_targets),
                            include_previous_target=True,
                        )
                        actions = np.asarray(trace.actions)
                        np.testing.assert_array_equal(states[0, 36:], saved["action"][prefix - 1])
                        np.testing.assert_array_equal(states[1:, 36:], actions[:-1])
                        numeric.append(
                            {
                                "observation.state": states,
                                "action": actions,
                                "texts": [task.texts[(len(numeric) - 129) % 2]] * len(actions),
                                "family": "backward",
                            }
                        )
        accepted = [r for r in report["candidates"] if r["quality"]["accepted"]]
        assert len(accepted) >= 4 and {r["seed"] for r in accepted} == {700, 730}, accepted
        report["stage"] = "complete"
        write(raw / "report.json", report)
    finally:
        sim.close()
    prepare_data(args, numeric)
    config = {
        "argv": sys.argv,
        "source_files": source_witness(),
        "initial_sha256": sha256(args.initial / "model.safetensors"),
        "paths": {
            k: str(getattr(args, k).resolve())
            for k in ("root", "replay", "initial", "upstream", "teacher")
        },
        "control_episodes": control_episodes,
        "mixed_episodes": control_episodes + list(range(129, len(numeric))),
        "updates_per_arm": {"discarded_smoke": 2, "formal": 1000},
        "max_updates": 2004,
        "eval_backward_bodies": [700, 730, 740, 741, 742],
        "regression_body": 730,
        "physics_paused": True,
        "external_sha256": {
            str(p): sha256(p)
            for p in (
                args.teacher,
                args.upstream / "scripts/infer_policy.py",
                args.upstream / "src/mjlab_microduck/robot/microduck/scene.xml",
            )
        },
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    write(root / "pair-config.json", config)
    (root / "prepare.exit-code").write_text("0")
    note(
        root,
        f"Preparation complete; {len(accepted)}/6 accepted across2 development bodies. "
        "No student inference or optimizer update during preparation.",
    )


def run_backward_pair(args: argparse.Namespace) -> None:
    """Native episode selection isolates focused fitting from added recovery coverage."""
    root = args.root
    cfg = json.loads((root / "pair-config.json").read_text())
    assert cfg["source_files"] == source_witness()
    assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
    assert all(sha256(Path(p)) == value for p, value in cfg["external_sha256"].items())
    audit = json.loads((root / "data-audit.json").read_text())
    assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
    assert not (root / "run.pid").exists()
    (root / "run.pid").write_text(str(os.getpid()))
    course = args.course

    def stage(name: str) -> None:
        (root / "run.stage").write_text(name)
        note(root, f"## {name}")

    def train(label: str, steps: int, episodes: list[int]) -> Path:
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
            "--episodes",
            *map(str, episodes),
        ]
        note(root, f"Native trainer argv: {command!r}")
        subprocess.run(command, check=True)
        checkpoint = root / label / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        processor_audit(checkpoint, root / "dataset")
        return checkpoint

    def evaluate_pair(checkpoint: Path, label: str) -> None:
        args.course = replace(
            course, tasks=("backward",), eval_seeds=tuple(cfg["eval_backward_bodies"])
        )
        evaluate(args, checkpoint, label + "-backward")
        args.course = replace(course, tasks=HEAD_TASKS, eval_seeds=(cfg["regression_body"],))
        evaluate(args, checkpoint, label + "-regression")

    try:
        for arm in ("control", "mixed"):
            stage("smoke-" + arm)
            smoke = train("smoke-" + arm, 2, cfg[arm + "_episodes"])
            args.course = replace(course, tasks=("backward",), eval_seeds=(700,))
            evaluate(args, smoke, "smoke-eval-" + arm, smoke=True)
        stage("zero")
        evaluate_pair(root / "zero-checkpoint", "zero")
        for arm in ("control", "mixed"):
            stage("train-" + arm)
            checkpoint = train("model-" + arm, 1000, cfg[arm + "_episodes"])
            stage("evaluate-" + arm)
            evaluate_pair(checkpoint, arm)
        stage("verify")
        assert cfg["source_files"] == source_witness()
        assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
        assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
        stage("complete")
        (root / "run.exit-code").write_text("0")
    except BaseException:
        (root / "run.exit-code").write_text("1")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "train-evaluate",
            "calibrate-gaze",
            "check-starts",
            "diagnose-backward",
            "prepare-backward-corrections",
            "train-backward-pair",
        ),
    )
    for name in ("root", "replay", "initial", "upstream", "teacher"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--course", choices=tuple(COURSES), default="heads")
    parser.add_argument("--diagnosis-root", type=Path)
    args = parser.parse_args()
    args.course = COURSES[args.course]
    {
        "prepare": prepare,
        "train-evaluate": run,
        "calibrate-gaze": calibrate_gaze,
        "check-starts": check_starts,
        "diagnose-backward": diagnose_backward,
        "prepare-backward-corrections": prepare_backward_corrections,
        "train-backward-pair": run_backward_pair,
    }[args.stage](args)


if __name__ == "__main__":
    main()
