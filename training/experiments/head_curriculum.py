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
    args: argparse.Namespace,
    checkpoint: Path,
    phase: str,
    *,
    smoke: bool = False,
    execute_steps: int = E,
    texts: dict[str, list[tuple[str, str]]] | None = None,
    forward_steps: int = 600,
) -> None:
    assert 1 <= execute_steps <= H
    course = args.course
    root = args.root
    out = root / phase
    out.mkdir(exist_ok=False)
    reference = args.replay if phase == "baseline" else root / "dataset"
    report = {
        "phase": phase,
        "execute_steps": execute_steps,
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
    language_cases = (
        texts
        if texts is not None
        else {
            name: (
                [("trained", SELECTED[name].texts[0])]
                if smoke
                else [("trained", SELECTED[name].texts[0]), ("unseen", course.unseen[name])]
            )
            for name in course.tasks
        }
    )
    assert set(language_cases) == set(course.tasks)
    for cases in language_cases.values():
        assert cases and len({label for label, _ in cases}) == len(cases)
        assert all(label.replace("-", "").isalnum() for label, _ in cases)
    seeds = course.eval_seeds[:1] if smoke else course.eval_seeds
    sim = Simulation(args.upstream, args.teacher)
    try:
        for seed in seeds:
            for name in course.tasks:
                task = SELECTED[name]
                for language, text in language_cases[name]:
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
                    steps = 10 if smoke else forward_steps if name == "forward" else 200
                    reason = "completed"
                    try:
                        for tick in range(steps):
                            if not sim.metrics()["stable"]:
                                reason = "unstable"
                                break
                            if tick % execute_steps == 0:
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
                            action = chunk[tick % execute_steps]
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
        report["scheduled_cases"] = len(seeds) * sum(
            len(cases) for cases in language_cases.values()
        )
        report["eligible_cases"] = len(report["trials"])
        assert report["scheduled_cases"] == len(report["trials"]) + len(report["invalid_starts"])
        if smoke:
            assert len(report["trials"]) == report["scheduled_cases"]
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
    finalize_backward_preparation(args)


def finalize_backward_preparation(args: argparse.Namespace) -> None:
    """Finish provenance from saved artifacts without repeating collection or training."""
    root = args.root
    assert not (root / "pair-config.json").exists() and not (root / "run.pid").exists()
    report = json.loads((root / "teacher/report.json").read_text())
    audit = json.loads((root / "data-audit.json").read_text())
    accepted = [r for r in report["candidates"] if r["quality"]["accepted"]]
    assert report["stage"] == "complete" and len(report["candidates"]) == 6
    assert len(accepted) >= 4 and {r["seed"] for r in accepted} == {700, 730}
    assert len(audit["episodes"]) == 129 + len(accepted)
    assert audit["frames"] == 30400 + 200 * len(accepted)
    for key in ("source_sha256", "dataset_sha256"):
        assert all(sha256(Path(p)) == value for p, value in audit[key].items())
    assert audit["initial_sha256"] == sha256(args.initial / "model.safetensors")
    assert sha256(root / "zero-checkpoint/model.safetensors") == audit["initial_sha256"]
    processor_audit(root / "zero-checkpoint", root / "dataset")
    control_episodes = [
        i for i, e in enumerate(audit["episodes"][:129]) if e["family"] == "backward"
    ]
    assert len(control_episodes) == 10
    revision = subprocess.run(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode:
        note(
            root,
            "Upstream Git revision unavailable; verified file hashes retain source identity: "
            + revision.stderr.strip(),
        )
    config = {
        "argv": sys.argv,
        "source_files": source_witness(),
        "initial_sha256": sha256(args.initial / "model.safetensors"),
        "paths": {
            k: str(getattr(args, k).resolve())
            for k in ("root", "replay", "initial", "upstream", "teacher")
        },
        "control_episodes": control_episodes,
        "mixed_episodes": control_episodes + list(range(129, len(audit["episodes"]))),
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
        "upstream_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "upstream_git_error": revision.stderr.strip() if revision.returncode else None,
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


def diagnose_correction_fit(args: argparse.Namespace) -> None:
    """Compare decoded predictions on the exact v21 training states, with no updates.

    spec: pair state, text, target, normalization and flow noise across checkpoints.
    link: docs/project/vla-correction-fit-2026-10-14.md
    """
    root, source = args.root, args.replay.parent
    root.mkdir(parents=True, exist_ok=False)
    (root / "run.pid").write_text(str(os.getpid()))
    witness = source_witness()
    audit = json.loads((source / "data-audit.json").read_text())
    pair = json.loads((source / "pair-config.json").read_text())
    assert (source / "run.exit-code").read_text().strip() == "0"
    assert len(audit["episodes"]) == 134 and audit["frames"] == 31400
    assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
    old_ids = pair["control_episodes"]
    correction_ids = sorted(set(pair["mixed_episodes"]) - set(old_ids))
    assert len(old_ids) == 10 and correction_ids == list(range(129, 134))
    checkpoints = {
        "zero": source / "zero-checkpoint",
        "control": source / "model-control/checkpoints/001000/pretrained_model",
        "mixed": source / "model-mixed/checkpoints/001000/pretrained_model",
    }
    model_hashes = {name: sha256(path / "model.safetensors") for name, path in checkpoints.items()}
    ds = dataset(args.replay, chunks=True)
    ticks = (0, 20, 60, 100, 140, 180)
    offsets = np.cumsum([0] + [e["frames"] for e in audit["episodes"]])
    original_root = Path(pair["paths"]["replay"]).parent
    original_report = json.loads((original_root / "teacher/report.json").read_text())
    original_rows = [
        r
        for r in original_report["episodes"]
        if r["name"] == "backward" and r["quality"]["accepted"]
    ]
    correction_report = json.loads((source / "teacher/report.json").read_text())
    correction_rows = [r for r in correction_report["candidates"] if r["quality"]["accepted"]]
    raw_paths = {
        **{
            episode: original_root / "teacher" / (row["id"] + ".npz")
            for episode, row in zip(old_ids, original_rows, strict=True)
        },
        **{
            episode: source / "teacher" / (row["id"] + ".npz")
            for episode, row in zip(correction_ids, correction_rows, strict=True)
        },
    }
    raw_hashes = {str(path): sha256(path) for path in raw_paths.values()}
    samples, states, targets, images = [], [], [], []
    for episode in old_ids + correction_ids:
        assert audit["episodes"][episode]["family"] == "backward"
        assert audit["episodes"][episode]["frames"] == 200
        with np.load(raw_paths[episode]) as raw:
            raw_state = raw["state"].copy()
            raw_prior = raw["previous_target"].copy()
            raw_action = raw["action"].copy()
        np.testing.assert_array_equal(raw_prior[1:], raw_action[:-1])
        expected_text = (
            SELECTED["backward"].texts[old_ids.index(episode) % 2]
            if episode in old_ids
            else SELECTED["backward"].texts[correction_ids.index(episode) % 2]
        )
        for tick in ticks:
            item = ds[int(offsets[episode]) + tick]
            assert int(item["episode_index"]) == episode
            assert int(item["frame_index"]) == tick and not item["action_is_pad"].any()
            assert not item[IMAGE_KEY].any()
            state, target = item["observation.state"].numpy(), item["action"].numpy()
            assert state.shape == (51,) and target.shape == (H, 15)
            np.testing.assert_array_equal(state[:36], raw_state[tick])
            np.testing.assert_array_equal(state[36:], raw_prior[tick])
            np.testing.assert_array_equal(target, raw_action[tick : tick + H])
            assert item["task"] == expected_text
            samples.append(
                {
                    "episode": episode,
                    "tick": tick,
                    "text": item["task"],
                    "group": "correction" if episode in correction_ids else "original",
                    "period": "early" if tick < 60 else "late",
                    "noise_seeds": [22000000 + episode * 1000 + tick * 2 + j for j in range(2)],
                }
            )
            states.append(state)
            targets.append(target)
            images.append(BLACK.copy())
    states, targets, images = np.asarray(states), np.asarray(targets), np.asarray(images)
    np.savez_compressed(root / "inputs.npz", state=states, target=targets, image=images)
    leg_indices = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(("left_", "right_"))]
    assert len(leg_indices) == 10
    report = {
        "stage": "started",
        "optimizer_updates": 0,
        "physics_rollouts": 0,
        "argv": sys.argv,
        "source_files": witness,
        "samples": samples,
        "model_sha256": model_hashes,
        "raw_sha256": raw_hashes,
        "dataset_sha256": audit["dataset_sha256"],
        "input_sha256": sha256(root / "inputs.npz"),
        "leg_indices": leg_indices,
        "inference_budget": 543,
        "inference_calls": 0,
        "arms": {},
    }
    note(
        root,
        "# Correction-state fitting diagnosis\n"
        "SHOULD: 0 updates/rollouts; "
        "15 episodes × 6 ticks × 2 noises × 3 models + 3 repeats = 543 calls.\n"
        "SHOULD: identical state, stored text, black image, noise and targets across models.\n"
        "Training-state fitting only; overlapping corrections from two bodies are not holdout.",
    )
    try:
        for arm, path in checkpoints.items():
            (root / "run.stage").write_text(arm)
            stats = processor_audit(path, args.replay)
            model = SmolPolicy(path)
            native = model.preprocess(
                {
                    "observation.state": torch.from_numpy(states[0].copy()),
                    IMAGE_KEY: torch.from_numpy(images[0].copy()).permute(2, 0, 1).float() / 255,
                    "task": samples[0]["text"],
                }
            )
            np.savez_compressed(
                root / f"{arm}-native-input.npz",
                **{k: v.cpu().numpy() for k, v in native.items() if isinstance(v, torch.Tensor)},
            )
            predictions, latencies, rows = [], [], []
            arm_report = {"checkpoint": str(path), "processors": stats, "rows": rows}
            report["arms"][arm] = arm_report
            try:
                for sample_id, sample in enumerate(samples):
                    state, target, image = states[sample_id], targets[sample_id], images[sample_id]
                    for noise in sample["noise_seeds"]:
                        torch.manual_seed(noise)
                        start = time.perf_counter()
                        chunk = model.predict(sample["text"], state[:36], image, state[36:])
                        elapsed = time.perf_counter() - start
                        report["inference_calls"] += 1
                        assert chunk.shape == (H, 15) and np.isfinite(chunk).all()
                        predictions.append(chunk)
                        latencies.append(elapsed)
                        if len(predictions) == 1:
                            torch.manual_seed(noise)
                            repeat = model.predict(sample["text"], state[:36], image, state[36:])
                            report["inference_calls"] += 1
                            np.savez_compressed(
                                root / f"{arm}-smoke.npz", first=chunk, repeated=repeat
                            )
                            np.testing.assert_array_equal(chunk, repeat)
                            arm_report["same_input_smoke"] = "exact"
                        error = (chunk - target) ** 2
                        persistence_error = (state[None, 36:] - target) ** 2
                        row = {
                            "sample_id": sample_id,
                            "noise_seed": noise,
                            "latency_s": elapsed,
                        }
                        for name, count in (("first", 1), ("executed5", E), ("chunk", H)):
                            row[name + "_per_joint"] = error[:count].mean(0).tolist()
                            row[name + "_persistence_per_joint"] = (
                                persistence_error[:count].mean(0).tolist()
                            )
                            row[name + "_mse"] = float(error[:count].mean())
                            row[name + "_legs_mse"] = float(error[:count, leg_indices].mean())
                            row[name + "_persistence_mse"] = float(persistence_error[:count].mean())
                            row[name + "_persistence_legs_mse"] = float(
                                persistence_error[:count, leg_indices].mean()
                            )
                        rows.append(row)
                    if sample["tick"] == ticks[-1]:
                        note(
                            root,
                            f"{arm}: episode{sample['episode']}; {len(rows)}/180 predictions",
                        )
                        write(root / "report.json", report)
            finally:
                np.savez_compressed(
                    root / f"{arm}-predictions.npz", prediction=predictions, latency_s=latencies
                )
                del model
                torch.cuda.empty_cache()
            arm_report["prediction_sha256"] = sha256(root / f"{arm}-predictions.npz")
            arm_report["summary"] = {}
            for group in ("original", "correction"):
                for period in ("early", "late"):
                    selected = [
                        row
                        for row in rows
                        if samples[row["sample_id"]]["group"] == group
                        and samples[row["sample_id"]]["period"] == period
                    ]
                    arm_report["summary"][group + "_" + period] = {
                        "predictions": len(selected),
                        **{
                            key: float(np.mean([row[key] for row in selected]))
                            for key in selected[0]
                            if key.endswith("mse")
                        },
                    }
            arm_report["per_episode"] = {
                str(episode): {
                    key: float(
                        np.mean(
                            [
                                row[key]
                                for row in rows
                                if samples[row["sample_id"]]["episode"] == episode
                            ]
                        )
                    )
                    for key in rows[0]
                    if key.endswith("mse")
                }
                for episode in old_ids + correction_ids
            }
            write(root / "report.json", report)
        assert report["inference_calls"] == 543
        assert source_witness() == witness
        assert all(sha256(Path(p)) == value for p, value in raw_hashes.items())
        assert all(sha256(Path(p)) == value for p, value in audit["dataset_sha256"].items())
        assert all(
            sha256(checkpoints[name] / "model.safetensors") == value
            for name, value in model_hashes.items()
        )
        assert sha256(root / "inputs.npz") == report["input_sha256"]
        report["stage"] = "complete"
        (root / "run.stage").write_text("complete")
        (root / "run.exit-code").write_text("0")
        note(
            root,
            "## Complete\n543 calls, zero updates/rollouts. Decoded MSE is not physical success.",
        )
    except BaseException:
        report["stage"] = "failed"
        (root / "run.exit-code").write_text("1")
        raise
    finally:
        write(root / "report.json", report)


def diagnose_feedback(args: argparse.Namespace) -> None:
    """Compare execution horizons on fixed checkpoints without optimizer updates.

    spec: hold model, normalization, initial state and common-tick flow seeds fixed.
    why: E1 also changes chunk offset and sampling frequency; it is not pure feedback isolation.
    link: docs/project/vla-feedback-2026-10-14.md
    """
    root, source = args.root, args.replay.parent
    root.mkdir(parents=True, exist_ok=False)
    (root / "run.pid").write_text(str(os.getpid()))
    started = time.perf_counter()
    witness = source_witness()
    audit = json.loads((source / "data-audit.json").read_text())
    assert (source / "run.exit-code").read_text().strip() == "0"
    assert all(sha256(Path(p)) == h for p, h in audit["dataset_sha256"].items())
    (root / "dataset").symlink_to(args.replay, target_is_directory=True)
    shutil.copy2(source / "data-audit.json", root / "data-audit.json")
    checkpoints = {
        arm: source / f"model-{arm}/checkpoints/001000/pretrained_model"
        for arm in ("control", "mixed")
    }
    checkpoint_hashes = {
        str(p): sha256(p)
        for checkpoint in checkpoints.values()
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    cfg = {
        "optimizer_updates": 0,
        "maximum_inference_calls": 4944,
        "main_rollouts": 40,
        "smoke_rollouts": 2,
        "max_ticks": 200,
        "bodies": [700, 730, 740, 741, 742],
        "execute_steps": [5, 1],
        "horizon": H,
        "physics_paused": True,
        "fresh_holdout": False,
        "argv": sys.argv,
        "source_files": witness,
        "checkpoint_files": checkpoint_hashes,
        "dataset_sha256": audit["dataset_sha256"],
    }
    write(root / "config.json", cfg)
    note(
        root,
        "# v23 execution-horizon comparison\n"
        "SHOULD: identical state/prior/physics at reset and common-tick noise seeds.\n"
        "SHOULD: E5 numeric traces reproduce v21; images in Trace are not model inputs.\n"
        "SHOULD: 0 updates, <=4944 inference calls, 40 main +2 smoke rollouts.\n"
        "E1 changes state feedback, chunk offset and flow sampling frequency together.\n",
    )
    course = replace(args.course, tasks=("backward",), eval_seeds=tuple(cfg["bodies"]))
    summary = {"stage": "started", "phases": [], "optimizer_updates": 0}
    try:
        for execute in (5, 1):
            phase = f"smoke-e{execute}"
            (root / "run.stage").write_text(phase)
            args.course = replace(course, eval_seeds=(700,))
            evaluate(args, checkpoints["mixed"], phase, smoke=True, execute_steps=execute)
            smoke = json.loads((root / phase / "report.json").read_text())
            assert all(row["stable"] for row in smoke["trials"])
        args.course = course
        for execute in (5, 1):
            for arm, checkpoint in checkpoints.items():
                phase = f"{arm}-e{execute}"
                (root / "run.stage").write_text(phase)
                torch.cuda.reset_peak_memory_stats()
                phase_started = time.perf_counter()
                evaluate(args, checkpoint, phase, execute_steps=execute)
                report = json.loads((root / phase / "report.json").read_text())
                assert report["eligible_cases"] == 10 and not report["invalid_starts"]
                if execute == 5:
                    # Keep E5 an exact numerical reproduction before interpreting E1.
                    for row in report["trials"]:
                        for suffix in (".npz", "-inputs.npz"):
                            with np.load(root / phase / (row["id"] + suffix)) as new:
                                with np.load(
                                    source / (arm + "-backward") / (row["id"] + suffix)
                                ) as old:
                                    for key in new.files:
                                        if key not in ("image", "latency_s"):
                                            np.testing.assert_array_equal(new[key], old[key])
                summary["phases"].append(
                    {
                        "phase": phase,
                        "accepted": sum(r["quality"]["accepted"] for r in report["trials"]),
                        "stable": sum(r["stable"] for r in report["trials"]),
                        "denominator": 10,
                        "elapsed_s": time.perf_counter() - phase_started,
                        "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                        "e5_reproduced": execute == 5,
                    }
                )
                write(root / "report.json", summary)
                note(root, f"Completed {phase}: {summary['phases'][-1]}")
        phases = ["smoke-e5", "smoke-e1", "control-e5", "mixed-e5", "control-e1", "mixed-e1"]
        calls = 0
        for phase in phases:
            report = json.loads((root / phase / "report.json").read_text())
            calls += len(report["fit_train_only"])
            for path in (root / phase).glob("*-inputs.npz"):
                with np.load(path) as inputs:
                    calls += len(inputs["tick"])
        assert calls <= cfg["maximum_inference_calls"]
        assert witness == source_witness()
        assert all(sha256(Path(p)) == h for p, h in checkpoint_hashes.items())
        assert all(sha256(Path(p)) == h for p, h in audit["dataset_sha256"].items())
        summary.update(
            stage="complete", inference_calls=calls, elapsed_s=time.perf_counter() - started
        )
        write(root / "report.json", summary)
        (root / "run.stage").write_text("complete")
        (root / "run.exit-code").write_text("0")
        note(root, f"## Complete\n{summary}\nPhysics paused; no real-time claim.")
    except BaseException:
        (root / "run.exit-code").write_text("1")
        raise


def diagnose_text_noise(args: argparse.Namespace) -> None:
    """Measure text and flow-noise sensitivity on frozen teacher observations.

    spec: every text/noise pair uses identical state, actual prior and black image.
    why: separate paraphrase dispersion from wrong-goal separation; neither proves motion.
    link: docs/project/vla-text-noise-2026-10-15.md
    """
    root, source = args.root, args.replay.parent
    root.mkdir(parents=True, exist_ok=False)
    (root / "run.pid").write_text(str(os.getpid()))
    started = time.perf_counter()
    language_fit = args.stage == "diagnose-language-fit"
    pair = json.loads((source / "pair-config.json").read_text())
    teacher_root = Path(pair["paths"]["replay"]).parent / "teacher"
    if language_fit:
        # spec: all six v25 training paraphrases share the same teacher target.
        # link: docs/project/vla-small-curriculum-2026-10-16.md
        assert args.diagnosis_root is not None
        assert (args.diagnosis_root / "run.exit-code").read_text().strip() == "0"
        texts = (*LANGUAGE_TEXTS, "保持站立", "向前走")
        roles = (*(["v25-trained"] * 6), "stand-control", "forward-control")
        seeds, ticks = (26001, 26002), (0, 20, 60, 140)
        accepted = [
            r
            for r in json.loads((source / "teacher/report.json").read_text())["candidates"]
            if r["quality"]["accepted"]
        ]
        raw_paths = [
            teacher_root / "train-700-backward.npz",
            teacher_root / "train-702-backward.npz",
            source / "teacher" / (accepted[0]["id"] + ".npz"),
        ]
        checkpoints = {
            "baseline": args.initial,
            "augmented": args.diagnosis_root
            / "model-augmented/checkpoints/001000/pretrained_model",
        }
    else:
        texts = (
            "向后走",
            "往后退一点",
            "朝身后退着走",
            "向后迈步",
            "往身后走",
            "保持站立",
            "向前走",
        )
        roles = (
            "trained",
            "trained",
            "old-development",
            "new-diagnostic",
            "new-diagnostic",
            "stand-control",
            "forward-control",
        )
        seeds, ticks = (24001, 24002, 24003, 24004), (0, 60, 140)
        raw_paths = [
            teacher_root / (name + ".npz")
            for name in ("train-700-backward", "calibration-720-backward")
        ]
        checkpoints = {
            arm: source / f"model-{arm}/checkpoints/001000/pretrained_model"
            for arm in ("control", "mixed")
        }
    sample_count = len(raw_paths) * len(ticks)
    text_count, noise_count = len(texts), len(seeds)
    backward_count = text_count - 2
    call_budget = len(checkpoints) * (sample_count * text_count * noise_count + 1)
    hashes = {str(p): sha256(p) for p in raw_paths}
    hashes.update(
        {str(p): sha256(p) for c in checkpoints.values() for p in c.rglob("*") if p.is_file()}
    )
    witness = source_witness()
    samples, states, priors, targets = [], [], [], []
    for path in raw_paths:
        with np.load(path) as raw:
            np.testing.assert_array_equal(raw["previous_target"][1:], raw["action"][:-1])
            for tick in ticks:
                samples.append(
                    {
                        "episode": path.stem,
                        "tick": tick,
                        "group": "original" if path.parent == teacher_root else "correction",
                        "period": "startup" if tick < 60 else "sustained",
                    }
                )
                states.append(raw["state"][tick])
                priors.append(raw["previous_target"][tick])
                targets.append(raw["action"][tick : tick + H])
    states, priors, targets = np.asarray(states), np.asarray(priors), np.asarray(targets)
    assert (
        states.shape == (sample_count, 36)
        and priors.shape == (sample_count, 15)
        and targets.shape == (sample_count, H, 15)
    )
    np.savez_compressed(
        root / "observations.npz", state=states, previous_target=priors, target=targets, image=BLACK
    )
    cfg = dict(
        texts=texts,
        roles=roles,
        noise_seeds=seeds,
        samples=samples,
        horizon=H,
        maximum_inference_calls=call_budget,
        optimizer_updates=0,
        physics_steps=0,
        source_files=witness,
        input_files=hashes,
        argv=sys.argv,
        processors={a: processor_audit(c, args.replay) for a, c in checkpoints.items()},
    )
    write(root / "config.json", cfg)
    note(
        root,
        f"# Frozen observation language fitting: {args.stage}\n"
        f"SHOULD: {call_budget} predictions including repeats; 0 updates and 0 physics steps.\n"
        "SHOULD: same state/prior/black image and pre-call RNG across texts and arms.\n"
        "SHOULD: finite H10x15 physical targets, unchanged source/checkpoints/raw.\n"
        f"Teacher backward error is meaningful only for first {backward_count} texts; "
        "controls measure separation.",
    )
    report = dict(stage="started", inference_calls=0, optimizer_updates=0, arms={})
    legs = [i for i, name in enumerate(JOINT_NAMES) if name.startswith(("left_", "right_"))]
    assert len(legs) == 10
    try:
        for arm, checkpoint in checkpoints.items():
            (root / "run.stage").write_text(arm)
            model = SmolPolicy(checkpoint)
            torch.cuda.reset_peak_memory_stats()
            arm_started = time.perf_counter()
            chunks, native, rng, rows = [], {}, {}, []
            for sample, (state, prior) in enumerate(zip(states, priors, strict=True)):
                for text_index, text in enumerate(texts):
                    # Same native preprocessing as predict; no sampling here.
                    batch = model.preprocess(
                        {
                            "observation.state": torch.from_numpy(
                                policy_state(state, prior, include_previous_target=True).astype(
                                    np.float32
                                )
                            ),
                            IMAGE_KEY: torch.from_numpy(BLACK.copy()).permute(2, 0, 1).float()
                            / 255,
                            "task": text,
                        }
                    )
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            native[f"{sample}:{text_index}:{key}"] = value.detach().cpu().numpy()
                            if (
                                text_index
                                and key.startswith("observation.")
                                and "language" not in key
                            ):
                                np.testing.assert_array_equal(
                                    native[f"{sample}:0:{key}"],
                                    native[f"{sample}:{text_index}:{key}"],
                                )
                    for noise in seeds:
                        torch.manual_seed(noise)
                        identity = f"{sample}:{text_index}:{noise}"
                        rng[identity + ":cpu"] = torch.get_rng_state().numpy()
                        rng[identity + ":cuda"] = torch.cuda.get_rng_state().cpu().numpy()
                        call_started = time.perf_counter()
                        chunk = model.predict(text, state, BLACK, prior)
                        latency = time.perf_counter() - call_started
                        rng[identity + ":cpu-after"] = torch.get_rng_state().numpy()
                        rng[identity + ":cuda-after"] = torch.cuda.get_rng_state().cpu().numpy()
                        report["inference_calls"] += 1
                        assert chunk.shape == (H, 15) and np.isfinite(chunk).all()
                        if not chunks:
                            torch.manual_seed(noise)
                            repeat = model.predict(text, state, BLACK, prior)
                            report["inference_calls"] += 1
                            np.savez_compressed(
                                root / f"{arm}-smoke.npz", prediction=chunk, repeat=repeat
                            )
                            np.testing.assert_array_equal(chunk, repeat)
                            note(root, f"{arm}: same-input smoke exact; shape H10x15 finite.")
                        chunks.append(chunk)
                        rows.append(
                            dict(
                                sample=sample,
                                text_index=text_index,
                                noise_seed=noise,
                                latency_s=latency,
                            )
                        )
                        # Persist each completed call, including partially completed states.
                        np.savez_compressed(root / f"{arm}-predictions.npz", prediction=chunks)
                        np.savez_compressed(root / f"{arm}-native.npz", **native)
                        np.savez_compressed(root / f"{arm}-rng.npz", **rng)
                        write(root / f"{arm}-rows.json", rows)
                note(
                    root,
                    f"{arm}: completed {sample + 1}/{sample_count} states; "
                    f"calls={report['inference_calls']}",
                )
            cube = np.asarray(chunks, dtype=np.float64).reshape(
                sample_count, text_count, noise_count, H, 15
            )[..., legs]
            target = targets[..., legs].astype(np.float64)
            prior = priors[..., legs].astype(np.float64)
            metrics = []
            for sample in range(sample_count):
                for text_index in range(text_count):
                    pred = cube[sample, text_index]
                    for horizon in (E, H):
                        p = pred[:, :horizon]
                        # Average unordered noise pairs, preserving state and text identity.
                        variance = [
                            np.mean((p[a] - p[b]) ** 2)
                            for a in range(noise_count)
                            for b in range(a + 1, noise_count)
                        ]
                        metrics.append(
                            dict(
                                sample=sample,
                                text_index=text_index,
                                horizon=horizon,
                                within_noise_rms_rad=float(np.sqrt(np.mean(variance))),
                                paired_text_rms_rad=float(
                                    np.sqrt(np.mean((p - cube[sample, 0, :, :horizon]) ** 2))
                                ),
                                teacher_mse_rad2=float(
                                    np.mean((p - target[sample, None, :horizon]) ** 2)
                                )
                                if text_index < backward_count
                                else None,
                                persistence_mse_rad2=float(
                                    np.mean((prior[sample] - target[sample, :horizon]) ** 2)
                                ),
                                target_offset_rms_rad=float(
                                    np.sqrt(np.mean((p - prior[sample]) ** 2))
                                ),
                            )
                        )
            report["arms"][arm] = dict(
                metrics=metrics,
                elapsed_s=time.perf_counter() - arm_started,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                median_latency_s=float(np.median([r["latency_s"] for r in rows])),
            )
            write(root / "report.json", report)
            note(
                root,
                f"{arm}: offline grid complete, elapsed={report['arms'][arm]['elapsed_s']:.2f}s",
            )
            del model
            torch.cuda.empty_cache()
        assert report["inference_calls"] == call_budget
        assert source_witness() == witness
        assert all(sha256(Path(p)) == digest for p, digest in hashes.items())
        report.update(stage="complete", elapsed_s=time.perf_counter() - started)
        (root / "run.stage").write_text("complete")
        (root / "run.exit-code").write_text("0")
        note(
            root,
            f"## Complete\n{call_budget} inference calls; 0 updates; 0 physics steps.\n"
            "Full predictions, native tensors and RNG states saved; "
            "per-state metrics in report.json.",
        )
    except BaseException:
        report["stage"] = "failed"
        (root / "run.exit-code").write_text("1")
        raise
    finally:
        write(root / "report.json", report)


LANGUAGE_TEXTS = ("向后走", "往后退一点", "朝身后退着走", "向后迈步", "往身后走", "往后退步")
LANGUAGE_EVAL = [
    ("original", "向后走"),
    ("augmented", "朝身后退着走"),
    ("fresh-a", "向身后迈几步"),
    ("fresh-b", "往后方走一段"),
]


def prepare_language_pair(args: argparse.Namespace) -> None:
    """Pair native dataset replicas; change only backward task strings.

    spec: both arms share source-frame exposure and physical normalization.
    link: docs/project/vla-language-coverage-2026-10-15.md
    """
    from collections import Counter

    from lerobot.datasets.dataset_tools import merge_datasets, modify_tasks
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.sampler import EpisodeAwareSampler

    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(root).free > 15 * 2**30
    source = dataset(args.replay)
    original = json.loads((args.replay.parent / "data-audit.json").read_text())["episodes"]
    assert source.num_episodes == len(original) == 134 and source.num_frames == 31400
    source_hashes = {str(p): sha256(p) for p in args.replay.rglob("*") if p.is_file()}
    backward = [i for i, e in enumerate(original) if e["family"] == "backward"]
    assert len(backward) == 15
    replay = []
    for family in HEAD_TASKS + MOTION_TASKS:
        if family != "backward":
            chosen = [i for i, e in enumerate(original) if e["family"] == family][:2]
            assert len(chosen) == 2 and all(original[i]["frames"] == 200 for i in chosen)
            replay.extend(chosen)
    selected = sorted(backward + replay)
    episodes = [r * 134 + i for r in range(6) for i in selected]
    assert len(episodes) == 210
    config = {
        "source_files": source_witness(),
        "source_dataset_sha256": source_hashes,
        "initial_sha256": sha256(args.initial / "model.safetensors"),
        "external_sha256": {
            str(p): sha256(p)
            for p in (
                args.teacher,
                args.upstream / "scripts/infer_policy.py",
                args.upstream / "src/mjlab_microduck/robot/microduck/scene.xml",
                *[p for p in args.initial.iterdir() if p.is_file()],
            )
        },
        "argv": sys.argv,
        "train_texts": LANGUAGE_TEXTS,
        "eval_texts": LANGUAGE_EVAL,
        "eval_bodies": [730, 760, 761],
        "selected_source_episodes": selected,
        "selected_episodes": episodes,
        "max_updates": 2004,
        "formal_steps_per_arm": 1000,
        "discarded_smoke_per_arm": 2,
        "batch_size": 8,
        "optimizer_seed": 42,
        "state_dim": 51,
        "horizon": H,
        "execute_steps": E,
        "physics_paused": True,
    }
    write(root / "language-config.json", config)
    note(
        root,
        "# v25 language coverage pair\n"
        "SHOULD: identical numeric data, native sampling and effective normalization; "
        "only backward text changes.\n"
        "SHOULD: each arm discards 2 smoke updates and restarts from the same initial "
        "weight for 1000 updates.\n"
        "SHOULD: 22 scheduled main physics trials per model, including old-task "
        "regression; no visual or real-time claim.",
    )
    copies = []
    for replica in range(6):
        path = root / "replicas" / f"source-{replica}"
        shutil.copytree(args.replay, path)
        copies.append(dataset(path))
    control = root / "dataset-control"
    merge_datasets(
        copies,
        "duckmind/language-control",
        output_dir=control,
        concatenate_data=False,
        concatenate_videos=False,
    )
    augmented = root / "dataset-augmented"
    shutil.copytree(control, augmented)
    modify_tasks(
        dataset(augmented),
        episode_tasks={
            r * 134 + i: text for r, text in enumerate(LANGUAGE_TEXTS) for i in backward
        },
    )
    # modify_tasks rewrites parquet in place, so discard its stale hf_dataset object.
    # Repair bookkeeping stats too; physical episode values are unchanged by replication.
    source_arrays = {
        k: np.asarray(source.hf_dataset[k])
        for k in ("observation.state", "action", "timestamp", "frame_index", "task_index")
    }
    offset = np.cumsum([0] + [e["frames"] for e in original])
    arm_arrays, audit_rows, hashes = {}, {}, {}
    for arm in ("control", "augmented"):
        path = root / ("dataset-" + arm)
        ds = dataset(path)
        assert ds.num_episodes == 804 and ds.num_frames == 188400
        arrays = {
            k: np.asarray(ds.hf_dataset[k]) for k in (*source_arrays, "episode_index", "index")
        }
        arm_arrays[arm] = arrays
        bookkeeping, declared = [], []
        for replica in range(6):
            for i, episode in enumerate(original):
                ep = replica * 134 + i
                start, stop = replica * 31400 + offset[i], replica * 31400 + offset[i + 1]
                for key in ("observation.state", "action", "timestamp", "frame_index"):
                    np.testing.assert_array_equal(
                        arrays[key][start:stop], source_arrays[key][offset[i] : offset[i + 1]]
                    )
                assert np.all(arrays["episode_index"][start:stop] == ep)
                np.testing.assert_array_equal(arrays["index"][start:stop], np.arange(start, stop))
                ids = np.unique(arrays["task_index"][start:stop])
                assert len(ids) == 1
                text = str(ds.meta.tasks.iloc[int(ids[0])].name)
                expected = (
                    LANGUAGE_TEXTS[replica]
                    if arm == "augmented" and i in backward
                    else episode["text"]
                )
                assert text == expected
                declared.append({**episode, "text": text, "source_episode": i, "replica": replica})
                bookkeeping.append(
                    {
                        k: arrays[k][start:stop].reshape(-1, 1)
                        for k in ("index", "episode_index", "task_index")
                    }
                )
        precise_numeric_stats(path, ds, bookkeeping)
        # Six identical copies have exactly the original physical distribution. Preserve
        # canonical float64 stats rather than introducing aggregation roundoff near zero.
        for key in ("observation.state", "action", "timestamp", "frame_index", IMAGE_KEY):
            ds.meta.stats[key] = {
                k: np.array(v, copy=True) for k, v in source.meta.stats[key].items()
            }
            ds.meta.stats[key]["count"] *= 6
        write_stats(ds.meta.stats, path)
        processor_audit(args.initial, path)
        audit_rows[arm] = declared
        hashes[arm] = {str(p): sha256(p) for p in path.rglob("*") if p.is_file()}
        note(
            root,
            f"{arm}: 804 replicas /188400 frames; full numerical and resolved-text audit passed.",
        )
    for key in arm_arrays["control"]:
        if key != "task_index":
            np.testing.assert_array_equal(arm_arrays["control"][key], arm_arrays["augmented"][key])

    def video_hashes(path: Path) -> list[str]:
        return sorted(sha256(p) for p in path.rglob("*.mp4"))

    assert video_hashes(control) == video_hashes(augmented) == sorted(video_hashes(args.replay) * 6)
    # A native sampler reconstruction is evidence of planned exposure, not a captured
    # training hook. Its seed/epoch are independent of the model's diffusion RNG.
    sampling = {}
    for arm in ("control", "augmented"):
        ds = LeRobotDataset(
            "duckmind/" + arm,
            root=root / ("dataset-" + arm),
            episodes=episodes,
            video_backend="pyav",
        )
        sampler = EpisodeAwareSampler(
            ds.meta.episodes["dataset_from_index"],
            ds.meta.episodes["dataset_to_index"],
            episode_indices_to_use=ds.episodes,
            shuffle=True,
            seed=42,
            absolute_to_relative_idx=ds.absolute_to_relative_idx,
        )
        relative = list(sampler)[:8000]
        global_indices = np.asarray(ds.hf_dataset["index"])[relative].astype(int)
        ep_indices = np.asarray(ds.hf_dataset["episode_index"])[relative].astype(int)
        tasks = [audit_rows[arm][i]["text"] for i in ep_indices]
        sampling[arm] = {
            "global_indices": global_indices.tolist(),
            "text_counts": dict(Counter(tasks)),
            "family_counts": dict(Counter(audit_rows[arm][i]["family"] for i in ep_indices)),
        }
    assert sampling["control"]["global_indices"] == sampling["augmented"]["global_indices"]
    write(root / "sampling.json", sampling)
    # evaluate uses a canonical physical dataset for processors and train-only probes.
    (root / "dataset").symlink_to(control.name)
    write(
        root / "data-audit.json",
        {
            "episodes": audit_rows["control"],
            "frames": 188400,
            "arm_episodes": audit_rows,
            "dataset_sha256": hashes,
            "source_dataset_sha256": source_hashes,
            "independent_source_episodes": 134,
            "selected_source_episodes": 35,
        },
    )
    raw = root / "teacher"
    raw.mkdir()
    report = []
    sim = Simulation(args.upstream, args.teacher)
    try:
        for seed in config["eval_bodies"]:
            task = SELECTED["backward"]
            preparation = sim.prepare(task, seed)
            initial_witness(root, sim, seed, task, preparation)
            sim.command(task)
            trace = expert_rollout(sim, 200)
            trace.save(raw / f"{seed}-backward.npz")
            quality = outcome(task, trace.metrics, 200, trace_gaze(sim, trace.physics))
            report.append({"body": seed, "ticks": len(trace.actions), "quality": quality})
            write(raw / "report.json", report)
            note(
                root,
                f"Teacher body={seed}: ticks={len(trace.actions)}, accepted={quality['accepted']}",
            )
        assert all(row["quality"]["accepted"] for row in report), report
    finally:
        sim.close()
    assert all(sha256(Path(p)) == value for p, value in source_hashes.items())
    (root / "prepare.exit-code").write_text("0")
    note(
        root,
        "Preparation complete. Zero model inference and zero optimizer updates; "
        "independent preflight required before training.",
    )


def run_language_pair(args: argparse.Namespace) -> None:
    root, course = args.root, args.course
    cfg = json.loads((root / "language-config.json").read_text())
    audit = json.loads((root / "data-audit.json").read_text())
    assert cfg["source_files"] == source_witness()
    assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
    assert all(sha256(Path(p)) == value for p, value in cfg["external_sha256"].items())
    assert (root / "prepare.exit-code").read_text().strip() == "0"
    assert (
        json.loads((root.parent / "independent-preflight-audit.json").read_text())["stage"]
        == "passed"
    )
    for hashes in audit["dataset_sha256"].values():
        assert all(sha256(Path(p)) == value for p, value in hashes.items())
    assert not (root / "run.pid").exists()
    (root / "run.pid").write_text(str(os.getpid()))

    def stage(name: str) -> None:
        (root / "run.stage").write_text(name)
        note(root, "## " + name)

    def train(arm: str, steps: int, label: str) -> Path:
        command = [
            sys.executable,
            "-c",
            "from duckmind_training.train import main; main()",
            "--dataset",
            str(root / ("dataset-" + arm)),
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
            *map(str, cfg["selected_episodes"]),
        ]
        note(root, f"Native trainer argv: {command!r}")
        subprocess.run(command, check=True)
        checkpoint = root / label / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        processor_audit(checkpoint, root / ("dataset-" + arm))
        return checkpoint

    def physical(checkpoint: Path, label: str) -> None:
        args.course = replace(course, tasks=("backward",), eval_seeds=tuple(cfg["eval_bodies"]))
        evaluate(args, checkpoint, label + "-backward", texts={"backward": LANGUAGE_EVAL})
        regression = tuple(t for t in HEAD_TASKS + MOTION_TASKS if t != "backward")
        args.course = replace(course, tasks=regression, eval_seeds=(730,))
        evaluate(
            args,
            checkpoint,
            label + "-regression",
            texts={name: [("original", SELECTED[name].texts[0])] for name in regression},
        )

    try:
        for arm in ("control", "augmented"):
            stage("smoke-" + arm)
            checkpoint = train(arm, 2, "smoke-" + arm)
            args.course = replace(course, tasks=("backward",), eval_seeds=(730,))
            evaluate(
                args,
                checkpoint,
                "smoke-eval-" + arm,
                smoke=True,
                texts={"backward": [("augmented", LANGUAGE_TEXTS[2])]},
            )
        stage("baseline")
        physical(args.initial, "baseline")
        for arm in ("control", "augmented"):
            stage("train-" + arm)
            checkpoint = train(arm, 1000, "model-" + arm)
            stage("evaluate-" + arm)
            physical(checkpoint, arm)
        stage("verify")
        assert all(sha256(Path(p)) == value for p, value in cfg["external_sha256"].items())
        assert cfg["source_files"] == source_witness()
        assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
        for hashes in audit["dataset_sha256"].values():
            assert all(sha256(Path(p)) == value for p, value in hashes.items())
        stage("complete")
        (root / "run.exit-code").write_text("0")
    except BaseException:
        (root / "run.exit-code").write_text("1")
        raise


SMALL_TEXTS = {
    "stand": ("保持站立", "站好别动", "在原地站稳", "保持站姿", "站着保持平衡", "停在原地站好"),
    "forward": ("向前走", "往前走一点", "朝前面走", "向前迈步", "往前方走", "往前走几步"),
    "backward": LANGUAGE_TEXTS,
}
SMALL_COURSE = Curriculum(
    "v27-small",
    tuple(SMALL_TEXTS),
    tuple(SMALL_TEXTS),
    (700, 730),
    (),
    (700, 730, 770, 771),
    {"stand": "请在这里站稳不动", "forward": "朝前方迈出几步", "backward": "朝后方迈出几步"},
    4000,
    0,
    0,
)


def prepare_small(args: argparse.Namespace) -> None:
    """Build six matched-reset trajectories and balanced language replicas.

    spec: only teacher actions label training; all texts of one physical trajectory stay together.
    why: this is a fitting diagnostic, not a broad replacement checkpoint.
    link: docs/project/vla-small-curriculum-2026-10-16.md
    """
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(root).free > 12 * 2**30
    (root / "teacher").mkdir()
    cfg = {
        "source_files": source_witness(),
        "initial": str(args.initial),
        "initial_sha256": sha256(args.initial / "model.safetensors"),
        "teacher_sha256": sha256(args.teacher),
        "external_sha256": {
            str(p): sha256(p)
            for p in (
                args.teacher,
                args.upstream / "scripts/infer_policy.py",
                args.upstream / "src/mjlab_microduck/robot/microduck/scene.xml",
                *[p for p in args.initial.iterdir() if p.is_file()],
            )
        },
        "texts": SMALL_TEXTS,
        "curriculum": asdict(SMALL_COURSE),
        "budget_updates": 4002,
        "normalization": "true small training dataset; zero-control changes stats only",
        "argv": sys.argv,
    }
    write(root / "small-config.json", cfg)
    note(
        root,
        "# v27 small language curriculum\n"
        "SHOULD: six physical episodes, 36 language replicas, 7200 frames; shared resets.\n"
        "SHOULD: 2 discarded smoke +4000 formal updates; native checkpoints every1000.\n"
        "SHOULD: new true float64 stats, zero unchanged weights; black51/H10/E5.\n"
        "Four seconds is the trained duration. No visual, realtime or broad capability claim.",
    )
    report = {"stage": "started", "episodes": []}
    numeric = []
    sim = Simulation(args.upstream, args.teacher)
    try:
        # No teacher screening of evaluation bodies: keep their student results honest.
        for seed in SMALL_COURSE.train_seeds:
            for name, texts in SMALL_TEXTS.items():
                task = SELECTED[name]
                preparation = sim.prepare(task, seed)
                initial_witness(root, sim, seed, task, preparation)
                sim.command(task)
                trace = expert_rollout(sim, 200)
                identity = f"train-{seed}-{name}"
                trace.save(root / "teacher" / f"{identity}.npz")
                gaze = trace_gaze(sim, trace.physics)
                quality = outcome(task, trace.metrics, 200, gaze)
                row = {
                    "id": identity,
                    "body": seed,
                    "task": name,
                    "quality": quality,
                    "command": asdict(task),
                    "ticks": len(trace.actions),
                    "wrong_goals": {
                        n: outcome(SELECTED[n], trace.metrics, 200, gaze)["accepted"]
                        for n in SMALL_TEXTS
                        if n != name
                    },
                }
                report["episodes"].append(row)
                write(root / "teacher/report.json", report)
                note(root, f"Teacher {identity}: accepted={quality['accepted']}")
                assert quality["accepted"] and not any(row["wrong_goals"].values()), row
                states = policy_state(
                    np.asarray(trace.states),
                    np.asarray(trace.previous_targets),
                    include_previous_target=True,
                )
                actions = np.asarray(trace.actions)
                np.testing.assert_array_equal(states[1:, 36:], actions[:-1])
                for text in texts:
                    numeric.append(
                        {
                            "observation.state": states,
                            "action": actions,
                            "texts": [text] * 200,
                            "family": name,
                        }
                    )
    finally:
        sim.close()
    assert len(numeric) == 36
    report["stage"] = "complete"
    write(root / "teacher/report.json", report)
    prepare_data(args, numeric)
    assert source_witness() == cfg["source_files"]
    (root / "prepare.exit-code").write_text("0")


def small_fit(args: argparse.Namespace, checkpoint: Path, label: str) -> None:
    """Every trained text, both bodies and six periods; fixed noise across paraphrases."""
    root = args.root
    out = root / (label + "-fit")
    out.mkdir(exist_ok=False)
    processor_audit(checkpoint, root / "dataset")
    ds = dataset(root / "dataset", chunks=True)
    model = SmolPolicy(checkpoint)
    ticks = (0, 5, 10, 20, 60, 140)
    legs = [i for i, n in enumerate(JOINT_NAMES) if n.startswith(("left_", "right_"))]
    rows, predictions, states, targets = [], [], [], []
    started = time.perf_counter()
    try:
        for episode in range(36):
            physical = episode // 6
            family = SMALL_COURSE.tasks[physical % 3]
            for tick in ticks:
                item = ds[episode * 200 + tick]
                assert int(item["episode_index"]) == episode and not item["action_is_pad"].any()
                assert item["task"] == SMALL_TEXTS[family][episode % 6]
                state, target = item["observation.state"].numpy(), item["action"].numpy()
                assert not item[IMAGE_KEY].any()
                for noise in (27001, 27002):
                    torch.manual_seed(noise)
                    call_start = time.perf_counter()
                    pred = model.predict(item["task"], state[:36], BLACK, state[36:])
                    assert pred.shape == (H, 15) and np.isfinite(pred).all()
                    predictions.append(pred)
                    states.append(state)
                    targets.append(target)
                    rows.append(
                        {
                            "episode": episode,
                            "physical_episode": physical,
                            "task": family,
                            "tick": tick,
                            "text": item["task"],
                            "noise": noise,
                            "latency_s": time.perf_counter() - call_start,
                            "leg_e5_mse": float(np.mean((pred[:E, legs] - target[:E, legs]) ** 2)),
                            "leg_h10_mse": float(np.mean((pred[:, legs] - target[:, legs]) ** 2)),
                            "persistence_leg_e5_mse": float(
                                np.mean((state[36:][legs] - target[:E, legs]) ** 2)
                            ),
                        }
                    )
                # Preserve partial diagnostics on interruption, not only a final aggregate.
                write(out / "rows.json", rows)
                np.savez_compressed(
                    out / "predictions.npz", prediction=predictions, state=states, target=targets
                )
        assert len(rows) == 432
        write(
            out / "report.json",
            {
                "stage": "complete",
                "rows": rows,
                "model_sha256": sha256(checkpoint / "model.safetensors"),
                "calls": len(rows),
                "training_states_only": True,
                "elapsed_s": time.perf_counter() - started,
            },
        )
        note(
            root,
            f"{label} fit:432 calls, {time.perf_counter() - started:.1f}s; teacher states only.",
        )
    finally:
        del model
        torch.cuda.empty_cache()


def run_small(args: argparse.Namespace) -> None:
    root = args.root
    cfg = json.loads((root / "small-config.json").read_text())
    audit = json.loads((root / "data-audit.json").read_text())
    assert (root / "prepare.exit-code").read_text().strip() == "0"
    assert cfg["source_files"] == source_witness()
    assert all(sha256(Path(p)) == v for p, v in cfg["external_sha256"].items())
    assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
    assert all(sha256(Path(p)) == digest for p, digest in audit["dataset_sha256"].items())
    assert not (root / "run.pid").exists()
    (root / "run.pid").write_text(str(os.getpid()))

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
            "--save-freq",
            "1000",
        ]
        note(root, f"Native trainer argv: {command!r}")
        subprocess.run(command, check=True)
        checkpoint = root / label / "checkpoints" / f"{steps:06d}" / "pretrained_model"
        processor_audit(checkpoint, root / "dataset")
        return checkpoint

    def physical(checkpoint: Path, phase: str, *, final: bool = False, smoke: bool = False) -> None:
        selected = argparse.Namespace(**vars(args))
        selected.course = replace(
            SMALL_COURSE,
            eval_seeds=(SMALL_COURSE.eval_seeds if final else SMALL_COURSE.train_seeds),
        )
        texts = {n: [("original", t[0])] for n, t in SMALL_TEXTS.items()}
        if final:
            texts = {
                n: [("original", t[0]), ("variant", t[-1]), ("unseen", SMALL_COURSE.unseen[n])]
                for n, t in SMALL_TEXTS.items()
            }
        evaluate(selected, checkpoint, phase, smoke=smoke, texts=texts, forward_steps=200)

    try:
        stage("smoke-train")
        smoke_checkpoint = train(2, "model-smoke")
        stage("smoke-evaluate")
        physical(smoke_checkpoint, "smoke", smoke=True)
        stage("zero-fit")
        small_fit(args, root / "zero-checkpoint", "zero")
        stage("zero-evaluate")
        physical(root / "zero-checkpoint", "zero-eval")
        stage("train")
        final = train(4000, "model")
        for step in (1000, 2000, 4000):
            checkpoint = root / "model/checkpoints" / f"{step:06d}" / "pretrained_model"
            stage(f"fit-{step}")
            small_fit(args, checkpoint, f"step-{step}")
            stage(f"evaluate-{step}")
            physical(checkpoint, f"step-{step}", final=step == 4000)
        stage("verify")
        assert source_witness() == cfg["source_files"]
        assert all(sha256(Path(p)) == v for p, v in cfg["external_sha256"].items())
        assert all(sha256(Path(p)) == digest for p, digest in audit["dataset_sha256"].items())
        assert cfg["initial_sha256"] == sha256(args.initial / "model.safetensors")
        assert final.exists()
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
            "diagnose-correction-fit",
            "diagnose-feedback",
            "diagnose-text-noise",
            "diagnose-language-fit",
            "prepare-backward-corrections",
            "finalize-backward-preparation",
            "train-backward-pair",
            "prepare-language-pair",
            "train-language-pair",
            "prepare-small",
            "train-small",
        ),
    )
    for name in ("root", "replay", "initial", "upstream", "teacher"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--course", choices=tuple(COURSES), default="heads")
    parser.add_argument("--diagnosis-root", type=Path)
    args = parser.parse_args()
    args.course = (
        SMALL_COURSE if args.stage in ("prepare-small", "train-small") else COURSES[args.course]
    )
    {
        "prepare": prepare,
        "train-evaluate": run,
        "calibrate-gaze": calibrate_gaze,
        "check-starts": check_starts,
        "diagnose-backward": diagnose_backward,
        "diagnose-correction-fit": diagnose_correction_fit,
        "diagnose-feedback": diagnose_feedback,
        "diagnose-text-noise": diagnose_text_noise,
        "diagnose-language-fit": diagnose_text_noise,
        "prepare-backward-corrections": prepare_backward_corrections,
        "finalize-backward-preparation": finalize_backward_preparation,
        "train-backward-pair": run_backward_pair,
        "prepare-language-pair": prepare_language_pair,
        "train-language-pair": run_language_pair,
        "prepare-small": prepare_small,
        "train-small": run_small,
    }[args.stage](args)


if __name__ == "__main__":
    main()
