"""Collect forward-walking expert corrections from replayed learner prefixes.

Each intervention is an independent episode. Learner actions and rejected corrections remain
raw evidence, never behavioral-cloning labels. This is a synchronous simulation collector,
not an online safety controller or a completed DAgger training algorithm.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from duckmind_training.collect import sha256
from duckmind_training.contract import (
    FPS,
    MICRODUCK_RL_REVISION,
    STATE_NAMES,
    TASKS,
    Task,
    split_for,
    validate_rollouts,
    wire_positions,
)
from duckmind_training.correction_quality import recovery_outcome
from duckmind_training.learned import SmolPolicy
from duckmind_training.simulation import Simulation

LOG = logging.getLogger(__name__)


@dataclass
class Trace:
    states: list = field(default_factory=list)
    images: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    previous_targets: list = field(default_factory=list)
    physics: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    metrics: list = field(default_factory=list)

    def mark(self, simulation: Simulation) -> None:
        self.physics.append(simulation.integration_state())
        self.targets.append(simulation.bam.q_target.copy())
        self.metrics.append(simulation.metrics())

    def step(self, simulation: Simulation, target: np.ndarray | None = None) -> None:
        state, image = simulation.observe()
        self.states.append(state)
        self.images.append(image)
        self.previous_targets.append(wire_positions(simulation.bam.q_target.copy()))
        if target is None:
            target = wire_positions(simulation.teacher_step())
        else:
            simulation.apply(target)
        self.actions.append(target.copy())
        self.mark(simulation)

    def save(self, path: Path) -> None:
        # Even rejected attempts retain their labels and state traces for diagnosis.
        np.savez_compressed(
            path,
            state=np.asarray(self.states),
            image=np.asarray(self.images),
            action=np.asarray(self.actions),
            previous_target=np.asarray(self.previous_targets),
            integration_state=np.asarray(self.physics),
            actuator_target=np.asarray(self.targets),
        )
        path.with_suffix(".json").write_text(json.dumps(self.metrics, indent=2))


def expert_rollout(simulation: Simulation, steps: int) -> Trace:
    trace = Trace()
    trace.mark(simulation)
    for _ in range(steps):
        if not simulation.metrics()["stable"]:
            break
        trace.step(simulation)
    return trace


def learner_rollout(
    simulation: Simulation, policy: SmolPolicy, text: str, steps: int, execute_steps: int
) -> tuple[Trace, list[float], str]:
    trace = Trace()
    trace.mark(simulation)
    latencies = []
    for tick in range(steps):
        if not simulation.metrics()["stable"]:
            return trace, latencies, "unstable"
        if tick % execute_steps == 0:
            state, image = simulation.observe()
            start = time.perf_counter()
            predictions = policy.predict(text, state, image, simulation.previous_target())
            latencies.append(time.perf_counter() - start)
        action = predictions[tick % execute_steps]
        if not np.isfinite(action).all() or np.any(np.abs(action) > np.pi):
            return trace, latencies, "invalid_target"
        trace.step(simulation, action)
    reason = "completed" if simulation.metrics()["stable"] else "unstable"
    return trace, latencies, reason


def check_handover(simulation: Simulation, task: Task, seed: int) -> dict:
    """A teacher passing through absolute targets must retain its subsequent behavior."""
    simulation.prepare(task, seed)
    simulation.command(task)
    reference = expert_rollout(simulation, 100)
    if len(reference.actions) != 100:
        raise ValueError("teacher-only handover control lost balance")
    simulation.prepare(task, seed)
    # Replay exact targets through the learner path; infer() is deliberately not called.
    for target in reference.actions[:20]:
        simulation.apply(target)
    before = simulation.integration_state()
    target_before = simulation.bam.q_target.copy()
    simulation.resume_teacher(task)
    np.testing.assert_array_equal(before, simulation.integration_state())
    np.testing.assert_array_equal(target_before, simulation.bam.q_target)
    np.testing.assert_allclose(before, reference.physics[20], rtol=0, atol=1e-9)
    expected_history = (
        target_before - simulation.expert.default_pose
    ) / simulation.expert.action_scale
    np.testing.assert_allclose(simulation.expert.last_action, expected_history, rtol=0, atol=1e-7)
    resumed = expert_rollout(simulation, 80)
    error = float(
        np.max(
            np.abs(np.asarray(resumed.actions) - reference.actions[20 : 20 + len(resumed.actions)])
        )
    )
    # Absolute->relative float32 conversion loses a few bits. Compare the immediate
    # prediction tightly; later contact dynamics can amplify that rounding over 80 ticks.
    np.testing.assert_allclose(resumed.actions[0], reference.actions[20], rtol=0, atol=1e-6)
    if len(resumed.actions) != 80 or not all(m["stable"] for m in resumed.metrics):
        raise ValueError("resumed teacher lost balance in handover control")
    return {
        "passed": True,
        "max_target_error_rad": error,
        "first_target_error_rad": float(np.max(np.abs(resumed.actions[0] - reference.actions[20]))),
        "physics_unchanged_at_handover": True,
        "teacher_terminal": reference.metrics[-1],
        "resumed_terminal": resumed.metrics[-1],
    }


def collect_corrections(
    upstream: Path,
    teacher: Path,
    checkpoint: Path,
    output: Path,
    seeds: list[int],
    prefix_steps: list[int],
    seconds: float,
    execute_steps: int,
) -> None:
    split_for(0, len(seeds))
    if len(set(seeds)) != len(seeds) or len(set(prefix_steps)) != len(prefix_steps):
        raise ValueError("use distinct seeds and prefix lengths")
    if seconds < 2 or not prefix_steps or min(prefix_steps) < 1:
        raise ValueError("need positive learner prefixes and at least two correction seconds")
    policy = SmolPolicy(checkpoint)
    if not 1 <= execute_steps <= policy.model.config.chunk_size:
        raise ValueError("execute_steps must fit the checkpoint prediction horizon")
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    steps = round(seconds * FPS)
    repetitions = len(seeds) * len(prefix_steps)
    manifest = {
        "kind": "expert_corrections",
        "status": "running",
        "mode": "synchronous_delay_free",
        "fps": FPS,
        "seconds": seconds,
        "repetitions": repetitions,
        "task_families": ["forward"],
        "state_names": STATE_NAMES,
        "action_space": "microduck-joints-v1/absolute_joint_position_rad",
        "reference_upstream_revision": MICRODUCK_RL_REVISION,
        "upstream_script_sha256": sha256(upstream / "scripts/infer_policy.py"),
        "teacher_sha256": sha256(teacher),
        "student_checkpoint": str(checkpoint),
        "student_sha256": sha256(checkpoint / "model.safetensors"),
        "student_config_sha256": sha256(checkpoint / "config.json"),
        "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "seeds": seeds,
        "prefix_steps": prefix_steps,
        "execute_steps": execute_steps,
        "episodes": [],
        "controls": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    simulation = Simulation(upstream, teacher)
    template = next(task for task in TASKS if task.name == "forward")
    try:
        manifest["handover_check"] = check_handover(simulation, template, seeds[0])
        for repetition, seed in enumerate(seeds):
            task = template.sample(seed)
            split = split_for(repetition, len(seeds))
            # Correlated prefixes from one physical reset ALWAYS share the same split.
            text = (
                task.texts[repetition % (len(task.texts) - 1)]
                if split == "train"
                else task.texts[-1]
            )
            simulation.prepare(task, seed)
            simulation.command(task)
            teacher_control = expert_rollout(simulation, steps)
            teacher_control.save(evidence / f"seed-{seed}-teacher.npz")
            teacher_outcome = recovery_outcome(teacher_control.metrics, steps)
            simulation.prepare(task, seed)
            torch.manual_seed(seed)
            # Familiar wording isolates collection-state difficulty from language generalization.
            baseline, latencies, stop_reason = learner_rollout(
                simulation, policy, template.texts[0], steps + max(prefix_steps), execute_steps
            )
            baseline.save(evidence / f"seed-{seed}-student.npz")
            manifest["controls"].append(
                {
                    "seed": seed,
                    "teacher": teacher_outcome,
                    "student_stop_reason": stop_reason,
                    "student_task": template.texts[0],
                    "student_steps": len(baseline.actions),
                    "inference_seconds": latencies,
                }
            )
            for index, prefix in enumerate(prefix_steps):
                episode_id = f"{repetition * len(prefix_steps) + index:04d}-forward"
                record = {
                    "id": episode_id,
                    "seed": seed,
                    "split": split,
                    "task": text,
                    "task_family": "forward",
                    "accepted": False,
                    "frames": 0,
                    "prefix_steps": prefix,
                    "teacher_command": asdict(task),
                    "student_evidence": f"evidence/seed-{seed}-student.npz",
                }
                if prefix > len(baseline.actions) or not baseline.metrics[prefix]["stable"]:
                    record["reason"] = "student_failed_before_handover"
                else:
                    simulation.prepare(task, seed)
                    for target in baseline.actions[:prefix]:
                        simulation.apply(target)
                    replay_error = float(
                        np.max(np.abs(simulation.integration_state() - baseline.physics[prefix]))
                    )
                    np.testing.assert_allclose(
                        simulation.integration_state(), baseline.physics[prefix], rtol=0, atol=1e-9
                    )
                    np.testing.assert_array_equal(simulation.bam.q_target, baseline.targets[prefix])
                    # Keep original task heading and live BAM state; no mark_origin/reset here.
                    simulation.resume_teacher(task)
                    correction = expert_rollout(simulation, steps)
                    correction.save(output / f"{episode_id}.npz")
                    outcome = recovery_outcome(correction.metrics, steps)
                    # No credit for an incapable teacher, even if a short correction looks lucky.
                    record.update(
                        {
                            "accepted": outcome["accepted"] and teacher_outcome["accepted"],
                            "frames": len(correction.actions),
                            "metrics": outcome,
                            "replay_max_error": replay_error,
                            "reason": "checked_correction",
                            "without_intervention": recovery_outcome(
                                baseline.metrics[prefix : prefix + steps + 1], steps
                            ),
                        }
                    )
                manifest["episodes"].append(record)
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False)
                )
                LOG.info(
                    "correction episode=%s seed=%s prefix=%s accepted=%s result=%s",
                    episode_id,
                    seed,
                    prefix,
                    record["accepted"],
                    record.get("metrics", record["reason"]),
                )
        manifest["status"] = "completed"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise
    finally:
        simulation.close()
    # Keep all attempted evidence even when no usable train/eval split was obtained.
    validate_rollouts(manifest["episodes"], repetitions, ["forward"])
    LOG.info(
        "accepted=%s attempted=%s output=%s",
        sum(e["accepted"] for e in manifest["episodes"]),
        repetitions,
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--prefix-steps", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--execute-steps", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    collect_corrections(
        args.upstream,
        args.teacher,
        args.checkpoint,
        args.output,
        args.seeds,
        args.prefix_steps,
        args.seconds,
        args.execute_steps,
    )


if __name__ == "__main__":
    main()
