# First Microduck VLA training run — 2026-10-11

This experiment establishes expert data collection, SmolVLA fine-tuning, checkpoint inference
and physical evaluation on one Microduck alpha body. It does not establish reliable learned
locomotion or a camera-enabled online robotd loop.

## Environment and data

The devbox is Linux amd64 with an NVIDIA L4 (24 GiB), Python 3.12.13 and PyTorch 2.7.1+cu126.
CUDA allocation and tensor computation passed. The training source is synchronized from the
Duckmind `worktree-vla-training` checkout; the devbox directory is a source snapshot.
The executed training code is in commit `55323b3`; local and deployed source digests matched.

- Runtime dependencies: `training/uv.lock`, SHA-256
  `13113b1b5b69e3a5881e31e1cc1e7f412ef53751e67560930f9aab0f03366b36`.
- SmolVLA base revision: `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- ONNX expert SHA-256: `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`.
- Upstream inference script SHA-256:
  `03bfd0642a880cd379e057536ee819bd1377d4af7b5206a57907d10afd1d4270`.

All 60 final episodes passed the collector's stability and task-outcome checks: 12 repetitions
of stand, forward, backward, look-left and look-right, each lasting 4 seconds at 50 Hz.
Seeds 42–51 supply 50 training episodes / 10,000 frames; seeds 52–53 supply 10 evaluation
episodes / 2,000 frames. Two expressions per task are used in training, a third in evaluation.
This small, highly regular dataset is a pipeline baseline, not broad skill coverage.

Native LeRobot loading verified state `[21]`, head RGB `[3, 256, 256]` and future action
`[50, 15]`. Targets are absolute radians; the alpha body's absent mouth occupies a constant-zero
slot. The head-camera image was visually inspected. Raw trajectories and both converted splits
are retained locally; no dataset or checkpoint was published to the Hub.

## Training and evaluation evidence

The VLM stays frozen. About 100 million of 450 million parameters are trainable, including the
Action Expert and state projection. Four contract tests, formatting and lint passed on macOS
and Linux. The collection, data decoding and GPU training checks ran on Linux.

| Check | Result |
|---|---|
| 20-step smoke training, batch 4 | Completed; native checkpoint saved at step 20 |
| Logged mean loss | 2.459 for steps 1–10; 1.294 for steps 11–20 |
| Recorded CUDA peak memory | About 2.07 GiB |
| Held-out action chunks | 40, across all 10 evaluation episodes |
| Held-out action MSE | 0.035318 rad² |
| Median chunk inference time | 0.356 seconds on this smoke run |
| Existing Policy interface | Real checkpoint returned a validated 50×15 Action Chunk at 20 ms per step |
| Physical evaluation | 0/5 successful; stability checks failed after 40–60 ticks |
| 1,000-step baseline, batch 8 | Completed; step-1,000 checkpoint and post-training evaluation saved |

The smoke model's loss is not physical success: the duck loses stability after roughly
0.8–1.2 seconds. Physics is paused while inference runs, so these failures occur even without
real-time inference delay. Inference time also exceeds the evaluation's 200 ms replanning
interval; this experiment does not validate a real-time control deadline.

The 1,000-step baseline finished with a logged loss of 0.125 and a recorded CUDA peak of
2.59 GiB. Held-out action MSE was 0.006728 rad² over the same 40 chunks; median inference time
was 0.339 seconds. Its physical outcomes were:

| Instruction | Outcome in the 4-second trial |
|---|---|
| 稳稳地站着 | Passed stability/drift checks for all 200 ticks |
| 看看左边 | Passed; final head yaw 0.212 rad against a 0.3 rad goal |
| 看看右边 | Passed; final head yaw −0.253 rad against a −0.3 rad goal |
| 往前迈步 | Failed stability at tick 50 (1.00 s) |
| 往后退步 | Failed stability at tick 69 (1.38 s); displacement was forward |

This is 3/5 successful trials at one held-out initial seed (10042), using the reserved task
expressions and the documented heuristic thresholds. It is partial learned capability, not a
robustness estimate. The two locomotion failures need investigation of observation coverage,
demonstration coverage and closed-loop distribution shift before claiming replacement of the
expert. More optimization steps alone are not established as the remedy.

## Artifacts and continuation

On the devbox, the experiment root is
`~/rustprojects/compforge/duckmind/.worktrees/vla-training/training`.
Paths below are relative to that directory:

| Artifact | Path |
|---|---|
| Raw data and provenance/outcomes | `artifacts/raw-v1/manifest.json` and episode NPZ files |
| Native LeRobot data | `artifacts/dataset-v1/{train,eval}` |
| Smoke checkpoint | `artifacts/smolvla-smoke-v2/checkpoints/last/pretrained_model` |
| Smoke results | `artifacts/evaluation-smoke.json` |
| Serialized Policy interface result | `artifacts/policy-contract-smoke.json` |
| Collection/conversion/smoke logs | `artifacts/logs/` |
| Baseline job script, PID and completion code | `artifacts/run-v1.sh`, `run-v1.pid`, `run-v1.exit-code` |
| Baseline log and output | `artifacts/train-v1.log`, `artifacts/smolvla-v1/` |
| Baseline evaluation after training | `artifacts/evaluation-v1.json` |

The bounded job launched as PID 2425400 and finished with `run-v1.exit-code = 0`. Both training
and evaluation commands completed; physical task success is the separate 3/5 result above.

The normal collection/training commands and boundaries are in
[training/README.md](../../training/README.md) and the [data design](../design/vla-training.md).
