# Microduck VLA training

Generate language-labelled Microduck demonstrations in MuJoCo, convert them to LeRobotDataset,
and fine-tune SmolVLA to produce joint Action Chunks. The teacher's command vector is used only
to generate demonstrations; the learned model receives language, head-camera RGB and body state.

The collection catalog covers standing, translation, turning, head direction and stopping from
motion. Select only tasks whose expert passes physical checks. This is a training pipeline, not
evidence of reliable learned walking or general language understanding.
Task outcomes must be checked separately from training loss.

## Environment

Training uses the `train` extra on Linux x86_64, Python 3.12 and a CUDA GPU. `pyproject.toml` pins
the LeRobot and BAM source revisions and the PyTorch CUDA build. `uv.lock` records the resolved
environment. Set your normal
HTTP(S) proxy when needed; credentials and private proxy configuration do not belong here.

```bash
cd training
uv sync --locked --extra train
export MUJOCO_GL=egl
export HF_HOME="$PWD/artifacts/huggingface"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2
```

Provide a checkout/snapshot of `pollen-robotics/microduck_rl` at
`8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec`, and the official v5 `velstand.onnx` expert.
The collector records a teacher checksum and upstream script checksum with every dataset.

## Collect and convert

```bash
uv run --extra train duckmind-collect \
  --upstream /path/to/microduck_rl \
  --teacher /path/to/velstand.onnx \
  --output artifacts/raw-v2 --repetitions 12 --seconds 4 --seed 242 \
  --tasks stand forward backward turn_left turn_right look_left look_right \
          look_up look_down stop_forward stop_backward
uv run --extra train duckmind-dataset --raw artifacts/raw-v2 --output artifacts/dataset-v2
```

Collection uses an isolated in-process simulator, so it does not commandeer a running robotd or
browser demo. At four seconds, each episode has 200 synchronized observation/target frames
at 50 Hz. Twelve
repetitions of the eleven selected tasks produce up to 132 accepted episodes. The raw files also
retain the preceding executed target. Sideways tasks are available in the catalog but excluded
from this recipe: the current expert either barely moved or developed substantial yaw in probes.
Complete initial-condition seeds
are held out together; evaluation additionally uses a reserved paraphrase. Failed/fallen episodes
are recorded in `manifest.json` but excluded from training. Empty task/split coverage fails the run.

Outputs are raw compressed trajectories plus provenance/outcome metadata, then independent
`train/` and `eval/` LeRobot datasets with Parquet state/actions and MP4 video. No data is uploaded.
The [data design](../docs/design/vla-training.md) explains units and model inputs.

## Collect corrections from learner states

Use a fixed learner checkpoint to discover forward-walking states, then let the expert continue
from selected learner prefixes. This does not update model weights or change the online Runner.

```bash
uv run --locked --extra train python -m duckmind_training.corrections \
  --upstream /path/to/microduck_rl --teacher /path/to/velstand.onnx \
  --checkpoint artifacts/smolvla-v4/checkpoints/004980/pretrained_model \
  --output artifacts/corrections-forward --seeds 350 351 352 353 354 355 \
  --prefix-steps 10 25 50 --seconds 4 --execute-steps 5
uv run --locked --extra train duckmind-dataset \
  --raw artifacts/corrections-forward --output artifacts/dataset-corrections-forward
```

The collector first checks expert handover continuity. It then saves an expert-only control and
a student-only rollout for each seed, replays exact student targets to each selected prefix, and
checks the full MuJoCo integration state and actuator target against the original prefix.
At handover it preserves physics/BAM and reconstructs the expert's previous-action input.
Only the following expert window is an eligible training episode; action chunks never span the
student/expert boundary. Prefix lengths are scheduled probes of learner-visited states, not a
learned risk detector. A prefix already fallen or never reached is explicitly rejected.

Every attempt stays in `manifest.json`; attempted expert windows have `.npz` arrays and `.json`
physical traces, including rejected windows. Student prefixes and teacher-only controls live in
`evidence/`. Conversion selects only accepted expert windows. Related prefixes share their
seed's train/eval split; reserved labels are applied only to the corresponding eval episodes.
The student's discovery prompt is always the familiar “向前走”. These eval episodes test
corrective-label prediction, not autonomous learned recovery or fresh language generalization.

Acceptance requires four seconds (or the configured duration) without losing balance, forward
progress relative to the handover position, bounded yaw relative to the original task heading,
limited lateral drift, and an upright final second with continued forward progress. The expert-only
control must pass the same gates. Failed attempts remain visible; a run lacking accepted data in
either split fails conversion. Detailed boundaries are in the [data design](../docs/design/vla-training.md).

## Add the previous applied target

Keep old datasets/checkpoints and create a new dataset version:

```bash
uv run --extra train duckmind-dataset --raw artifacts/raw-v2 \
  --output artifacts/dataset-v7-forward-previous-target \
  --tasks forward --include-previous-target
```

This reuses the original forward train/eval episodes. State grows from 36 to 51 features;
labels remain future absolute joint targets. The new 15 features are the target held before
this observation, including the first frame after preparation. New checkpoints require this
signal at inference; the evaluator reads it from BAM after the actually executed chunk prefix.
This is one previous action, not a multi-frame history. See [semantics](../docs/design/vla-training.md#optional-previous-target-input).

## Fine-tune and evaluate

```bash
# Smoke run first; each output directory belongs to one run.
uv run --extra train duckmind-train --dataset artifacts/dataset-v2/train \
  --output artifacts/smolvla-v2-smoke --steps 20 --batch-size 4
uv run --extra train duckmind-train --dataset artifacts/dataset-v2/train \
  --output artifacts/smolvla-v2 --steps 2000 --batch-size 8

uv run --extra train duckmind-evaluate \
  --checkpoint artifacts/smolvla-v2/checkpoints/last/pretrained_model \
  --raw artifacts/raw-v2 --output artifacts/evaluation-v2.json \
  --upstream /path/to/microduck_rl --teacher /path/to/velstand.onnx \
  --execute-steps 5 10 --seeds 10042 10043
```

Training freezes the VLM and trains the Action Expert and state projection from the pretrained
SmolVLA checkpoint (revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`). Native LeRobot processors
normalize using the training set and are saved with the checkpoint. Native trainer logs/checkpoints
are retained, and Hub/W&B publishing is disabled.
For long runs use your process manager or `nohup`; record PID, log, configuration and output path.

Evaluation reports held-out action MSE, inference latency and (when upstream/teacher are supplied)
physical task outcomes. Physics pauses while inference runs; this is a **delay-free synchronous
evaluation**, not proof that real-time execution meets the robotd deadline. There is no claimed
real-hardware validation. A low loss alone is not sufficient to replace the balancing controller.

To separate motor imitation failure from paraphrase generalization failure, repeat the same
checkpoint, seeds and execution horizons with `--language-split trained`. This uses the first
training phrase for each physical task; the default `held-out` uses a reserved phrase. Raw-data
action MSE always uses held-out episodes and language. Save the two reports to different paths.
Use `--tasks forward backward` to restrict both prediction-error samples and physical trials
to selected dataset tasks; the report records this subset.

`duckmind-train --chunk-size 10` uses SmolVLA's native horizon setting to learn 200 ms of future
targets instead of the default 50 targets (one second). The evaluator reads the horizon from
the checkpoint and refuses execution windows longer than the prediction. Sampling remains at
50 Hz. Shorter predictions reduce future motion to learn but also leave less time for an online
runner to obtain its next chunk; success with paused physics does not resolve that latency tradeoff.
Compare physical outcomes across horizons, not aggregate MSE over different future lengths.

`SmolPolicy` implements `duckmind_policy.Policy.infer`; it requires joint velocities, head-camera images and IMU.
The existing stock robotd Runner collects body states only, so this change does not by itself
provide a camera-enabled online robotd loop. The adapter can be passed to the existing HTTP
`create_app` after a camera-aware caller is supplied. See the [model contract](../docs/design/policy-interface.md).

## Validation

Formatting, lint and contract tests use a small CPU-only environment and also run on macOS.
Collection, dataset conversion, CUDA training and checkpoint evaluation are separate Linux checks.

```bash
make fix
make lint
make test
```

See [design and data semantics](../docs/design/vla-training.md). Dated execution results belong in
`docs/project/`, with their evidence paths and limitations.
