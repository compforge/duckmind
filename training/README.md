# Microduck VLA training

Generate language-labelled Microduck demonstrations in MuJoCo, convert them to LeRobotDataset,
and fine-tune SmolVLA to produce joint Action Chunks. The teacher's command vector is used only
to generate demonstrations; the learned model receives language, head-camera RGB and body state.

The first dataset covers standing, walking forward/backward and looking left/right. This is a
training-pipeline baseline, not evidence of reliable walking or general language understanding.
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
  --output artifacts/raw-v1 --repetitions 12 --seconds 4
uv run --extra train duckmind-dataset --raw artifacts/raw-v1 --output artifacts/dataset-v1
```

Collection uses an isolated in-process simulator, so it does not commandeer a running robotd or
browser demo. Each episode has 200 synchronized observation/target frames at 50 Hz. Twelve
repetitions of five tasks produce up to 60 accepted episodes. Complete initial-condition seeds
are held out together; evaluation additionally uses a reserved paraphrase. Failed/fallen episodes
are recorded in `manifest.json` but excluded from training. Empty task/split coverage fails the run.

Outputs are raw compressed trajectories plus provenance/outcome metadata, then independent
`train/` and `eval/` LeRobot datasets with Parquet state/actions and MP4 video. No data is uploaded.
The [data design](../docs/design/vla-training.md) explains units and model inputs.

## Fine-tune and evaluate

```bash
# Smoke run first; each output directory belongs to one run.
uv run --extra train duckmind-train --dataset artifacts/dataset-v1/train \
  --output artifacts/smolvla-smoke --steps 20 --batch-size 4
uv run --extra train duckmind-train --dataset artifacts/dataset-v1/train \
  --output artifacts/smolvla-v1 --steps 1000 --batch-size 8

uv run --extra train duckmind-evaluate \
  --checkpoint artifacts/smolvla-v1/checkpoints/last/pretrained_model \
  --raw artifacts/raw-v1 --output artifacts/evaluation-v1.json \
  --upstream /path/to/microduck_rl --teacher /path/to/velstand.onnx
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

`SmolPolicy` implements `duckmind_policy.Policy.infer`; it requires head-camera images and IMU.
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
