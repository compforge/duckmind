#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/../policy/src"
OLD="$HOME/rustprojects/compforge/duckmind/.worktrees/vla-training/training"
export HF_HOME="$OLD/artifacts/huggingface" HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT=/srv/duckmind-training/v26-owner/run
[ ! -e "$ROOT" ] || { echo "Refusing existing output: $ROOT" >&2; exit 1; }
trap 'code=$?; if [ "$code" -ne 0 ] && [ -d "$ROOT" ]; then printf "%s\n" "$code" > "$ROOT/run.exit-code"; fi' EXIT
"$OLD/.venv/bin/python" experiments/head_curriculum.py diagnose-language-fit --course motion \
 --root "$ROOT" --replay /srv/duckmind-training/v21-owner/run/dataset \
 --initial /srv/duckmind-training/v21-owner/run/zero-checkpoint \
 --diagnosis-root /srv/duckmind-training/v25-owner/run \
 --upstream "$HOME/rustprojects/pollen-robotics/microduck_rl" \
 --teacher "$HOME/myprojects/mono-robot/MicroduckSim/policies/velstand.onnx"
