#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/../policy/src"
OLD="$HOME/rustprojects/compforge/duckmind/.worktrees/vla-training/training"
export HF_HOME="$OLD/artifacts/huggingface" HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT=/srv/duckmind-training/v19-owner/run
exec "$OLD/.venv/bin/python" experiments/head_curriculum.py "$1" --course motion \
 --root "$ROOT" \
 --replay /srv/duckmind-training/v18-owner/run/dataset \
 --initial /srv/duckmind-training/v18-owner/run/model/checkpoints/002000/pretrained_model \
 --upstream "$HOME/rustprojects/pollen-robotics/microduck_rl" \
 --teacher "$HOME/myprojects/mono-robot/MicroduckSim/policies/velstand.onnx"
