"""Use LeRobot's native trainer/checkpoints; keep robotd independent of training."""

import argparse
import json
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

from duckmind_training.contract import PREVIOUS_TARGET_STATE_NAMES, STATE_NAMES
from duckmind_training.initialization import expanded_state_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Train on selected native episode indices; normalization uses the full dataset stats",
    )
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, choices=range(1, 101), default=50)
    parser.add_argument("--save-freq", type=int, default=500)
    args = parser.parse_args()
    if args.save_freq < 1:
        parser.error("--save-freq must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads((args.dataset / "meta/info.json").read_text())
    if metadata["fps"] != 50 or metadata["features"]["action"]["shape"] != [15]:
        raise ValueError("expected the Microduck 50 Hz / 15-joint dataset")
    state_names = tuple(metadata["features"]["observation.state"]["names"])
    if state_names not in (STATE_NAMES, PREVIOUS_TARGET_STATE_NAMES):
        raise ValueError("dataset state names differ from the model adapter")
    if metadata["features"]["observation.state"]["shape"] != [len(state_names)]:
        raise ValueError("dataset state shape and names disagree")
    revision = (
        "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6" if args.model == "lerobot/smolvla_base" else None
    )
    initial = expanded_state_checkpoint(
        args.model, revision, args.output.with_name(args.output.name + "-initial"), len(state_names)
    )
    policy = SmolVLAConfig.from_pretrained(str(initial))
    policy.pretrained_path = initial.resolve()
    policy.pretrained_revision = None
    # The CLI merges feature dictionaries, so passing {} retains the original three
    # cameras. Clear them explicitly: native make_policy derives our camera/state/action
    # features from the dataset, using the explicitly adapted state projection.
    policy.input_features = {}
    policy.output_features = {}
    policy.device = "cuda"
    policy.push_to_hub = False
    policy.load_vlm_weights = False  # The full checkpoint already supplies VLM weights.
    policy.train_expert_only = True
    policy.train_state_proj = True
    # Native LeRobot derives future action labels from this horizon. Shorter chunks
    # test near-term imitation without changing the body's 50 Hz target contract.
    policy.chunk_size = args.chunk_size
    policy.n_action_steps = min(10, args.chunk_size)
    policy.use_amp = True
    policy.scheduler_warmup_steps = min(100, max(1, args.steps // 10))
    policy.scheduler_decay_steps = args.steps
    # Use a dedicated train dataset: its normalization stats cannot include held-out episodes.
    train(
        TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id="duckmind/microduck-train",
                root=str(args.dataset.resolve()),
                episodes=args.episodes,
                video_backend="pyav",
            ),
            policy=policy,
            output_dir=args.output.resolve(),
            job_name="microduck-smolvla",
            steps=args.steps,
            batch_size=args.batch_size,
            num_workers=2,
            env_eval_freq=0,
            log_freq=10,
            save_freq=min(args.steps, args.save_freq),
            seed=42,
        )
    )
