"""Use LeRobot's native trainer/checkpoints; keep robotd independent of training."""

import argparse
import json
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    metadata = json.loads((args.dataset / "meta/info.json").read_text())
    if metadata["fps"] != 50 or metadata["features"]["action"]["shape"] != [15]:
        raise ValueError("expected the Microduck 50 Hz / 15-joint dataset")
    revision = (
        "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6" if args.model == "lerobot/smolvla_base" else None
    )
    policy = SmolVLAConfig.from_pretrained(args.model, revision=revision)
    policy.pretrained_path = Path(args.model)
    policy.pretrained_revision = revision
    # The CLI merges feature dictionaries, so passing {} retains the original three
    # cameras. Clear them explicitly: native make_policy derives our camera/state/action
    # features from the dataset, while the padded weight dimensions remain unchanged.
    policy.input_features = {}
    policy.output_features = {}
    policy.device = "cuda"
    policy.push_to_hub = False
    policy.load_vlm_weights = False  # The full checkpoint already supplies VLM weights.
    policy.train_expert_only = True
    policy.train_state_proj = True
    policy.chunk_size = 50
    policy.n_action_steps = 10
    policy.use_amp = True
    policy.scheduler_warmup_steps = min(100, max(1, args.steps // 10))
    policy.scheduler_decay_steps = args.steps
    # Use a dedicated train dataset: its normalization stats cannot include held-out episodes.
    train(
        TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id="duckmind/microduck-train",
                root=str(args.dataset.resolve()),
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
            save_freq=min(args.steps, 500),
            seed=42,
        )
    )
