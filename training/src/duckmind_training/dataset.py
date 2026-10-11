"""Convert accepted physical rollouts to native LeRobot datasets, split by episode."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from duckmind_policy.schema import JOINT_NAMES
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from duckmind_training.contract import FPS, IMAGE_KEY, IMAGE_SIZE, STATE_NAMES, validate_rollouts


def convert(raw: Path, output: Path) -> None:
    manifest = json.loads((raw / "manifest.json").read_text())
    validate_rollouts(manifest["episodes"], manifest["repetitions"], manifest["task_families"])
    if manifest["state_names"] != list(STATE_NAMES):
        raise ValueError("raw state layout differs from this training adapter; recollect the data")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)},
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
    }
    if output.exists():
        raise FileExistsError(output)
    for split in ("train", "eval"):
        episodes = [e for e in manifest["episodes"] if e["accepted"] and e["split"] == split]
        dataset = LeRobotDataset.create(
            repo_id=f"duckmind/microduck-{split}",
            fps=FPS,
            features=features,
            root=output / split,
            robot_type="microduck",
            use_videos=True,
            image_writer_threads=4,
            video_backend="pyav",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast", crf=23),
        )
        try:
            for episode in episodes:
                with np.load(raw / f"{episode['id']}.npz") as frames:
                    for state, action, image in zip(
                        frames["state"], frames["action"], frames["image"], strict=True
                    ):
                        dataset.add_frame(
                            {
                                "observation.state": state,
                                "action": action,
                                IMAGE_KEY: image,
                                "task": episode["task"],
                            }
                        )
                dataset.save_episode()
        finally:
            dataset.finalize()
        (output / split / "demonstrations.json").write_text(
            json.dumps({**manifest, "episodes": episodes}, indent=2, ensure_ascii=False)
        )
        logging.info("wrote %s episodes to %s", len(episodes), output / split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(args.raw, args.output)
