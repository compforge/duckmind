"""Convert accepted physical rollouts to native LeRobot datasets, split by episode."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from duckmind_policy.schema import JOINT_NAMES
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from duckmind_training.contract import (
    FPS,
    IMAGE_KEY,
    IMAGE_SIZE,
    PREVIOUS_TARGET_STATE_NAMES,
    STATE_NAMES,
    TASKS,
    policy_state,
    validate_rollouts,
)


def convert(
    raw: Path,
    output: Path,
    *,
    include_previous_target: bool = False,
    task_names: list[str] | None = None,
) -> None:
    manifest = json.loads((raw / "manifest.json").read_text())
    validate_rollouts(manifest["episodes"], manifest["repetitions"], manifest["task_families"])
    if manifest["state_names"] != list(STATE_NAMES):
        raise ValueError("raw state layout differs from this training adapter; recollect the data")
    families = manifest["task_families"] if task_names is None else task_names
    if not families or not set(families).issubset(manifest["task_families"]):
        raise ValueError("conversion tasks must be present in the raw manifest")
    state_names = PREVIOUS_TARGET_STATE_NAMES if include_previous_target else STATE_NAMES
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
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
        episodes = [
            e
            for e in manifest["episodes"]
            if e["accepted"] and e["split"] == split and e["task_family"] in families
        ]
        numeric_episodes = []
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
                    previous = frames["previous_target"] if include_previous_target else None
                    if previous is not None and not np.array_equal(
                        previous[1:], frames["action"][:-1]
                    ):
                        raise ValueError(
                            f"{episode['id']}: previous_target is not the preceding applied action"
                        )
                    states = policy_state(
                        frames["state"], previous, include_previous_target=include_previous_target
                    )
                    numeric_episodes.append(
                        {"observation.state": states, "action": frames["action"]}
                    )
                    for state, action, image in zip(
                        states, frames["action"], frames["image"], strict=True
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
        precise_numeric_stats(output / split, dataset, numeric_episodes)
        (output / split / "demonstrations.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "episodes": episodes,
                    "task_families": families,
                    "raw_state_names": manifest["state_names"],
                    "state_names": state_names,
                    "include_previous_target": include_previous_target,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        logging.info("wrote %s episodes to %s", len(episodes), output / split)


def precise_numeric_stats(root: Path, dataset: LeRobotDataset, episodes: list[dict]) -> None:
    """Keep both episode and aggregate stats accurate for near-constant IMU channels.

    The pinned LeRobot calculator accumulates float32 second moments in float32.
    Use its native calculator in float64, without changing stored float32 samples.
    Repair episode metadata too, so a later native split/merge retains the correction.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import write_stats

    stats = [
        {
            key: get_feature_stats(values.astype(np.float64), axis=0, keepdims=False)
            for key, values in episode.items()
        }
        for episode in episodes
    ]
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        for key, values in stats[0].items():
            for name in values:
                column = f"stats/{key}/{name}"
                i = table.schema.get_field_index(column)
                if i < 0:
                    raise ValueError(f"missing native episode statistics: {column}")
                data = [stats[index][key][name].tolist() for index in indices]
                table = table.set_column(
                    i, table.schema.field(i), pa.array(data, type=table.schema.field(i).type)
                )
        pq.write_table(table, path)
    dataset.meta.stats.update(aggregate_stats(stats))
    write_stats(dataset.meta.stats, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-previous-target", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=[task.name for task in TASKS])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(
        args.raw,
        args.output,
        include_previous_target=args.include_previous_target,
        task_names=args.tasks,
    )
