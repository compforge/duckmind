"""Native conversion regression; runs in the Linux training environment."""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")

import pyarrow.parquet as pq  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from duckmind_training.contract import IMAGE_KEY, PREVIOUS_TARGET_STATE_NAMES, STATE_NAMES  # noqa: E402
from duckmind_training.dataset import convert  # noqa: E402


def test_native_conversion_keeps_previous_targets_and_precise_episode_stats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    episodes = []
    expected = []
    for i, split in enumerate(("train", "eval")):
        name = f"{i:04d}-forward"
        state = np.zeros((8, 36), dtype=np.float32)
        # Near-constant gravity reproduces cancellation in float32 second moments.
        state[:, 20] = -1 + np.arange(8) * 0.00001
        action = np.arange(8 * 15, dtype=np.float32).reshape(8, 15) / 100
        previous = np.concatenate((np.full((1, 15), 0.3, dtype=np.float32), action[:-1]))
        image = np.zeros((8, 256, 256, 3), dtype=np.uint8)
        np.savez(
            raw / f"{name}.npz", state=state, action=action, previous_target=previous, image=image
        )
        episodes.append(
            dict(id=name, split=split, accepted=True, task_family="forward", task="向前走")
        )
        expected.append(np.concatenate((state, previous), axis=1))
    (raw / "manifest.json").write_text(
        json.dumps(
            dict(
                episodes=episodes, repetitions=2, task_families=["forward"], state_names=STATE_NAMES
            )
        )
    )
    out = tmp_path / "converted"
    convert(raw, out, include_previous_target=True, task_names=["forward"])
    for i, split in enumerate(("train", "eval")):
        data = LeRobotDataset(f"duckmind/{split}", root=out / split, video_backend="pyav")
        assert data.features["observation.state"]["names"] == list(PREVIOUS_TARGET_STATE_NAMES)
        actual = np.asarray(data.hf_dataset["observation.state"])
        np.testing.assert_array_equal(actual, expected[i])
        direct = actual.astype(np.float64).std(axis=0)
        np.testing.assert_allclose(data.meta.stats["observation.state"]["std"], direct, atol=1e-9)
        episode_meta = pq.read_table(next((out / split / "meta/episodes").glob("*/*.parquet")))
        np.testing.assert_allclose(
            episode_meta["stats/observation.state/std"][0].as_py(), direct, atol=1e-9
        )
        sample = data[0]
        assert tuple(sample[IMAGE_KEY].shape) == (3, 256, 256)
    # A future-label leak cannot silently become a history feature.
    with np.load(raw / "0000-forward.npz") as frames:
        values = dict(frames)
    values["previous_target"] = values["action"]
    np.savez(raw / "0000-forward.npz", **values)
    with pytest.raises(ValueError, match="preceding applied action"):
        convert(raw, tmp_path / "bad", include_previous_target=True)
