"""Run against actual collector artifacts; CPU checks skip when none are supplied."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from duckmind_training.contract import STATE_NAMES, validate_rollouts
from duckmind_training.correction_quality import recovery_outcome


@pytest.mark.skipif(
    not os.getenv("DUCKMIND_CORRECTIONS_RAW"), reason="requires collected artifacts"
)
def test_correction_artifacts_preserve_handover_labels_and_split():
    raw = Path(os.environ["DUCKMIND_CORRECTIONS_RAW"])
    manifest = json.loads((raw / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["handover_check"]["passed"]
    validate_rollouts(manifest["episodes"], manifest["repetitions"], ["forward"])
    split_by_seed = {}
    for episode in manifest["episodes"]:
        seed, split = episode["seed"], episode["split"]
        assert split_by_seed.setdefault(seed, split) == split
        if not episode["frames"]:
            assert not episode["accepted"]
            continue
        with (
            np.load(raw / f"{episode['id']}.npz") as correction,
            np.load(raw / episode["student_evidence"]) as student,
        ):
            prefix = episode["prefix_steps"]
            assert correction["state"].shape == (episode["frames"], len(STATE_NAMES))
            assert correction["action"].shape == (episode["frames"], 15)
            assert correction["image"].shape == (episode["frames"], 256, 256, 3)
            assert correction["image"].dtype == np.uint8
            for key in ("state", "action", "previous_target"):
                assert np.isfinite(correction[key]).all()
            np.testing.assert_array_equal(correction["action"][:, 9], 0)
            np.testing.assert_array_equal(
                correction["previous_target"][1:], correction["action"][:-1]
            )
            np.testing.assert_array_equal(
                correction["previous_target"][0], student["action"][prefix - 1]
            )
            np.testing.assert_allclose(
                correction["integration_state"][0],
                student["integration_state"][prefix],
                rtol=0,
                atol=1e-9,
            )
            if prefix < len(student["state"]):
                np.testing.assert_array_equal(correction["state"][0], student["state"][prefix])
                # Re-rendered EGL pixels can differ by a few uint8 levels although the
                # integration-state and proprioception witnesses above are exactly equal.
                np.testing.assert_allclose(
                    correction["image"][0].astype(np.int16),
                    student["image"][prefix].astype(np.int16),
                    rtol=0,
                    atol=3,
                )
        metrics = json.loads((raw / f"{episode['id']}.json").read_text())
        outcome = recovery_outcome(metrics, round(manifest["seconds"] * manifest["fps"]))
        assert outcome == episode["metrics"]
        if episode["accepted"]:
            assert outcome["accepted"]
            assert episode["frames"] == round(manifest["seconds"] * manifest["fps"])
