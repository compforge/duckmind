import numpy as np
import pytest

from duckmind_training.contract import (
    TASKS,
    episode_success,
    split_for,
    state_vector,
    validate_rollouts,
    wire_positions,
)


def test_mouth_mapping_preserves_right_leg_and_state_has_no_command():
    joints = np.arange(14, dtype=np.float32)
    mapped = wire_positions(joints)
    np.testing.assert_array_equal(mapped[:9], joints[:9])
    assert mapped[9] == 0
    np.testing.assert_array_equal(mapped[10:], joints[9:])
    state = state_vector(joints, np.array([1, 2, 3]), np.array([0, 0, -1]))
    assert state.shape == (21,)
    np.testing.assert_array_equal(state[-6:], [1, 2, 3, 0, 0, -1])


def test_split_reserves_complete_rollouts():
    assert [split_for(i, 12) for i in range(12)] == ["train"] * 10 + ["eval"] * 2
    with pytest.raises(ValueError):
        split_for(0, 1)


def test_reject_fallen_and_wrong_direction_demonstrations():
    forward = TASKS[1]
    assert episode_success(forward, 0.1, 0, True) is True
    assert not episode_success(forward, -0.1, 0, True)
    assert not episode_success(forward, 0.1, 0, False)
    assert not episode_success(TASKS[3], 0, -0.3, True)


def test_conversion_requires_complete_collection_and_successful_coverage():
    episodes = [
        {"task_family": task.name, "split": split, "accepted": True}
        for split in ("train", "eval")
        for task in TASKS
    ]
    validate_rollouts(episodes, 2)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rollouts(episodes[:-1], 2)
    episodes[-1]["accepted"] = False
    with pytest.raises(ValueError, match="missing successful tasks.*eval"):
        validate_rollouts(episodes, 2)
