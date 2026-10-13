import numpy as np
import pytest

from duckmind_training.contract import (
    STATE_NAMES,
    TASKS,
    episode_success,
    split_for,
    state_vector,
    validate_rollouts,
    wire_positions,
)


def test_joint_mapping_and_state_have_no_teacher_command():
    joints = np.arange(14, dtype=np.float32)
    mapped = wire_positions(joints)
    np.testing.assert_array_equal(mapped[:9], joints[:9])
    assert mapped[9] == 0
    np.testing.assert_array_equal(mapped[10:], joints[9:])
    state = state_vector(mapped, np.array([1, 2, 3]), np.array([0, 0, -1]), mapped + 20)
    assert state.shape == (len(STATE_NAMES),) == (36,)
    np.testing.assert_array_equal(state[15:21], [1, 2, 3, 0, 0, -1])
    np.testing.assert_array_equal(state[21:], mapped + 20)


def test_split_reserves_complete_rollouts():
    assert [split_for(i, 12) for i in range(12)] == ["train"] * 10 + ["eval"] * 2
    with pytest.raises(ValueError):
        split_for(0, 1)


def test_outcomes_reject_wrong_direction_falls_and_false_stops():
    tasks = {t.name: t for t in TASKS}
    metrics = dict(forward_m=0.1, left_m=0.2, yaw_rad=0.3, head_yaw_delta=0.3, tail_speed_mps=0.03)
    assert episode_success(tasks["forward"], metrics, True)
    assert episode_success(tasks["strafe_left"], metrics, True)
    assert episode_success(tasks["turn_left"], metrics, True)
    assert not episode_success(tasks["backward"], metrics, True)
    assert not episode_success(tasks["strafe_right"], metrics, True)
    assert not episode_success(tasks["strafe_left"], {**metrics, "yaw_rad": 1.0}, True)
    assert not episode_success(tasks["turn_right"], metrics, True)
    assert not episode_success(tasks["forward"], metrics, False)
    assert not episode_success(tasks["look_right"], metrics, True)
    assert episode_success(tasks["stop_forward"], metrics, True)
    assert not episode_success(tasks["stop_forward"], {**metrics, "tail_speed_mps": 0.2}, True)


def test_conversion_requires_declared_task_coverage_and_complete_ids():
    families = [t.name for t in TASKS]
    episodes = [
        {"id": f"{i:04d}-{task.name}", "task_family": task.name, "split": split, "accepted": True}
        for i, split in enumerate(("train", "eval"))
        for task in TASKS
    ]
    validate_rollouts(episodes, 2, families)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rollouts(episodes[:-1], 2, families)
    with pytest.raises(ValueError, match="incomplete"):
        validate_rollouts([e for e in episodes if e["task_family"] != "stand"], 2, families)
    episodes[-1]["accepted"] = False
    with pytest.raises(ValueError, match="missing successful tasks.*eval"):
        validate_rollouts(episodes, 2, families)


def test_previous_target_batch_and_online_state_are_identical():
    from duckmind_training.contract import PREVIOUS_TARGET_STATE_NAMES, policy_state

    states = np.arange(4 * 36, dtype=np.float32).reshape(4, 36)
    actions = np.arange(4 * 15, dtype=np.float32).reshape(4, 15) / 100
    # First observation follows warmup; it must not use zeros or action[0].
    previous = np.concatenate((np.full((1, 15), 0.7, dtype=np.float32), actions[:-1]))
    batch = policy_state(states, previous, include_previous_target=True)
    assert batch.shape == (4, len(PREVIOUS_TARGET_STATE_NAMES)) == (4, 51)
    for i in range(4):
        online = policy_state(states[i], previous[i], include_previous_target=True)
        np.testing.assert_array_equal(batch[i], online)
    np.testing.assert_array_equal(batch[:, :36], states)
    np.testing.assert_array_equal(batch[:, 36:], previous)
    np.testing.assert_array_equal(policy_state(states, None, include_previous_target=False), states)
    for bad in (None, np.zeros((4, 14)), np.full((4, 15), np.nan)):
        with pytest.raises(ValueError, match="previous_target"):
            policy_state(states, bad, include_previous_target=True)
