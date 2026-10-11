import pytest

from duckmind_training.correction_quality import recovery_outcome


def trace():
    return [
        dict(
            forward_m=1.0 + i * 0.004,
            left_m=0.0,
            yaw_rad=0.1,
            gravity_z=-0.98,
            trunk_z_m=0.116,
            stable=True,
        )
        for i in range(201)
    ]


def test_correction_counts_only_progress_after_handover():
    metrics = trace()
    result = recovery_outcome(metrics, 200)
    assert result["accepted"]
    assert result["forward_m"] == pytest.approx(0.8)
    assert not recovery_outcome(metrics[:150], 200)["accepted"]
    for item in metrics:
        item["forward_m"] = 10.0
    # Large learner-prefix displacement cannot make a stationary correction successful.
    assert not recovery_outcome(metrics, 200)["accepted"]


@pytest.mark.parametrize("field,value", [("stable", False), ("yaw_rad", 0.6)])
def test_transient_fall_or_heading_error_cannot_be_hidden_by_recovery(field, value):
    metrics = trace()
    metrics[30][field] = value
    assert not recovery_outcome(metrics, 200)["accepted"]


def test_lateral_walk_and_stopped_tail_are_not_forward_recovery():
    metrics = trace()
    metrics[-1]["left_m"] = 1.0
    assert not recovery_outcome(metrics, 200)["accepted"]
    metrics = trace()
    for item in metrics[150:]:
        item["forward_m"] = metrics[150]["forward_m"]
    assert not recovery_outcome(metrics, 200)["accepted"]
