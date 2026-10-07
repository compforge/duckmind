import base64
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from conftest import serve
from pydantic import ValidationError

from duckmind_policy.policy import FakePolicy
from duckmind_policy.robot import Session
from duckmind_policy.runner import submission
from duckmind_policy.schema import (
    JOINT_NAMES,
    STEP_NS,
    ActionChunk,
    Observation,
    PredictRequest,
)
from duckmind_policy.service import HttpPolicy


def request(task="向左看", t_ns=1_000_000_000):
    return PredictRequest(task=task, observation=Observation(t_ns=t_ns, positions=[0.0] * 15))


def test_contract_tracks_robotd_joint_order():
    root = Path(__file__).resolve().parents[2]
    source = (root / "duck-ipc-proto/src/lib.rs").read_text()
    array = re.search(r"pub const JOINT_NAMES[^=]*=\s*\[(.*?)\];", source, re.S).group(1)
    assert tuple(re.findall(r'"([a-z_]+)"', array)) == JOINT_NAMES
    assert (
        f"ACTION_STEP_NS: u64 = {STEP_NS:_}" in (root / "duck-ipc-proto/src/actions.rs").read_text()
    )


@pytest.mark.parametrize("bad", [[0.0] * 14, [float("nan")] * 15, [4.0] * 15])
def test_rejects_invalid_model_actions(bad):
    with pytest.raises(ValidationError):
        ActionChunk(observation_t_ns=1, positions=[bad])


def test_text_images_history_share_one_contract():
    data = request().model_dump()
    data["observation"]["images"] = {
        "front": {"t_ns": 900_000_000, "jpeg_base64": base64.b64encode(b"\xff\xd8demo").decode()}
    }
    data["history"] = [request(t_ns=800_000_000).observation.model_dump()]
    parsed = PredictRequest.model_validate(data)
    assert parsed.observation.images["front"].t_ns < parsed.observation.t_ns
    data["history"][0]["t_ns"] = data["observation"]["t_ns"]
    with pytest.raises(ValidationError, match="history"):
        PredictRequest.model_validate(data)


def test_fake_paraphrases_and_absolute_goal_preserve_other_joints():
    policy = FakePolicy()
    chunk = policy.infer(request())
    assert chunk == policy.infer(request("往左看"))
    assert chunk.positions[-1][7] == pytest.approx(0.15)
    assert all(q == 0.0 for frame in chunk.positions for i, q in enumerate(frame) if i != 7)
    req = request(t_ns=2_000_000_000)
    req.observation.positions = chunk.positions[-1]
    assert policy.infer(req).positions[-1] == chunk.positions[-1]
    assert policy.infer(request("hold")).positions[-1] == [0.0] * 15


def test_http_transport_is_an_interchangeable_policy():
    with serve(FakePolicy()) as url:
        policy = HttpPolicy(url)
        try:
            assert policy.infer(request()) == FakePolicy().infer(request())
            with pytest.raises(httpx.HTTPStatusError) as exc:
                policy.infer(request("推球"))
            assert exc.value.response.status_code == 422
        finally:
            policy.close()


def test_service_rejects_misaligned_response_and_invalid_request():
    class WrongTime:
        def infer(self, req):
            return ActionChunk(observation_t_ns=req.observation.t_ns + 1, positions=[[0.0] * 15])

    with serve(WrongTime()) as url, httpx.Client(base_url=url) as client:
        assert client.post("/v1/policy/predict", json=request().model_dump()).status_code == 500
        assert client.post("/v1/policy/predict", json={"task": "hold"}).status_code == 422
        assert client.get("/v1/policy").json()["joint_names"] == list(JOINT_NAMES)


def test_busy_backend_does_not_queue_stale_observations():
    entered, release = threading.Event(), threading.Event()

    class Slow:
        def infer(self, req):
            entered.set()
            assert release.wait(3)
            return FakePolicy().infer(req)

    with serve(Slow()) as url, ThreadPoolExecutor(1) as pool:
        first = pool.submit(httpx.post, f"{url}/v1/policy/predict", json=request().model_dump())
        try:
            assert entered.wait(2)
            assert (
                httpx.post(f"{url}/v1/policy/predict", json=request().model_dump()).status_code
                == 503
            )
        finally:
            release.set()
        assert first.result().status_code == 200


def test_runner_trims_elapsed_prefix_without_rebasing_time():
    req = request()
    chunk = FakePolicy().infer(req)
    session = Session(
        session_id="s", t_ns=1, step_ns=STEP_NS, joint_names=list(JOINT_NAMES), positions=[0.0] * 15
    )
    payload = submission(chunk, req, session, 3, req.observation.t_ns + 5 * STEP_NS)
    assert payload["start_t_ns"] == req.observation.t_ns + 7 * STEP_NS
    assert payload["positions"] == chunk.positions[7:]
    assert payload["sequence"] == 3
    with pytest.raises(ValueError, match="expired"):
        submission(chunk, req, session, 4, req.observation.t_ns + 50 * STEP_NS)
    with pytest.raises(ValueError, match="timestamp"):
        submission(chunk, request(t_ns=2_000_000_000), session, 4, 2_000_000_000)
