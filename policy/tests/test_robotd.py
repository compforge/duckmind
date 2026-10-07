"""Real HTTP → Runner → Unix IPC → robotd/FakeIo → measured state, no replacement executor.

Set ROBOTD_BIN to a robotd built from this checkout. Without it these integration tests skip.
"""

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from conftest import serve

from duckmind_policy.policy import FakePolicy
from duckmind_policy.robot import RobotClient, RpcFailure, StateFeed
from duckmind_policy.runner import run
from duckmind_policy.service import HttpPolicy


@pytest.fixture
def robot(tmp_path):
    binary = os.environ.get("ROBOTD_BIN")
    if not binary:
        pytest.skip("set ROBOTD_BIN for real robotd/FakeIo integration")
    socket = tmp_path / "robot.sock"
    params = tmp_path / "robotd.toml"
    params.write_text("[audio]\nenabled=false\n[chorale]\naccept=false\n")
    with subprocess.Popen(
        [binary, "--fake", "--no-policy", "--socket", str(socket), "--params", str(params)],
        env={**os.environ, "RUST_LOG": "error"},
        stdout=subprocess.DEVNULL,
    ) as process:
        try:
            deadline = time.monotonic() + 8
            while not socket.exists():
                assert process.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            client = RobotClient(socket)
            client.call("robot.init")
            # Readiness includes warm IMU and completion of the home transition.
            while True:
                try:
                    session = client.call("robot.actions.begin")
                    client.call("robot.actions.end", {"session_id": session["session_id"]})
                    break
                except RpcFailure:
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_state(feed, predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        state, _ = feed.latest()
        if predicate(state):
            return state
        time.sleep(0.01)
    pytest.fail("expected measured feedback not received")


def test_http_policy_drives_measured_joints_and_releases_session(robot):
    feed = StateFeed(robot)
    with serve(FakePolicy()) as url, ThreadPoolExecutor(1) as pool:
        policy = HttpPolicy(url)
        try:
            initial, _ = feed.latest()
            result = pool.submit(run, robot, policy, "向左看", 1.5)
            reached = wait_state(feed, lambda s: abs(s.joints[7] - 0.15) < 1e-6)
            assert reached.actions.last_write_ok is True
            assert reached.joints[:7] == pytest.approx(initial.joints[:7])
            assert reached.joints[8:] == pytest.approx(initial.joints[8:])
            assert result.result(timeout=4) >= 2  # replacement/replanning, not one static write
            wait_state(feed, lambda s: s.actions.phase == "ended")
        finally:
            policy.close()
            feed.close()


def test_stop_during_inference_discards_late_prediction(robot):
    entered, release = threading.Event(), threading.Event()

    class Delayed:
        def infer(self, req):
            entered.set()
            assert release.wait(3)
            return FakePolicy().infer(req)

    feed = StateFeed(robot)
    with serve(Delayed()) as url, ThreadPoolExecutor(1) as pool:
        policy = HttpPolicy(url, timeout=2)
        try:
            result = pool.submit(run, robot, policy, "向左看", 3)
            assert entered.wait(2)
            robot.call("robot.stop")
            stopped = wait_state(feed, lambda s: s.actions.phase == "ended")
            release.set()
            # Feedback and HTTP arrive independently: either the runner or robotd may reject it.
            with pytest.raises(RpcFailure):
                result.result(timeout=3)
            time.sleep(0.1)
            after, _ = feed.latest()
            assert after.joints == stopped.joints
            assert after.actions.phase == "ended"
        finally:
            release.set()
            policy.close()
            feed.close()


def test_http_failure_ends_session(robot):
    feed = StateFeed(robot)
    with serve(FakePolicy()) as url:
        policy = HttpPolicy(url)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                run(robot, policy, "unsupported task")
            wait_state(feed, lambda s: s.actions.phase == "ended")
        finally:
            policy.close()
            feed.close()


def test_inference_timeout_cancels_previously_admitted_tail(robot):
    entered, release = threading.Event(), threading.Event()

    class TimeoutOnReplan:
        calls = 0

        def infer(self, req):
            self.calls += 1
            if self.calls > 1:
                entered.set()
                assert release.wait(3)
            return FakePolicy().infer(req)

    feed = StateFeed(robot)
    with serve(TimeoutOnReplan()) as url, ThreadPoolExecutor(1) as pool:
        policy = HttpPolicy(url, timeout=0.15)
        try:
            result = pool.submit(run, robot, policy, "向左看", 3)
            assert entered.wait(2)
            moving, _ = feed.latest()
            assert moving.actions.phase == "running"
            with pytest.raises(httpx.TimeoutException):
                result.result(timeout=2)
            stopped = wait_state(feed, lambda s: s.actions.phase == "ended")
            release.set()
            time.sleep(0.1)
            after, _ = feed.latest()
            assert after.joints == stopped.joints
        finally:
            release.set()
            policy.close()
            feed.close()
