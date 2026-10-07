"""Small Unix JSON-RPC client and a latest-only robot.state subscription."""

import json
import socket
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, JsonValue

from duckmind_policy.schema import Imu, Observation, Vector15


class RpcFailure(RuntimeError):
    pass


class RpcError(BaseModel):
    code: int
    message: str


class Envelope(BaseModel):
    id: int | None = None
    method: str | None = None
    result: JsonValue = None
    params: JsonValue = None
    error: RpcError | None = None


class ActionStatus(BaseModel):
    session_id: str | None = None
    phase: Literal["idle", "waiting", "running", "ended"]
    t_ns: int
    reason: str | None = None
    remaining: int = 0
    last_write_ok: bool | None = None


class State(BaseModel):
    # robot.state has unrelated telemetry; this consumer reads only its declared subset.
    t_ns: int
    joints: Vector15
    velocities: list[float] | None = None
    imu: Imu | None = None
    actions: ActionStatus | None = None

    def observation(self) -> Observation:
        return Observation(
            t_ns=self.t_ns,
            positions=self.joints,
            velocities=self.velocities or None,
            imu=self.imu,
        )


class Session(BaseModel):
    session_id: str
    t_ns: int
    step_ns: int
    joint_names: list[str]
    positions: Vector15


class RobotClient:
    def __init__(self, path: str | Path, timeout: float = 1.0):
        self.path = str(path)
        self.timeout = timeout

    def connect(self) -> socket.socket:
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            stream.settimeout(self.timeout)
            stream.connect(self.path)
        except OSError:
            stream.close()
            raise
        return stream

    def call(self, method: str, params: dict[str, JsonValue] | None = None) -> JsonValue:
        with self.connect() as stream, stream.makefile("rb") as reader:
            stream.sendall(encode(method, params or {}))
            response = Envelope.model_validate_json(reader.readline())
            if response.error is not None:
                raise RpcFailure(f"{method}: {response.error.code} {response.error.message}")
            if response.id != 1:
                raise RpcFailure(f"{method}: mismatched response ID")
            return response.result


def encode(method: str, params: dict[str, JsonValue]) -> bytes:
    return (
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}) + "\n"
    ).encode()


class StateFeed:
    """Drain feedback during inference; never replay a backlog as fresh observations."""

    def __init__(self, robot: RobotClient):
        self._stream = robot.connect()
        self._stream.sendall(encode("robot.subscribe", {"hz": 50}))
        self._condition = threading.Condition()
        self._state: State | None = None
        self._received_ns = 0
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._read, daemon=True, name="robot-state")
        self._thread.start()

    def _read(self) -> None:
        try:
            with self._stream.makefile("rb") as reader:
                while line := reader.readline():
                    response = Envelope.model_validate_json(line)
                    if response.error is not None:
                        raise RpcFailure(response.error.message)
                    if response.method != "robot.state":
                        continue
                    state = State.model_validate(response.params)
                    with self._condition:
                        if self._state is not None and state.t_ns <= self._state.t_ns:
                            raise RpcFailure("robot clock stopped or regressed")
                        self._state = state
                        self._received_ns = time.monotonic_ns()
                        self._condition.notify_all()
                raise RpcFailure("robot.state stream closed")
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def latest(self, timeout: float = 1.0) -> tuple[State, int]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._state is not None or self._error is not None, timeout=timeout
            )
            if self._error is not None:
                raise RpcFailure("robot.state unavailable") from self._error
            if self._state is None:
                raise RpcFailure("no robot.state received")
            age = time.monotonic_ns() - self._received_ns
            if age > 250_000_000:
                raise RpcFailure("robot.state is stale")
            # Local elapsed time advances the last robot timestamp; Unix IPC adds little delay.
            # This is not a wall-clock conversion and does not assume shared host boot times.
            return self._state, self._state.t_ns + age

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._stream.close()
        self._thread.join(timeout=2)
