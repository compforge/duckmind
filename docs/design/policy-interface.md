# Policy inference interface

The model boundary is `Policy.infer(PredictRequest) → ActionChunk`, implemented in
[`policy/`](../../policy/README.md). A rule, local neural network, or remote model adapter can
implement it. Inference does not acquire a robot or write motors; the runner owns a robotd
Action Session and submits predictions to the existing executor.

```text
text + latest body observation + history
                  ↓
        Policy.infer / HTTP service       FakePolicy today; learned adapter later
                  ↓
        decoded joint ActionChunk
                  ↓
        Runner on robotd's machine        discard expired prefix; submit remaining timeline
                  ↓ Unix JSON-RPC
        robotd → Safety → RobotIo → body
                  ↑
              robot.state
```

This separates the **model contract** from the **execution contract**. The latter, including
control ownership, admission, replacement, stops and hold behavior, belongs to
[Action Chunks](action-chunks.md). No Rust IPC or control-loop behavior changes in this layer.

## Model contract: microduck-joints-v1

The Pydantic types in [`schema.py`](../../policy/src/duckmind_policy/schema.py) are the source
of truth; FastAPI publishes their JSON Schema through `/openapi.json`. This is a Duckmind
body-specific contract, not an industry standard and not a promise of cross-embodiment control.

`POST /v1/policy/predict` accepts:

| Field | Meaning |
|---|---|
| `schema_version` | `microduck-joints-v1`; defaults to this value, other versions reject. |
| `task` | Free-form nonempty text. The API has no command enum. |
| `observation.t_ns` | Positive robotd monotonic timestamp in nanoseconds, from `robot.state`. |
| `observation.positions` | 15 measured absolute joint angles, radians, in robotd `JOINT_NAMES` order. |
| `observation.velocities` | Optional 15 measured velocities, rad/s; absent means unavailable. |
| `observation.imu` | Optional `gyro` (trunk rad/s), `quat` (trunk→world, scalar-first wxyz). |
| `observation.images` | Optional map of camera name to `{t_ns, jpeg_base64}`; image capture time uses the same robot clock and cannot be newer than the observation. |
| `history` | Up to eight earlier observations in strictly increasing timestamp order; excludes the current sample. |

History is explicit and each inference request is independent: there is no server-side
conversation or hidden per-robot temporal state. A model needing a context window builds it
from these samples; a model needing a different history contract requires a versioned change.
The stock runner sends recent body states only, with IMU/velocities when available. Camera
capture, synchronization, and JPEG conversion are not implemented by this runner yet. An
image-aware collector should use the existing WebRTC consumer path, preserve capture times,
and form the same `PredictRequest`; do not invent a separate camera transport.

The response contains:

| Field | Meaning |
|---|---|
| `schema_version` | `microduck-joints-v1`. |
| `observation_t_ns` | Exact timestamp from the request; ties the prediction to its observation. |
| `step_ns` | Exactly 20,000,000 ns (50 Hz). |
| `positions` | H × 15 finite absolute-radian targets, 1 ≤ H ≤ 100. |

Frame `i` belongs to `[observation_t_ns + i*step_ns, observation_t_ns + (i+1)*step_ns)`.
Inference latency consumes part of this horizon. The response contains no session ID or
execution sequence: those belong to the runner, not the model. A response predicts motion;
it does not report task success.

A model adapter owns preprocessing, image decoding/resizing, state normalization, inference,
action denormalization and joint mapping. It must return this decoded action space. Upstream
Microduck's 14 normalized locomotion outputs are **not** directly compatible: the adapter must
apply its model's offsets/scales and supply all 15 targets, including the mouth. The ±π wire
bound is only a coarse numeric envelope, not collision checking or a balance guarantee.

`GET /v1/policy` exposes joint order, timestep and action-space metadata. The runner also checks
robotd's acquisition response against its expected joint order/timestep before inference.

## Replace a backend

The Python protocol has one method. No particular model architecture, trainer or GPU is required.

```python
from duckmind_policy.policy import FakePolicy, Policy
from duckmind_policy.schema import ActionChunk, PredictRequest
from duckmind_policy.service import HttpPolicy, create_app
import uvicorn

# Local implementation, useful before training exists:
policy: Policy = FakePolicy()

# The HTTP client implements the very same interface:
remote: Policy = HttpPolicy("http://127.0.0.1:8081")

# A future adapter implements this signature and performs its model-specific conversions:
# def infer(self, request: PredictRequest) -> ActionChunk: ...
uvicorn.run(create_app(policy), host="127.0.0.1", port=8081)
```

The built-in FakePolicy supports `向左看`/`往左看`/`look left`, right equivalents, and
`保持姿势`/`hold`. It copies observed non-head joints and approaches an absolute head-yaw target
at at most 0.5 rad/s in its predictions. Unsupported tasks fail explicitly; these aliases are
only the fake backend's test vocabulary. A learned implementation can accept any task it has
learned without changing the API.

The server serializes inference and rejects overlap with HTTP 503 rather than queueing old
observations. Invalid requests/unsupported tasks return 422, backend failures return 500.
The HTTP client has one pooled connection, configured I/O timeout, and no automatic retries.
A timed-out server inference may finish, but its result cannot submit itself to robotd. The
service binds to loopback by default; remote access/authentication deployment is outside this
prototype. The runner and robotd remain on the same machine; a GPU policy service can be remote.

## Runner timing and lifecycle

The runner acquires one Action Session, drains `robot.state` on a separate thread, and runs one
inference request at a time. It keeps only the latest feedback and rejects feedback older than
250 ms. It uses robot timestamps plus locally elapsed receive time; wall clocks and the model
server's clock do not determine execution time.

After inference, it verifies the session is still active, removes expired frames plus two ticks
of admission lead, and submits the remaining targets with their **original timestamps**.
It never rebases old predictions to "now". A completely expired response fails. Replanning occurs
at most 200 ms after admission (earlier for a short remaining horizon); robotd retains the prior
prefix until replacement starts. Models must produce a horizon long enough for inference,
admission and the next prediction. A 1–2-frame response is schema-valid but too short for this
runner's two-tick lead; slow inference can exhaust even a longer horizon. Admission failures stop
the run rather than silently skipping, retrying or acquiring a new session.

Normal duration completion, inference failure, Ctrl-C and lost feedback all leave through a
`finally` block that ends this runner's session. An external stop invalidates it; a late result
is discarded or rejected by robotd. Cleanup never issues a global stop against a newer owner.
If the process dies or IPC is unavailable, robotd's own bounded timeline/expiry remains the
fallback. Session end/expiry holds the last pose; it is not an active balance or recovery policy.

## Run and verify

Build robotd using the repository's Rust prerequisites, then start an isolated fake instance:

```sh
cargo build -p robotd -p robotctl
printf '[audio]\nenabled=false\n[chorale]\naccept=false\n' > /tmp/duckmind-policy.toml
./target/debug/robotd --fake --no-policy --socket /tmp/duckmind-policy.sock \
  --params /tmp/duckmind-policy.toml
```

In another terminal, initialize it and allow the home transition to finish:

```sh
./target/debug/robotctl --robot-socket /tmp/duckmind-policy.sock robot init
cd policy
uv sync --locked
uv run duckmind-policy serve
```

Then start inference/execution:

```sh
cd policy
uv run duckmind-policy run --socket /tmp/duckmind-policy.sock --task '向左看' --duration 5
```

Unit/HTTP tests run with `make test`. Setting `ROBOTD_BIN` also runs real-daemon tests that
launch their own fake robotd, initialize it, serve HTTP, submit/replan chunks, assert measured
joint feedback, and verify a stop during inference rejects the late result. Without that variable
those tests report skips, not success. CI sets it after building robotd.

This validates the software path using FakeIo. A frozen-leg head-turn rule is not a MuJoCo
walking/balancing policy; neither physics performance, camera-driven VLA behavior nor a physical
robot is validated here. The existing robotd Action Chunk API remains restricted to fake/sim.

## Precedents

[OpenPI BasePolicy](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/base_policy.py)
provides an interchangeable `infer(obs)` boundary and its WebSocket client implements that same
abstraction. [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)
separates the model server from robot execution.
[LeRobot PreTrainedPolicy](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py)
exposes `predict_action_chunk`, and its
[async server tests](https://github.com/huggingface/lerobot/blob/main/tests/async_inference/test_policy_server.py)
use a mock policy. Duckmind borrows the separation and replaceability, not their wire protocols
or dependencies. Continuous execution during synchronous remote inference is supported; RTC,
parallel inference requests and model training are not implemented by this package.
