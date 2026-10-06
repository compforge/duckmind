# Action Chunk execution

Duckmind accepts model-generated joint trajectories through `robot.actions.*`. A Python policy runner owns language, images, history, inference and model-specific action decoding. `robotd` owns the action timeline and the only motor-writing path.

The API is enabled only with `--fake` or `--sim`, at a 50 Hz control rate. Physical hardware rejects these calls. BLE and WebRTC do not forward them; a policy runner on the simulation host uses the existing Unix JSON-RPC socket. The runner may call a remote model service.

## Start and observe

```bash
cargo run -p robotd -- --fake --no-policy --socket /tmp/duckmind.sock
# In another terminal; wait for homing to complete before acquiring the joints.
cargo run -p robotctl -- --robot-socket /tmp/duckmind.sock robot init
cargo run -p robotctl -- --robot-socket /tmp/duckmind.sock robot actions begin
```

`begin` requires fresh sensor feedback, warmed IMU and completed bring-up, with no shutdown, mode change, policy swap or limp-fall sequence in progress. It returns `session_id`, robot-clock `t_ns`, `step_ns`, `joint_names` and measured `positions`.

Acquisition disables the on-board movement policy and clears its velocity intent. Theremin and chorale joint writers are deactivated. Another movement writer receives `BUSY` while the session owns the body. Stop, disable, init, relax, motor reboot and shutdown can preempt the session.

## Submit a chunk

Use `robotctl robot actions submit chunk.json`, or send a JSON-RPC request with method `robot.actions.submit` and these parameters:

| Field | Meaning |
|---|---|
| `session_id` | ID returned by `begin`; invalid after cancellation, exhaustion or daemon restart. |
| `sequence` | Increasing inference-request number within this session. |
| `observation_t_ns` | Robot-clock timestamp of the state used for inference. |
| `start_t_ns` | Start of the first action interval, on the same clock. |
| `step_ns` | Exactly `20000000` (20 ms). |
| `positions` | 1–100 arrays of 15 absolute joint angles in radians, ordered as `joint_names`. |

The 15-joint body contract includes the mouth. The upstream locomotion model produces 14 normalized offsets; a model adapter must decode those and supply a mouth target before submitting. `robotd` does not guess normalization, HOME offsets or joint mappings.

Angles must be finite and within the actuator travel range. This is not a complete anatomical or collision constraint model. A chunk's end and its source observation must fall within the bounded two-second admission window; timestamps that overflow or use a future observation are refused.

Admission happens on the motor loop. A successful RPC means the timeline accepted the chunk, not that joints reached their targets. The IPC caller waits at most 250 ms for admission. A request whose reply was abandoned before processing is not executed later.

## Time and replacement

Action `i` applies during `[start_t_ns + i * step_ns, start_t_ns + (i + 1) * step_ns)`. Cloud wall time is irrelevant: use the clock already reported by `robot.state.t_ns`. The client maintains alignment to that clock from received robot state.

Past intervals are discarded. A late tick selects the current target instead of replaying missed targets rapidly. A future replacement preserves the old prefix before its start and discards the old tail from that start onward. An entirely expired or out-of-order response cannot replace the current timeline.

Waiting for a future first action holds the measured pose captured at acquisition. Replacements must overlap or continue the buffered timeline; a gap is rejected without modifying the old buffer. There is no automatic interpolation, action averaging or RTC inpainting. Plan continuous boundaries in the policy runner and validate them in simulation.

## Feedback and termination

`robot.subscribe` adds an `actions` block on eligible backends:

- `phase`: `idle`, `waiting`, `running` or `ended`.
- `session_id`, latest accepted `sequence` and current robot-clock `t_ns`.
- `selected_sequence` / `selected_index`: most recently selected command, not measured task completion.
- `remaining`: buffered intervals, including the currently active interval.
- `last_write_ok`: whether the session's most recent bus write succeeded.
- `reason`: why control ended, such as `cancelled`, `operator_preempted`, `buffer_exhausted`, `body_not_ready` or `bus_write_failed`.

Measured motion remains in `robot.state.joints` and `velocities`; `targets` reports the selected target. A successful write is not proof that the mechanism moved, and chunk exhaustion is not task success.

```bash
robotctl --robot-socket /tmp/duckmind.sock robot actions end SESSION_ID
```

End, stop, buffer exhaustion or loss of fresh/ready body state invalidates the session and discards its timeline. A session with no first chunk expires after two seconds. Except for explicit power/mode operations, the loop captures the last valid measured pose and holds it; the old policy is not automatically resumed. Holding a pose does not guarantee dynamic balance on a biped. A lost client can execute only the remaining bounded timeline, not an unbounded backlog.

## Ownership and implementation

`duck-ipc-proto/src/actions.rs` owns the wire types. `robotd/src/action_chunk.rs` owns admission, session generations and timed replacement. `main.rs` selects the source before the shared `Safety::apply → RobotIo` write; IPC never writes motors. Stop generations invalidate commands that were queued before the stop as well as already active sessions.

This follows [LeRobot asynchronous inference](https://huggingface.co/docs/lerobot/async) and [OpenPI's action-chunk broker](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/action_chunk_broker.py) in separating inference from action consumption. [RTC](https://huggingface.co/docs/lerobot/main/rtc) additionally conditions generation on the previous action prefix; it belongs in the policy runner and is not implemented by this queue.

## Verification

```bash
cargo test -p duck-ipc-proto -p robotd -p robotctl
cargo test -p robotd --test action_chunk_ipc
```

Unit tests cover late responses, tick skips, future replacement, stale sessions, queue admission and invalid actions. The integration test launches the real `robotd --fake`, sends JSON-RPC chunks, observes joint feedback, preempts execution and verifies the cancelled tail never moves the joint. MuJoCo checks exercise the same `RobotIo` path against physics; neither test establishes a learned language skill or balance policy.
