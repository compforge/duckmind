# Expanded Microduck demonstrations — 2026-10-11

The second experiment expands the expert-generated curriculum and supplies joint velocities to
SmolVLA. It is a new baseline combining data, observation and training changes; it does not isolate
which change improves control. The initial 1,000-step results remain in
[vla-baseline-2026-10-11.md](vla-baseline-2026-10-11.md).

## Data and expert probes

The catalog has 13 task families. Eleven passed the expert smoke: stand, forward/backward,
turn left/right, look left/right/up/down, and stop from forward/backward walking. All 22 smoke
trajectories passed the documented heuristic checks. The full collection then accepted all
132 trajectories (12 repetitions, 4 seconds at 50 Hz), or 26,400 frames:
110 training episodes and 22 evaluation episodes, separated by complete initial-condition seeds
242–251 and 252–253, with a held-out language expression.

The selected command magnitudes come from physical probes of the installed v5 velstand expert.
Smaller reverse/turn commands can produce almost no movement. Forward/backward use nominal
0.4 m/s magnitude; turns use 1.5 rad/s; head yaw uses 0.3 rad; pitch uses −0.4 / +0.25 rad.
Each sampled magnitude is 1.00–1.05 times nominal and is recorded per episode. These are teacher
inputs, not claims of measured tracking accuracy or limits for another checkpoint.

Sideways commands at ±0.3 m/s barely moved. Larger values produced substantial unwanted yaw.
Sideways tasks remain in the catalog with a heading/drift gate but are excluded from this run.
No failed or mislabeled side-motion trajectory enters the dataset. Sit/stand transitions,
recovery and other specialist policies are outside this run.

Small post-settling joint-position (±0.005 rad), planar trunk-velocity (±0.02 m/s) and angular-rate
(±0.1 rad/s) perturbations make starts physically distinct. Stops begin after two seconds of
verified teacher walking. This is still a small flat-floor dataset, not broad domain randomization.

## Engineering checks

- State is 36 floats: positions, gyro, projected gravity, joint velocities. Native checkpoints
  have a 36-column state projection, expanded from the pretrained 32 columns with zero new weights.
- Raw trajectories retain the previously applied target. It is not a learner feature because
  the current inference request has no applied-action feedback field.
- The full raw-data audit verified finite 200×36 states, 200×15 actions, zero mouth slots and
  exact equality of `previous_target[t]` to `action[t-1]` for every accepted episode.
- Training formatting/lint and four contract tests passed. Policy validation passed with
  10 tests and four pre-existing opt-in tests skipped; lint/type checking passed.
- Native conversion of the 22-episode smoke set and two CUDA training steps completed. The
  checkpoint reloaded through `SmolPolicy.infer` and returned validated 50×15 absolute-radian
  targets, a 20 ms step, the request observation timestamp and a zero mouth target.
- Local/deployed Python source digest (sorted filenames plus bytes):
  `966fd99be5fbe0c1ebc80456ab1aa0dc68fc648ae3365bec980e9ccc061ac52e`.

## Running experiment

At this record's update, full collection is complete and dataset conversion is running. The
bounded job uses PID 2576163 and automatically gates each stage: collection, conversion,
20-step smoke training, checkpoint inference, 2,000-step training (batch 8), then synchronous
physical evaluation. Evaluation requests execution horizons 5/10 and seeds 10042/10043.
The 2-step engineering preflight is separate; it does not establish physical task success.

On the devbox, paths below are relative to
`~/rustprojects/compforge/duckmind/.worktrees/vla-training/training/artifacts/`:

| Evidence | Path |
|---|---|
| Expert parameter probes | `probe-v2.json`, `grid-v2.json` |
| Full raw data and quality evidence | `raw-v2/manifest.json`, `audit-v2.json` |
| Preflight native training | `preflight-v2.log`, `smolvla-v2-preflight/` |
| Existing Policy contract check | `policy-contract-v2.json`, `contract-v2.log` |
| Job definition and live status | `run-v2.sh`, `run-v2.pid`, `run-v2.stage`, `run-v2.log` |
| Final process result (written on exit) | `run-v2.exit-code` |
| Full dataset (conversion in progress at this update) | `dataset-v2/{train,eval}` |
| Scheduled model and evaluation outputs | `smolvla-v2/`, `evaluation-v2.json` |

A running stage or successful process exit is not a passing motion result. Consult the final
physical trials before claiming improvement or replacement of the official policy. Physics pauses
during inference; this does not validate real-time execution or hardware. Visual grounding also
remains untested because these instructions can mostly be solved from proprioception.
