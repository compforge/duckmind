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

## Completed 2,000-step experiment

The job completed with exit code 0 after collection, conversion, 20-step smoke training,
checkpoint inference, 2,000-step training (batch 8) and synchronous physical evaluation.
The final training log reports loss 0.118. Held-out action MSE is 0.01325 rad² and median
inference time is 353 ms. The task set differs from v1, so aggregate MSE is not a matched comparison.

Only 8/44 physical trials passed: stand and look-right passed at both seeds (10042/10043)
and both execution horizons (5/10 ticks). All other tasks failed. 33/44 trials remained stable,
but often barely moved; stopping from walking could lose balance. Each horizon passed 4/22,
so increasing feedback frequency alone did not solve this checkpoint's failures. These results
do not establish replacement of the command-conditioned expert.

The model saw 16,000 sampled frames during training, about 0.73 dataset-equivalent passes.
Undertraining is a hypothesis to test, not an explanation established by the current loss.
A paired physical evaluation with the first training phrase holds seed and horizon fixed to
separate failure on familiar wording from held-out paraphrase failure. The evaluator records
`physics_language_split`; held-out raw action MSE keeps its original meaning.

This diagnostic completed at seed 10042/horizon 5: familiar wording passed 5/11 (stand and all
four head directions), versus 2/11 for held-out wording under the same conditions. Walking,
turning and both stopping cases failed even with familiar wording. Language generalization and
motor imitation both need improvement; this single paired seed does not quantify their general rates.

## Completed follow-up: more updates on the same demonstrations

Follow-up round 1 completed with exit code 0 (PID 2844949). It kept the dataset and architecture
fixed, warm-started from v2 checkpoint `002000`, and restarted the native optimizer/scheduler.
It checked dataset metadata, ran a separate 20-step smoke and reloaded it for inference, then
trained 4,980 steps at batch 8 from the original v2 weights. Smoke weights did not flow into the
formal run. This consumed 5,000 updates including preflight.

The job repeated the same 44 held-out-language physical trials and 11 familiar-language
trials. Held-out success improved from 8/44 to 12/44: stand and both horizontal head directions
passed at both seeds/horizons; all other tasks failed. Familiar wording improved from 5/11 to
6/11 by additionally passing stop-from-forward. Walking, turning and stop-from-backward still
failed. Final training loss was 0.040, held-out action MSE 0.01223 rad², and median inference
329 ms. These reused seeds are development diagnostics, not a fresh final test set. A later
replacement claim needs untouched wording/initial conditions and execution with real inference
latency. There is no expert correction during learned rollouts.

Evidence under `artifacts/`: `run-v3.sh`, `run-v3.pid`, `run-v3.stage`, `run-v3.log`,
`run-v3.inputs.sha256`, `run-v3.exit-code` (on exit), `smolvla-v3-smoke/`,
`evaluation-v3-smoke.json`, `smolvla-v3/`, `evaluation-v3.json`, and
`evaluation-v3-trained-language.json`. The evaluation CLI passed local formatting, lint and four tests,
and its new familiar-language option completed a real checkpoint/MuJoCo diagnostic.

## Final bounded follow-up: shorter predicted chunks

Before allocating the final round, a familiar-language diagnostic replanned every single control
tick for the five remaining failed motion cases, using seed 10042. All five still failed:
forward/backward, left/right turning and stop-from-backward. Forward survived 171 ticks instead
of 54 at horizon 5, but remained unsuccessful. See `evaluation-v3-feedback1.json`. Feedback
frequency alone is insufficient for this checkpoint; this experiment still paused physics while
waiting for each prediction and does not establish feasible 50 Hz VLA inference.

Round 2 is running as PID 3437215. It uses SmolVLA's native `chunk_size=10` rather than 50,
predicting 200 ms of targets to focus training on nearer actions. Dataset, body-state layout,
model architecture and 20 ms target interval are unchanged. It warm-starts from v3 checkpoint
`004980`, restarts the optimizer/scheduler, gates on 20 smoke updates and a 10×15 Policy.infer
contract check, then performs 4,980 formal updates from v3 (not smoke weights). It repeats both
language evaluations with the original seeds/horizons 5/10. The shorter prediction horizon plus
additional updates is a directional experiment, not a strict single-variable ablation. MSE over
10 future targets cannot be compared directly with MSE over 50.

All two follow-up rounds are now allocated. After this round's evaluations, no further training
is authorized by this experiment budget. Evidence uses the `run-v4.*`, `smolvla-v4*`,
`evaluation-v4*.json` and `policy-contract-v4-smoke.json` artifact names. At this update the run
is in preflight; no improvement from shorter chunks is claimed. Shorter chunks also provide less
buffer against inference delay, so physical imitation success would still need real-time testing.

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
| Full dataset | `dataset-v2/{train,eval}` |
| Model and completed physical evaluation | `smolvla-v2/`, `evaluation-v2.json` |
| Paired familiar-language diagnostic | `evaluation-v2-trained-language.json` |

A running stage or successful process exit is not a passing motion result. Consult the final
physical trials before claiming improvement or replacement of the official policy. Physics pauses
during inference; this does not validate real-time execution or hardware. Visual grounding also
remains untested because these instructions can mostly be solved from proprioception.
