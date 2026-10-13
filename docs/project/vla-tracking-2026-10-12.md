# Separating expert replacement, velocity tracking and sustained balance

The user authorized continuation after [v15](vla-heading-2026-10-12.md). This read-only
analysis reuses all 54 v14 learner traces and the nine unique v15 expert traces. It adds zero
training, inference or simulation. Historical strict success gates and outcomes are unchanged.

## What the learner has acquired

The v14 correction-trained policy remains stable through four seconds in **18/18** body/noise
cases, versus 12/18 for original-data training and 8/18 before either update. Its strict four-second
success remains **15/18**, because stability is only one requirement. At eight seconds, stable
completion is 10/18, versus 9/18 and 5/18 respectively. The expert remains stable in all nine
body conditions. These are one optimizer seed and nine initial states crossed with two inference
noise seeds, not eighteen independent bodies or training repetitions.

| Existing policy | Stable 4s | Stable 8s | Strict 4s | Strict 8s |
|---|---:|---:|---:|---:|
| Correction mixture | 18/18 | 10/18 | 15/18 | 4/18 |
| Original-data continuation | 12/18 | 9/18 | 12/18 | 5/18 |
| Zero updates, common normalization | 8/18 | 5/18 | 8/18 | 5/18 |

All eight mixed-policy failures first cross the existing stability condition at 4.26–7.82s.
These are sampled detection times: the crossing lies between that sample and the preceding
20ms sample. Trials stable at eight seconds are right-censored, not proven stable indefinitely.
Thus the mixture improved short-duration balance in this experiment; sustainable walking and
original-heading tracking remain distinct unsolved requirements.

## Velocity measurement and controls

The trace records horizontal displacement in the task's initial-heading frame and accumulated
relative yaw. Take each 20ms displacement difference, then rotate it by the interval's midpoint
yaw into a horizontal heading frame:

```text
vx = cos(yaw_mid) * delta_forward / dt + sin(yaw_mid) * delta_left / dt
vy = -sin(yaw_mid) * delta_forward / dt + cos(yaw_mid) * delta_left / dt
yaw_rate = delta_yaw / dt
```

This is finite-displacement velocity in a gravity-aligned heading frame, not exact instantaneous
three-dimensional body-frame velocity. Roll/pitch motion and fast within-tick dynamics are not
fully represented. Three constant-velocity synthetic trajectories at different headings verify
rotation direction; a stationary control gives forward-speed RMSE0.4m/s against command0.4m/s.
No new success threshold is chosen from these results.

Each learner is compared with the same body's expert over exactly the learner's observed
prefix, including its terminal unstable sample if any. Every trajectory is weighted equally
in median/min/max summaries; longer rollouts do not contribute more samples to the aggregate.
The expert is reused for both inference-noise cases without increasing its independent sample
count. Initial metric records match, and v15 independently verified complete initial physics.

For the mixed policy's eighteen matched prefixes, median forward speed is **0.15147m/s**, versus
**0.15089m/s** for the expert. The expert's nine full eight-second trajectories have median forward
speed0.15162m/s, ranging0.14753–0.15554m/s, despite command0.4m/s. Therefore numerical command
tracking and reproducing this expert's behavior are different benchmarks. These measurements do
not establish why this particular expert/environment undershoots the commanded speed.

Velocity alone also misses failure. The zero-update policy has a lower forward-speed RMSE than
its expert prefix in14/18 cases, yet remains stable for eight seconds in only5/18. Falling or
otherwise moving somewhat faster can improve a single RMSE against an unmet0.4m/s command.
Do not use that score alone to declare capability parity.

## Survivor selection and angular motion

Separate full-window summaries select only trials that remain stable throughout4s or8s.
Those populations differ by policy. To avoid comparing different selected bodies silently,
the report also includes their common-ID intersections: five cases survive4s in all three
phases, and only two survive8s. Even these are survivor-selected development subsets, not
independent evidence of generalization.

On the common five four-second cases, median forward speeds are0.14762/0.14255/0.15738m/s for
zero/control/mixed, versus0.14685m/s for the corresponding expert. The common two eight-second
cases are too small to infer broader superiority. Complete values and IDs remain in the report.

On mixed-policy matched prefixes, median yaw-rate RMS is0.83054rad/s, versus0.75425rad/s for the
expert; it is lower than the matched expert in only2/18 cases. Mean yaw rate is+0.06837rad/s
versus-0.07553rad/s. RMS includes the normal alternating yaw motion of the gait as well as drift;
it must not be interpreted as sustained turn rate or proven cause of falling. Lateral velocity
similarly includes gait oscillation. The raw interval signals are preserved for inspection.

## Evidence and next learning experiment

The analysis completed with exit0 in0.25s and66 JSON input hashes unchanged. It reads only
existing artifacts; no production source, dataset, model or normalization changed. Exact script,
full log and exit code are `training/artifacts/run-v16-tracking.py`, `.log`, `.exit-code`.
`training/artifacts/diagnostic-v16-tracking/` contains `report.json`, `run.md`, per-case student
and matched-expert interval arrays, and the independent audit. Independent NumPy review passed3,082 checks, reconstructed
velocities directly from stored world positions and quaternions, verified20ms timestamps and
exact initial physics/actuator/state/prior equality for all54 pairs, and checked129 raw inputs
unchanged. See `independent-audit.md` and `independent-audit.json`.

The next learning question should target **sustained balance beyond the demonstrated short
windows**. Original episodes and accepted v14 correction windows each contain200 ticks/four
seconds. This does not prove short episodes cause the later failures: a stationary feedback
policy can generalize beyond episode duration, and v14 starts from previously trained weights.
It does make longer continuous demonstrations and corrections from later learner states a
more specific candidate than blindly increasing optimizer steps or appending heading alone.

Proposed comparison: retain the fixed image and51-dimensional state, collect8–12s expert
trajectories plus recovery windows from the current learner's late failure states, and compare
an equal-update-budget continuation against original-data training with shared normalization.
Keep previous data and strict heading results, label the data task as body-relative walking,
and report stability, measured velocities and heading separately. A reference used for labels
must first demonstrate the requested duration; a deliberately straight-line task additionally
needs a reference that actually corrects heading. No new collection or training is queued here.

Storage is a concrete constraint: devbox has5.3GiB free. Another full paired checkpoint run
needs a storage plan; no old checkpoints or data have been deleted. Realtime inference is also
unresolved: all of these saved learner rollouts paused physics during predictions.
