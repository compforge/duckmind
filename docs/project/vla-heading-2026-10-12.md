# Heading observability and an eight-second expert reference

After [v14](vla-fixed-corrections-2026-10-12.md), the user authorized continuation. This
diagnostic tests the expert against the same longer-horizon task and checks whether existing
sensory history can reveal accumulated yaw. It adds **zero optimizer updates and zero VLA calls**.

## Decisive result: the expert also misses the heading requirement

Use the existing ONNX expert, native `Simulation`/`Trace`, actuator path and unchanged quality
function. The command remains body-forward velocity `(0.4, 0, 0)`. Each expert rollout starts
from exactly the saved v14 physical state, body observation and previous target. All nine
initial-state seeds are now development conditions: 392, 410, 412, 411, 413, 420–423.

| Reference | Stable for 8s | Strict first 4s | Strict full 8s |
|---|---:|---:|---:|
| Original expert, nine initial conditions | 9/9 | 9/9 | 0/9 |

Every eight-second failure is **only** `original_heading`: maximum absolute yaw relative to
the initial heading exceeds the existing 0.5 rad gate. Maximum excursions span 0.565–0.790 rad
(32.4–45.3 degrees). All other existing checks pass: duration, balance, forward progress,
lateral drift, final-second progress and uprightness. Seed420 finishes at -0.495 rad but
exceeds the gate earlier; the criterion applies to the whole trajectory, not just its endpoint.

An extra seed392 rollout reproduces states, actions, previous targets, integration state,
actuator targets and metrics exactly. It is a determinism check, not a tenth independent body
condition. The expert has no sampled VLA inference noise. Its nine cases must not be described
as equivalent independent samples to the student's eighteen body/noise combinations.

This changes the interpretation of v14: **expert replacement and sustained initial-heading
tracking are different goals**. The expert supplying the demonstrations does not reliably
demonstrate the latter at eight seconds in this environment. The existing strict result remains
valid as a higher target; it cannot by itself quantify failure to imitate the expert. The student
still has a real stability gap: v14 mixed stays stable for eight seconds in 10/18 body/noise
cells, while the expert stays stable in every corresponding body condition. This diagnostic
does not establish accurate velocity tracking or task completion merely from stable walking.

## Source and observation contract

The pinned upstream source is `microduck_rl` commit
`8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec`. Relevant local and devbox source hashes agree.
`scripts/infer_policy.py` builds the expert's 61-dimensional observation from angular velocity,
projected gravity, relative joint positions, joint velocities, previous action and commands.
It has no accumulated task-relative heading input. The current velocity training configuration
sets `rel_heading_envs=0.0`; its velocity tracking objective is not an initial-heading target.
This current source does not prove the full reward/config provenance of the exported ONNX file.

The learner's 51-dimensional input likewise has instantaneous body state and the previous
executed joint target, with a fixed black image. A synthetic static check rotates the body by
0 and ±0.6 rad about world vertical while keeping the task reference fixed. Both learner state
and expert observation remain equal within 9.32e-10, while relative heading changes. Sensors
are refreshed with `mj_forward` in every branch. These are synthetic observation examples,
not reachable training trajectories or additional learned rollouts.

Thus neither instantaneous observation directly identifies accumulated yaw. This does not
prove that missing heading caused every fall, or that an angular-rate controller cannot walk
straight without explicit heading feedback.

## Existing sensor history contains useful relative-heading information

Analyze all 54 saved v14 trajectories, including truncated failures. Each trace contains N
pre-action body states and N+1 physical metrics. Integrate at 50 Hz using only samples available
before the corresponding action. The initial relative-heading estimate is zero; simulator yaw
is used only to measure error, never fed back into the estimate or policy.

For body-frame angular velocity `(p,q,r)` and normalized projected gravity `(gx,gy,gz)`, the
ZYX yaw rate away from pitch singularity is:

```text
yaw_rate = -(gy*q + gz*r) / (gy² + gz²)
relative_yaw[k+1] = relative_yaw[k] + 0.02 * yaw_rate[k]
```

The implementation rejects a near-zero denominator instead of silently clipping it. Three
synthetic pure-yaw cases validate the sign and tilt correction. The IMU site uses the trunk's
axes. Compare against both constant-zero heading and simply integrating gyro z:

| Estimator | Median per-trajectory RMSE (rad), lower is better | Worst absolute error (rad), lower is better |
|---|---:|---:|
| Tilt-corrected gyro integration | 0.01728 | 0.10850 |
| Gyro-z integration | 0.02243 | 0.22363 |
| Constant-zero heading | 0.17626 | 1.75746 |

Tilt correction improves RMSE over gyro-z integration in 47/54 traces, not universally.
These are diagnostic results on existing development traces, not a learned-policy improvement.
The worst tilt-corrected error on stable samples is 0.09286 rad. On the nine full expert runs,
its maximum error ranges from 0.03118 to 0.04170 rad. Simulation gyro noise is zero and projected
gravity is obtained from simulator orientation; real IMU bias, attitude estimation and long-term
drift remain untested. Sensor timing and 50 Hz integration also limit accuracy. No estimated
heading has been added to the deployed model or dataset contract.

## Evidence and bounded execution

The diagnostic completed in 32.47 seconds with exit code0. A ten-tick native expert smoke
preceded the longer rollouts. All 212 recorded v14 dependency/input hashes matched before and
after execution; original datasets, models, normalization and production source remain unchanged.
No new training, expert intervention in learned evaluation, hardware access or paid compute ran.

Artifacts under `training/artifacts/` on devbox:

- `run-v15-heading.py`, `.log`, `.exit-code`: exact diagnostic and full execution evidence.
- `diagnostic-v15-heading/report.json`, `run.md`: configuration, predictions and results.
- `diagnostic-v15-heading/teacher/`: complete expert states/actions/metrics, including smoke
  and the determinism repeat.
- `diagnostic-v15-heading/estimates/`: causal estimates, rates and comparison truth for all
  saved student and new expert trajectories.
- `diagnostic-v15-heading/yaw-invariance.npz`: actual synthetic observation pairs.
- `diagnostic-v15-heading/final-audit.md`: complete ML-debug form and interpretation limits.

Independent NumPy/stdlib review passed 2,073 checks without importing the production estimator
or gate function. It reconstructed expert yaw from root quaternions and displacement/height
from stored positions, checked 20 ms timestamps, recomputed all quality/estimation metrics and
verified initial-state equality, action causality and provenance. Details are retained in
`diagnostic-v15-heading/independent-final-audit.md` and `v15-independent-final.json`.

## Next research decision

Keep the historical strict heading gate and all failures. For the near-term expert-replacement
goal, evaluate balance, body-relative forward velocity and angular-rate tracking separately from
accumulated heading drift. First measure those signals for the expert and student on the same
traces; stable completion alone is insufficient. This makes the capability gap explicit without
changing previous results or declaring an easier replacement success after seeing the outcomes.

If the intended task is instead to preserve the original direction, first obtain demonstrations
that achieve that task over the requested duration, and provide causal task-relative heading or
adequate history. An explicitly labeled heading controller can help generate expert data; its
success must not be presented as learned VLA ability. Then use a new state/dataset contract and
a bounded paired learning comparison. Merely appending heading while retaining labels that drift
does not supply the missing corrective behavior.

The remaining stability error, action fitting, state coverage and inference latency are still
live problems. These results neither disprove language-conditioned motion learning nor prove
that one additional observation will fix it. This diagnostic is complete; no new training or
automatic follow-up has been queued.
