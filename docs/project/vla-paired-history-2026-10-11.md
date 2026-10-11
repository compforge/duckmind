# Previous-target controlled experiment

The user authorized an equal-budget comparison after the [51-state preparation](vla-previous-target-2026-10-11.md).
The question is whether one previous applied target improves forward walking with the same expert demonstrations.
This is behavior cloning with SmolVLA's flow-matching objective, not a new reinforcement-learning run.

## Fixed comparison

| Setting | Control | Previous-target input |
|---|---|---|
| Body input | 36 measured values | Same 36 + 15 previous applied targets |
| Native state projection | 51 inputs; native padding fills the last 15 with zero | 51 inputs |
| Initial weights | v6 step2000, expanded with 15 zero projection columns | Exact same weight tensors |
| Task/image/action | Same task text, head-camera frames, 10×15 absolute-radian targets | Same |
| Data | `dataset-v7-forward-control` | `dataset-v7-forward-previous-target` |
| Formal updates | 2000, batch8, seed42, fresh optimizer | Same |
| Smoke | 2 updates, discarded | Same |

Both arms have the same architecture and initialization RNG consumption. Only the additional information
changes; the control's zero input gives the additional weight columns no data gradient. The pinned native
trainer supplies explicit seed42 generators for data loading/sampling. One optimizer seed remains one
paired experiment; multiple physics seeds are not independent training replications.

The converter reuses unchanged `raw-v2`: forward train seeds242–251, 10 episodes/2000 frames, and held-out
seeds252–253, 2 episodes/400 frames. Per-episode and global numerical statistics use float64 calculations.
Control data is reconverted using the same video encoder as the 51-state data, avoiding v6's extra video
re-encoding. Preflight requires exact shared frame values, task/index alignment, shared normalization and
video hashes. No new expert trajectories, recovery examples or language paraphrases are added.

The two zero-update checkpoints share immutable model bytes but retain independent processor/config files.
The control declares state36 and max_state_dim51; native SmolVLA pads after normalization. The experiment
normalizes all51 values. The first36 statistics and all action statistics must match. Nine observations
with identical inference-noise seeds must produce outputs within1e-5rad before optimization starts.

## Execution and acceptance

Devbox `training/artifacts/run-v7-pair.sh` owns the bounded run. Its `.log`, `.stage`, `.exit-code`,
`.inputs.sha256` and `.md` retain progress and provenance. The launch shell reported PID1388674.
The run is limited to4004 new optimizer updates: two discarded smoke updates and2000 formal updates per arm.
Formal training never starts from smoke weights. Source code is not modified during execution.

Before formal training, each smoke checkpoint runs one seed242 physics trial. The common zero baseline
runs seeds242 and380 under familiar and reserved wording. Final checkpoints each run18 trials:

- Seed242: training-state diagnostic, one trial per wording.
- Seeds370–373: reused development states, four trials per wording.
- Seeds380–383: predeclared states introduced this run, four trials per wording.

Wording is “向前走” versus the existing reserved “往前迈步”. The latter has been evaluated in earlier runs;
this comparison does not claim a fresh language holdout. Seed380 is also used by the predeclared zero
baseline; no training or acceptance settings are tuned from that outcome. Each prediction executes five
frames before observing again. The teacher only prepares the initial state; it never takes over a rollout.

Strict success uses the existing `forward_quality.accepted` / `recovery_outcome`, with full per-tick traces.
`len(trace)-1` is actual executed ticks. Report balance, forward progress, lateral/yaw drift, wording and
latency separately. Offline MSE uses8 held-out chunks; repeating the previous target is a null baseline.
Synchronous physics pauses during inference, so success cannot establish real-time deployment feasibility.

## Status and evidence

Preflight passed: all2400 common state/action rows, task/index columns and shared numeric statistics are exact; both splits have identical video hashes. Nine zero-update predictions are bit-identical (maximum difference0rad). Saved zero-checkpoint state/action mean/std exactly match train statistics. The8-chunk persistence baseline has MSE0.101871rad². Control smoke completed2 updates and saved its checkpoint; physical smoke and the remaining stages are running. No formal training outcome is claimed yet.
Artifacts use `audit-v7-pair-*`, `smolvla-v7-pair-{control,previous}*` and
`evaluation-v7-pair-{zero,control,previous}-{trained,held-out}.json` prefixes.

Independent pre-run/script review found no blocking issue. It confirmed native padding and projection
shape, causal previous-target alignment, equal4004-update budget, separate smoke initialization, same-noise
zero comparison and persistence baseline. Final audit must directly compare saved effective processor
state/action mean/std tensors, as well as dataset statistics and prediction equivalence.

Working hypotheses (subjective priors, not measured probabilities): hidden actuator-command state35%;
closed-loop data coverage35%; optimization10%; training adapter bug5%; evaluation/teacher limitations5%;
unknown10%. Lower offline error without higher physical success supports interpolation improvement only.
A failed paired trial would not rule out longer history, alternative conditioning or better recovery data.
No additional training is automatically authorized by this run's result.
