# Forward-only specialization — 2026-10-11

The v6 run specializes the existing SmolVLA checkpoint on forward walking. Its first paired
test observed **3/10 strict successes versus 0/10 for unchanged weights with the same new
normalization**. This is partial improvement in a small synchronous test, not reliable walking
or replacement of the expert. The user authorized this run after the
[v5 correction experiment](vla-correction-training-2026-10-11.md).

## Restored v5 checkpoint

After Docker cleanup recovered disk capacity, native LeRobot resumed mixed001500 with saved
optimizer, scheduler, RNG and processors, performed500updates and saved002000 successfully.
A disposable two-update smoke preceded it; its shorter schedule was automatically rescaled,
but formal2000 retained the original schedule. The new model and optimizer tensors are finite;
effective state/action normalization is unchanged. Original failed-save artifacts are preserved.

Matched control002000/mixed-resumed002000 each remained strict0/12 on seeds360–365 and two
wordings. Mixed had one complete stable trial that failed heading, and11 instability failures.
This completed the missing-checkpoint comparison rather than improving reliable forward success.
Evidence: `training/artifacts/run-v5-resume.md`, `.log`, `audit-v5-resume-results.json` and
`evaluation-v5-2000-*.json`. This recovery consumed502 new optimizer updates including smoke.

## Experiment and controls

- Native `split_dataset` selects the ten original forward training episodes, seeds242–251:
  2,000 frames, two familiar task texts, no held-out episodes or new correction trajectories.
- Initialize from `smolvla-v5-mixed-resumed/checkpoints/002000/pretrained_model`. These weights
  already learned from mixed/corrective data; “forward-only” describes this run's new data.
- Train native SmolVLA Action Expert/state projection with frozen VLM, H10 at50Hz, batch8,
  optimizer seed42,20 disposable smoke updates then2,000 formal updates with a fresh optimizer.
  No new architecture, loss, command dispatch or expert fallback is added.
- Zero-control retains identical starting weights but uses the same corrected subset
  normalization as training. It tests normalization alone, not an unmodified old model.
- Evaluate five physical initial states with “向前走” and “往前迈步”. Seed242 is a training
  initial condition;370–373 are new state seeds. Execute five predictions before replanning.
  Each trial targets four simulated seconds. Physics pauses for inference.

Changing task mixture, sampling frequency and normalization together prevents an isolated
causal claim about task competition. One training seed and four new physical seeds are a small
engineering experiment. A single forward task also cannot demonstrate semantic discrimination
between different instructions, even when a reserved paraphrase succeeds.

## Data audit and numerical issue

All2,000 numeric action/state frames exactly match their source;30 first/middle/final samples
also match labels and terminal padding. Native video re-encoding gives maximum sampled mean
pixel error0.003075 on a0–1 scale. No evaluation frames enter training statistics.

The first preparation attempt stopped before training: native float32 standard deviations
did not match direct float64 recomputation. Independent review reproduced every original
episode's float32 statistics using LeRobot's `get_feature_stats`. Its
`RunningQuantileStats` computes variance as `E[x²]−E[x]²`, which loses precision near a
constant nonzero mean. The gravity_z aggregate standard deviation was0.000762049 instead of
0.000681964; one episode reported zero instead of0.000660695. This was not split leakage.

Only the new subset's global action/state statistics are recomputed through public native
`get_feature_stats(float64)` and `write_stats`, then checked against direct means/stds. Zero,
smoke and final processors use identical effective state/action statistics. Original statistics
remain in `meta/stats.float32-native.json`; the failed audit remains in `run-v6-attempt1.*`.
Per-episode metadata retains original statistics: do not split/merge this subset again without
correcting those episode statistics. Source datasets and the installed library are unchanged.

## Physical results

Independent review recomputed all gates from the20 zero/final traces. Recorded initial metrics
match across paired trials. Strict success includes full execution, stability, forward and
last-second progress, bounded lateral drift, max absolute yaw below0.5rad, and upright final
posture; `forward_quality.accepted` is authoritative, not the older5cm criterion.

| Group | Zero strict | Final strict |
|---|---:|---:|
| Known initial state242, both texts |0/2|1/2|
| New states370–373, familiar text |0/4|2/4|
| New states370–373, reserved text |0/4|0/4|
| Total |0/10|3/10|

Final strict passes are familiar371/373 and reserved242. Five of ten final trials complete
all200targets stably, versus zero of ten for zero-control. Reserved371/373 complete stably but
reach max yaw0.50806/0.50447rad, just beyond the fixed0.5rad gate. The remaining five terminate
with excessive tilt. The threshold is not relaxed after seeing results.

Standalone expert controls at the same five initial states complete stably5/5 but pass strict
3/5: expert371/372 reach max yaw0.50034/0.51553rad. Replaying expert on all ten training initial
states passes strict10/10. These are new expert rollouts, not recovered original traces. They
show that some direction failures also occur in the teacher; the student still has a substantial
balance gap. Teacher control never intervenes in student evaluation.

## Training and ML-debug audit

| Item | Evidence and limit |
|---|---|
| Budget/config |20smoke +2,000formal updates; 2,000train frames, batch8, H10, seed42; formal8:45 |
| Full log |2,761 normalized lines,202 metric rows; finite loss and total gradient norms |
| Final training point |loss0.045,grad_norm0.788,LR2.5e−6; logged memory2.33GB is not complete peak GPU usage |
| Saved model |500 tensors all finite; successful final checkpoint save and native inference reload |
| Baseline/offline |Same8 held-out chunks: zero MSE0.0187113 → final0.00142111rad²; repeated language reports reuse these chunks |
| Input/output sample |Reports retain task, H10×15 target example and all per-tick physical metrics |
| Processor control |Effective state/action tensors identical between zero and final; no stats mismatch explains the paired difference |
| Null/shortcut |Stationary or transient forward movement fails progress gates. Random/shuffled-label runs were not performed; one-task model may ignore language |
| Alternative causes |Specialization can improve fitting; normalization and initialization interact; missing previous targets/history and action-generation stochasticity remain candidates |
| Reference audit |Native SmolVLA masks padded steps/dimensions; collect observes before target/application; no detected time shift or cross-episode label leakage |
| Missing diagnostics |No module-wise gradients or full offline train-fit score; known-start rollout is not an overfit-one-batch test |
| Real time |Final offline inference medians369–389ms exceed100ms replanning and200ms prediction horizons |

This supports continuing to investigate single-task fit, state/history and stable execution.
It does not establish broad language generalization, visual grounding, other-task retention,
real-time deployment or physical-hardware readiness. The teacher sees `last_action`; the
student's36D state has no previous target/history, an unresolved observation-sufficiency assumption.

## Repeatability check

A no-training repeat of all five seeds and both texts again passes exactly the same three
cases, strict3/10. Do not combine repeats into20 independent tests. Four repeated trials
complete stably versus five originally: reserved371 now tilts after111targets rather than
finishing200, and reserved373 remains a heading failure with max yaw0.803rad. Other failure
durations also change. Recorded starts and seeds do not establish bitwise simulator/inference
determinism. The repeat confirms the observed pass set, not stable performance across all cases.

## Evidence

All paths below are relative to remote `training/artifacts/`; data/checkpoints stay outside Git.

- `run-v6.sh`, `.md`, `.log`, `.stage`, `.exit-code` and `.inputs.sha256`: ordered run/provenance;
  main pipeline completed exit0 at2026-10-11 14:50UTC.
- `audit-v6-data.json`, `audit-v6-results.json`: selected episodes, numeric audit and trial rows.
- `dataset-v6-forward/train`, `smolvla-v6-zero`, `smolvla-v6-forward/checkpoints/002000`:
  training data, normalization control and final saved training state.
- `evaluation-v6-{zero,final}-{trained,held-out}.json`: paired results and raw traces.
- `evaluation-v6-expert.json`, `evaluation-v6-expert-train-states.json`: separate expert controls.
- `run-v6-repeat.*`, `evaluation-v6-repeat-*.json`: no-training repeats, reported separately.

The2,020-update budget is consumed; no further training round is queued.
