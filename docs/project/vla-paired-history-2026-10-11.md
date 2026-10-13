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

## Result

The bounded run completed at 2026-10-11 16:23:03 UTC, exit0, with exactly4004 updates including discarded
smokes. **Adding one previous applied target did not improve strict forward success in this experiment.**
This is a single-seed negative result for this conditioning/training recipe, not evidence that history is
unnecessary. Two trained checkpoints and all raw reports are preserved; no additional training was launched.

| Evaluation metric | Control36 | Previous-target51 |
|---|---:|---:|
| Strict success, familiar “向前走” | 2/9 | 2/9 |
| Strict success, reserved “往前迈步” | 1/9 | 0/9 |
| Strict success, total | 3/18 | 2/18 |
| Full-duration balance | 7/18 | 8/18 |
| Strict success, predeclared states380–383, both wordings | 1/8 | 0/8 |
| Strict success, training state242, both wordings | 1/2 | 1/2 |
| Held-out target MSE, rad²,8 chunks | 0.00148528 | 0.00145744 |
| Offline sample median inference, seconds, two evaluation processes | 0.393–0.401 | 0.321–0.366 |

MSE is only1.87% lower with the previous target and both final models are worse than the common zero-update
MSE0.00142064. Repeating the previous target gives0.101871 on the same8 held-out chunks, showing that the
learned predictors beat a persistence shortcut, not that they walk reliably. The offline chunks are the
same in both language reports and must not be counted twice. On the exactly matched zero-evaluation subset
(seeds242/380, both wordings), zero, control and previous-target models each achieve1/4 strict success.

The control has11 unstable trials and4 balanced-but-heading failures; the previous-target model has10
unstable trials and6 balanced-but-heading failures. In particular, all4 balanced trials for the previous
model's reserved wording violate the yaw gate. The old permissive `success` would report7/18 and8/18,
respectively, hiding these direction failures. All36 paired trials have matching initial recorded metrics;
actual execution counts equal `len(trace)-1` and range up to200 ticks/4seconds.

Inference medians exceed both the100ms execution window and the200ms prediction horizon. These small
samples include initial inference effects and are not a dedicated warm latency benchmark. Physics was
paused during inference; neither model has established real-time walking or robotd deployment readiness.

## Reproduction, controls and audit

Preflight checked all2400 common numeric rows, task/index columns, shared statistics and video-file hashes.
All are exact. Nine zero-update predictions are bit-identical (max difference0rad). Saved zero, smoke and
final state/action mean/std tensors exactly match their respective training datasets. The control's
additional projection columns remain zero; the previous model has13440 nonzero added weights with L2 norm
0.185754. Thus the added path was trained, but this does not establish meaningful use of history.

The source SHA checks still pass after completion. The complete CR-expanded run log has5685 lines
(482312 UTF-8 bytes). Each formal arm contributes200 logged measurements. Both start with logged loss0.061
at step10 and finish at0.040 at step2000; global gradient norms at the end are0.733/0.736. No NaN, OOM or
traceback was observed. Native logged allocated training memory peaks at2.33GB, not whole-process peak GPU
memory. Formal stages including loading/checkpointing took9m21s and9m27s; the whole pipeline took32m06s.
The learning schedule warms up100steps to1e-4 and decays to2.5e-6. Falling training loss did not translate
into held-out error or strict physical improvement.

After completion, both checkpoints were independently rerun on familiar wording, seeds380 and381, without
training. Both remain0/2 strict. Control executed ticks changed95→158 and133→103; previous-target changed
200→200 (heading failure) and55→56. Initial recorded metrics match, but trace differences begin at tick1.
The result therefore reproduces these failures, **not bitwise trajectories**. The source of numerical/run
order variation is unisolated; these repeats do not provide independent training seeds or justify treating
3/18 versus2/18 as a statistically established ranking.

Artifacts under `training/artifacts/`:

- `run-v7-pair.{sh,log,stage,exit-code,inputs.sha256,md}` and `run-v7-pair-repeat.log`.
- `audit-v7-pair-{preflight,processors-initial,processors-final,summary,repeat}.json` and
  `audit-v7-pair-source-check.txt`.
- `evaluation-v7-pair-{zero,control,previous}-{trained,held-out}.json`, both smoke reports and
  `evaluation-v7-pair-{control,previous}-repeat.json`; each retains full physical traces.
- `smolvla-v7-pair-{control,previous}/checkpoints/002000/pretrained_model` and all earlier checkpoints.

An independent reviewer found no blocking comparison defect. They independently verified effective
normalization, common starting metrics, projection updates and result counts, and warned about single-seed
claims, repeated language, the unmatched zero-evaluation denominator and permissive success. They found an
untriggered evaluator bookkeeping issue: rejection before an action can overstate its `ticks` field;
all this run's counts match the trace, and this report consistently uses the trace count.

## ML decision record

| Audit item | Evidence / interpretation |
|---|---|
| Config and log | Native SmolVLA, frozen VLM/trainable Action Expert and state projection; H10,batch8,seed42,2000steps each. All400 measurements and resolved configs retained. |
| SHOULD: shared inputs and zero behavior | All common frames/stats and video hashes exact;9 same-noise outputs maxdiff0. |
| SHOULD: bounded budget and no takeover |4004 total optimizer updates; teacher used only for initial preparation; full traces saved. |
| Init versus final | Same4 physical conditions strict1/4 for all three checkpoints; held-out MSE worsens from0.00142064 to0.00148528/0.00145744. |
| Null scale | Persistence MSE0.101871; no new acceptance thresholds chosen. A random-actuator baseline is not needed to establish parity between these two trained arms. |
| Samples | Raw-v2 episodes, first zero chunk in preflight, and complete per-tick physical reports permit reconstruction. Final reports do not retain every intermediate model input tensor/image. |
| Gradient / worst evidence | Finite global gradients; previous peak1.75, below clip10. Per-module gradients were not recorded; added projection weights do change. |
| Surprising measurement | Full balance7→8 but strict success3→2: heading failures explain the divergence. Repeat trajectories differ despite same seed; mechanism remains unknown. |
| Three ways a ranking would mislead | One optimizer seed; mostly reused state/wording; forward-only learning can ignore language/vision. This experiment does not prove either modality is meaningfully used. |
| Competing causes after audit | Subjective working priorities: coverage/closed-loop error45%; insufficient temporal conditioning20%; optimization10%; training-adapter bug5%; evaluation/numerical variability10%; unknown10%. Input checks argue against simple alignment/stats errors; the other causes are not isolated. |
| Cheapest next distinction | Before more broad training, test fitting/replaying a tiny fixed trajectory and inspect the first closed-loop divergence. Poor expert-state predictions suggest fitting/conditioning; good predictions but rollout failure suggests distribution shift/control sensitivity. This is a proposal, not an additional run. |
| Another implementation of history | Multi-step state/action conditioning could still help even though a single zero-initialized previous-target projection did not improve this recipe. |

The immediate objective—reliable forward walking—remains unmet. Keep the fixed-body VLA interface and the
captured failures; do not expand the command set or claim a replacement motion policy from this run.
Automatic follow-up is paused after result delivery; further training requires a new bounded experiment.
