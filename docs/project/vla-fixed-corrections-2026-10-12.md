# Targeted expert corrections for deterministic fixed-image failures

The user authorized the next step after [v13](vla-fixed-image-2026-10-12.md): locate the
reproducible forward-walking failures, collect successful expert recovery demonstrations, and
compare equal-budget learning with and without those demonstrations. Camera observations remain
constant black during training and learned inference. No language or visual generalization claim
is intended; the task is early closed-loop motion learning on one fixed Microduck body.

## Frozen collection and evidence

Use only repeat0 of the three unique failed v13 fixed-image cells. Repeat1 was exactly identical
and is not duplicated as new data. Predeclared interventions span early, middle and later portions
of each saved trajectory; they are not claimed to identify the first causative error.

| Saved source | Original instability tick | Intervention ticks | Strict recovery accepted |
|---|---:|---|---|
| 392-390-0 |84|20,40,60|3/3|
| 410-392-0 |146|20,70,125|2/3;70 rejected for heading|
| 412-390-0 |195|20,95,175|2/3;95 rejected for heading|

Collection completed in24.72s with exit0, zero new VLA predictions and zero optimization updates.
All nine windows execute200 expert-controlled ticks and remain stable. Seven also satisfy the
unchanged direction/progress/tail checks: **1,400 accepted expert frames**. The two rejected windows
remain evidence, not training labels. Later interventions125/175 succeed despite earlier70/95
rejections, so these results do not establish a monotonic late-recovery limit.

The implementation reuses `Simulation`, `Trace`, `check_handover`, `expert_rollout`,
`resume_teacher` and `recovery_outcome`. It replays every saved learner action through the normal
actuator/physics path. All three full failure trajectories reproduce state, previous target,
integration state, actuator target and metrics exactly; all candidate prefixes are also verified.
A physics vector is only a witness, not a complete saved BAM snapshot: full sequential replay
preserves the live actuator evolution. Three fresh teacher-only controls pass the strict gate.

At handover, preserve the original heading, physical state and last applied target. Reconstruct
the expert's previous normalized action from that applied target. Only post-handover expert
commands become labels; previous_target on the first correction frame remains the learner's last
applied target. Each subsequent previous_target equals the preceding action. Independent CPU
collection audit passed344 checks. No rule dispatch or expert takeover is used in learned evaluation.

Original learner suffixes terminate at their actual instability, with fewer than200 remaining
steps. Their `without_intervention` completeness failure is not an equal-duration causal control;
the separately reproduced84/146/195-tick instability is the relevant failure evidence.

## Native data and normalization

`training/artifacts/dataset-v14-union` is a new native LeRobot dataset containing:

- Episodes0–9: the original ten forward trajectories,2,000 frames, copied numerically from the
  v13 fixed-image root.
- Episodes10–16: all seven accepted recovery windows,1,400 frames. These are correlated development
  examples from three known failure conditions; none is treated as an independent held-out episode.

Every encoded frame is black256×256 RGB. The original raw correction images remain in the evidence
archives. Native create/add_frame/save_episode/finalize is used; the existing float64 numerical
statistics helper updates both episode metadata and global statistics. Original state/action,
task text, timestamps and indices remain unchanged. Corrections concatenate the pre-action36-body
state with the15 previous-target values; action is15 absolute joint targets in rad.

Preparation passed exact numerical checks for all3,400 rows, original task/time/index preservation,
finite values/statistics, all3,400 decoded black frames, native sample checks and H10 padding at every
episode boundary. Control selects episodes0–9 through the native episode selector; mixed samples
all17. Both use the common union statistics, so control has access to the correction distribution's
summary too. Old datasets are preserved.

The initial model is v13 fixed step1,000. A zero-update checkpoint hard-links the identical model
bytes and installs the new union processors. This distinguishes normalization change from weight
updates. Original-normalization failure behavior is already verified by exact replay; the zero
checkpoint is not mislabeled as unchanged original-policy behavior.

## Bounded learning comparison

Both arms initialize from the same v13 fixed checkpoint with fresh optimizers, batch8, seed42,
H10 and E5. Native SmolVLA flow matching, frozen VLM, action expert/state projection and image
presence remain unchanged; augmentation is disabled. Each arm has two discarded smoke updates
followed by1,000 formal updates: **2,004 total maximum**. Smoke weights never initialize the formal
run. This is one optimization seed, not replicated training.

Control samples2,000 frames; mixed samples3,400. At8,000 formal sampled examples per arm this is
approximately4.0 versus2.35 passes, and the mixed correction fraction is41.2%. Consequently the
comparison estimates the fixed-budget correction-mixture recipe, not coverage alone or a full
DAgger implementation. No automatic extra training is authorized by this experiment.

Prior v13 artifacts occupy approximately3.1GiB per formal run and1.6GiB per smoke. Roughly15GiB
was free before preparation, sufficient for the equivalent9.4GiB checkpoint budget plus small data
and trace outputs. The zero model is a hard link. No old images, checkpoints or datasets are deleted.

## Predeclared evaluation and interpretation

Zero, control and mixed each evaluate the same eighteen cells: known initial-state seeds392,410,412
(failures),411,413(regression), plus fresh420–423, crossed with inference-noise390/392. An artifact
seed inventory found no prior use of420–423. Zero sees the same reserved matrix, but no tuning or
selection based on its outcomes is allowed before final runs. There are four fresh initial-state
conditions, not eight independent robots or independent training seeds.

Each trial runs up to400 ticks/eight seconds with fixed black effective images. Compute strict
four-second quality on the first200 actions and strict eight-second quality on the full trajectory,
using the same existing gate function. The four-second result is a nested prefix of the longer
trial, not another rollout. An early failure retains its actual shorter trace. Ten-tick smoke
strict failure is expected and is not an execution failure.

Save every actual input image/state/prior, predicted chunk, inference latency and complete physical
trace. Require identical initial state/prior/physics across phases and exact effective processors.
Offline fit uses twelve original training chunks with a persistence floor; it is not independent
predictive validation. Report known and fresh results, completion/balance/direction and4s/8s
separately. Physics pauses for inference, so this is not a real-time runtime test.

Improvement only on the three collection conditions would support local fitting. Improvement on
untouched initial states and sustained8s behavior would support broader closed-loop benefit under
this recipe. No benefit despite correct labels would motivate inspecting the earliest prediction
errors, data weighting or representation; it would not disprove correction learning in general.

## Artifact ownership and status

All artifacts live under `training/artifacts/`:

- `run-v14-collected.py` preserves the exact completed collector; collection log/exit-code remain.
- `run-v14-correction.py` owns collection, preparation, smoke and evaluation stages; native training
  is called by `run-v14-pair.sh`.
- `diagnostic-v14-correction/recovery/`: report, all nine attempts, full failed-prefix replays and
  teacher controls; `run.md` records stage decisions.
- `diagnostic-v14-correction/data-audit.json`, `dataset-v14-union/`, `smolvla-v14-zero/`.
- Planned `smolvla-v14-{control,mixed}[-smoke]` and
  `diagnostic-v14-correction/{zero,control,mixed,smoke-control,smoke-mixed}/`.

Collection and preparation are complete; independent preflight passed74 checks plus full video decoding.
The bounded launcher PID673267 has started on the amd devbox. Training stage and actual checkpoints must be checked
against `run-v14-pair.stage/log/exit-code` and live processes, never inferred from this design.
No robotd/production Policy source or blog content is changed by this experiment.
