# Fixed-budget training with accepted recovery demonstrations

The [v11 collection](vla-recovery-2026-10-12.md) produced ten strictly accepted expert-recovery
windows from learner deviations. This v12 comparison asks whether adding those demonstrations
improves learned forward walking at a fixed optimization budget. Expert recovery alone was not
learner improvement; the new comparison must measure the learned policy without expert takeover.

## Data and native framework boundary

Both arms use one native LeRobot root, `training/artifacts/dataset-v12-union`, built from:

- Ten original forward episodes from `dataset-v7-forward-previous-target/train`: 2,000 frames,
  original reset seeds242–251.
- Ten accepted v11 recovery episodes: 2,000 frames, all derived from known body392 learner failures.
  These correlated windows are training data only, not an independent held-out split.

Control selects native episode indices0–9; mixed selects all20. `duckmind-train --episodes` forwards
the optional selection to LeRobot's existing `DatasetConfig.episodes`. The framework filters sampled
rows and its episode sampler, while metadata statistics remain those of the common root. Thus both arms
share accurate union statistics, including the control arm's exposure to the correction distribution's
summary. This is intentional and is not a no-information control.

The pinned LeRobot revision is `d40e8709cffb93644db66e30604ef50fdec003cb`. Native merge uses
`concatenate_videos=False, concatenate_data=False`: it copies source videos and remaps bookkeeping,
without rewriting original numerical observations/actions. New correction videos use the established
H264 conversion. Correction episode and aggregate numerical statistics are calculated in float64 via
the existing `precise_numeric_stats`; saved frames remain float32. All old datasets are retained.

Observed preparation checks pass: control2,000/mixed4,000 rows, episode sets0–9/0–19, exact numerical
state/action preservation over all4,000 frames, global statistics against float64 concatenation, equal
control/mixed statistics, and a matching multiset of SHA256 for every source/merged MP4. Original
frame1999 and final mixed frame3999 return H10 copies of the last action with padding mask
`[False, True, ... True]`; mixed frame2000 is the first correction frame. Native SmolVLA masks padded
labels. Correction `previous_target[1:]` equals `action[:-1]`; first history values retain actual learner
commands from v11, not measured positions or future predictions.

Preparation initially stopped after native merge because the audit called `.numpy()` on a datasets
`Column`. Replacing it with `np.asarray` and resuming only the audit preserved generated data. No
optimizer had run; the failed script and log remain as `run-v12-pair.prepare-attempt1.py` and
`run-v12-prepare.log`. The successful resumed audit is `run-v12-audit.log` and
`diagnostic-v12-pair/data-audit.json`.

## Fixed training comparison

Both arms initialize from the same v8 single-trajectory step1,000 checkpoint. Body state is51 values
including the previous actually executed target; actions are H10×15 absolute joint targets in rad.
Image and language interfaces stay unchanged. Each arm uses batch8, optimizer seed42, the existing
native flow-matching objective, expert/state-projection training and a fresh optimizer.

Each arm has two discarded smoke updates, then1,000 formal updates: **2,004 updates total**. Smoke
weights do not initialize formal training. Native schedule/configuration is saved in each checkpoint.
No automatic extra training is authorized by this experiment definition.

The union normalization differs from v8's single-episode normalization. A zero-update checkpoint keeps
exact v8 model bytes and installs union processors. This baseline separates normalization change from
weight updates. For zero, smoke and final checkpoints, the audit follows `state_file` references in
`policy_preprocessor.json` and `policy_postprocessor.json`, then compares all six effective state/action
mean/std tensors exactly after casting union statistics to the loaded dtype. It does not audit arbitrary
unused files by filename glob.

Fixed update count does not mean fixed original-data exposure: control draws8,000 samples from2,000
original frames (approximately four passes); mixed draws8,000 from4,000 frames (approximately two passes).
The comparison estimates the accepted-data mixture's effect under a fixed update budget. It cannot
isolate OOD coverage from changed example weighting or prove a general benefit of correction learning.

## Predeclared physical acceptance

Zero, control and mixed each run the same matrix without expert takeover:

| Group | Body reset seeds | Diffusion seeds | Trials per checkpoint |
|---|---|---|---:|
| Known diagnostic |242,390,392|390,392|6|
| Fresh acceptance |400,401,402,403|390,392|8|

Before launch, an inventory of existing experiment JSON seed fields found no400–403 references.
These four bodies are reserved before results; they are not fourteen independent held-out states.
There is one optimization seed, no training replication, and one phrase `向前走`; this does not test
language generalization. Both noise seeds were already used for development.

Each trial uses200 control ticks/four seconds, H10 prediction and E5 execution. The original
`recovery_outcome` strict conjunction remains unchanged: complete execution, stability, forward progress,
lateral drift, original heading, tail progress and upright tail. Invalid predictions or exceptions retain
partial traces and input requests. No rule-based action or expert fallback can count as learned success.

Matching body seeds must start with identical state, prior target and integration state. Initial images
may differ slightly due to the previously observed renderer variation; initial image MAE is recorded,
not asserted zero. Each actual model input and predicted chunk is saved, independently of Trace's
additional render. Diffusion RNG uses `noise_seed*10000+tick`, independent of body reset RNG.

Twelve native training chunks (original episodes0/9 and correction episodes10/19, three times each)
provide an offline fit diagnostic against target persistence. This is not a validation score. Physical
results and latency are separate: the simulator pauses while inferring, so success would not establish
real-time deployment. Record latency relative to100ms execution and200ms prediction windows.

## Evidence and status

The launcher started on2026-10-12 at13:58UTC, PID4127004. Runtime files are in the remote worktree's
`training/artifacts/`:

- `run-v12-pair.py`, `.sh`, `.log`, `.stage`, `.exit-code`, `.inputs.sha256`.
- `diagnostic-v12-pair/run.md`, `data-audit.json`, and `zero/control/mixed/report.json` with per-trial
  traces and actual model requests.
- `smolvla-v12-{control,mixed}` native checkpoints, their resolved configs and optimizer state.
- `smolvla-v12-zero`, retaining the initial model by hardlink and replacing only processors.

At this document revision, data preparation/audit is complete and the pipeline is in smoke training.
No formal-training or physical-success result is claimed. Check the live stage/exit status before
continuing; do not launch a duplicate or change running scripts. Final results and the independent
ml-debug audit will extend this record. Local training checks: Ruff format/lint passed; pytest9passed,
2skipped. Skipped native/GPU coverage is supplied by the actual remote smoke, not these CPU tests.

## Research decision and competing explanations

The selected intervention is additional accepted recovery demonstrations. Image regularization,
architecture and loss changes remain alternatives, held fixed here. Subjective pre-run priorities are
coverage45%, fit/conditioning25%, visual sensitivity15%, training bug5%, evaluation bug5%, unknown5%;
these percentages express investigation priorities, not estimated causal probabilities.

Improvement on392 only would support local fitting, not transfer. Improvement on fresh states would
justify broader collection and replicated training, subject to exact pipeline checks. Better offline fit
without physical benefit leaves feedback sensitivity, sampling and coverage unresolved. A negative
comparison would reject this particular mixture/budget, not the broader correction-learning method.

Three ways an apparently positive result could mislead: union normalization accounts for the effect
(check zero); extra exposure/weighting accounts for it (fixed-update conditional interpretation);
known-body memorization accounts for it (report fresh4-body matrix separately). Renderer variability
and a single optimizer seed limit small differences even when those controls pass.
