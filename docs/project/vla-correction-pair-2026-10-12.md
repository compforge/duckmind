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

The launcher started on2026-10-12 at05:58UTC, PID4127004. Runtime files are in the remote worktree's
`training/artifacts/`:

- `run-v12-pair.py`, `.sh`, `.log`, `.stage`, `.exit-code`, `.inputs.sha256`.
- `diagnostic-v12-pair/run.md`, `data-audit.json`, and `zero/control/mixed/report.json` with per-trial
  traces and actual model requests.
- `smolvla-v12-{control,mixed}` native checkpoints, their resolved configs and optimizer state.
- `smolvla-v12-zero`, retaining the initial model by hardlink and replacing only processors.

Data preparation, both smoke stages, formal training, physical acceptance and targeted no-training
repeats are complete. Original pipeline exit0; results and interpretation are below. Local training
Ruff format/lint and pytest9passed/2skipped checks passed. The real remote smoke supplies native/GPU
coverage absent from those CPU checks. Independent reports remain in the artifact directory.

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

## Completed results

The pipeline completed at2026-10-12 06:18:45UTC with exit0. It used exactly2,004 optimizer updates:
two discarded two-update smokes and two1,000-update formal arms. All declared source checks passed
at completion. **This fixed-budget correction mixture has not demonstrated a reproducible strict-forward gain.**
It learned the sampled correction targets better offline, while its first fresh-body score was lower.
The targeted repeat below changes several physical outcomes, preventing a reliable ranking from the
initial two-trial difference.

| Model | Known bodies: strict ↑ | Fresh bodies: strict ↑ | All trials: strict ↑ | Full-horizon stable ↑ |
|---|---:|---:|---:|---:|
| Original-data control |3/6|5/8|8/14|8/14|
| Original + recovery mixture |2/6|4/8|6/14|9/14|
| Zero updates, union processors |0/6|0/8|0/14|0/14|

Fresh acceptance means four body resets with two diffusion seeds each, not eight independent bodies.
This is one optimizer seed and one language phrase, so the two-trial difference is descriptive, not
statistically established general harm from correction data. Control/mixed success sets differ:
mixed rescues400/noise390 and402/noise392 but loses390/noise392,400/noise392,401/noise390 and402/noise390.
The normalization-zero result is not a rerun of v8's original effective policy: union statistics change
its input/output mapping. Both trained arms recover some behavior relative to that shared zero.

All six control failures lose balance. Mixed has five balance failures and three stable directional
failures. In the targeted known condition392/noise390, mixed completes200ticks but ends with yaw1.19953rad
and fails the original-heading gate; control loses balance at127ticks. Mixed390/noise392 completes but
reaches yaw1.83332rad and0.31034m lateral displacement, also failing tail progress. Mixed403/noise392
completes with yaw0.96082rad and0.20698m lateral displacement. Extra survival is not successful straight
walking; no thresholds were relaxed.

### Offline fitting and runtime

These twelve chunks are sampled from training episodes, six original and six recovery. They are fit
diagnostics only. The persistence baseline repeats the actual prior target across H10.

| Model | Original6 chunks MSE ↓, rad² | Recovery6 chunks MSE ↓, rad² | Combined12 MSE ↓, rad² |
|---|---:|---:|---:|
| Mixture |0.000619681|0.008778543|0.004699112|
| Zero, union processors |0.000551201|0.015287043|0.007919122|
| Control |0.000551968|0.015821645|0.008186806|
| Persistence baseline |0.086800896|0.128471419|0.107636161|

Recovery-sample MSE is about44.5% lower for mixture versus control; original-sample MSE is about12.3%
higher. These few samples are consistent with a fitting tradeoff, not proof of its causal contribution
to closed-loop outcomes. The changed data mixture and reduced original-example exposure remain coupled.

Both formal arms have100 logged metric rows. Control starts at step10 loss0.082 and finishes at0.051;
mixture starts0.173 and finishes0.072. Final gradient norms are0.746/0.991 and LR2.5e-6. During warmup,
control's largest logged loss/gradient is0.116/1.592 at step120; mixture's maximum loss0.207 is at step70,
and its maximum gradient2.173 at step10. No NaN, OOM, disk error or traceback appears in the completed
training log. The groups sample different distributions, so their training losses are not an isolated
quality comparison. Module-wise gradients were not recorded.

Zero/control/mixed physical-evaluation inference median is0.24154/0.24929/0.25450s; p95 is
0.27813/0.28309/0.29035s. Every one of1,279 rollout inference calls exceeds200ms, while E5 covers100ms
and H10 covers200ms. Physics was paused; none of these models is validated for sustained real-time
control. Evaluation durations including loading/fit/rendering are109.19/177.05/188.74s, with peak PyTorch
allocated memory1,262,706,688bytes each. Native training logs report2.33GB allocated memory; this is not
whole-process or device peak memory. Each formal1,000-step training loop takes about4m24s.

## ml-debug result audit

The complete normalized log has3,239 lines and251,804 UTF-8 bytes. It retains resolved configs, four
native training stages, physical failures and final input checks. `result-summary.json` is derived from
raw reports/fit tensors; it is not a replacement for them. The independently verified initial audit
checks and the final result audit are stored beside `run.md`.

| Form item | Observed evidence and limitations |
|---|---|
| SHOULD budget/config | Four completed stages:2+2+1000+1000 updates, native batch8/H10/seed42, resumeFalse. Formal configs differ only output directory and selected episodes. |
| SHOULD source data | All4000 numerical frames match source, native samplers select2000/4000 rows, source MP4 hashes preserved, shared float64 statistics and H10 boundary masks checked. |
| SHOULD actual processors | Zero/smoke/final JSON-referenced state/action mean/std tensors equal union stats exactly; original v8 bytes equal zero model. |
| SHOULD physical inputs/execution | Actual input state/prior/physics align with Trace and applied prediction prefixes;20ms ticks and strict gates independently recomputed. Matched initial physics/state/prior are exact; renderer variation is explicitly retained. |
| Initial/baseline | Zero weights with union processors fail14/14. This controls normalization for the arms, not original-normalization v8 capability. |
| Dummy | Same12-chunk persistence MSE0.107636 versus0.007919/0.008187/0.004699. No random/shuffled training or persistence physical baseline; standing still fails progress gates by definition. |
| Validation and transfer | Fresh strict control5/8,mixture4/8 over four bodies; known3/6 versus2/6. Offline12 chunks are training samples, not validation. One optimizer seed, no language/visual-grounding claim. |
| LR/worst step | Warmup100 then decay to2.5e-6; finite loss and total gradient norms. Higher mixed warmup losses may reflect harder/different samples; per-module and same-batch loss are unavailable. |
| Full sample | `mixed/392-390-inputs.npz` and matching trace contain actual task-conditioned51D inputs,H10x15 predictions and200applied ticks; stable outcome fails yaw1.19953rad. |
| Surprising result / second cause | Better recovery-target fit and more stable finishes9vs8 coexist with lower strict success6vs8. Original-heading/lateral/tail gates explain the distinction. More stable finishes could reflect a different gait, not learned direction recovery. |
| Missing evidence | Optimizer repetitions, original-normalization matched baseline on all new states, broader recovery origins, more phrases, per-module gradients and real-time inference remain absent. |
| Working diagnoses | Subjective next-investigation priorities: correlated mixture/exposure35%, conditioning and recoverability25%, visual/sampling sensitivity20%, training bug5%, evaluation bug5%, unknown10%. These are not probabilities inferred statistically. |
| Evidence for/against | Recovery fitting improves while original fitting slightly worsens; correction origins sharebody392. But improvement on two new cells argues against universal mixture failure. Exact data/processor/execution checks constrain pipeline bugs; they do not prove observation sufficiency or identify a physical mechanism. |
| Cheapest distinction | Reproduce targeted392 failures without updates. A future bounded comparison could vary recovery sampling while preserving original examples, or diversify recovery origins; do not combine both changes and claim one cause. |
| Independent review | Source-first CPU reviewer recomputes gates, temporal alignment, all hashes and effective processors; final report and checks retained. |
| Cost/shorter loop | Whole pipeline20m45s; formal training~4m24s each, inference~250ms/request. Smaller diagnostic matrices shorten investigations but do not replace acceptance. |

Three ways to overclaim: a single-task policy can ignore language and still score well; correlated reset/
noise trials inflate apparent sample size; stable but turning trajectories can appear successful in a
video. Single-phrase scope, explicit grouping and strict per-tick direction gates keep those separate.
A negative result applies to this50% recovery mixture, shared-normalization warm start and1,000-update
budget. It does not rule out correction learning, lower recovery weights, broader data, a different
history representation or a different learner.

The experiment budget is consumed. No new training is queued. The initial matrix favors control, but the observed repeat variability prevents promoting it as
a reliably superior model. Neither is an expert replacement or a real-time deployment candidate.
Retain both checkpoints and all failures. The next design decision should first address repeatability and closed-loop sensitivity, then the
fit/control tradeoff and narrowly sourced recovery data before spending another training budget.

## Targeted no-training repeat

After original completion and exit0, archive the exact executed script as
`run-v12-pair.completed.py`, then add only an optional body-seed selection argument to the same evaluator.
The archived script matches its original manifest SHA. `run-v12-repeat.log/.exit-code` records four more
rollouts: both trained checkpoints, known body392, noise390/392. No optimizer update, model, processor,
threshold or production-source change is involved. These are repeats, not four new held-out conditions,
and are not pooled into the original14-trial denominators.

| Model / body392 noise | Original strict / ticks | Repeat strict / ticks |
|---|---|---|
| Control /390 |Fail /127|Fail /197|
| Control /392 |Fail /134|Pass /200|
| Mixed /390 |Fail, stable heading /200|Pass /200|
| Mixed /392 |Fail /149|Pass /200|

The repeated subset is control1/2 versus mixed2/2, compared with0/2 each originally. Each checkpoint's
12-chunk fixed-input fit metric remains exactly equal. Initial physics/state/prior also remain exact.
Actual model input images vary slightly: mixed390 first-request image differs in nine RGB channels by
one intensity level; mixed392 differs in one channel by one level. Their first predicted-chunk RMSEs
are0.000530985/0.000517193rad, maximum absolute difference0.0023590/0.0022158rad.

Both control first requests are pixel/state/prior/chunk-identical. The first prediction differences occur
later: control390 at tick10 (three image channels differ by one; chunk maximum0.0026898rad) and control392
at tick15 (one image channel differs by one; chunk maximum0.0036775rad), while state/prior still match.
This is descriptive evidence that visual/numerical sensitivity precedes some divergence, consistent with
v8/v9; it is not a new intervention proving that those particular pixels caused every physical failure.

The independent audit passed804 original checks plus300 repeat checks, including the198 unchanged
simulation/BAM/teacher witnesses. Full reports are `independent-results.md`, `independent-checks.json`,
`repeat-checks.json` and `repeat-first-requests.json`. Original and repeated failures remain preserved.

**Decision after repeat:** there is no reproducible correction-mixture benefit established here, and
there is also insufficient evidence to rank control as generally superior. The next useful experiment
should test robustness to observation variation (for example, a controlled training-time visual
augmentation comparison) and repeated physical acceptance, before interpreting small one-pass score
differences. Any new training needs its own bounded plan; none is launched by this run.
