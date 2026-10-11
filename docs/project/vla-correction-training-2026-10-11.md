# Forward corrective training — 2026-10-11

This bounded experiment tests whether adding accepted expert corrections improves autonomous
forward walking. It follows [corrective collection](vla-corrections-2026-10-11.md). The user
explicitly authorized training after that collection; this is a new budget, not an extension of
the exhausted v3/v4 automation. No hardware, browser deployment or robotd behavior changes.

## Frozen plan

Both formal arms start from the same v4 `004980/pretrained_model`, use native SmolVLA with frozen
VLM, train the Action Expert and state projection, and restart the optimizer/scheduler. Each gets
2,000 updates, batch eight, seed 42, chunk length ten at 50 Hz. Each first gets a disposable
20-update smoke run and native checkpoint reload/inference; smoke weights never initialize a
formal run. Total new optimization budget: 4,040 updates. No automatic extra training rounds.

| Arm | Training episodes / frames | Change |
|---|---|---|
| Control | 110 / 22,000 | Original v2 train set |
| Mixed | 125 / 25,000 | Original + 15 accepted correction episodes from seeds 350–354 |

LeRobot native `aggregate_datasets` merges data, videos, task indices and statistics. Correction
seed 355 and all original eval episodes are excluded. Native loaders compare the first, middle
and final frame of all 125 episodes against their source: action, state, text, video and terminal
padding. Three prefixes per correction seed are correlated, not independent demonstrations.

The native non-resume trainer replaces checkpoint normalization with training-dataset statistics.
We preserve this behavior and explicitly evaluate a **mixed-zero** checkpoint: unchanged v4
weights plus the mixed-data native state/action normalization, without optimization. Thus the experiment
compares complete training recipes; it cannot isolate corrective-state coverage from changed
normalization or forward sampling weight (approximately 9.1% to 20%). Optimizer state is not
resumed. Statistics never use eval data.

## Evaluation contract

Evaluate v4, mixed-zero, control and mixed on the same fresh physical seeds 360–365, with both the
familiar forward instruction and the reserved paraphrase. Execute five targets before each new
prediction; each trial lasts four simulated seconds unless unstable/invalid. Only initialization
uses the expert: no takeover, rule dispatch or fallback occurs during evaluated student control.

Retain legacy success for historical comparison, but use the existing collection
`recovery_outcome` as the stricter forward measure: complete execution, stability throughout,
forward displacement, limited lateral drift, original heading, continued forward progress and
upright posture during the last second. Save per-tick physical traces, including initial state.
Trace length minus one is the number of applied targets; the old `ticks` field counts attempts.
These flat-floor heuristic gates do not certify real-world locomotion or velocity tracking.

Physics pauses during inference. Report latency separately; successful synchronous motion does
not establish real-time feasibility. Six physical seeds and two texts per arm are paired small
engineering trials, not multiple independent training seeds or broad language generalization.

## ML-debug pre-run audit

| Item | Evidence / predicted discriminator |
|---|---|
| Question | Does the mixed recipe improve unassisted strict forward success relative to equal-update original-data training? |
| Main change | Data and its native normalization; model architecture, update budget and starting weights fixed |
| Options | More original updates is the control; corrective recipe is selected; new architecture/history and latency changes deferred |
| Prediction | If coverage is limiting, mixed should improve completion and strict success across fresh starts; lower offline MSE alone is insufficient |
| Alternative explanations | More forward exposure, normalization changes, observation/history limitations, or pipeline/eval bugs can explain a difference |
| Working diagnostic weights | Coverage 45%, model/history 25%, pipeline/evaluation 20%, unknown 10%; subjective planning weights, not measured causal probabilities |
| Evidence against certainty | Expert takeover success only proves labels can be obtained. No autonomous post-correction result existed before this run |
| Independent review | Found normalization and task-weight confounds; warned legacy 5 cm success permits diagonal/transient movement and attempted ticks can differ from applied steps |
| Response to review | Add mixed-zero control and per-tick trace; reuse stricter tested collection gates. Do not claim isolated DAgger benefit |
| Null/shortcut | Stationary policy fails forward/tail gates; sideways and transient movements fail tested strict gates. No new shuffled-label optimization arm is included |
| Leakage check | Only train sources aggregated; all prefixes of seed355 withheld; physics360–365 not used for collection |
| Sample check | Native merged loader compares 375 source samples including terminal H10 action padding; source and script hashes saved |
| Loss/LR/gradient diagnosis | Pending formal logs; module-level gradient norms are not provided by current native logger |
| Missing evidence | Real-time rollout, unseen task semantics, multiple training seeds, severe-state recovery and real hardware remain untested |
| Failure interpretation | Failure would reject this small-data recipe at this budget, not corrective imitation or VLA in general |

## Artifacts and execution

Paths are relative to remote `training/`. `artifacts/run-v5.sh` owns the ordered merge, smoke,
formal training and evaluation pipeline; `run-v5.stage`, `.log`, `.exit-code` record progress and
failure. `run-v5.inputs.sha256` freezes sources, scripts and input manifests. No in-flight source
files are edited. Raw data, models, native resolved configs and metrics remain outside Git.

| Evidence | Path |
|---|---|
| Merge and source comparison | `artifacts/prepare-v5.py`, `artifacts/audit-v5-merge.json` |
| Mixed native train set | `artifacts/dataset-v5-mixed/train` |
| Untrained normalization control | `artifacts/smolvla-v5-mixed-zero` |
| Trained models | `artifacts/smolvla-v5-{control,mixed}/checkpoints/last/pretrained_model` |
| Smoke models / inference | `artifacts/smolvla-v5-{control,mixed}-smoke`, `evaluation-v5-*-smoke.json` |
| Paired physical reports and traces | `artifacts/evaluation-v5-{baseline,mixed-zero,control,mixed}-{trained,held-out}.json` |

## Launch evidence

At 2026-10-11 12:45 UTC, `run-v5.sh` started on the existing L4 devbox (launcher PID 173220).
Merge passed: 125 episodes / 25,000 frames and all 375 source samples matched. A separate raw
Parquet audit verified 45 samples from the 15 correction train episodes, all seeds350–354,
with seed355 excluded (`artifacts/audit-v5-split.json`). Mixed-zero model SHA256 remains
`83f399f3f8ee957de369bbdbfe07308cc8fdb9e7b1e8886fa28d30ba7f749f5a`.

Control smoke completed 20 updates and checkpoint inference; the formal control run is active.
Native logger reports 99,884,832 trainable parameters of 450,050,016 total, batch8 and approximately
four updates/second. Local training lint/format pass; eight CPU tests pass, one raw-artifact test
skips locally because its data stays on devbox. Previous collection's actual-artifact test passed;
this run's full physical evidence is still pending.

Independent script review found no blocking issue. The zero-control state/action processors match
the native warm-start path. Image statistics differ because native training sets ImageNet stats;
current VISUAL=IDENTITY means they do not change model inputs. No pure data-causality claim is made.

## Checkpoint save failure and bounded recovery

Both arms executed all 2,000 updates, but mixed failed while saving its final checkpoint:
`SafetensorError: I/O error: No space left on device (os error 28)`. `run-v5.exit-code` is 1.
The failed `002000/pretrained_model` contains config only, not model weights. The /data00
filesystem had only 389 MiB available. All logs and the failed directory are retained.

Both arms have valid `001500` checkpoints, confirmed by successful-save log entries, 500 readable
model tensors and native `training_step.json`. The final comparison therefore uses **1,500-step
checkpoints from the original 2,000-step schedules**, selected before viewing any v5 physical
outcome. It is not a new 1,500-step training run. The full 4,040-update budget was consumed; no
updates are repeated or added. Control002000 remains available but is not the matched comparator.

One byte-identical v4/mixed-zero model copy was replaced atomically with a verified same-filesystem
hardlink, recovering about 1.2 GB without losing model content. No datasets, unique checkpoints or
other projects were deleted. Recovery details and hashes are in `artifacts/audit-v5-recovery.json`.

`artifacts/run-v5-evaluation.sh` resumes **evaluation only** with its own stage/log/exit files,
preserving the original failed run. Four checkpoints, the same six fresh seeds and two language
conditions remain fixed. Reports use `evaluation-v5-1500-{arm}-{language}.json`, with explicit
001500 paths for both trained arms. Independent review verified that actual saved train configs
differ only in dataset root and output directory, and all 70 non-image normalization statistics
per processor match within baseline/control and mixed-zero/mixed pairs. These exact comparisons
are saved in `artifacts/audit-v5-processors.json`.

Before another training run, reserve space for all checkpoint weights and optimizer states;
checking only GPU memory is insufficient.

## Final paired physical results

Evaluation-only continuation completed with exit0. All 48 planned trials are present: six physical
seeds times two wordings times four checkpoints. An independent reviewer recomputed every gate
from the stored traces using separate formulas and verified identical initial metrics within each
seed. The task texts are “向前走” and “往前迈步”. No teacher takeover occurred.

| Model | Strict success | Legacy success | Complete and stable for 4 s |
|---|---:|---:|---:|
| v4 baseline | 0/12 | 0/12 | 0/12 |
| Same weights, mixed normalization | 0/12 | 0/12 | 0/12 |
| Original data, checkpoint1500 | 0/12 | 0/12 | 0/12 |
| Added corrections, checkpoint1500 | 0/12 | 1/12 | 1/12 |

The mixed model changes behavior, but has **not improved strict forward-task success** in this
small paired test. With familiar wording its median forward displacement at termination is
0.268 m versus control 0.048 m. However, its median executed duration is only 1.47 s versus
control 1.62 s; larger displacement before instability is not stable locomotion. With the reserved
wording the median durations are 1.41 s and 2.18 s respectively. These durations include the first
unstable tick for terminated trials; they are not survival estimates beyond the four-second cap.

Forty-seven trials terminate at the first projected-gravity tilt violation (`gravity_z >= -0.65`),
before trunk height breaches the height gate. Report these as **instability / excessive tilt**,
not proof that the body had already hit the floor.

The remaining trial is mixed / held-out / seed365: it completes200ticks, advances0.275m and stays
within the balance gate, but maximum absolute yaw is0.970rad (about56 degrees), above the fixed
0.5rad bound. Its final yaw is only−0.019rad. This is exactly why a final-pose-only or legacy5cm
success check would overstate capability. The entire trace is preserved in
`evaluation-v5-1500-mixed-held-out.json`.

As a concrete paired failure, control / familiar / seed360 starts upright at height0.11655m,
gravity_z−0.99989; after76 applied targets it is at height0.09594m, gravity_z−0.59355 and total
forward displacement−0.01099m. Mixed at the same initial seed/text advances0.30062m but loses
stability after91 targets. Neither is a successful walk.

Offline eight-chunk action MSE is0.003412rad² for v4,0.003111 for mixed-zero,0.004165 for control
and0.003211 for mixed. This narrow held-out sample repeats across the two physics-language reports;
it is not16 independent samples per model and is not an all-task generalization score.
Mixed-zero even has lower MSE than mixed while still failing every task, reinforcing that offline
fit does not certify closed-loop success.

Reported median inference is324–366ms, measured on offline held-out chunks, not each live rollout
call. Replanning every five50Hz targets requires100ms, and the ten-target prediction covers200ms.
All measured medians exceed both intervals. Since physics pauses for inference, this run provides
no real-time deployment evidence.

## Final ML-debug audit and interpretation

`artifacts/audit-v5-results.json` holds per-language rows, every seed's applied steps, displacements,
failed gates and source paths, plus training-curve summaries. The original log has5,357 lines after
normalizing carriage returns. There are 404 logged optimizer observations:200 per formal arm and
two per smoke. All logged loss and total gradient-norm values are finite.

| Audit item | Observation and boundary |
|---|---|
| SHOULD merged size / split / alignment |125train episodes/25,000frames;375merged samples and45raw split samples passed; seed355 excluded |
| SHOULD same starts and budgets |Both arms execute 2,000 updates from v4, plus20smoke each; saved configs differ only in dataset root/output directory; usable paired checkpoint is1500 |
| SHOULD fresh unassisted physics |48/48 completed; trace gates independently recalculated; strict0/12 per arm |
| Baseline and zero-update comparison |v4 and mixed-zero strict0/12; normalization alone changes behavior but does not solve walking |
| Null / shortcut control |Stationary, excessive sideways drift and transient progress are rejected by existing unit-tested gates; no new random-policy/shuffled-label optimization run was performed |
| Initialization / samples |Source labels/boundaries were audited; mixed-zero preserves exact v4 model hash; concrete paired physical sample above and first predicted target arrays remain in raw reports |
| Loss and schedule |Formal control/mixed final logged losses0.062/0.051; final20logged-point means0.0407/0.0513. Both reach lr2.5e−6. Warmup/restart and different data/statistics prevent interpreting these as a causal model ranking |
| Worst-looking finite points |Both peak formal loss at step320,lr9.4e−5:control0.139,mixed0.176. Peak total gradient norms1.789/1.902; native logger lacks per-module gradients, so no module-specific diagnosis follows |
| Surprising log line |Normalized log5355 reports `No space left on device (os error 28)` during mixed final-save; explained by filesystem capacity, not loss divergence |
| Timing / memory |Formal loops about 9 min each; native logger allocated memory about 2.33 GB, not a complete device-memory peak. Offline inference remains slower than100ms replanning |
| Independent audit |Valid matched1500checkpoints and exact state/action processor stats; all 48 initial-state pairings and strict gate calculations checked; warned that apparent forward motion and terminal pose can hide instability/yaw |
| Competing learning causes |Corrective coverage/weight insufficient, model/history or action representation limits, optimizer schedule, training/evaluation bugs and unknown factors remain candidates. Prior subjective planning weights are not a posterior inferred from this negative score |
| Evidence against simple conclusions |More original updates also fail; normalization-only still fails; exact raw label and processor checks reduce some pipeline concerns. None rules out missing history or insufficient effective training |
| What is still missing |Mixed002000model, multiple training seeds, task-weight-matched ordinary demonstrations, long-horizon and real-time evaluation, visual grounding and other-task regressions |

Three misleading conclusions were checked: (1) a lower MSE can coexist with zero physical success;
(2) the one legacy pass conceals a large transient yaw; (3) more forward displacement does not
mean longer stable execution. Small correlated data, one optimizer seed and changed normalization /
task weights additionally prevent isolating a pure corrective-state coverage effect.

This tests one small corrective-data recipe, not the general viability of VLA or corrective
imitation learning. A further investigation should first isolate forward-only imitation and
observation/history sufficiency instead of blindly repeating the same mixed training budget.
No further training was launched. All 4,040 authorized updates have been consumed.

## Repeatability check and closure

A separate no-training repeat evaluated mixed / held-out seeds360 and365. Both again fail the
strict criterion. Seed360 executes61 rather than60 steps; seed365 executes185 rather than200,
so the sole legacy success does **not** reproduce. These two repeated trials are excluded from
the original 48-trial denominator. Evidence: `evaluation-v5-1500-mixed-repeat.json` and
`audit-v5-repeat.json`.

Independent review finds exactly matching initial trace metrics, but the first recorded numerical
differences occur at tick11 and tick1 respectively, initially on the order of 1e−5 rad. Small
differences amplified by closed-loop contact dynamics are a plausible mechanism, not an
established source diagnosis. CUDA inference, image rendering and unrecorded internal state
have not been isolated. Matching the recorded metrics does not establish identical complete
MuJoCo/BAM state or pixel input. Same-seed bitwise trajectory reproducibility remains unresolved.

No learning improvement or deployment readiness is established. The budget is exhausted,
evaluation and result delivery are complete, and automatic follow-up is paused. A subsequent run
needs a fresh scoped training decision, a disk-capacity budget and stronger repeatability evidence.
