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

Results will be recorded after the bounded pipeline finishes. A queued run is not a passing result.
