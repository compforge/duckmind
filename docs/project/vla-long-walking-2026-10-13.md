# Longer walking demonstrations and late recovery

Following [v16](vla-tracking-2026-10-12.md), the user authorized another bounded learning
experiment. The immediate question is whether longer expert trajectories and recovery from
later learner states improve sustained walking. This remains fixed-image, language-conditioned
motion on one Microduck configuration; no visual or cross-embodiment claim is intended.

## Data intervention

The old-data control is all seventeen v14 union episodes,3,400 frames. The candidate addition is:

- Ten600-tick/12-second native expert trajectories, seeds242–251, with the original sampled
  command and original episode text. Their first200 body/prior/action frames must match the
  original demonstrations exactly. These2,000 repeated frames increase original-example
  exposure; only the later4,000 frames extend these trajectories.
- Three400-tick/eight-second recovery windows. Select the three latest v14 mixed failures with
  distinct body seeds:392-390-0 fails at391,422-392-0 at364,411-390-0 at331. Replay every action
  from reset, verify the entire original failure exactly, then replay again and hand over to
  the expert50 ticks before failure: prefixes341/314/281, all after four seconds. Preserve
  physics, actuator targets and actual previous target through `resume_teacher`; only expert
  actions after handover become labels.

Keep every candidate and failure. Eligibility requires the full requested duration, all existing
balance checks, finite body/prior/action values and positive integrated horizontal heading-frame
forward displacement. This is a coarse training-data filter, not a new task-success benchmark.
Historical strict quality, including original heading, is recorded independently even when false.
The gate to proceed is at least8/10 eligible long and2/3 eligible recovery episodes; failure stops
this experiment rather than weakening criteria. Task semantics are body-relative forward walking,
matching the teacher's command. No rejected historical sample is silently relabeled.

The native LeRobot dataset starts with the unchanged v14 episodes, then appends eligible samples.
Images are all black256×256 RGB; state51 contains the actual previous target; actions are15 absolute
joint targets. Native episode boundaries, H10 padding, all numerical rows and task/index/time
preservation are checked, along with every decoded image and float64 episode/global statistics.
The model adapter and robotd execution contract do not change.

## Paired learning and normalization controls

Both arms start from v14 mixed step1,000 with fresh optimizers, batch8, seed42 and H10/E5.
Control selects native episodes0–16; mixed selects the full new union. Both use the same union
normalization, including the correction distribution's summary. Each arm has two discarded smoke
updates followed by1,000 formal updates: **2,004 updates total**, with no automatic expansion.

Evaluation includes four models on the same initial-state/inference-noise matrix:

1. Original v14 mixed weights and original processors, preserving the prior baseline.
2. Identical weights with new union processors and zero optimizer updates.
3. Original-data continuation using the new processors.
4. Long/recovery mixture continuation using the same new processors.

The source model is copied across filesystems for the zero control and its weight SHA is checked.
Smoke models never initialize formal training. Full evaluation occurs after both formal runs;
its outcomes cannot change either training recipe. Effective processor tensor checks happen
before and after training. Native trainer/checkpoint behavior remains unchanged.

Mixed changes episode lengths, late-state coverage, original-prefix duplication, language mixture
and sample exposure. At equal updates, it sees fewer examples from each original episode. Therefore
the contrast estimates this complete data-mixture recipe, not the isolated effect of longer
sequences or corrections. A stationary policy can generalize past training episode duration;
the prior4s/late-failure correlation is not proof of causality.

## Predeclared evaluation

Known body seeds392/422/411 are the recovery sources. Fresh body seeds440–443 had no occurrence
in the30 existing diagnostic reports scanned for seed/body_seed; the inventory is retained.
Cross each body with inference-noise390/392: fourteen trials per model, one training seed. Four
fresh bodies are not eight independent bodies, and the known cases are not held out.

Each learned rollout executes up to600 ticks/12s, with no expert takeover. Save all actual
state/prior/black-image inputs, predicted chunks, inference latency and raw physics/metrics.
All models must start from identical physical state/prior. Report balance and historical strict
quality at4/8/12s; shorter horizons are nested prefixes, not separate trials. Report measured
heading-frame speed, original-heading displacement and yaw alongside stability to expose a
stationary shortcut. Compare velocities over matched durations and mark survivor-selected subsets.
Physics pauses while predicting; this experiment does not establish real-time control.

## Storage, provenance and execution

The data disk has about5.2GiB free while the root disk has about64GiB. New artifacts live under
`/srv/duckmind-training/v17`, exposed at the usual worktree path through
`training/artifacts/experiment-v17`. Existing datasets/checkpoints are retained; nothing is pruned.
Preparation requires at least14GiB free on the output filesystem. Cached base models remain on
the data disk; new checkpoints and evidence go to the root disk.

Scripts are `training/artifacts/run-v17-long.py` and `run-v17-pair.sh`. The experiment directory
owns collection log/exit code, recovery traces, native dataset, data audit, frozen source hashes,
run stage/log/exit code, all four model reports and the independent audits. Formal execution is
smoke-control, smoke-mixed, train-control, train-mixed, evaluate-baseline/zero/control/mixed, then
input-hash verification. The experiment is not complete until physical evaluation and result
review finish. No hardware or additional compute purchase is involved.

## Collection outcome

Collection completed with exit0:10/10 long trajectories and3/3 late recoveries satisfy the
predeclared coarse eligibility filter, adding7,200 rows. Of these,2,000 duplicate original
prefixes,4,000 extend their tails and1,200 are late recovery labels. Historical strict quality
passes only1/10 long trajectories and0/3 recoveries; nine long trajectories fail heading, and
all three recoveries fail heading plus original-frame lateral drift. These are demonstrations
of sustained body-relative walking, not proof of straight-line navigation.

Independent collection review passed591 checks: original200-frame state/prior/action and text,
sampled commands, full failed-trajectory replay, first-failure and handover indices, unchanged
physical state/targets through handover, and all recorded provenance. Raw rejected strict outcomes
remain in the report. The v17 union therefore has30 episodes and10,600 frames. At8,000 formal
sample draws, control sees about2.35 passes over its3,400 frames; mixed sees about0.75 over10,600.
Duplicated prefixes alter the weight of original rows; this is not a pure episode-length ablation.

## Launch status

Native preparation completed with exit0. Independent data audit passed11,692 checks, including
all numerical/task/time records, float64 episode/global statistics,150 native boundary samples,
all10,600 decoded black images, six effective zero processor arrays and weight identity.
`independent-collection-audit.json` and `independent-data-audit.json` retain the checks.

The fixed-budget launcher started on2026-10-13 around09:46 Asia/Shanghai, PID3597970.
Both smoke runs completed two optimizer updates and ten physical ticks. Independent review
checked 54 raw-evidence assertions: six effective processor tensors, checkpoint identity, black
inputs, state/prior alignment, predicted-to-executed targets and recomputed fit metrics. Both
short rollouts remained stable; strict success is false because 0.2 seconds cannot satisfy the
4/8/12-second duration gates. Both formal arms subsequently completed exactly 1,000 updates. `run.stage`, `run.log` and `run.exit-code`
are authoritative. Scheduled follow-up is paused after final audit and delivery; no further training is authorized by this run.
The launch checks establish pipeline integrity; physical results are reported below.

## Evaluation output collision and bounded recovery

Post-launch review found an orchestration defect: preparation creates `experiment-v17/zero`
as a checkpoint directory, while the frozen launcher later uses the same name for evaluation
output. Its exclusive output-directory creation failed before zero-model inference. This
does not affect training or baseline evaluation, and neither running script is modified.

`training/artifacts/run-v17-eval-resume.sh` continued evaluation after the original launcher exited with code 1 at 02:05 UTC.
It requires the original launcher to have exited at `evaluate-zero` with code 1 and the expected
`FileExistsError`, unchanged input hashes, both formal checkpoints at exactly 1,000 updates,
and a complete baseline report with the fourteen predeclared unique cases. It also refuses
existing destination directories. After those gates, it evaluates the unchanged zero checkpoint
into `zero-eval`, then control and mixed, and checks hashes again. No optimizer update is possible
in this continuation. Preserve the original failure and separate `resume.stage/log/exit-code`;
follow-up must not restart the original launcher or overwrite prior results.

## Completed result

The evaluation-only continuation completed at **2026-10-13 02:29 UTC** with exit code **0**.
The original failed launch remains recorded with exit code 1. No training was repeated: two
smoke runs of two updates and two formal runs of 1,000 updates total **2,004 updates**.
Independent training audit checked 2,203 assertions, all optimizer state was finite, effective
processors matched the union, and source/data/model provenance checks passed.

**The mixture improves sustained balance in this evaluation, but does not solve straight walking.**
All rows use fourteen paired trials; 4/8/12-second outcomes are nested prefixes. Stability alone
means completing the window without violating the existing balance checks. Strict success also
requires forward progress, bounded original-heading and lateral drift, and end-window checks.

| Model / evidence | Stable 4s ↑ | Stable 8s ↑ | Stable 12s ↑ | Strict 4s ↑ | Strict 8s ↑ | Strict 12s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| *Original v14 weights + original stats (`baseline/report.json`)* | 13/14 | 7/14 | 5/14 | 11/14 | 4/14 | 0/14 |
| *Same weights + union stats (`zero-eval/report.json`)* | 13/14 | 9/14 | 7/14 | **12/14** | **6/14** | **4/14** |
| Original-data training (`control/report.json`) | 11/14 | 8/14 | 6/14 | 7/14 | 0/14 | 0/14 |
| Long/recovery mixture (`mixed/report.json`) | 13/14 | **12/14** | **12/14** | 11/14 | 2/14 | 1/14 |

On the four fresh bodies crossed with two inference-noise seeds, stable 12-second outcomes are
4/8 baseline, 4/8 zero, 3/8 control and 7/8 mixed. Strict outcomes are 0/8, 2/8, 0/8 and 1/8.
Known-body stable outcomes are respectively 1/6, 3/6, 3/6 and 5/6. These are four fresh physical
initial conditions, not eight independently trained policies; only one optimizer seed was used.

Of the twelve stable mixed rollouts, eleven fail the original-heading condition; two also fail
lateral-drift limits. All twelve pass forward-progress and final-window forward-progress checks.
Thus the dominant remaining issue is direction retention, rather than an apparent stability gain
from simply standing still. The two unstable cases remain in the denominator and all raw traces
are retained. Zero-update normalization alone changes outcomes substantially, and has more strict
12-second successes than either trained arm. This is not evidence of overall command replacement.

Median inference latency is 252.7 / 246.1 / 253.6 / 253.9 ms for baseline / zero / control / mixed;
p95 is 297.4 / 288.6 / 300.7 / 299.5 ms. Five executed targets cover only 100 ms. Because physics
pauses for inference, these results do not establish a real-time robotd deployment.

Offline fit on twelve old training chunks yields MSE 0.000603 / 0.000662 / 0.000642 / 0.000536
rad², versus persistence 0.094813. This is a training diagnostic, not a held-out score; lower
error does not override the physical heading failures. Training losses likewise come from
different sampling distributions and cannot rank task competence.

## Research decision

Keep the mixed checkpoint as a sustained-balance candidate and retain the zero checkpoint as a
useful direction-sensitive comparison. Do not replace the existing expert or claim language,
vision or real-time capability. No further training is launched automatically.

The next experiment should distinguish **body-relative forward walking** from **holding the
initial task heading**. Long expert demonstrations here usually fail the latter condition too;
more of those examples need not teach straight-line motion. The policy observes joint state,
angular velocity, projected gravity and the previous target, but not accumulated task-relative
yaw error. That observation boundary is a plausible contributor, not a proven diagnosis.
Before another training run, establish a heading-consistent expert/data target and test whether
explicit task-relative heading observation is necessary. Keep normalization and update budgets
controlled; do not attribute the present mixture result solely to episode length or corrections.

All raw evidence remains under `training/artifacts/experiment-v17`, including both launch logs,
checkpoints, source hashes, every trace and actual model input. Independent collection, dataset,
smoke, training and evaluation audits are retained beside the results.
