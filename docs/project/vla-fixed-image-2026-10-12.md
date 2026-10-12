# Fixed-image conditioning for early language-conditioned walking

The user wants to defer camera-dependent behavior while stabilizing basic motion. This v13
experiment keeps the native SmolVLA architecture but supplies a constant black image during
both training and inference. Its result is a language-conditioned motion baseline, not evidence
of visual grounding or obstacle avoidance. The long-term task + vision + body-state interface
remains the goal.

## Question and controls

Does removing camera variation improve repeatability and strict forward walking? The preceding
[v12 comparison](vla-correction-pair-2026-10-12.md) showed outcome changes on repeated evaluations.
Small image differences could change predictions, but that did not establish the cause of every
closed-loop failure. This comparison separates image replacement from subsequent learning.

Both arms initialize from `smolvla-v12-control/checkpoints/001000/pretrained_model`, select the
same original ten forward episodes (2,000 frames), and share the true v12 union numerical
normalization. The raw arm uses recorded camera images; the fixed arm uses uint8 zeros of shape
256 x 256 x 3. Both retain state51, previous executed target, H10 x 15 absolute joint targets,
batch8, optimizer seed42, native flow matching, and a restarted optimizer/scheduler.

Each arm receives two discarded smoke updates and 1,000 formal updates: **2,004 total**.
Formal training starts from the common initial checkpoint, never smoke weights. No further
optimization is included. This is one optimizer seed, not replicated training.

The fixed image remains a present camera input. SmolVLA's internal image scaling maps black to
-1; the image encoder and image tokens remain in the architecture. VISUAL normalization is
IDENTITY and image augmentation is disabled. This experiment neither removes the visual branch
nor tests a smaller state-only architecture. Its warm start previously learned with real images.

## Native data and preparation evidence

`training/artifacts/dataset-v13-fixed-image` was rebuilt using LeRobot's native
create/add_frame/save_episode/finalize path. All twenty v12 union episodes are retained so both
roots have the same numerical statistics; both trainers select only episodes0–9. Correction
frames are not sampled, though their summary contributes to the shared union normalization.
The original root and raw trajectories remain unchanged.

Preparation passed:

- All 4,000 rows: state, action, task/index fields and timestamps equal the original values.
- All 4,000 video frames decoded to exactly black RGB, including native dataset sample checks.
- Each training selection contains exactly 2,000 frames from the same ten episodes; H10 padding
  at episode ends remains correct.
- Numerical statistics were recomputed in float64 and checked against the source. Global
  state/action statistics are then canonicalized to the source values; episode-statistic byte
  identity is not claimed because aggregation grouping can introduce rounding differences.
- Effective processor state/action mean/std tensors match the common root exactly. Independent
  CPU review also confirmed every numerical frame and source numerical statistic is finite.
- Simulation dependency hashes and initial model/data hashes are recorded. Source review found
  no blocking issue; the evaluator additionally rejects checkpoints with augmentation enabled.

The fixed root's `image-contract.json` records the required train/inference behavior. Generic
camera inference must not be used to deploy this checkpoint without honoring that contract.
No production Policy adapter or robotd execution path changes in this experiment.

## Predeclared evaluation

Both smoke evaluations execute ten ticks; strict failure is expected because full acceptance
requires 200 ticks. They verify the real loader, training, processor, inference and physics paths.

Before formal updates, the same initial weights are evaluated once with raw images and once with
fixed images: body242/392 crossed with inference-noise390/392, four trials per mode. A worse
fixed-image zero baseline is possible because its conditioning changed; it is not the result of
fixed-image retraining.

Each final arm runs twenty trials: known body392 and fresh bodies410–413, crossed with two
inference-noise seeds and two repeats. A saved artifact inventory found no prior use of410–413.
There are four fresh body conditions, not sixteen independent fresh bodies. The second repeat
reverses body order. One text phrase, `向前走`, is used throughout; this does not test language
recognition or paraphrase generalization.

Every trial runs at most 200 control ticks, replanning every five ticks and predicting ten.
The existing `recovery_outcome` strict gate is unchanged. Initial physics, body state and previous
target must match exactly across arms/repeats; initial raw image differences are reported.
Actual model input images, state, prior target, prediction chunks and complete physical traces
are saved. Fixed mode asserts black input during offline fit and every closed-loop prediction.
If repeated fixed-mode outcomes differ, inspect the first input/prediction divergence rather than
attributing it to camera variation. Twelve original training chunks provide offline fit diagnostics
with persistence as a floor; they are not held-out validation.

Report strict success, balance, heading, repeat agreement and inference latency separately.
Simulation pauses for inference, so physical success does not establish real-time execution.
No expert takeover, rule-based action dispatch, threshold change or real hardware is involved.

## Run identity and current status

Launched on the amd devbox on 2026-10-12 at approximately 06:36 UTC (14:36 Asia/Shanghai),
launcher PID233824. Completed at 06:59:56 UTC (14:59:56 Asia/Shanghai), exit0.
Both 1,000-step final checkpoints were saved; all 2,004 authorized updates are accounted for.
The source hash check passed. No additional training was launched.

Artifacts under `training/artifacts/`:

- `run-v13-fixed-image.sh`, `.py`, `.stage`, `.log`, `.exit-code`, `.inputs.sha256`.
- `run-v13-prepare.log`, `run-v13-prepare.exit-code`, `v13-seed-inventory.json`.
- `diagnostic-v13-fixed-image/run.md` and `data-audit.json`.
- `diagnostic-v13-fixed-image/{smoke-raw,smoke-fixed,zero-raw,zero-fixed,raw,fixed}/`:
  report, fit arrays, actual inputs and full traces.
- `smolvla-v13-{raw,fixed}[-smoke]`: native configs, processors and checkpoints.

The phase order is smoke/raw, smoke/fixed, both zero controls, raw training/evaluation, fixed
training/evaluation, then source-hash verification. Do not duplicate a job or modify running
scripts. Completion must include independent result auditing before claiming improvement.


## Completed comparison and decision

The fixed-image recipe made repeated evaluation exactly reproducible in this run, but did not
improve strict walking success. This supports using it as an early motion-debugging baseline,
while retaining the better-scoring raw checkpoint. Neither is promoted as a reliable replacement
for the expert or as a real-time robotd policy.

| Source | Strict success ↑ | Fresh initial states ↑ | Stable throughout ↑ | Equal repeat outcomes ↑ | Exact repeat trajectories ↑ |
|---|---:|---:|---:|---:|---:|
| [Raw final](../../training/artifacts/diagnostic-v13-fixed-image/raw/report.json) | **18/20** | **15/16** | **18/20** | 8/10 pairs | 0/10 pairs |
| [Fixed final](../../training/artifacts/diagnostic-v13-fixed-image/fixed/report.json) | 14/20 | 12/16 | 14/20 | **10/10 pairs** | **10/10 pairs** |

The fresh column covers four initial-state seeds, each evaluated with two noise seeds and two
repeats. Known392 contributes3/4 raw and2/4 fixed. These are small perturbations of the same
Microduck body on flat ground, not morphology or hardware variation. There is only one training
seed and one task phrase; the table does not establish general superiority of camera conditioning.

Zero-update controls use a different, smaller known-state matrix and must be kept separate:

| Source | Strict success ↑ | Twelve training chunks MSE rad² ↓ |
|---|---:|---:|
| [Raw zero](../../training/artifacts/diagnostic-v13-fixed-image/zero-raw/report.json) | **2/4** | **0.000464750** |
| [Fixed zero](../../training/artifacts/diagnostic-v13-fixed-image/zero-fixed/report.json) | 0/4 | 0.001194787 |

Final offline MSE is0.000435761 raw and0.000523839 fixed, versus persistence0.094813421 on the
same twelve training chunks. These are fit diagnostics, not held-out predictive performance.
Native end loss is0.049 raw/0.054 fixed; the lower offline error alone does not certify walking.

### What repeated failures explain

All ten fixed body/noise pairs have exactly equal saved effective input state, previous target,
black image, predicted chunks and physical trajectories across repeats. Three pairs fail both times:

| Initial-state seed | Inference-noise seed | Executed ticks, each repeat | Failure |
|---|---|---:|---|
| 392 | 390 | 84 | unstable before200 ticks |
| 410 | 392 | 146 | unstable before200 ticks |
| 412 | 390 | 195 | unstable before200 ticks |

Every failure violates only the completion/stability checks; heading and measured progress do not
explain these particular rejections. Raw failures occur only on repeat1:392/392 at147 ticks and
412/392 at119 ticks. All raw repeated trajectories differ eventually. The first effective
image/prediction difference precedes physical-state divergence; in three pairs the first camera
image already differs by one RGB level, yielding initial chunk max differences0.00128–0.00195rad.
This local evidence is consistent with image-sensitive predictions amplifying into trajectory
differences. It does not prove all past walking failures were caused by pixels.

Fixed conditioning removes that observed repeat variability here, but reproducible falls remain.
The next useful experiment would target these deterministic failure states, or examine the first
prediction error leading to each fall, using the fixed-image model as a reproducible baseline.
It is not sufficient to add paraphrases or assume more training steps will resolve balance.
This report recommends that direction; it does not launch another experiment or allocate updates.

### Timing and audit limits

Formal inference took a median0.24866s raw and0.25000s fixed. All1,506 formal requests exceeded
0.2s, longer than both the0.1s executed prefix and the0.2s predicted horizon. Keeping a black image
in the existing VLM does not remove its compute. Physics paused during inference; this is not
real-time deployment evidence. Zero/raw, zero/fixed and final raw/fixed evaluation stages took
72.8s,42.4s,260.6s and248.7s respectively; evaluation peak CUDA allocation was1,262,706,688bytes.
Native logged training memory was2.33GB for both arms.

Independent CPU review passed83 preflight checks and856 final checks. It recomputed all strict
quality fields from complete traces, matched executed actions to saved chunks, checked effective
processors, numerical data, hashes and all fixed input images, and confirmed final native configs
differ only in dataset root/output directory. Saved optimizer states sum to2,004 updates. No
reviewer training or GPU inference was used. Failure evidence, checkpoints and original data remain.

Audit files in `training/artifacts/diagnostic-v13-fixed-image/` include
`v13-independent-preflight.md`, `v13-independent-preflight-checks.json`,
`v13-independent-results.json`, `v13-independent-results.md`, `v13-audit-form.md`, and
`v13-training-metrics.json`/`.md`; `run.md` links the result audit. The experiment is complete;
automatic follow-up is paused after delivery rather than silently starting another training run.
