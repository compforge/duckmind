# Single-trajectory fitting and closed-loop diagnosis

The user authorized this bounded diagnostic after the previous-target comparison failed to establish
reliable forward walking. The question is whether the policy can fit a single successful expert episode,
and where its own physical rollout diverges from that episode. This is supervised imitation learning
with the existing native SmolVLA flow-matching objective, not reinforcement learning.

## Experiment

The source is raw-v2 `0000-forward`, seed242, task “向前走”, 200 frames at50Hz. Native LeRobot
`split_dataset` selects episode0 from `dataset-v7-forward-previous-target/train` into the new
`dataset-v8-single/train`. All200 state51/action15 rows equal the original raw state36 + previous target15
and action. Float64-derived mean/std are checked against all rows. Three decoded video samples have
mean absolute pixel errors0.006222,0.006702,0.007002 on the0–1 scale against raw images; native splitting
can re-encode video, so image identity is not assumed.

Initialization is v7 previous-target step2000. A zero-update checkpoint keeps identical model bytes
(SHA256 `5296888d23445118ee9c67f4627238f9bc4253b2e5d97bb9ed8400d0ba47580f`) with new subset normalization.
Zero, smoke and final effective state/action statistics must match this same subset. Zero versus source
predictions on9 identical raw inputs quantify normalization effects separately from weight updates.

Budget: **2 discarded smoke updates +1000 formal updates**, batch8, H10, optimizer seed42, fresh optimizer.
Formal training starts from zero, never from smoke weights. The native frozen-VLM/trainable Action Expert
and state projection configuration is unchanged. There is no automatic extension beyond1002 updates.

`training/artifacts/run-v8-single.sh` owns preparation, smoke training/physics, zero diagnosis, formal
training, final diagnosis and two development-state trials. It started at2026-10-12 02:59UTC
(10:59Asia/Shanghai), launcher PID3517943. Status and evidence are in the matching `.stage`, `.log`,
`.exit-code`, `.inputs.sha256` and `.md` files. Source and running script remain unchanged.

## Controls and measurements

Before training, original targets were replayed twice through the existing `Simulation` and `Trace`.
Both runs reproduced every state36 row and previous-target15 row exactly and passed the existing strict
forward criterion for200 ticks. Both moved0.564347m forward with-0.056168m lateral displacement and
-0.327600rad terminal yaw. Tiny render differences remained: raw0–255 image MAE8.29e-6 and2.83e-5.
This validates this target replay, not arbitrary policy rollouts or pixel-exact rendering.

Zero and final diagnosis use the same procedure:

- Predict on all200 native decoded training inputs: first-action MSE, per-joint RMSE, largest absolute
  error, and full-chunk MSE for the191 samples that have all10 future targets. No fabricated tail labels.
- Every5 frames, predict again on the original raw image with identical state, prior target and diffusion
  noise. Report40 paired prediction differences and image differences separately from target error.
- Use Torch seed`2420000+tick` before each prediction, including closed-loop predictions at common ticks.
  This pairs diffusion noise; it differs from older evaluator runs that seed only at rollout start.
- Run seed242 closed-loop with1 and5 executed actions between observations, without expert takeover.
  Preserve every applied action/physical trace and every predicted chunk with the actual image, state
  and previous target used for that prediction. `Trace.step` renders again; use `predictions-*.npz` for
  actual model-input image analysis, not the second rendered image stored by Trace.
- Report strict forward result, balance, direction, actual ticks (`len(trace.metrics)-1`), and distance to
  the time-aligned expert state in separate position/gyro/gravity/velocity units. Reference distance alone
  does not prove distribution shift or its causal role.
- Evaluate final on seeds243/380 with the existing evaluator. These are reused development states, and243
  was seen in pretraining. Neither is a fresh holdout, and this single-task experiment cannot establish
  language or vision use, generalization, or a replacement motion policy.

Persistence (repeat previous applied target) supplies an offline null. The unchanged existing strict
physical criterion is used; no acceptance threshold is chosen from this run. Physics pauses for inference,
so even successful closed-loop trials do not establish real-time feasibility.

## Decision criteria and review

| Candidate | Distinguishing evidence |
|---|---|
| Incomplete fitting or insufficient conditioning | Poor predictions even on native expert inputs after single-episode optimization. |
| Visual input mismatch | Good native-input predictions but worse raw-image predictions under paired noise/state. |
| Feedback sensitivity or missing recovery coverage | Accurate expert-state predictions but divergence when the model controls its own observations. |
| Evaluation/adapter issue | Failed raw-target replay, inconsistent effective normalization, or inconsistent saved inputs/action timing. |

These are competing explanations, not causal conclusions. Initial subjective priorities were
fit/conditioning35%, feedback30%, evaluation/input variation20%, adapter5%, unknown10%.

An independent read-only reviewer checked the shell/Python syntax, native split/statistics, zero
processors, smoke isolation, future-target indexing and noise pairing, and found no reason to stop.
They identified an untriggered empty-trace diagnostic branch: rejection before the first applied action
can make reference-distance array shapes invalid. If triggered, retain its failure and fix only diagnosis,
reusing checkpoints without additional training. The actual rejected prediction/input is saved first.

## Results

The bounded pipeline completed at2026-10-12 03:14:33UTC (11:14:33Asia/Shanghai), exit0, using exactly
1002 updates including the discarded smoke. **Single-trajectory target fitting improved substantially;
reliable closed-loop replacement of the expert is still unproven.** No further training was started.

| Training-episode metric ↓ | Zero, subset statistics | Final1000 |
|---|---:|---:|
| First-action MSE,200 frames,rad² | 0.000311658 | 0.000158463 |
| Full H10 MSE,191 complete overlapping chunks,rad² | 0.001064854 | 0.000166581 |
| Largest first-action absolute error,rad | 0.253054 | 0.124403 |
| Native paired first-action MSE,40 frames,rad² | 0.000248786 | 0.000147502 |
| Raw-image paired first-action MSE,same40 frames,rad² | 0.000252630 | 0.000144010 |
| Raw/native prediction RMSE,same noise and state,rad | 0.00372649 | 0.00321947 |

First-action MSE falls49.2%; full-chunk MSE falls84.4%. These are in-sample, overlapping predictions,
not200/191 independent trials. All15 dimensions include the constant mouth target. Repeating the previous
target gives first-action MSE0.0123318. An offline win against this null demonstrates fitted targets,
not language understanding, visual grounding or reliable physical control.

Both models' maximum first-action error occurs at tick12, right_knee: expert target-0.197474rad,
zero+0.055580rad, final-0.073071rad. Error falls from14.5° to7.1° but remains nonzero during the startup
transient. The model has not perfectly memorized this demonstration. Native and raw-image first-action
errors are close under paired state/noise; this sample does not identify video decoding as the main cause.

The9 old-versus-subset-statistics predictions, with identical v7 weights, differ by0.006989rad RMSE and
0.036922rad maximum. Therefore comparing v7 directly to final would mix normalization and optimization.
All reported zero/final fitting comparisons use identical subset statistics.

### Initial physical trials

| Seed242 condition | Strict result | Executed ticks | Forward,m | Lateral,m | Terminal yaw,rad |
|---|---|---:|---:|---:|---:|
| Zero, replan every1step | Fail: heading | 200 | 0.50823 | -0.17400 | -0.81070 |
| Final, replan every1step | Fail: heading | 200 | 0.55718 | -0.04528 | -0.51290 |
| Zero, replan every5steps | Pass | 200 | 0.53645 | +0.00878 | -0.19492 |
| Final, replan every5steps | Pass | 200 | 0.53766 | -0.01679 | -0.12936 |

All four remain balanced. Strict acceptance is unchanged1/2 before/after; the5step zero trial already
passes. The final1step trial first breaches the existing absolute-yaw0.5rad criterion on the last physical
step: tick199=-0.495333, tick200=-0.512897. Zero breaches it at tick117. Report the threshold failure without
describing it as falling or proving a statistically reliable improvement.

Body states and previous targets begin exactly equal to the raw demonstration. Position error begins
after the first action (tick1 RMSE approximately0.00116rad zero /0.00103rad final). Final1step
position-reference RMSE at ticks5/10/12/20/50 is0.002799/0.006144/0.008630/0.012487/0.030995rad.
Final5step is0.003563/0.005009/0.008111/0.005771/0.023248rad. These measure divergence from the time-aligned
reference; different valid gait phases could also produce distance, so they do not alone prove an
out-of-distribution state or failed recovery.

The two initial5step development-state tests pass2/2: seed243 forward0.52633m, yaw+0.16904rad; seed380
forward0.54445m,yaw-0.30175rad. They reuse known states and do not establish generalization. The same8 held-out
raw chunks used by earlier evaluators give final MSE0.00105407rad², a separate distribution from the
single-episode fit table. Repeated runs and a matched zero baseline are reported below.

### Repetition and audit

A no-training repeat uses the exact original closed-loop body with the same checkpoint and per-tick
noise seeding. Repeated seed242 now passes both1step and5step, terminal yaw-0.307970/-0.047947rad.
The original1step heading failure is therefore not a robustly reproduced result. Alternate seed243 fails
at149ticks from instability with1step, but completes200ticks and passes with5step. These few trials support
retaining5step as a candidate execution setting, not a universal ranking or independent training repeats.

A matched native-evaluator zero baseline on243/380 gives1/2 strict:243 passes200ticks,380 loses balance at177.
Final gives2/2 on the same conditions. This is encouraging development evidence, too small and numerically
variable to establish a reliable gain. Native held-out8-chunk MSE changes0.00148195→0.00105407; these chunks
come from seeds252/253 and wording“往前迈步”, distinct from the physical trials'“向前走”.

CPU comparison locates the first visible difference between the two final1step seed242 runs. Inputs and
predictions at ticks0–8 are bit-identical. At tick9, state and previous target are still identical, but one
of196608 image channel values changes by one intensity level; predicted chunk RMSE differs0.000397191rad.
The first body/previous-target difference appears at tick10, after that action. Position differences grow
from8.808e-6rad RMSE at10 to0.055883rad at199. Rendering variation precedes the physical divergence, but
that timing alone cannot exclude GPU arithmetic variability or establish an exclusive cause.

A fixed-input control isolates that sampled prediction difference. Keep tick9 state, previous target and
noise seed2420009 fixed; alternate the two saved images, four predictions per image. Same-image outputs
are bit-identical across all four repeats, and each image's prediction exactly equals its originally saved
chunk. Swapping the sole differing image channel reproduces0.000397191rad chunk RMSE. Thus image variation
explains this first prediction difference under this setup; unexplained GPU nondeterminism is not needed
to explain this sample. It does not prove all failures have this cause or that removing visual input is
appropriate. Closed-loop robustness to small observations remains a concrete next investigation.

Saved zero/smoke/final state/action mean/std tensors exactly match dataset-v8-single statistics, and all
original source/script/manifest SHA checks pass. Prediction inputs' state and previous target equal their
Trace rows; previous_target[t] equals the prior applied action, and executed targets equal the selected
chunk prefixes. The untriggered empty-trace branch needed no repair.

The complete log contains1647 CR-expanded lines and100 formal logged measurements. Training starts at
step10 loss0.060; near the warmup peak, step120 has loss0.123, global gradient1.629 and lr9.7e-5; the peak
logged global gradient is1.666, below clip10. At step1000 loss0.039, gradient0.608, lr2.5e-6; the last10 logged
losses average0.0298. Finite losses fluctuate with minibatches/flow noise, so this does not isolate a
learning-rate cause. Per-module gradient norms were not recorded. No training exception or OOM occurred.
TorchCodec load warnings explicitly fall back to the configured pyav decoder; they did not stop the run.

The full pipeline took15m09s: prepare10s, smoke86s, zero diagnosis250s, formal stage305s, final diagnosis200s,
transfer58s. Native logged allocated training memory peaked2.33GB (not whole-process peak). Median forward
prediction on200 expert inputs is0.262s zero /0.269s final; this exceeds the100ms5step execution window and
200ms chunk horizon. Physics was paused for inference; real-time execution remains unverified.

The independent reviewer reproduced the reported MSEs from tensors, checked all exercised action/input
timing and normalization paths, and independently localized the repeated-run divergence. No exercised-path
alignment bug was found. Review and raw controls remain in the artifact audit; latent diagnostic errors
not triggered by these trials are not explanations of the observed behavior.

## Research decision

| ML-debug form item | Evidence / interpretation |
|---|---|
| Config and budget | Native SmolVLA,100M trainable/450M total parameters,state51,H10,batch8,seed42,2+1000 updates; frozen VLM,trainable Action Expert/state projection. |
| SHOULD: replay and data alignment | Two200tick target replays reproduce every state/previous target and pass strict walking. All subset numeric rows/stats match raw; video is explicitly lossy. |
| SHOULD: normalization and no takeover | Effective zero/smoke/final stats exact; immutable initial weights; smoke not used for initialization. Students use explicit targets; expert only prepares resets. |
| Null and initial demonstration | Persistence first-action MSE0.0123318; zero beats it, and its5step rollout already passes. Final full-chunk fit improvement therefore cannot be equated with newly acquired walking. |
| Full sample and worst error | All actual model inputs/chunks retained in predictions files. Tick12 right-knee residual falls0.253054→0.124403rad; exact target and outputs quoted above. |
| Training dynamics |100 measurement rows retained in run-v8-single.metrics.csv; warmup loss rise followed by decline, finite global gradients. Module-level attribution is unknown. |
| Alternative causes for low MSE | Memorization, constant language and near-repeated camera backgrounds can score well. This run intentionally tests fitting, not modality use. |
| What remains unknown | Cross-seed training variability, broader language/state robustness, noisy-input sensitivity, complete recovery coverage and real-time performance. |
| Why a causal claim could fail | Only one optimizer seed/trajectory; reused development states; physical repeats vary despite matching recorded state/noise seed. Zero already passes the matched5step case. |
| Updated working priorities | Fit/startup residual30%,feedback sensitivity/coverage30%,visual-input/feedback sensitivity20%,training/adapter bug5%,unknown15%. These are subjective triage weights, not fitted probabilities. |
| Next distinguishing experiment | Fix the checkpoint/execution schedule; compare coherent repeated rollouts and controlled tiny observation perturbations, then use a broader reserved set before more training. Robust expert-state error with fragile rollouts favors sensitivity/coverage; reproducibly large startup errors favor targeted fitting/conditioning work. |
| Other viable implementations | Longer history, improved transient coverage or a more deterministic/coherent action generator may work even though this particular state51/H10 recipe is not yet reliable. No such experiment was launched here. |

Artifacts under `training/artifacts/`: `diagnostic-v8-preflight`, `audit-v8-single-{data,processors,training-log}.json`,
`diagnostic-v8-{zero,final}`, `smolvla-v8-single-zero`, `smolvla-v8-single-smoke`, `smolvla-v8-single`,
`run-v8-single.{sh,log,stage,exit-code,inputs.sha256,md,metrics.csv}`, `evaluation-v8-{single,zero}-transfer.json`,
`diagnostic-v8-repeat-{242,243}`, `diagnostic-v8-image-control`, `run-v8-repeat.{py,log}`,
`run-v8-image-control.py`, and `audit-v8-independent.md`.
Models, data, full traces and operational logs remain outside Git. Automatic follow-up pauses after result
delivery; the training allocation is exhausted and no additional optimization is authorized by this run.
