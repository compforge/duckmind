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

Pending. Preparation and the replay control are complete; training and learned-policy diagnosis are in
progress. This document is not evidence of successful single-trajectory fitting or physical control.

Artifacts are under `training/artifacts/`: `diagnostic-v8-preflight`, `audit-v8-single-data.json`,
`audit-v8-single-processors.json`, `diagnostic-v8-{zero,final}`, `smolvla-v8-single-zero`,
`smolvla-v8-single-smoke`, `smolvla-v8-single`, and `evaluation-v8-single-transfer.json`.
Models, data, full traces and operational logs remain outside Git. Automatic follow-up is bounded to
finishing diagnosis and result delivery; it must pause on delivery or user stop.
