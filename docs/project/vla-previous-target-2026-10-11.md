# Previous applied targets: dataset and adapter preparation

The new dataset adds one previous applied joint target to the VLA input. It tests a specific
missing state signal; it does not claim that one-step history resolves the walking failures.
No formal training or capability comparison belongs to this preparation run.

## Input and ownership

- Input: task text, head-camera RGB, 36 body values, then 15 previous absolute targets (51 values).
- Output: 10 future frames × 15 absolute-radian joint targets, sampled at 50 Hz (200 ms).
- Dataset: `training/artifacts/dataset-v7-forward-previous-target/{train,eval}` on the devbox.
- Raw source: existing `raw-v2`, unchanged. Forward train seeds 242–251 (10 episodes/2000 frames),
  eval seeds 252–253 (2 episodes/400 frames), same episode split as the original forward task.
- Training, offline evaluation and synchronous MuJoCo rollout share `policy_state`. Collection
  already recorded the applied target before executing the next expert action.
- `previous_target[t]` equals `action[t-1]` inside an episode. The first frame uses the actual
  warmup/intervention target; neither zero filling nor measured position is an equivalent input.
- Partial chunk execution feeds back the last frame actually applied, not the predicted tail.

`Observation.previous_target` extends the model request as an optional field. A 51-state checkpoint
requires it; a 36-state checkpoint continues to use its original input. The generic robotd Runner
still lacks applied-target feedback and camera acquisition, so this is not a deployed online
robotd controller. Older strict-schema services must be upgraded before clients send the new field,
including a null field from default serialization.

## Reproduction and artifacts

`training/artifacts/run-v7-prepare.sh` owns conversion, data audit, 20-update smoke training,
`Policy.infer` checks and a seed242 synchronous physics smoke. It preserves its source hashes,
log, stage and exit code under the same prefix. No automatic follow-up or long training was started.
The model initializes from `smolvla-v6-forward/checkpoints/002000/pretrained_model` and writes
`smolvla-v7-previous-target-smoke/checkpoints/000020/pretrained_model`.

```bash
cd training
duckmind-dataset --raw artifacts/raw-v2 \
  --output artifacts/dataset-v7-forward-previous-target \
  --tasks forward --include-previous-target
duckmind-train --dataset artifacts/dataset-v7-forward-previous-target/train \
  --model artifacts/smolvla-v6-forward/checkpoints/002000/pretrained_model \
  --output artifacts/smolvla-v7-previous-target-smoke \
  --steps 20 --batch-size 8 --chunk-size 10
```

Dataset conversion repairs both numerical per-episode and global stats with the native LeRobot
float64 calculator; saved frames remain float32. Near-constant gravity previously suffered
float32 second-moment cancellation. Native split is checked against direct float64 statistics,
so it cannot silently restore the old inaccurate episode statistics. Train and eval stats stay
separate. Head video is re-encoded from the original RGB, whereas v6 had a native split/re-encode;
future paired comparisons should use a common decoder/source and normalization controls.

## Verification

Data audit `audit-v7-data.json`: all 2400 state/action rows exactly match their raw source plus
previous-target concatenation; every interior previous target matches the preceding label;
all episode and aggregate numerical statistics match direct float64 calculations. Native split
of train episodes 0 and 9 retains correct statistics. Input split seeds remain disjoint.

Local gates: training lint, 9 tests passed with 2 environment/artifact skips; policy lint including
mypy, 11 tests passed with 4 robotd-binary skips. Linux complete training tests: 10 passed, 1 artifact-dependent skip (including the native conversion regression).
The first native test attempt failed because the lightweight `meta.episodes` view omits stats;
the corrected test reads the actual persisted episode Parquet statistics. The original failure
is retained in the session; no dataset or algorithm failure was hidden by changing a threshold.

Preparation completed at 2026-10-11 15:31:14 UTC, exit 0. Interface audit
`audit-v7-interface.json` verifies 51-state → 10×15 inference, missing-target rejection, exact
first warmup target replay, and frame4 feedback after applying five predictions (not frame9).
Initial expansion compares every tensor: original weights are unchanged and only 15 zero
columns were appended. `audit-v7-processors.json` verifies the saved pre/postprocessor state/action
mean/std exactly match new train statistics after float32 conversion. Independent review repeated
that comparison. Old 36-state inference also produced 10×15 without a previous-target argument
(`evaluation-v7-compat36.json`).

The single smoke physics trial is **not successful**: seed242 with “向前走”, executing five frames
per inference, became unstable after 59 actual control ticks (1.18 s). It moved 0.202 m forward;
terminal projected gravity z was −0.630, and strict `forward_quality.accepted=false`.
`evaluation-v7-prepare-smoke.json` preserves the full trace. This is not an efficacy comparison or
evidence against the history hypothesis; only 20 smoke updates were run.

Native logs contain 554 lines after CR expansion (43,853 characters). Logged step10/step20
losses are 0.122/0.127 and global gradient norms 1.709/1.538. No improvement is inferred from
these two finite smoke measurements. The 20 updates/checkpoint took about 14 s after loading;
logged training allocated memory reached 2.33 GB (not whole-process peak VRAM). Conversion took
37 s, data audit 8 s, model smoke/load/save 54 s, interface/load 35 s and physics/load 38 s.
Two held-out predictions yielded MSE 0.003681 rad² and median latency 1.074 s including cold
inference. This tiny cold sample is not a steady-state latency benchmark. A TorchCodec shared-library
warning during native split fell back to PyAV; all subsequent stages completed. Explicit train and
evaluation decode uses PyAV.

## ML audit and next decision

Question: can we supply the same, causally valid previous target during training and inference?
The only learned-input change is one applied-target vector. Alternatives left for later include
multi-step observation/action history, more recovery data and objective/architecture changes.
This preparation cannot determine which improves balance or language generalization.

| Audit item | Evidence / limit |
|---|---|
| SHOULD: new dataset, original split, 51 input / 15 action | `audit-v7-data.json`, all 2400 rows checked. |
| SHOULD: no next-action leakage | Exact previous-to-prior-label equality on every interior frame; first-frame warmup replay. |
| SHOULD: coherent normalization | Direct float64 per-episode/global checks and native split check. |
| Init control | Every original weight is preserved; only 15 zero columns appended to state projection. Recomputed statistics can still change behavior. |
| Null / persistence / baseline comparison | Not run as an efficacy comparison. Copying the previous target could lower short-horizon MSE without walking; physical closed-loop comparison remains required. |
| Learning schedule | 20 smoke updates, batch8, seed42, H10; VLM frozen, Action Expert and state projection trainable. This is behavioral cloning / flow matching, not new RL training. |
| Full sample | `audit-v7-interface.json` retains the complete request, JPEG and output chunk. |
| Gradient evidence | Native global gradient norm only; no per-module causal diagnosis is inferred from this smoke. |
| Competing diagnoses of earlier walking failure | Prior working hypotheses, not estimates from this smoke: missing temporal information 35%; coverage/optimization 35%; adapter/evaluation issue 15%; unknown 15%. Input alignment audit narrows one adapter concern but does not prove the temporal hypothesis. |
| Independent review | Reviewer found no blocking code issue; requested infer/missing-field/partial-execution and native split checks, plus the corrected stale documentation paragraph. |
| Cheapest next comparison | Paired equal-budget 36-state control vs 51-state model, common starting model and sources, plus zero-update normalization control. Keep task success, balance, language and latency separate. |
| What remains unknown | Whether this input makes walking more reliable; runtime latency and integration with robotd; sufficiency of one-step versus longer history. |

The next formal run should start from the recorded initial model, not silently reuse the smoke's
20 updates. The old v6 dataset/checkpoint remains available as a reference. Do not claim an
improvement from this short pipeline validation or substitute expert interventions for learned
control. Preserve failure traces even if the model still falls over.
