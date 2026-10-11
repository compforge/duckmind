# Forward corrective collection — 2026-10-11

The collector now obtains expert-labelled windows from states visited by the v4 learner.
This run validates collection and data conversion; it does not update the learner or establish
that corrective training improves autonomous walking. No hardware, online Runner or browser demo
was changed. The previous autonomous training budget remains exhausted.

## Question and controls

Can the existing expert produce valid forward-walking labels from learner-visited states without
resetting the body or mistaking an assisted rollout for learner success?

| Option | Predicted observation | What this run can distinguish |
|---|---|---|
| Corrective states are useful | Expert recovers where the unassisted learner fails | Demonstrations are obtainable; training benefit remains untested |
| The teacher cannot handle learner states | Teacher-only succeeds but takeover falls/drifts | Reject those windows rather than label them successful |
| Handover history/state is wrong | Teacher-through-absolute-targets differs immediately from teacher-only | Verify live physics/BAM continuity and normalized previous-action reconstruction |
| Existing observation/model or evaluation limits dominate | Corrections can succeed without improving the learner | Needs a later matched training experiment; collection alone cannot resolve it |

The selected change is collection only. Checkpoint v4, native 10-target horizon, 36-float state,
head camera, 15 absolute-radian target slots, 50 Hz physics and five-target student execution remain
unchanged. There is no new model, reward, optimizer, rule-based student or expert intervention in
ordinary autonomous evaluation. A negative collection result would reject this expert/window
combination, not imitation learning as a whole.

## Handover and data boundaries

A student-only rollout supplies exact target prefixes and unassisted continuations. Each prefix
is replayed after the same seeded initialization, verifying the full MuJoCo integration state and
BAM target. At the intervention point the same simulator continues without a reset. The expert's
previous normalized action is reconstructed from the applied target, not actual joint position.
Its old warm-up history would otherwise be stale after student execution.

Only the following expert window becomes a training candidate. Learner frames and rejected
windows remain raw evidence. All prefixes of one seed share a split. Quality subtracts takeover
position but retains the original task heading, bounds lateral/yaw drift, checks stability at each
tick and demands forward progress and upright posture over the final second.

LeRobot's pinned `rollout/strategies/dagger.py` already represents corrections as separate episodes.
We reuse that boundary and native Dataset conversion. Its interactive teleoperator/real-time
engine cannot directly drive the local ONNX expert and synchronous body, so takeover is in the
existing training adapter. No alternative dataset format or training loop was introduced.

## Physical results

| Run | Initial seeds | Requested windows | Accepted expert windows | Unassisted learner |
|---|---|---|---|---|
| Smoke | 340, 341 | 6: prefixes 10/50/100 ticks | 3/6 | Both lost balance at 78/69 ticks |
| Collection v1 | 350–355 | 18: prefixes 10/25/50 ticks | 18/18 | All six lost balance at 74/77/115/80/89/97 ticks |

In smoke, four prefixes were reached while stable: three expert windows passed; one was rejected
for yaw and lateral drift. Two 100-tick prefixes were unreachable because the student had already
fallen. All rejected cases remain recorded; the unsuccessful teacher window retains raw frames.

In v1, the six teacher-only controls passed. The 18 expert windows each lasted four simulated
seconds and advanced 0.582–0.663 m after handover. None of the 18 corresponding unassisted
continuations passed. These are three correlated prefixes per seed, not 18 independent initial
conditions or a general recovery success rate. The gates were unchanged after smoke.

Every replayed student prefix had maximum integration-state error 0.0. Handover changed neither
physics nor the applied target. In the v1 teacher-through-targets control, immediate target error
was 5.96e-8 rad; maximum difference over the subsequent 80 ticks was 3.04e-5 rad, with both
trajectories stable and practically identical terminal displacement.

The first smoke attempt rejected a long-rollout comparison at an overly strict 1e-4-rad tolerance
(maximum difference 1.70e-4). Absolute-to-relative float32 inversion loses a few bits and contact
dynamics can amplify them. The corrected gate preserves exact instantaneous physics continuity,
checks the next prediction at 1e-6 rad and requires sustained stability; longer target differences
remain diagnostic output. The original failed log is retained.

The first artifact-image equality check also failed: EGL re-rendering differed by up to two uint8
levels in a small fraction of pixels despite identical integration state and proprioception.
The raw-image comparison allows three levels; physical-state and label checks retain strict
alignment. This is distinct from lossy MP4 encoding in the native dataset.

## ML-debug audit

| Check | Evidence / limitation |
|---|---|
| Run configuration and full traces | `corrections-forward-v1/manifest.json`, full 92-line `.log`, per-window `.json` and `.npz`; config and source/checkpoint hashes are recorded |
| Before/after baseline | Same v4 learner, all six unassisted runs fail; teacher-only 6/6 and assisted windows 18/18 pass the same quality gates |
| Null/shortcut | A stationary correction fails forward/tail gates; moving sideways or briefly moving then stopping also fails tested gates. An assisted success cannot count as learned success |
| Sample alignment | Actual-artifact test checks all saved windows: first observation/previous target match student handover, subsequent previous targets equal preceding expert labels, state/action shapes and finite values checked |
| Loss, LR, module gradients | Not applicable: this run performs no optimization. Earlier v4 loss does not diagnose the cause of walking failure |
| Independent review | Reviewer identified stale expert action history and warned against reset/heading redefinition. Implementation review caught a terminal-step failure reason bug, which was fixed |
| Competing causes | Working hypothesis only: coverage mismatch 45%, model/history limitation 25%, pipeline/evaluation issue 20%, unknown 10%. These are subjective planning weights, not measured probabilities; multiple causes can coexist |
| Evidence against simple explanations | Teacher recovery shows some learner states are recoverable; it does not exclude observation limitations or prove more data will improve the model |
| Cheapest next discrimination | Fixed update budget and identical initial checkpoint: original data versus original plus accepted corrections; fresh unassisted seeds, same language/horizon and metrics |
| Latency | Student median measured inference 253 ms in this collector, still longer than its 200 ms chunk; physics pauses. Timing differs from prior runs and is not a latency optimization result |
| Missing evidence | No post-correction training, fresh learned-policy success, true real-time execution or visual grounding result |

Collection demonstrates a necessary condition for this data strategy, not sufficient evidence for
expert replacement. Rejecting bad late windows biases data toward recoverable states; coverage of
more severe failures and stopping an unsafe real robot are separate problems. Frozen seeds and
checkpoint plus exact prefix replay make within-run comparisons strong, but the small correlated
sample, heuristic gates and paused physics limit generalization.

## Validation and artifacts

Local training format/lint pass; CPU tests pass with the artifact test skipped when no raw data is
provided. On devbox, all nine tests pass against both smoke and v1 raw artifacts. Native conversion
and action-window validation completed with exit code 0. The native dataset contains 15 training
episodes / 3,000 frames and three evaluation episodes / 600 frames. Its loader was checked at
the first, middle and final frame of every episode: 10×15 action chunks exactly match raw labels,
terminal padding never crosses the episode boundary, state/text match and video frames decode.
The actual-artifact test passes against all raw windows, including smoke rejects.

Final manifest SHA-256: `111d31c0979613992f84099e8a25056c4113babd3ceda8a119a1144b4f509aa3`.
Native audit SHA-256: `a11c7aa78ae02ad7d35a3ed0177ce2673cff867e75375538214cebdf9973ac36`.

Paths are relative to the devbox training directory
`~/rustprojects/compforge/duckmind/.worktrees/vla-training/training/`:

| Evidence | Path |
|---|---|
| Failed first handover threshold check | `artifacts/corrections-forward-smoke.log` |
| Smoke data, rejects and controls | `artifacts/corrections-forward-smoke2/` |
| Final raw windows and controls | `artifacts/corrections-forward-v1/` |
| Collection log and exit status | `artifacts/corrections-forward-v1.log`, `.exit-code` |
| Raw artifact regression | `artifacts/corrections-forward-v1-tests.log` |
| Native datasets | `artifacts/dataset-corrections-forward-v1/{train,eval}` |
| Native action-window audit | `artifacts/audit-corrections-native.py`, `artifacts/audit-corrections-v1.json` |
| Conversion/audit exit status | `artifacts/corrections-forward-v1-validation.exit-code` |

Models and raw data stay outside Git. Usage is in [training/README.md](../../training/README.md).
