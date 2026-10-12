# Expert recovery from reproducible learner deviations

[v10](vla-crossed-seeds-2026-10-12.md) found that the selected body392 condition failed with both
sampling schedules, while the original expert walked successfully from that reset. This experiment
asks whether the expert can also recover from the learner's early deviations. It does not train.

## Predeclared collection

Replay each of v10's six saved body392 learner trajectories for 10, 20 and 30 control ticks, then let
the original expert control for up to 200 ticks / four seconds. This produces at most 18 candidate
windows. Prefixes come from both diffusion schedules and all three repetitions, but all share one
previously studied physical reset; they are correlated development data, not 18 independent states.

There are zero new VLA predictions and zero optimizer updates. The implementation reuses `Simulation`,
`Trace`, `check_handover`, `expert_rollout` and `recovery_outcome`. The ordinary correction collector
would generate new learner trajectories and assign a train/eval split, which is inappropriate for this
fixed saved-prefix diagnosis; an artifact-owned orchestration script uses its existing primitives.

Before collection, run the existing mechanistic handover check and one fresh teacher-only 200-tick
control. The teacher command is fixed to forward 0.4 m/s. Replay saved absolute targets through the
normal `Simulation.apply` path. At every prefix tick, compare the full MuJoCo integration witness with
the original trace (absolute tolerance 1e-9), and require exact applied targets. At the intervention,
measured state and previous target must equal the saved learner state.

Call `resume_teacher` without resetting physics, applied targets or original heading. It reconstructs
the teacher's previous normalized action from the last applied absolute target. Assert that handover
leaves physics and targets unchanged and that the reconstructed history matches. Correction observations
precede expert actions; learner-prefix targets never become demonstration labels.

Every candidate, including rejection or partial exception, retains trace and replay evidence. Acceptance
uses the unchanged strict recovery gate, conditional on the teacher-only control passing. Candidate
metadata is `split=development_only`; it cannot be honestly divided into independent train/eval subsets
by shuffling these correlated prefixes. No LeRobot dataset is created or old dataset modified here.

## Interpretation

Success at all prefixes makes targeted expert correction feasible at these sampled deviations. Success
only at early prefixes suggests a narrower recovery region. Failure at every prefix calls for a teacher,
handover or earlier-intervention diagnosis before new training. Expert recovery does not prove that a
fine-tuned learner can reproduce it, nor does it separate visual shortcuts from proprioceptive weakness.

The no-intervention comparison is the original failed learner trace after the same prefix. It terminates
at the first instability; it is not extended using an expert or fabricated standing actions. Its partial
length fails the same 200-tick gate. Original task heading is retained throughout, so steering into a new
direction cannot hide heading error. Synchronous rendering/physics is not a real-time runtime test.

## Evidence

Scripts: `training/artifacts/run-v11-recovery.py` and `.sh`. Output:
`training/artifacts/diagnostic-v11-recovery/` with `report.json`, `run.md`, teacher-control trace, each
correction trace and matching `-replay` trace. The source v10 report, action traces and actual request
files are hashed before replay and checked unchanged at completion; upstream simulation/BAM/teacher
hashes from v10 are verified. Runtime scripts/source hashes and exit status remain alongside the run.

## Results and next decision

Completed in 34.81 s with exit 0: **10/18 candidates accepted, 2,000 accepted expert frames**.
There were zero new VLA predictions and zero optimizer updates. All 18 candidates complete 200 ticks
and remain stable; rejection is about direction, not renewed falling.

| Learner prefix / handover time | Strict accepted ↑ | Stable and complete ↑ | Rejected conditions |
|---|---:|---:|---|
| 10 ticks / 0.2 s | 1/6 | 6/6 | Five exceed original-heading limit |
| 20 ticks / 0.4 s | 6/6 | 6/6 | None |
| 30 ticks / 0.6 s | 3/6 | 6/6 | Three exceed heading limit; two also exceed lateral-drift limit |

The exact maximum absolute heading values explain the binary table. At prefix10 they span
0.49164–0.52419 rad around the unchanged 0.5-rad limit: the five rejected examples reach
0.50268–0.52419 rad. Thus 1/6 versus 6/6 must not be interpreted as five failures to regain balance.
Prefix20 maxima are 0.35138–0.42372 rad and all pass; prefix30 maxima are 0.46528–0.69149 rad.
All three prefix30 accepts come from noise390 source trajectories; noise392's three are rejected.
These trajectories are correlated and were selected from a known failure condition, not a general
estimate of recovery probability or an optimized universal intervention time.

The fresh teacher-only control passes (0.57514 m forward in four seconds). The existing mechanistic
handover control also passes: first-target discrepancy is 5.96e-8 rad, maximum over the resumed control
is 7.22e-5 rad; its physics and applied target are unchanged at intervention. For all 18 learner-prefix
replays, the maximum physics discrepancy versus the original v10 trajectory is exactly 0.0, not merely
within tolerance. This supports that the expert was queried at the intended learner-visited states.

The original six learner traces all actually terminate unstable at ticks40–53; their recorded remaining
suffix after each prefix is only10–43 ticks. `without_intervention` retains that short suffix and an
expected200-step gate; its completeness failure is not an independently run matched-duration control.
Its observed terminal instability is real evidence of failure, distinct from incomplete recording length.

**Decision:** targeted corrective supervision is feasible. Keep all 10 strict successes as labeled
expert candidates, preserve all eight rejections as evidence, and retain `development_only` grouping.
Prefix20 offers the most consistent direction-preserving candidates in this small matrix, but selecting
it is now a development decision. Do not turn its six correlated traces into six held-out successes.

The next proposed training comparison should hold starting weights, optimizer budget, state layout and
effective normalization fixed, then compare original-data training against original-plus-accepted-correction
training. Record changed sampling weights explicitly. Use body392 for development diagnosis, retain390
as an existing regression check, and predeclare new reset conditions for final acceptance. Do not replace
all original data with these correlated corrections. No conversion, training or learner improvement has
yet occurred; this run establishes data feasibility, not a repaired policy.

## ml-debug audit

The full74-line log contains all18 candidate outcomes, and all raw successful/rejected trajectories are
retained. Preflight review caught and corrected an exception-reporting bug before launch: an assertion
failure now forces `accepted=False` even if earlier metrics passed. No runtime source changed afterward.

| Form item | Evidence / limitation |
|---|---|
| Config and log | Fixed body392,6 saved failed v10 trajectories, prefixes10/20/30,200 expert steps, fixed0.4m/s command; log `complete accepted=10/18 frames=2000`. |
| SHOULD exact replay |18/18 saved action-prefix replays reproduce every recorded integration state and applied target; maximum error0.0. State and actual previous target at handover are exact. |
| SHOULD unchanged handover | Assertions preserve integration state and BAM target; expert previous normalized action is reconstructed from applied target. Existing control passes with first-target error5.96e-8rad. |
| SHOULD expert-only labels / unchanged gate | Correction trace uses `Trace.step(sim)` with expert generation only; learner prefixes are saved separately. Acceptance uses the original seven-check conjunction, without threshold edits. |
| SHOULD no new prediction/training/split | Entrypoint does not initialize SmolPolicy or an optimizer; report0/0, all candidates development_only. No LeRobot dataset created. |
| Null/baseline | Original saved learner suffixes all terminate unstable; not matched200-step controls. Teacher-only reference1/1 strict,200ticks,0.57514m. Stationary null would fail forward and tail-progress gates by definition, but no random/persistence physical baseline was run. |
| Before updates | Same v10 saved actions define all learner prefixes; no learned weights updated before or after collection. |
| Held-out / dummy | One old body seed, two old sampling schedules, three repeats, correlated intervention lengths. No independent test split, language test or deployable dummy comparison. |
| LR, loss, grad norms | Not applicable: expert inference and simulation only, no training. |
| Full sample | `392-392-0-learner-prefix20.npz/.json` contains200 post-prefix observation/expert-action rows; matching-replay trace contains the20 learner actions and preserved task frame. |
| Surprise / second cause | Early handover10 passes only1/6 despite6/6 balance; explained by maximum yaw straddling0.5rad. Stable completion alone would admit all18, hiding direction errors. |
| Missing evidence | Correction learning benefit, broader reset coverage, exact input-cause separation, independent train/eval resets and real-time deployment remain untested. |
| Working diagnoses | Subjective priorities: state-conditioned transient steering/recovery50%, threshold-near heading variation20%, correlated sampling15%, handover/evaluation bug5%, unknown10%; these are not statistical probabilities. |
| Evidence for / against | Prefix-dependent yaw outcomes and exact source replay support a transient-control question; all18 stay upright, opposing a universal inability to recover. Handover/source checks constrain an adapter bug; they do not identify gait phase or a specific dynamics mechanism. |
| Cheapest next distinction | Equal-budget fixed-normalization data comparison tests whether these corrections improve the learner. A benefit on392 alone is memorization evidence; separately reserved resets must test transfer. |
| Time / memory | Whole run34.81s; no VLA latency, per-stage duration or peak-memory instrumentation. Expert ONNX uses the existing CPU path; rendering still uses EGL. |

Three ways a stronger claim would be false: counting18 interventions as18 independent reset conditions;
calling all18 balance recoveries successful straight walking; calling10 accepted demonstrations a learned
policy improvement. Grouping, full gate breakdown and zero updates directly prevent these claims.
A failure to transfer in a future experiment would characterize that particular data/training recipe,
not disprove expert correction or language-conditioned locomotion in general.

The independent CPU reviewer passed297/297 structural, gate, source/hash, replay and alignment checks.
Its conclusion: “No remaining blocking artifact defect found.” All correction state/action/prior arrays
are finite and correctly aligned to pre/post-action targets. Eighteen windows contain3,600 expert frames;
only the10 strict successes (2,000 frames) are accepted. The other1,600 frames remain rejection evidence.
First correction images match source pixels in10/18 cases; eight differ by at most one intensity level,
consistent with the known renderer variation. No new VLA consumes them during this run.
Full evidence is retained as `independent-audit.md` and `independent-checks.json` in the output directory.
No further collection or training was queued by this bounded run.
