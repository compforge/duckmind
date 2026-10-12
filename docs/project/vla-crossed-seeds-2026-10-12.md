# Cross body initialization and action-sampling noise

Following [v9](vla-observation-2026-10-12.md), this bounded no-training diagnostic separates two
variables that previously shared one seed: the simulated body's reset and the VLA's diffusion noise.
Seeds 390 and 392 were selected from development results, not held out for a new generalization claim.

## Predeclared experiment

Use the unchanged v8 single-trajectory checkpoint at step 1000 and its complete processor/config files.
Input remains task “向前走”, image, measured state and the previous applied target (51 state values).
Output is H10 joint targets; execute the first E5 at 50 Hz, then infer again. No injected image noise.

| Body reset | Diffusion schedule | Repetitions |
|---|---|---:|
| 390 | 390 | 3 |
| 390 | 392 | 3 |
| 392 | 390 | 3 |
| 392 | 392 | 3 |

Each schedule sets `torch.manual_seed(noise_seed*10000+tick)`. It cannot alter `Simulation.prepare`,
which takes only the body seed. Conditions are interleaved in a predeclared order; the same body seed
must have bit-exact measured state, applied target and integration state at reset across all cells.
An extra initial image is saved alongside that witness; actual model requests are saved separately.
Natural renderer variability remains, so the physical comparison is not pixel-identical.

Six teacher-only rollouts (three per body seed) use the fixed existing forward command 0.4 m/s.
This is a feasibility baseline, not expert takeover of learner trajectories. It deliberately avoids
sampling teacher speed from the body seed. Learner inputs contain no velocity command or expert label.
All rollouts last up to 200 ticks / four simulated seconds and use the unchanged strict recovery gate.

Budget is zero optimizer updates and at most 482 VLA calls: two same-input smoke calls plus 12×40
rollout predictions. Smoke outputs are saved before exact-equality assertion. Each trial saves full
trace and requests, including partial failures. Inputs, checkpoint files and dependencies are recorded;
checkpoint file hashes are verified again at completion. No training, new collection dataset, real
hardware action or paid compute is included.

## Interpretation fixed before results

Failure following the body seed supports sensitivity to that body/observation condition; it does not
separate physical state from its camera image or prove a data-coverage cause. Failure following noise
supports action-sampling sensitivity. An interaction or variable repeats leaves multiple explanations.
Teacher success on a failing learner reset establishes that the task is feasible for the reference
controller under these conditions; it does not show that the learner needs only more copies of it.

Three repeats are repeated executions, not new initial states or independent training seeds. Only one
language phrase and one trained checkpoint are tested. Physics pauses for inference; latency and real
control readiness must remain separate from physical success.

## Evidence and status

The existing `Simulation`, `Trace`, `SmolPolicy` and `recovery_outcome` own the implementation; the
artifact-owned `training/artifacts/run-v10-cross.py` orchestrates the bounded matrix. The launcher
`run-v10-cross.sh` saves exit status, source/script hashes and installed package versions. Output is
`training/artifacts/diagnostic-v10-cross/`, with `report.json`, `run.md`, smoke, full traces, separate
actual request NPZ files and initial-state witnesses. Existing outputs are never overwritten.

Preflight: AST / shell syntax checks pass; remote CUDA is idle and the data disk has 34 GB free.
Completed: exit 0, 123.26 s, 299 VLA calls including smoke, zero updates. All 10 checkpoint files
have identical before/after hashes. The manifest also records 198 upstream simulation/BAM/asset/teacher
files and installed dependency versions. No additional experiment was launched.

## Results

| Body reset | Noise 390: strict pass ↑ | Noise 392: strict pass ↑ | Original expert: strict pass ↑ |
|---|---:|---:|---:|
| 390 | 3/3 | 3/3 | 3/3 |
| 392 | 0/3 | 0/3 | 3/3 |

The same counts apply to all-tick stability. Every successful run completes 200 control ticks (4 s).
The six body-392 learner trials fail after 40, 41, 41 ticks with noise392 and 49, 47, 53 with noise390:
0.80–1.06 s. Sampling noise changes failure timing but does not reverse success in this selected matrix.
The expert starts from the same asserted state/prior/integration witness and passes all six trials,
using 0.4 m/s commanded forward speed; measured displacement is about 0.574–0.575 m over four seconds.
This benchmark does not verify commanded-speed tracking, only the unchanged strict forward-walking gate.

The body-reset condition predicts the binary outcome across both tested noise schedules. That narrows
v9's confound: merely swapping these schedules does not rescue body392. A body reset also changes the
camera observation, however, and natural render variation persists. This result cannot separate physical
conditioning, visual overfitting, or insufficient recovery behavior, and cannot by itself prove that
more demonstrations are the remedy. Two selected development resets are not a population estimate.

The proposed next intervention is targeted correction data at body392's early learner deviations, while
retaining successful examples. Query the expert before the observed collapse (for example, predeclared
10/20/30-tick learner prefixes), and accept only corrections that satisfy the existing full recovery gate.
Compare a bounded mixed-data fine-tune with an equal-budget original-data control from identical weights;
first prove the correction path is feasible. Original-expert success from reset does not guarantee recovery
from an already deviated learner state. Collection and training are proposals, not completed work here.

Real-time remains unresolved: across 297 learner rollout requests, latency min/median/p95/max is
0.21577/0.24695/0.28020/0.31055 s, versus the 0.10 s execution prefix and 0.20 s full chunk. All requests
exceed both intervals. Peak PyTorch CUDA allocated memory is 1,262,706,688 bytes (~1.18 GiB); this excludes
MuJoCo/OpenGL and non-PyTorch allocations and is not total GPU memory. Overall 123.26 s includes loading,
smoke, simulation and saving; separate stage durations were not recorded.

## ml-debug audit and decision

The full log has 79 lines, including all 18 trial results and final completion. The explicit logger
also propagates to the root handler, so INFO messages appear twice; raw JSON trial IDs are the denominator,
not log-line counts. No runtime code was changed after launch. All 2,671 physical ticks and raw failed
traces are retained under the output directory.

| Form item | Evidence / limitation |
|---|---|
| Actual config | Report: fixed v8 step1000, H10/E5, state51, action15, task 向前走, 2×2×3 learner trials +6 expert, zero updates. |
| SHOULD: frozen model/config | `checkpoint_unchanged=true`; all 10 checkpoint files hashed before/after. |
| SHOULD: seed separation | `prepare(TASK, body_seed)` and `manual_seed(noise_seed*10000+tick)` are separate; crossed cells finish without reset assertion failure. |
| SHOULD: same-body witness | Initial state/prior/MuJoCo integration witnesses saved and exact equality asserted. This is not a complete BAM hidden-state snapshot. |
| SHOULD: exact smoke | `smoke_same_input_exact=true`, `(10,15)` outputs and frozen inputs saved before assertion. |
| SHOULD: expert boundary | Only expert trials call `command(TASK)` and `Trace.step(sim)`; learner calls use explicit predicted targets, no takeover. |
| Before updates | Every stage has the same checkpoint. The frozen-input smoke checks reproducibility, not task success. |
| Null / baseline and floors | No chance success estimate. Original expert is a feasible reference: 6/6 strict/stable, about 0.575 m in 4 s. No acceptance threshold changed. A single-task policy ignoring language can score perfectly here. |
| Dummy | No persistence physical test or new fit metric in this run; v9 offline persistence cannot substitute for a closed-loop baseline. |
| Held-out | Body/noise390/392 selected using v9; three repetitions per cell share seeds, one trained checkpoint, one phrase. No new language/generalization claim. |
| LR, gradient norms, losses | Not applicable: inference-only, no backward/optimizer operation. |
| Full sample | `392-392-0-learner-inputs.npz` stores actual image/state/prior and chunks; matching trace/JSON preserves 40 actions ending in instability. |
| Surprising result | Both body392 noise schedules are 0/3 while both expert resets are 3/3; explained: these noise schedules alone do not account for the gap. Specific conditioning/coverage cause remains unresolved. |
| Missing evidence | Expert recovery from learner-deviated states, image-versus-body isolation, other initial conditions/training seeds, longer horizons and asynchronous real-time control. |
| Working priorities | Subjective, not inferred probabilities: body/observation conditioning or recovery coverage50%, visual shortcut20%, sampled-action interaction10%, training/adapter bug5%, eval/reset bug5%, unknown10%. |
| Evidence for/against | Conditioning: failure follows reset, expert passes; physical and camera causes still coupled. Visual: previous fixed-input sensitivity, but no vision ablation. Sampling: timing changes, but neither schedule rescues392. Training bug remains untested by this inference run; timing/target audits constrain adapter/eval errors without proving their absence. |
| Cheapest next distinction | Expert recovery at saved early learner deviations: success enables targeted correction collection; failure calls for earlier prefixes or controller/task diagnosis before training. |
| Time/memory | 123.26 s total, 297 rollout latency samples, peak PyTorch allocated memory 1.18 GiB; stage timing and total GPU allocation unknown. |

Three stronger claims would be false: calling twelve learner runs twelve independent states; calling a
body-dependent outcome proof that only the proprioceptive input matters; calling expert start-up success
proof that all learner-deviated states are recoverable. The crossed matrix and actual saved requests expose
the first two; the third needs the proposed recovery check. Failure of this particular single-trajectory
checkpoint does not reject VLA locomotion: conditioning, data coverage and learned recovery remain live
alternatives. No acceptance gate or source behavior was changed for reporting.

The fresh independent reviewer recomputed every gate from raw JSON without calling the project gate
function. No check errors were found across 18 traces: 1,471 learner +1,200 expert actions and 297
learner requests +2 smoke calls reconcile. Request/action indices and applied prior targets agree;
all 10 checkpoint, 198 simulation and 13 source/entrypoint hashes verify. Expert requests contain no
VLA chunks. The six body392 failures violate completeness, stability and original heading.

The reviewer cautioned: same-body initial physical state/prior is exact, but initial images have three
or four distinct variants. Repeated first chunks can differ by roughly 0.0012–0.0015 rad despite a
fixed sampling seed. The observed binary outcome nevertheless follows the body/observation bundle in
this selected matrix; neither proprioception alone nor data shortage is established. Full independent
form and per-trace recomputation are retained in `independent-audit.md` and `independent-checks.json`
inside the output directory. Review and numerical audit were CPU-only and made no new predictions.
