# Observation perturbation and repeated closed-loop evaluation

The user authorized continued diagnosis after [v8](vla-single-trajectory-2026-10-12.md). This experiment
holds the learned policy fixed and asks whether modest observation changes predict materially different
closed-loop outcomes beyond natural repeat variability. It performs **zero optimizer updates**.

## Predeclared comparison

Use `smolvla-v8-single/checkpoints/001000/pretrained_model`, its unchanged state51/action15 processors,
H10 and execution length5. Task is “向前走”, each rollout lasts at most200 ticks/4seconds. Diffusion noise
is reseeded with `physics_seed*10000+tick` before each prediction, unchanged across conditions/repeats.
This is the explicit v8 diagnostic protocol, not the native evaluator's once-per-rollout noise schedule.

| Scope | Conditions | Count / interpretation |
|---|---|---|
| Fixed expert observations | Raw image; one RGB channel±1; all channels random±1 then clipped |30 predetermined frames, ticks0..174 every6;90 predictions |
| Same-input smoke | Repeat the first raw image/state/previous target/noise |1 prediction; must equal its earlier output exactly, or save evidence and stop |
| Closed loop | Raw observation; one RGB channel±1 at every prediction |4 initial-state seeds ×3 repetitions ×2 conditions=24 rollouts |

Maximum budget is1051 predictions:90 offline +1 smoke +24×40 closed-loop predictions. Early physical
failures consume fewer. Dense noise is an offline, larger-amplitude stress control: up to196608 modified
channels versus one. It cannot isolate sparse versus dense structure at equal perturbation energy.

Seed242 is already studied. Seeds390,391,392 were absent from current scanned evaluation reports and
raw manifests; the inventory is saved before execution. They remain a small test of initial-state
variation, not independent model training seeds. Three repeats with identical diffusion schedules measure
run variability; they are not three additional initial states. No parameters or acceptance gates are
selected from these outcomes, and no language-generalization claim follows from one repeated task.

The sparse perturbation picks an index/sign using NumPy RNG seed`17000000+physics_seed*10000+tick`.
At0/255 the sign reverses to retain exactly one changed uint8 channel. Every request retains its actual
image/state/previous target, chunk, and the changed channel's index, old value, new value and RNG seed.
The fixed-input offline comparisons reuse exactly the same state, previous target and diffusion noise.
Closed-loop raw/perturbed pairs can have naturally differing rendered images after reset, so they are
not pixel-identical causal pairs. Condition order alternates by seed-index plus repeat for balance.

## Execution and evidence

`training/artifacts/run-v9-observation.sh` runs the single artifact-owned diagnostic script of the same
name with `.py`; it uses the existing `SmolPolicy`, `Simulation`, `Trace` and `recovery_outcome`. It creates
`diagnostic-v9-observation` once, refusing overwrite. Launcher PID3757919 is an observation, not the source
of truth: check `.stage`, `.log`, `.exit-code` and the actual processes. Original source and script hashes
are retained in `.inputs.sha256`.

No training or robotd source changes are needed. The teacher only prepares the starting body state and
never takes over a learner rollout. Actual applied targets are the prediction's first5 entries followed
by a fresh observation. Full trace, inputs and chunks persist even when an exception interrupts a trial;
the exception remains fatal rather than silently retrying. The smoke saves both outputs and its input
before asserting equality. Independent preflight review specifically checked these failure paths,
noise/perturbation seed separation and actual executed tick counting.

Report two different quantities offline: target MSE against expert first actions (`30×15` values), and
prediction differences caused by the image intervention (`30×10×15` values). Neither establishes physical
robustness. Closed-loop reporting uses the unchanged strict gate, all-tick stability, executed ticks,
forward/lateral displacement, maximum and terminal yaw. Group old versus new initial states and retain
every per-seed3-repeat result rather than presenting24 trials as24 independent conditions.

Inference latency is measured separately. Simulation pauses for inference, so passing these trials does
not demonstrate real-time control. At v8's roughly0.27s per prediction, raw inference alone can take about
4.7minutes, plus loading/rendering/saving;96seconds denotes aggregate simulated physical duration only.

## Results and research decision

Completed with exit code 0 in 313.32 s: **823 predictions, zero optimizer updates**. The checkpoint
SHA-256 is `d762b5e9246b0712e0130deb534a6975e28c57392874f7731465c14c14fc3668`.
All 24 physical trials retain their inputs, chunks and full traces, including eight early failures.

| Seed / status | Raw strict pass ↑ | Sparse strict pass ↑ | Physical outcome |
|---|---:|---:|---|
| 242 / training-state diagnostic | 3/3 | 3/3 | All completed 200 ticks |
| 390 / newly evaluated condition | 3/3 | 3/3 | All completed 200 ticks |
| 391 / newly evaluated condition | 3/3 | 1/3 | Sparse failures at ticks 91 and 95 |
| 392 / newly evaluated condition | 0/3 | 0/3 | All failed at ticks 41–44 |

Strict success and all-tick stability have the same counts here. Raw totals are 9/12, sparse 7/12;
excluding seed 242, they are 6/9 and 4/9. These are descriptive counts over three repeats per condition,
not 12 independent initial states, training repeats or a statistically established treatment effect.
The physics seed also selects diffusion noise, so differences across seeds cannot be assigned solely
to body initialization. Each successful rollout covers four simulated seconds, not sustained walking.

The fixed-input smoke was bit-exact. Offline comparisons used 30 predetermined expert frames:

| Input | Expert first-action MSE ↓ (rad², 450 scalars) | Chunk difference from raw (RMSE rad, 4500 scalars) | Maximum absolute difference (rad) |
|---|---:|---:|---:|
| Raw | 0.000194474 | 0 | 0 |
| One RGB channel ±1 | 0.000194364 | 0.000433191 | 0.00423850 |
| All RGB channels ±1 | 0.000196279 | 0.000904928 | 0.00837928 |

These small input interventions change predictions without materially changing average first-action
fit on these frames. That does not show that images are unnecessary, nor that a small action error is
safe under feedback. Dense noise changes far more channels and is only a larger-energy stress control.

The clearest recurring failure is condition 392: instability appears within 0.82–0.88 simulated seconds
both with and without the injected perturbation. In `392-0-raw.json`, ticks 36–41 show body height falling
from 0.10679 to 0.08448 m, gravity_z changing from -0.88834 to -0.64407, and yaw increasing from 0.33019
to 0.59949 rad. It makes forward progress while tipping; forward distance alone would conceal failure.
Condition 391 is consistent with sensitivity near a stability boundary, but reset/render variability
prevents attributing its two outcome changes solely to the deliberately changed channel.

Across 732 closed-loop predictions, latency median is 0.24734 s and p95 is 0.59684 s (minimum 0.20337 s).
The execution prefix spans 0.10 s and the entire chunk 0.20 s. Physics was paused for inference;
these results still do not establish real-time feasibility. Peak GPU memory and stage-specific timing
were not recorded, so their values are unknown.

**Next proposed experiment, not launched:** separate initialization sensitivity from action-sampling
sensitivity by crossing the failing 392 and successful 390 body resets with their diffusion schedules,
then check the original expert on the failing reset. If failure follows the body and the expert succeeds,
prioritize start-up/recovery demonstrations around the failing prefix. If failure follows the diffusion
schedule, prioritize sampled-action variability and prediction consistency before collecting broadly.
This is more discriminating than adding an undirected batch of demonstrations. No further training or
collection was launched by this evaluation; the bounded automatic follow-up pauses after delivery.

## ml-debug result audit

The full stdout log contains 39 lines; it is retained with the script-owned `run.md`, `report.json`,
NPZ requests and physical traces. INFO diagnostic messages did not reach stdout, so the structured
artifacts carry the actual experiment settings and results. No missing stdout message is counted as
an executed check. The run used the existing native SmolVLA inference adapter and unchanged physical
acceptance function; no model or acceptance parameters were selected from these results.

| Form item | Evidence / limitation |
|---|---|
| Configuration and log | 39 stdout lines; report: `execute_steps=5`, `horizon=10`, `training_updates=0`, `actual_predictions=823`; task 向前走, 51 state inputs and 15 outputs per tick. |
| SHOULD: unchanged model, prior, execution | Script uses one fixed checkpoint and actual previous executed target; source/processor and trace audit recorded below. |
| SHOULD: one changed channel | Offline raw and sparse arrays plus per-request index/before/after/RNG metadata retained; dense control has much greater energy. |
| SHOULD: exact same-input smoke | `smoke_same_input_exact=true`, both outputs and input preserved in `smoke.npz`. |
| Null and scale | No-intervention prediction difference is exactly 0 by construction; raw input is the paired offline baseline. Chance/random physical success is unknown; no new threshold was introduced. |
| Before updates / baseline | No updates occur at any stage. Raw and sparse use the same v8 final checkpoint; this is not a training improvement comparison. |
| Dummy / persistence | CPU recomputation on the same 30 frames: previous-target persistence first-action MSE 0.0104658632 rad² versus learned raw 0.000194474 (about 53.8× lower). No persistence physical rollout. Walking four seconds can be solved by a single-task policy ignoring language, so these scores do not validate instruction following. |
| Held-out interpretation | 242 is old; 390–392 were absent from scanned artifacts. Only one task phrase and one trained checkpoint; neither language generalization nor training-seed robustness is tested. |
| LR, gradient norms and loss terms | Not applicable: no backward pass or optimizer. Expert-target MSE measures fit, not a training loss trajectory. |
| Full sample | `392-0-raw-inputs.npz` stores each actual image, measured state, previous target and H10 output; matching trace and JSON record its 41 executed ticks and instability. |
| Surprising observation | 392 fails all six trials without requiring injected noise; explained: observed failure is not confined to the perturbation condition. Root cause remains under investigation. |
| Missing evidence | Factorial body/noise control, expert capability on 392, more initial conditions, multiple training seeds, real-time inference and language controls remain untested. |
| Working diagnoses | Subjective priorities, not measured probabilities: coverage/closed-loop control 40%, action-sampling sensitivity 20%, visual-input sensitivity 20%, training/adapter bug 5%, evaluation/reset issue 5%, unknown 10%. |
| Evidence for / against hypotheses | Coverage: repeatable 392 failure supports a narrow robustness gap, but body/noise are coupled. Sampling: noise schedule varies across conditions, untested in isolation. Visual: fixed-input intervention changes outputs; raw 392 failure opposes it as sole cause. Adapter bug: long prior history warrants checks, but saved timing/target alignment is auditable. Evaluation issue: reset images vary; fixed-input smoke is exact and complete traces retain gate evidence. |
| Cheapest distinguishing test | Cross body reset and diffusion seed, then expert baseline on failing reset; proposals above, not additional authorized runs. |
| Time and memory | Whole run 313.32 s; offline median prediction 0.24746 s; closed-loop median/p95 0.24734/0.59684 s. Peak GPU memory and per-stage elapsed time unknown. |

Three ways a stronger claim would be false: counting repetitions as independent states inflates evidence;
attributing every sparse/raw difference to one channel ignores reset variability; calling physical success
real-time readiness ignores the paused simulation and latency. The saved per-seed table, actual requests
and timings expose those limitations. One short-chunk SmolVLA implementation failing on 392 does not
reject language-conditioned locomotion: a policy with better recovery coverage, conditioning or temporal
consistency can still succeed. No architecture change is justified by this experiment alone.

The fresh independent CPU audit found: “No denominator, gate, executed-chunk-index, or actual
previous-target bug found by these checks.” It independently recomputed all 24 strict gates, 3,638
executed actions and prior-target alignment, and 732 saved model requests. All 13 source/script manifest
entries and the model hash match. Effective processors consume finite state `[1,51]`, RGB/255 images
and task text `向前走\n`; output shape is `[1,15]` per action.

The audit also found initial physical integration state, measured state and previous target bit-exact
across same-seed repetitions, while rendered initial images could differ by 804 channels. At seed 391,
raw-only first-chunk differences reached RMSE 0.000454 rad, comparable to its controlled initial sparse
intervention (0.000366 rad). This strengthens the reset/render confound warning. Trace images can differ
from actual request images by 889 channels because they are rendered again; use `-inputs.npz` for
prediction evidence.

Provenance limitation: the startup manifest includes source/scripts and report includes the model hash,
but it did not hash processors, tokenizer, simulator assets or dependency versions at launch. The CPU
audit records current processor hashes and the script performs no normalization writes; historical
immutability of omitted files is not cryptographically proven. Full reviewer findings and recomputation
are preserved as `diagnostic-v9-observation/independent-audit.md` and `independent-checks.json`.

Artifacts under `training/artifacts/`:

- `audit-v9-seed-inventory.json` and `run-v9-observation.{sh,py,log,stage,exit-code,inputs.sha256}`.
- `diagnostic-v9-observation/run.md` and `report.json` with resolved settings and per-trial outcomes.
- `smoke.npz`, `offline.npz`, and each `<seed>-<repeat>-<condition>.npz/.json` physical trace.
- Each matching `-inputs.npz`, containing the actual prediction requests and produced chunks.

Weights and normalization remain unchanged. Automatic follow-up is limited to finishing this bounded
evaluation and recording/delivering its result, then pauses; it does not authorize new training.
