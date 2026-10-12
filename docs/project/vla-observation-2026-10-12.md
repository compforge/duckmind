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

## Status and decision criteria

Launched; results pending. First inspect the same-input smoke, offline intervention sizes and prediction
deltas, then compare between-condition differences against the spread of raw repeats. If closed-loop
outcomes vary comparably without intervention, do not attribute a few flips to the injected pixel.
If fixed-input deltas and physical degradation recur across states, prioritize controlled robustness work.
If difficult startup targets remain the clearer recurring failure, revisit those demonstrations or the
conditioning/training objective. This run itself does not queue either training intervention.

Artifacts under `training/artifacts/`:

- `audit-v9-seed-inventory.json` and `run-v9-observation.{sh,py,log,stage,exit-code,inputs.sha256}`.
- `diagnostic-v9-observation/run.md` and `report.json` with resolved settings and per-trial outcomes.
- `smoke.npz`, `offline.npz`, and each `<seed>-<repeat>-<condition>.npz/.json` physical trace.
- Each matching `-inputs.npz`, containing the actual prediction requests and produced chunks.

Weights and normalization remain unchanged. Automatic follow-up is limited to finishing this bounded
evaluation and recording/delivering its result, then pauses; it does not authorize new training.
