# Language-conditioned Microduck training

The training boundary is a language task and the current physical observation mapped to future
joint targets. The existing command-conditioned ONNX policy supplies demonstrations; SmolVLA
learns these targets through behavioral cloning with its native flow-matching objective.
LeRobot owns dataset serialization, model training, normalization and checkpoints. Duckmind owns
the body/task adapter and verification of what a demonstration means.

## Data semantics

At physical time t: render/capture the observation, ask the expert for its action, record the
absolute target, then integrate four 5 ms MuJoCo steps using the upstream BAM actuator model.
Each record therefore pairs the state **before** a command with the command subsequently applied.
The target is taken from BAM's position target, not `data.ctrl` (which contains motor torque).

| Field | Meaning |
|---|---|
| `task` | Natural-language instruction. Multiple expressions describe each expert task. |
| `observation.state` | 36 floats: 15 measured absolute joint angles, 3 trunk angular velocities, 3 projected gravity components, then 15 joint velocities. Units rad, rad/s, and a unit gravity direction. |
| `observation.images.head` | 256×256 RGB head-camera observation; upstream camera orientation correction with display rotation applied. |
| `action` | 15 absolute joint position targets in radians, in the existing Policy/robotd joint order. |
| Sample period | 20 ms (50 Hz); LeRobot forms future action chunks with the checkpoint's configured horizon (default 50 steps). |

The 14-actuator alpha model has no physical mouth joint. Both measured mouth position and commanded
mouth position are recorded as zero at robotd's mouth slot. This is an explicit simulation mapping,
not a demonstration of mouth actuation. Measured joint velocities are required at inference as well as collection. The raw files also
retain `previous_target`, the last actually applied joint target before each observation. It stays
out of model features because the current Policy request does not carry applied-action feedback;
using a predicted but unexecuted action there would create a train/inference mismatch.

The state width exceeds SmolVLA base's 32-column state projection. Initialization creates a native
checkpoint with a 36-column projection, preserves existing columns and zero-initializes the added
columns; all other weights are unchanged. The state projection is then trained. Checkpoint and raw
state layouts are checked before use; historical 21-state artifacts require their original adapter.

Command parameters and task-family names are provenance, not model features. Physical seeds stay
in one split across tasks; paraphrases do not duplicate a rollout into both training and evaluation.
Train and evaluation datasets are separate so held-out values cannot contaminate normalization.
Evaluation combines new initial yaw and reserved expressions; it does not isolate their individual
effects or establish generalization to unseen skills, terrain or body parameters.

## Demonstration quality and learning boundary

The collector records every attempted episode and its terminal metrics. Movement needs displacement in the requested direction; turning needs accumulated yaw in the
requested direction; looking needs measured head-joint displacement near the sampled goal;
standing needs bounded planar drift. Stop demonstrations start after two seconds of verified
expert walking and must end with low average speed over the final second. Walking starts and
stopping therefore have distinct physical initial states, without an unobserved timed task switch. All require maintained trunk height and an upright gravity direction.
Rejected rollouts never enter the behavioral-cloning dataset. These are baseline heuristics, not
collision, energy or full gait-quality certification.

The expert is used for initial settling before collection/evaluation. During a learned-policy
rollout, it never corrects the model's actions. The simulator uses the same task adapter for expert
and learner so changing the controller does not silently change joint ordering or actuator physics.

The Action Expert predicts normalized action chunks, then saved LeRobot postprocessors decode
physical joint targets. `SmolPolicy` wraps this as the existing `Policy.infer → ActionChunk`
contract. It does not acquire the body or send commands. Real-time alignment, expired predictions,
ownership and safety remain with the [Runner and robotd](policy-interface.md).

## Evidence and limits

Separate environment/CUDA availability, dataset validity, optimization progress, held-out prediction
error, synchronous physical task success and real-time robotd operation. Native checkpoint creation
proves a run saved weights, not that a biped can balance. Holding physics during inference removes
latency from the control problem; an online integration must evaluate latency against the action
timeline separately. Camera collection for that integration remains a distinct runtime concern.

Operational commands are in [training/README.md](../../training/README.md).


## Coverage and comparisons

The catalog includes continuous motion, head direction and stopping from motion. A selected task
list is saved in the manifest; conversion requires every declared family in both splits and every
attempted episode ID. A rejected attempt stays visible even though its frames are not converted.
Command amplitudes vary within a narrow, recorded interval, while the text specifies direction.
Small joint-position and trunk-velocity perturbations after settling provide physically distinct
starts in addition to camera yaw; these are controlled simulation perturbations, not hardware DR.
No side-motion demonstration is accepted just because the policy received a nonzero command.

Physical evaluation takes explicit initial seeds and execution horizons. A horizon of 1, 2, 5 or
10 means consuming that many predicted targets before obtaining another observation, independent
of the model's configured prediction horizon. Execution cannot exceed that horizon. Shorter
training chunks use native SmolVLA configuration: they focus supervision on nearer targets but
reduce how much inference delay a runner can hide. This isolates feedback frequency under paused-physics
inference; it cannot establish whether the GPU meets real-time deadlines. Data/input changes and
additional training together are a new baseline, not a causal ablation of any single factor.
