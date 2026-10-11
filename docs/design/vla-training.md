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
| `observation.state` | 21 floats: 15 measured absolute joint angles, 3 trunk angular velocities, 3 projected gravity components. Units rad, rad/s, and a unit gravity direction. |
| `observation.images.head` | 256×256 RGB head-camera observation; upstream camera orientation correction with display rotation applied. |
| `action` | 15 absolute joint position targets in radians, in the existing Policy/robotd joint order. |
| Sample period | 20 ms (50 Hz); LeRobot forms future 50-step action chunks from these frames. |

The 14-actuator alpha model has no physical mouth joint. Both measured mouth position and commanded
mouth position are recorded as zero at robotd's mouth slot. This is an explicit simulation mapping,
not a demonstration of mouth actuation. Velocities of the joints and history are omitted from the
first learner input; the existing inference contract can carry them when a later experiment needs them.

Command parameters and task-family names are provenance, not model features. Physical seeds stay
in one split across tasks; paraphrases do not duplicate a rollout into both training and evaluation.
Train and evaluation datasets are separate so held-out values cannot contaminate normalization.
Evaluation combines new initial yaw and reserved expressions; it does not isolate their individual
effects or establish generalization to unseen skills, terrain or body parameters.

## Demonstration quality and learning boundary

The collector records every attempted episode and its terminal metrics. A successful movement
needs displacement in the requested direction; looking needs measured head yaw near the goal;
standing needs bounded drift. All require maintained trunk height and an upright gravity direction.
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
