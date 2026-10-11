# Duckmind

Duckmind is an experimental project for learning how natural language becomes robot motion,
built from [Microduck](https://github.com/pollen-robotics/microduck).

The goal is to walk through the complete learning loop on one Microduck body:
collect demonstrations, train a policy, execute its predictions in MuJoCo, and evaluate the result.

## The idea

1. Use Microduck's existing command-driven motion policies to generate demonstrations.
2. Pair each command with multiple equivalent text instructions and the corresponding state/action trajectories.
3. Train a VLA to predict **action chunks**—sequences of joint targets—from text, camera images and body state.

```text
LLM → text task + camera/body observations → VLA → action chunk → robotd → robot motion
```

The high-level LLM handles conversation and task planning; the VLA learns motion from demonstrations.
`robotd` owns timed execution and feedback. A [replaceable Policy interface and Fake backend](policy/README.md) exercise this
execution path before training a model. [Training tools](training/README.md) collect expert
demonstrations and fine-tune SmolVLA, with separate offline and physical evaluation. Learned
robotd integration and reliable motion remain subject to evaluation.

## Origin

Microduck provides the robot runtime and hardware/simulation interfaces.
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) provides the upstream motion-policy
training stack. See [CONTRIBUTING.md](CONTRIBUTING.md) for building the runtime and
[the simulation guide](docs/robot/simulation.md) for running a simulated body.
