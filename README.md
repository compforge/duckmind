# Duckmind

Duckmind is an experimental project for learning how natural language becomes robot motion,
built from [Microduck](https://github.com/pollen-robotics/microduck).

The goal is to walk through the complete learning loop on one Microduck body:
collect demonstrations, train a policy, execute its predictions in MuJoCo, and evaluate the result.

## The idea

1. Use Microduck's existing command-driven motion policies to generate demonstrations.
2. Pair each command with multiple equivalent text instructions and the corresponding state/action trajectories.
3. Train a policy to predict **action chunks**—sequences of joint targets—from text and body observations.

```text
Text + body state/history → learned policy → action chunk → robotd → robot motion
```

Start with a language-conditioned motion policy, then add visual observations to explore
vision-language-action (VLA) models. The model learns the motion; `robotd` owns timed execution
and feedback. Training and model integration are ongoing work.

## Origin

Microduck provides the robot runtime and hardware/simulation interfaces.
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) provides the upstream motion-policy
training stack. See [CONTRIBUTING.md](CONTRIBUTING.md) for building the runtime and
[the simulation guide](docs/robot/simulation.md) for running a simulated body.
