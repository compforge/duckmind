# Duckmind Policy

A replaceable `Policy.infer(request) → ActionChunk` boundary. The same request/response can
run locally or over HTTP; the runner connects predictions to the existing robotd executor.
The included FakePolicy exercises plumbing with head turns and hold. It is not a trained VLA.

From this directory:

```sh
uv sync --locked
uv run duckmind-policy serve
# Another terminal, after initializing robotd --fake or the simulated robot:
uv run duckmind-policy run --socket /tmp/robot.sock --task '向左看' --duration 5
```

`--url` selects another compatible policy service; `--timeout` bounds HTTP I/O (default 0.8 s).
The model service exposes `POST /v1/policy/predict`, `GET /v1/policy`, and OpenAPI at `/docs`.
To serve a learned adapter, implement `Policy.infer` and pass it to `create_app(adapter)`.
No runner or robotd changes are required if the adapter meets the contract.

See [the Policy interface design](../docs/design/policy-interface.md) for schemas, timing,
backend replacement, simulation setup, and current limits.

```sh
make fix
make lint
make test
# Run the real HTTP → robotd → FakeIo integration tests too:
ROBOTD_BIN="$(pwd)/../target/debug/robotd" make test
```
