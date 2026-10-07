"""Local serving and a bounded simulation runner."""

import argparse
import logging

import uvicorn

from duckmind_policy.policy import FakePolicy
from duckmind_policy.robot import RobotClient
from duckmind_policy.runner import run
from duckmind_policy.service import HttpPolicy, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve the built-in FakePolicy")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8081)
    runner = commands.add_parser("run", help="run an initialized fake/sim robot through HTTP")
    runner.add_argument("--socket", required=True)
    runner.add_argument("--url", default="http://127.0.0.1:8081")
    runner.add_argument("--task", required=True)
    runner.add_argument("--duration", type=float, default=5.0)
    runner.add_argument("--timeout", type=float, default=0.8, help="HTTP timeout in seconds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "serve":
        uvicorn.run(create_app(FakePolicy()), host=args.host, port=args.port)
    else:
        policy = HttpPolicy(args.url, timeout=args.timeout)
        try:
            run(RobotClient(args.socket), policy, args.task, args.duration)
        except KeyboardInterrupt:
            pass  # run's finally block releases its action session.
        finally:
            policy.close()
