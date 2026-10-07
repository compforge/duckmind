"""The HTTP transport carries the same typed contract as local Policy.infer."""

import logging
import threading

import httpx
from fastapi import FastAPI, HTTPException

from duckmind_policy.policy import Policy, UnsupportedTask
from duckmind_policy.schema import ActionChunk, PolicyInfo, PredictRequest

logger = logging.getLogger(__name__)


def create_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="Duckmind Policy", version="1")
    gate = threading.Lock()

    @app.get("/v1/policy")
    def info() -> PolicyInfo:
        return PolicyInfo()

    @app.post("/v1/policy/predict")
    def predict(request: PredictRequest) -> ActionChunk:
        # GPU backends need bounded concurrency; never queue stale observations behind inference.
        if not gate.acquire(blocking=False):
            raise HTTPException(503, "policy is busy")
        try:
            result = policy.infer(request)
            if result.observation_t_ns != request.observation.t_ns:
                raise ValueError("policy returned a chunk for a different observation")
            return result
        except UnsupportedTask as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("policy inference failed")
            raise HTTPException(500, "policy inference failed") from exc
        finally:
            gate.release()

    return app


class HttpPolicy:
    def __init__(self, url: str, timeout: float = 0.8):
        self._client = httpx.Client(
            base_url=url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        )

    def infer(self, request: PredictRequest) -> ActionChunk:
        response = self._client.post("/v1/policy/predict", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return ActionChunk.model_validate_json(response.content)

    def close(self) -> None:
        self._client.close()
