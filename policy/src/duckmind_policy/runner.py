"""Receding-horizon inference. robotd remains the only timed action executor."""

import logging
import time

from pydantic import JsonValue

from duckmind_policy.policy import Policy
from duckmind_policy.robot import RobotClient, RpcFailure, Session, StateFeed
from duckmind_policy.schema import JOINT_NAMES, STEP_NS, ActionChunk, Observation, PredictRequest

logger = logging.getLogger(__name__)


def submission(
    chunk: ActionChunk,
    request: PredictRequest,
    session: Session,
    sequence: int,
    now_ns: int,
) -> dict[str, JsonValue]:
    if chunk.observation_t_ns != request.observation.t_ns:
        raise ValueError("policy returned a different observation timestamp")
    # Actions are anchored at the observation, not inference completion. Allow two robot ticks
    # for IPC admission, trim elapsed predictions, and never move an old prediction into the future.
    first = max(0, (now_ns + 2 * STEP_NS - chunk.observation_t_ns + STEP_NS - 1) // STEP_NS)
    if first >= len(chunk.positions):
        raise ValueError("entire predicted chunk expired during inference")
    return {
        "session_id": session.session_id,
        "sequence": sequence,
        "observation_t_ns": chunk.observation_t_ns,
        "start_t_ns": chunk.observation_t_ns + first * STEP_NS,
        "step_ns": chunk.step_ns,
        "positions": [list(frame) for frame in chunk.positions[first:]],
    }


def run(
    robot: RobotClient,
    policy: Policy,
    task: str,
    duration: float = 5.0,
) -> int:
    """Run an already-initialized fake/sim robot. Return admitted chunk count, not task success."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    session = Session.model_validate(robot.call("robot.actions.begin"))
    feed: StateFeed | None = None
    count = 0
    try:
        if tuple(session.joint_names) != JOINT_NAMES or session.step_ns != STEP_NS:
            raise ValueError("robotd body contract differs from microduck-joints-v1")
        feed = StateFeed(robot)
        history: list[Observation] = []
        deadline = time.monotonic() + duration
        logger.info("policy session acquired: %s", session.session_id)
        while time.monotonic() < deadline:
            state, _ = feed.latest()
            ensure_active(
                state.actions.session_id if state.actions else None,
                state.actions.phase if state.actions else None,
                session,
            )
            observation = state.observation()
            request = PredictRequest(task=task, observation=observation, history=history)
            chunk = policy.infer(request)
            # A stop/timeout while inference was running must not reacquire control.
            state, now_ns = feed.latest()
            ensure_active(
                state.actions.session_id if state.actions else None,
                state.actions.phase if state.actions else None,
                session,
            )
            if time.monotonic() >= deadline:
                break
            payload = submission(chunk, request, session, count + 1, now_ns)
            robot.call("robot.actions.submit", payload)
            count += 1
            logger.info("chunk admitted: session=%s sequence=%d", session.session_id, count)
            history = [*history, observation][-8:]
            # Replan before the tail expires; execution continues independently inside robotd.
            end_ns = chunk.observation_t_ns + len(chunk.positions) * STEP_NS
            delay = min(0.2, max(0.0, (end_ns - now_ns) / 1e9 / 2))
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        return count
    finally:
        if feed is not None:
            feed.close()
        try:
            robot.call("robot.actions.end", {"session_id": session.session_id})
        except (OSError, RpcFailure) as exc:
            # Ending is scoped to our session. Never issue a global stop against a newer owner.
            logger.warning(
                "could not end policy session %s; it may already be invalidated: %s",
                session.session_id,
                exc,
            )
        logger.info("policy session finished: %s admitted=%d", session.session_id, count)


def ensure_active(owner: str | None, phase: str | None, session: Session) -> None:
    if owner != session.session_id or phase not in ("waiting", "running"):
        raise RpcFailure("action session ended; discard prediction without reacquiring control")
