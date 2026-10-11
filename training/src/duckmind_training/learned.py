"""A trained SmolVLA implements the existing Policy boundary without commanding motors."""

import base64
import io
from pathlib import Path

import numpy as np
import torch
from duckmind_policy.schema import ActionChunk, PredictRequest
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from PIL import Image

from duckmind_training.contract import IMAGE_KEY, STATE_NAMES, state_vector


class SmolPolicy:
    def __init__(self, checkpoint: Path, device: str = "cuda"):
        self.model = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
        if tuple(self.model.config.input_features["observation.state"].shape) != (
            len(STATE_NAMES),
        ):
            raise ValueError("checkpoint state layout differs from this training adapter")
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    @torch.inference_mode()
    def predict(self, task: str, state: np.ndarray, image: np.ndarray) -> np.ndarray:
        # Each request is independent, as required by the existing Policy protocol.
        self.model.reset()
        batch = self.preprocess(
            {
                "observation.state": torch.from_numpy(state.astype(np.float32)),
                IMAGE_KEY: torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
                "task": task,
            }
        )
        chunk = self.model.predict_action_chunk(batch)
        decoded = torch.stack([self.postprocess(chunk[:, i]) for i in range(chunk.shape[1])], dim=1)
        return decoded[0].cpu().numpy()

    def infer(self, request: PredictRequest) -> ActionChunk:
        observation = request.observation
        if (
            observation.imu is None
            or observation.velocities is None
            or "head" not in observation.images
        ):
            raise ValueError(
                "SmolPolicy requires joint velocities, IMU and a head-camera observation"
            )
        q = np.asarray(observation.imu.quat, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            raise ValueError("invalid orientation quaternion")
        w, x, y, z = q / norm
        gravity = np.array([2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1])
        state = state_vector(
            np.asarray(observation.positions),
            np.asarray(observation.imu.gyro),
            gravity,
            np.asarray(observation.velocities),
        )
        image = np.asarray(
            Image.open(
                io.BytesIO(base64.b64decode(observation.images["head"].jpeg_base64))
            ).convert("RGB")
        )
        positions = self.predict(request.task, state, image)
        return ActionChunk(observation_t_ns=observation.t_ns, positions=positions.tolist())
