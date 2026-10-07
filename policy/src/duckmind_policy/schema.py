"""Versioned model boundary. Every action is already decoded into robotd joint space."""

import base64
import binascii
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

STEP_NS: Literal[20_000_000] = 20_000_000
JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)
Vector15 = Annotated[list[FiniteFloat], Field(min_length=15, max_length=15)]
Target15 = Annotated[
    list[Annotated[FiniteFloat, Field(ge=-math.pi, le=math.pi)]],
    Field(min_length=15, max_length=15),
]
Timestamp = Annotated[int, Field(strict=True, gt=0)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Imu(Contract):
    """Trunk angular velocity (rad/s), orientation trunk→world as [w,x,y,z]."""

    gyro: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    quat: Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class Image(Contract):
    """JPEG bytes in base64; capture time must be expressed in robotd's clock."""

    t_ns: Timestamp
    jpeg_base64: Annotated[str, Field(min_length=1, max_length=4_000_000)]

    @field_validator("jpeg_base64")
    @classmethod
    def encoded_jpeg(cls, value: str) -> str:
        try:
            data = base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64 image") from exc
        if not data.startswith(b"\xff\xd8"):
            raise ValueError("image must contain JPEG bytes")
        return value


class Observation(Contract):
    t_ns: Timestamp
    positions: Vector15
    velocities: Vector15 | None = None
    imu: Imu | None = None
    images: Annotated[dict[str, Image], Field(max_length=4)] = Field(default_factory=dict)

    @model_validator(mode="after")
    def capture_times(self) -> "Observation":
        if any(image.t_ns > self.t_ns for image in self.images.values()):
            raise ValueError("image capture time is newer than the observation")
        return self


class PredictRequest(Contract):
    schema_version: Literal["microduck-joints-v1"] = "microduck-joints-v1"
    task: Annotated[str, Field(min_length=1, max_length=4096)]
    observation: Observation
    history: Annotated[list[Observation], Field(max_length=8)] = Field(default_factory=list)

    @field_validator("task")
    @classmethod
    def nonblank_task(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task must not be blank")
        return value

    @model_validator(mode="after")
    def ordered_history(self) -> "PredictRequest":
        times = [sample.t_ns for sample in self.history] + [self.observation.t_ns]
        if any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("history must precede the observation in timestamp order")
        return self


class ActionChunk(Contract):
    schema_version: Literal["microduck-joints-v1"] = "microduck-joints-v1"
    observation_t_ns: Timestamp
    step_ns: Literal[20_000_000] = STEP_NS
    positions: Annotated[list[Target15], Field(min_length=1, max_length=100)]


class PolicyInfo(Contract):
    schema_version: Literal["microduck-joints-v1"] = "microduck-joints-v1"
    joint_names: tuple[str, ...] = JOINT_NAMES
    step_ns: Literal[20_000_000] = STEP_NS
    action_space: Literal["absolute_joint_position_rad"] = "absolute_joint_position_rad"
