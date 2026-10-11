"""Reuse the pinned upstream rehearsal, including its fitted BAM actuator physics."""

import importlib.util
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from duckmind_training.contract import IMAGE_SIZE, JOINT_NAMES, Task, state_vector, wire_positions


class Simulation:
    def __init__(self, upstream: Path, teacher: Path):
        spec = importlib.util.spec_from_file_location(
            "microduck_rehearsal", upstream / "scripts/infer_policy.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("cannot load the upstream inference script")
        upstream_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream_module)
        scene = upstream / "src/mjlab_microduck/robot/microduck/scene.xml"
        bam = upstream_module.load_bam_model(kp_fw=200, vin=7.4, max_current=None)
        self.model, self.data, self.bam, _ = upstream_module.load_mujoco_with_bam(
            str(scene), bam, timestep=0.005, vin_drop_gain=0.0, vin_min=6.0
        )
        self.expert = upstream_module.PolicyInference(
            self.model,
            self.data,
            walking_onnx_path=str(teacher),
            bam_ctrl=self.bam,
            use_projected_gravity=True,
            new_cmd_obs=True,
        )
        # The upstream interactive demo creates a default CPU thread pool. A tiny MLP
        # running alongside physics/rendering is faster with one non-spinning worker.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.expert.walking_session = ort.InferenceSession(
            str(teacher), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.expert.ort_session = self.expert.walking_session
        expected = tuple(name for name in JOINT_NAMES if name != "mouth")
        actual = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
            for jid in self.model.actuator_trnid[:, 0]
        )
        if actual != expected:
            raise ValueError(f"upstream actuator order changed: {actual}")
        # Use the upstream head-camera orientation correction, without its streaming server.
        sys.path.insert(0, str(upstream / "src"))
        from mjlab_microduck.sim.camera import Camera

        self.camera = Camera(self.model, "head_camera", width=IMAGE_SIZE, height=IMAGE_SIZE)
        self.root = self.expert._trunk_qpos_adr
        root_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        self.root_velocity = int(self.model.jnt_dofadr[root_joint])
        self.tail_speeds = deque(maxlen=50)
        self.reset(0)

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.initial_yaw = float(rng.uniform(-0.3, 0.3))
        self.data.qpos[self.root : self.root + 3] = [0.0, 0.0, 0.125]
        self.data.qpos[self.root + 3 : self.root + 7] = [
            np.cos(self.initial_yaw / 2),
            0,
            0,
            np.sin(self.initial_yaw / 2),
        ]
        self.data.qpos[self.expert.joint_qpos_indices] = self.expert.default_pose
        self.bam.reset(self.data.qpos)
        self.expert.last_action[:] = 0
        self.expert.head_offset[:] = 0
        self.expert.set_vel_cmd()
        mujoco.mj_forward(self.model, self.data)
        self.mark_origin()
        # Warm up under the expert before recording; settling is not task demonstration.
        for _ in range(50):
            self.teacher_step()
        # Break rotationally equivalent resets: the learner must handle small physical errors.
        self.data.qpos[self.expert.joint_qpos_indices] += rng.uniform(-0.005, 0.005, 14)
        self.data.qvel[self.root_velocity : self.root_velocity + 2] += rng.uniform(-0.02, 0.02, 2)
        self.data.qvel[self.root_velocity + 3 : self.root_velocity + 6] += rng.uniform(-0.1, 0.1, 3)
        mujoco.mj_forward(self.model, self.data)
        self.mark_origin()

    def mark_origin(self) -> None:
        self.origin = self.data.qpos[self.root : self.root + 2].copy()
        self.previous_xy = self.origin.copy()
        self.tail_speeds.clear()
        self.origin_yaw = self.yaw()
        self.yaw_travel = 0.0
        self.previous_yaw = self.origin_yaw

    def prepare(self, task: Task, seed: int) -> None:
        self.reset(seed)
        if any(task.initial_twist):
            self.expert.set_vel_cmd(*task.initial_twist)
            for _ in range(100):
                self.teacher_step()
            # Stop examples start from an actually moving body, never a stationary reset.
            if not self.metrics()["stable"] or np.mean(self.tail_speeds) < 0.06:
                raise ValueError(f"expert cannot prepare moving start for {task.name}")
        self.mark_origin()

    def command(self, task: Task) -> None:
        self.expert.head_offset[:] = task.head
        self.expert.set_vel_cmd(*task.twist)

    def yaw(self) -> float:
        w, x, y, z = self.data.qpos[self.root + 3 : self.root + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.data.qpos[self.expert.joint_qpos_indices]
        state = state_vector(
            wire_positions(positions),
            self.expert.get_base_ang_vel(),
            self.expert.get_projected_gravity(),
            wire_positions(self.expert.get_joint_vel()),
        )
        self.camera.renderer.update_scene(self.data, camera=self.camera.camera)
        image = np.rot90(self.camera.renderer.render(), k=-1).copy()
        return state, image

    def teacher_step(self) -> np.ndarray:
        action = self.expert.infer()
        self.expert.apply_action(action)
        targets = self.bam.q_target.copy()
        self.advance()
        return targets

    def advance(self) -> None:
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        xy = self.data.qpos[self.root : self.root + 2].copy()
        self.tail_speeds.append(float(np.linalg.norm(xy - self.previous_xy) * 50))
        self.previous_xy = xy
        yaw = self.yaw()
        self.yaw_travel += float(
            np.arctan2(np.sin(yaw - self.previous_yaw), np.cos(yaw - self.previous_yaw))
        )
        self.previous_yaw = yaw

    def apply(self, targets: np.ndarray) -> None:
        self.expert.set_position_targets(np.delete(targets, JOINT_NAMES.index("mouth")))
        self.advance()

    def metrics(self) -> dict[str, float | bool]:
        delta = self.data.qpos[self.root : self.root + 2] - self.origin
        c, s = np.cos(self.origin_yaw), np.sin(self.origin_yaw)
        z = float(self.data.qpos[self.root + 2])
        gravity_z = float(self.expert.get_projected_gravity()[2])
        relative = self.expert.get_joint_pos_relative()
        return {
            "forward_m": float(delta @ np.array([c, s])),
            "left_m": float(delta @ np.array([-s, c])),
            "yaw_rad": self.yaw_travel,
            "tail_speed_mps": float(np.mean(self.tail_speeds)) if self.tail_speeds else 0.0,
            "trunk_z_m": z,
            **{
                name: float(relative[i + 5])
                for i, name in enumerate(
                    ("neck_pitch_delta", "head_pitch_delta", "head_yaw_delta", "head_roll_delta")
                )
            },
            "stable": bool(z > 0.075 and gravity_z < -0.65 and np.isfinite(self.data.qpos).all()),
        }

    def close(self) -> None:
        self.camera.renderer.close()
