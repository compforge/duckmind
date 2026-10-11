"""Reuse the pinned upstream rehearsal, including its fitted BAM actuator physics."""

import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from duckmind_training.contract import IMAGE_SIZE, JOINT_NAMES, Task, state_vector


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
        # Warm up under the expert before recording; settling is not task demonstration.
        for _ in range(50):
            self.teacher_step()
        self.origin = self.data.qpos[self.root : self.root + 2].copy()

    def command(self, task: Task) -> None:
        self.expert.head_offset[2] = task.head_yaw
        self.expert.set_vel_cmd(lin_vel_x=task.vx)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self.data.qpos[self.expert.joint_qpos_indices]
        state = state_vector(
            positions, self.expert.get_base_ang_vel(), self.expert.get_projected_gravity()
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

    def apply(self, targets: np.ndarray) -> None:
        self.expert.set_position_targets(np.delete(targets, JOINT_NAMES.index("mouth")))
        self.advance()

    def metrics(self) -> dict[str, float | bool]:
        delta = self.data.qpos[self.root : self.root + 2] - self.origin
        displacement = float(delta @ np.array([np.cos(self.initial_yaw), np.sin(self.initial_yaw)]))
        z = float(self.data.qpos[self.root + 2])
        gravity_z = float(self.expert.get_projected_gravity()[2])
        return {
            "displacement_m": displacement,
            "trunk_z_m": z,
            "head_yaw_rad": float(self.data.qpos[self.expert.joint_qpos_indices[7]]),
            "stable": bool(z > 0.075 and gravity_z < -0.65 and np.isfinite(self.data.qpos).all()),
        }

    def close(self) -> None:
        self.camera.renderer.close()
