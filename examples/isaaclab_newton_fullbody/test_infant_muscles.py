"""Parity checks against MIMo's original antagonist force equations."""

import ast
from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch

from infant_muscles import InfantMuscles


MIMO_ROOT = Path(__file__).resolve().parents[3] / "MIMo"
MIMO_SCENE = MIMO_ROOT / "mimoEnv/assets/benchmarkv2_scene.xml"
MIMO_MUSCLE = MIMO_ROOT / "mimoActuation/muscle.py"


def reference_model():
    syntax = ast.parse(MIMO_MUSCLE.read_text())
    muscle = next(node for node in syntax.body
                  if isinstance(node, ast.ClassDef) and node.name == "MuscleModel")
    functions = [node for node in muscle.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"fl", "fv", "fp", "_update_torque"}]
    muscle.bases = []
    muscle.body = functions
    bump = next(node for node in syntax.body
                if isinstance(node, ast.FunctionDef) and node.name == "bump")
    module = ast.Module(body=[muscle, bump], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(MIMO_MUSCLE), "exec"), namespace)
    return namespace["MuscleModel"]


@unittest.skipUnless(MIMO_SCENE.exists() and MIMO_MUSCLE.exists(), "MIMo reference checkout unavailable")
class InfantMuscleTests(unittest.TestCase):
    def test_torque_matches_mimo_reference(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        batch = InfantMuscles(model, worlds=3)
        reference_type = reference_model()
        rng = np.random.default_rng(29)
        joint_ids = model.actuator_trnid[:, 0]
        qpos = np.tile(model.qpos0.astype(np.float32), (3, 1))
        qvel = np.zeros((3, model.nv), dtype=np.float32)
        joint_range = model.jnt_range[joint_ids]
        joint_positions = rng.uniform(joint_range[:, 0], joint_range[:, 1],
                                      size=(3, model.nu)).astype(np.float32)
        qpos[:, model.jnt_qposadr[joint_ids]] = joint_positions
        joint_speeds = rng.normal(0, 2, size=(3, model.nu)).astype(np.float32)
        qvel[:, model.jnt_dofadr[joint_ids]] = joint_speeds
        actions = rng.uniform(0, 1, size=(3, 2 * model.nu)).astype(np.float32)
        actual = batch.step(torch.from_numpy(qpos), torch.from_numpy(qvel),
                            torch.from_numpy(actions), model.opt.timestep).numpy()
        expected = []
        for world in range(3):
            reference = reference_type.__new__(reference_type)
            reference.lmin, reference.lmax = 0.5, 1.6
            reference.fvmax, reference.fpmax = 1.2, 1.3
            reference.vmax = model.actuator_user[:, 0]
            reference.fmax = np.r_[model.actuator_user[:, 1], model.actuator_user[:, 2]]
            reference.n_actuators = model.nu
            reference.moment_1 = batch.moment_negative.numpy()
            reference.moment_2 = batch.moment_positive.numpy()
            angle = joint_positions[world] - batch.spring.numpy()
            reference.lce_1 = angle * reference.moment_1 + batch.reference_negative.numpy()
            reference.lce_2 = angle * reference.moment_2 + batch.reference_positive.numpy()
            reference.lce_dot_1 = joint_speeds[world] * reference.moment_1
            reference.lce_dot_2 = joint_speeds[world] * reference.moment_2
            reference.activity = actions[world] * (model.opt.timestep / 0.01)
            reference._update_torque()
            expected.append(reference.joint_torque)
        np.testing.assert_allclose(actual, np.array(expected), rtol=2e-5, atol=2e-5)

    def test_activation_and_reset(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        batch = InfantMuscles(model, worlds=2)
        qpos = torch.from_numpy(np.tile(model.qpos0.astype(np.float32), (2, 1)))
        qvel = torch.zeros((2, model.nv))
        action = torch.ones((2, 2 * model.nu))
        batch.step(qpos, qvel, action, 0.005)
        self.assertTrue(torch.allclose(batch.activity, torch.full_like(batch.activity, 0.5)))
        batch.reset([1])
        self.assertTrue(torch.all(batch.activity[1] == 0))
        self.assertTrue(torch.all(batch.activity[0] == 0.5))


if __name__ == "__main__":
    unittest.main()
