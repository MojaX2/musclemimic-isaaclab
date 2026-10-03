"""Check anatomically meaningful lying resets for the infant model."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from run_infant_babbling import initial_pose, load_initial_pose


MIMO_SCENE = Path(__file__).resolve().parents[3] / "MIMo/mimoEnv/assets/benchmarkv2_scene.xml"


@unittest.skipUnless(MIMO_SCENE.exists(), "MIMo reference checkout unavailable")
class InfantBabblingTests(unittest.TestCase):
    def test_supine_faces_up_and_prone_faces_down(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        data = mujoco.MjData(model)
        head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
        eye_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_eye")
        for posture, faces_up in (("supine", True), ("prone", False)):
            data.qpos[:] = initial_pose(model, posture)
            mujoco.mj_forward(model, data)
            self.assertEqual(bool(data.xpos[eye_id, 2] > data.xpos[head_id, 2]), faces_up)
            self.assertGreaterEqual(min((contact.dist for contact in data.contact), default=0), -0.01)
        np.testing.assert_array_equal(initial_pose(model, "upright"), model.qpos0)

    def test_explicit_pose_overrides_posture_and_checks_shape(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        with TemporaryDirectory() as directory:
            pose_path = Path(directory) / "pose.npz"
            expected = model.qpos0.copy()
            expected[2] += 0.01
            np.savez(pose_path, qpos=expected)
            np.testing.assert_array_equal(load_initial_pose(model, "supine", pose_path), expected)
            np.savez(pose_path, qpos=expected[:-1])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_initial_pose(model, "supine", pose_path)


if __name__ == "__main__":
    unittest.main()
