"""Check anchor splitting preserves kinematics and adjacent-body filtering."""

import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from split_infant_joint_anchors import split_joint_anchors


SCENE = """
<mujoco>
  <option jacobian="dense"/>
  <worldbody>
    <body name="parent" pos="0 0 0.2">
      <freejoint/>
      <geom name="parent_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="child" pos="0 0 0.02">
        <joint name="first" axis="0 0 1" pos="0 0 0"/>
        <joint name="second" axis="0 1 0" pos="0.01 0 0"/>
        <geom name="child_geom" type="box" size="0.04 0.04 0.04" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class SplitJointAnchorTests(unittest.TestCase):
    def test_kinematics_and_parent_collision_filter(self):
        tree = ET.ElementTree(ET.fromstring(SCENE))
        self.assertEqual(split_joint_anchors(tree), ["child"])
        self.assertEqual(tree.find("option").get("jacobian"), "sparse")
        original = mujoco.MjModel.from_xml_string(SCENE)
        split = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot()))
        self.assertEqual((original.nq, original.nv, original.ngeom),
                         (split.nq, split.nv, split.ngeom))
        self.assertEqual(split.nexclude, original.nexclude + 1)
        for angles in ((0.0, 0.0), (0.3, -0.2)):
            original_data = mujoco.MjData(original)
            split_data = mujoco.MjData(split)
            for model, data in ((original, original_data), (split, split_data)):
                data.qpos[:] = model.qpos0
                data.qpos[-2:] = angles
                mujoco.mj_forward(model, data)
            np.testing.assert_allclose(original_data.geom_xpos, split_data.geom_xpos,
                                       atol=1e-7, rtol=0)
            np.testing.assert_allclose(original_data.geom_xmat, split_data.geom_xmat,
                                       atol=1e-7, rtol=0)
            self.assertEqual(original_data.ncon, split_data.ncon)
            self.assertEqual(split_data.ncon, 0)

    def test_rejects_nonpositive_mass(self):
        tree = ET.ElementTree(ET.fromstring(SCENE))
        with self.assertRaises(ValueError):
            split_joint_anchors(tree, mass=0)


if __name__ == "__main__":
    unittest.main()
