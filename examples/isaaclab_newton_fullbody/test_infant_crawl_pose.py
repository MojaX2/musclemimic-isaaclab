"""Check geometric floor-clearance estimates used by palm-support fitting."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from prepare_infant_crawl_pose import support_clearance


class InfantCrawlPoseTests(unittest.TestCase):
    def test_box_clearance_uses_oriented_half_extents(self):
        model = SimpleNamespace(geom_size=np.array([[0.01, 0.02, 0.03]]),
                                geom_type=np.array([mujoco.mjtGeom.mjGEOM_BOX]))
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        data = SimpleNamespace(geom_xpos=np.array([[0, 0, 0.015]]),
                               geom_xmat=rotation.reshape((1, 9)))
        self.assertAlmostEqual(support_clearance(model, data, 0), 0.005)

    def test_capsule_clearance_uses_radius_and_axial_length(self):
        model = SimpleNamespace(geom_size=np.array([[0.01, 0.03, 0]]),
                                geom_type=np.array([mujoco.mjtGeom.mjGEOM_CAPSULE]))
        data = SimpleNamespace(geom_xpos=np.array([[0, 0, 0.05]]),
                               geom_xmat=np.eye(3).reshape((1, 9)))
        self.assertAlmostEqual(support_clearance(model, data, 0), 0.01)


if __name__ == "__main__":
    unittest.main()
