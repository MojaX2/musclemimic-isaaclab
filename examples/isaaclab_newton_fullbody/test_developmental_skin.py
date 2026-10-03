import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from developmental_skin import (REGIONS, SkinContactSensor, accumulate_foot_pressure,
                                accumulate_ground_contacts, geom_region_name)


class SkinRegionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = "/tmp/infant-skin-test-warp-cache"

    def test_newton_fixed_joint_merge_uses_geom_path(self):
        name = "MyoFullBody/worldbody/Full Body/pelvis/femur_l/tibia_l/talus_l/calcn_l/l_foot_col1_2398"
        self.assertEqual(geom_region_name(name, "femur_l"), "left_foot")

    def test_plain_geom_uses_body(self):
        self.assertEqual(geom_region_name("humerus_r_coll", "humerus_r"), "right_upper_arm")

    def test_ground_support_excludes_self_contact(self):
        active = wp.array([3], dtype=int, device="cpu")
        world = wp.array([0, 0, 0], dtype=int, device="cpu")
        pairs = wp.array(np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32),
                         dtype=wp.vec2i, device="cpu")
        regions = wp.array([-1, REGIONS.index("right_hand"), REGIONS.index("right_shin")],
                           dtype=int, device="cpu")
        forces = wp.array(np.array([[5., 0., 0., 0., 0., 0.],
                                    [11., 0., 0., 0., 0., 0.],
                                    [7., 0., 0., 0., 0., 0.]], dtype=np.float32),
                          dtype=wp.spatial_vector, device="cpu")
        output = wp.zeros((1, len(REGIONS)), dtype=float, device="cpu")
        wp.launch(accumulate_ground_contacts, dim=3,
                  inputs=[active, world, pairs, regions, forces, 0, output], device="cpu")
        self.assertEqual(float(output.numpy()[0, REGIONS.index("right_hand")]), 5.0)
        self.assertEqual(float(output.numpy()[0, REGIONS.index("right_shin")]), 7.0)

    def test_foot_pressure_centroid_excludes_nonfoot_and_self_contact(self):
        active = wp.array([5], dtype=int, device="cpu")
        world = wp.array([0, 0, 0, 0, 1], dtype=int, device="cpu")
        pairs = wp.array(np.array([[0, 1], [0, 1], [0, 2], [1, 2], [0, 3]], dtype=np.int32),
                         dtype=wp.vec2i, device="cpu")
        positions = wp.array(np.array([[1., 2., 0.], [3., 4., 0.],
                                       [9., 9., 0.], [9., 9., 0.], [5., 6., 0.]],
                                      dtype=np.float32), dtype=wp.vec3, device="cpu")
        regions = wp.array([-1, REGIONS.index("right_foot"),
                            REGIONS.index("left_hand"), REGIONS.index("left_foot")],
                           dtype=int, device="cpu")
        forces = wp.array(np.array([[2., 0., 0., 0., 0., 0.],
                                    [6., 0., 0., 0., 0., 0.],
                                    [100., 0., 0., 0., 0., 0.],
                                    [100., 0., 0., 0., 0., 0.],
                                    [4., 0., 0., 0., 0., 0.]], dtype=np.float32),
                          dtype=wp.spatial_vector, device="cpu")
        output = wp.zeros((2, 2, 3), dtype=float, device="cpu")
        wp.launch(accumulate_foot_pressure, dim=5,
                  inputs=[active, world, pairs, positions, regions, forces, 0,
                          REGIONS.index("right_foot"), REGIONS.index("left_foot"), output],
                  device="cpu")
        np.testing.assert_allclose(output.numpy()[0, 0], [8., 20., 28.])
        np.testing.assert_allclose(output.numpy()[0, 1], [0., 0., 0.])
        np.testing.assert_allclose(output.numpy()[1, 1], [4., 20., 24.])

    def test_selected_ground_geom_map(self):
        sensor = SkinContactSensor.__new__(SkinContactSensor)
        sensor.device = "cpu"
        sensor.region = wp.array([-1, 2, 3, 4], dtype=int, device="cpu")
        sensor.data = SimpleNamespace(qpos=wp.zeros((2, 7), dtype=float, device="cpu"))
        sensor.configure_ground_geoms([[2, 3], [1]])
        np.testing.assert_array_equal(sensor.selected_ground_region.numpy(), [-1, 1, 0, 0])
        self.assertEqual(tuple(sensor.selected_ground_tensor.shape), (2, 2))
        with self.assertRaises(ValueError):
            sensor.configure_ground_geoms([1, 1])


if __name__ == "__main__":
    unittest.main()
