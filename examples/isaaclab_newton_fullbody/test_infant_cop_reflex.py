"""Check plantar feedback routing and loss-of-contact behavior."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import warp as wp

from infant_crawl_env import CRAWL_JOINTS
from sweep_infant_cop_reflex import cop_reflex_drives, evaluate, trunk_reflex_drives


class InfantCopReflexTests(unittest.TestCase):
    def test_trunk_reflex_routes_pitch_and_mirrored_roll_to_hips(self):
        position = torch.zeros((1, 7))
        position[0, 3:6] = torch.tensor([1., 0.1, 0.2])
        environment = SimpleNamespace(worlds=1, root_address=0,
                                      position=position,
                                      velocity=torch.zeros((1, 6)))
        drives = trunk_reflex_drives(
            environment, torch.zeros((1, len(CRAWL_JOINTS))),
            torch.tensor([[2., 0., 3., 0.]]))
        for side in ("right", "left"):
            self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
                f"robot:{side}_hip1")]), 0.8, places=5)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
            "robot:right_hip2")]), 0.6, places=5)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
            "robot:left_hip2")]), -0.6, places=5)

    def test_evaluation_uses_requested_excitation_range(self):
        measure = {"chest_height": torch.tensor([0.5]),
                   "ground_touch": torch.full((1, 15), 5.)}
        environment = SimpleNamespace(worlds=1, device="cpu", reset=Mock(),
                                      measure=Mock(return_value=measure),
                                      feedback_drives=Mock(return_value=torch.zeros((1, 22))),
                                      action_from_drives=Mock(return_value=torch.zeros((1, 180))),
                                      step=Mock(return_value=measure))
        with patch("sweep_infant_cop_reflex.cop_reflex_drives",
                   side_effect=lambda env, sensed, drives, gains: drives):
            standing, final, _, late = evaluate(
                environment, torch.zeros(22), torch.zeros((1, 2)), None, 2,
                return_late=True, baseline=0.5, amplitude=0.5,
                root_vertical_unload=0.25)
        self.assertEqual(environment.action_from_drives.call_count, 2)
        self.assertEqual(environment.action_from_drives.call_args.kwargs,
                         {"baseline": 0.5, "amplitude": 0.5})
        self.assertEqual(environment.step.call_args.kwargs,
                         {"root_vertical_unload": 0.25,
                          "root_planar_stiffness": 0.0,
                          "root_planar_damping": 0.0})
        torch.testing.assert_close(standing, torch.ones(1))
        torch.testing.assert_close(final, torch.ones(1))
        torch.testing.assert_close(late, torch.ones(1))

    def test_pressure_center_changes_both_ankles(self):
        com = wp.array(np.array([[[0.03, 0., 0.4]]], dtype=np.float32),
                       dtype=wp.vec3, device="cpu")
        env = SimpleNamespace(worlds=1, data=SimpleNamespace(subtree_com=com),
                              velocity=torch.zeros((1, 6)), root_address=0)
        measure = {"foot_force": torch.tensor([[5., 5.]]),
                   "foot_center_xy": torch.tensor([[[0.01, -0.05], [0.01, 0.05]]])}
        drives = cop_reflex_drives(env, measure, torch.zeros((1, len(CRAWL_JOINTS))),
                                   torch.tensor([[10., 0.]]))
        for side in ("right", "left"):
            ankle = CRAWL_JOINTS.index(f"robot:{side}_foot1")
            self.assertAlmostEqual(float(drives[0, ankle]), 0.2, places=5)

    def test_no_floor_force_disables_feedback(self):
        com = wp.array(np.array([[[0.03, 0., 0.4]]], dtype=np.float32),
                       dtype=wp.vec3, device="cpu")
        env = SimpleNamespace(worlds=1, data=SimpleNamespace(subtree_com=com),
                              velocity=torch.zeros((1, 6)), root_address=0)
        measure = {"foot_force": torch.zeros((1, 2)),
                   "foot_center_xy": torch.zeros((1, 2, 2))}
        base = torch.zeros((1, len(CRAWL_JOINTS)))
        self.assertTrue(torch.equal(cop_reflex_drives(env, measure, base,
                                                      torch.tensor([[10., 0.]])), base))

    def test_four_gains_route_sagittal_feedback_to_both_hips(self):
        com = wp.array(np.array([[[0.03, 0., 0.4]]], dtype=np.float32),
                       dtype=wp.vec3, device="cpu")
        env = SimpleNamespace(worlds=1, data=SimpleNamespace(subtree_com=com),
                              velocity=torch.zeros((1, 6)), root_address=0)
        measure = {"foot_force": torch.tensor([[5., 5.]]),
                   "foot_center_xy": torch.tensor([[[0.01, -0.05], [0.01, 0.05]]])}
        base = torch.zeros((1, len(CRAWL_JOINTS)))
        drives = cop_reflex_drives(env, measure, base,
                                   torch.tensor([[10., 0., -5., 0.]]))
        for side in ("right", "left"):
            self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
                f"robot:{side}_foot1")]), 0.2, places=5)
            self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
                f"robot:{side}_hip1")]), -0.1, places=5)

    def test_six_gains_route_lateral_feedback_with_mirrored_ankles(self):
        com = wp.array(np.array([[[0., 0.04, 0.4]]], dtype=np.float32),
                       dtype=wp.vec3, device="cpu")
        env = SimpleNamespace(worlds=1, data=SimpleNamespace(subtree_com=com),
                              velocity=torch.zeros((1, 6)), root_address=0)
        measure = {"foot_force": torch.tensor([[5., 5.]]),
                   "foot_center_xy": torch.tensor([[[0., -0.05], [0., 0.05]]])}
        base = torch.zeros((1, len(CRAWL_JOINTS)))
        drives = cop_reflex_drives(env, measure, base,
                                   torch.tensor([[0., 0., 0., 0., 5., 0.]]))
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
            "robot:right_foot2")]), 0.2, places=5)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index(
            "robot:left_foot2")]), -0.2, places=5)


if __name__ == "__main__":
    unittest.main()
