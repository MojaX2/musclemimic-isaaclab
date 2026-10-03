"""Check selective antagonist excitation limits."""

from types import SimpleNamespace
import unittest

import torch

from evaluate_infant_stand_adaptive_excitation import adaptive_antagonistic_action


class AdaptiveStandingExcitationTests(unittest.TestCase):
    def test_zero_midpoint_and_saturated_drives(self):
        environment = SimpleNamespace(
            worlds=1, crawl_actuators=torch.tensor([0, 2]),
            source=SimpleNamespace(nu=3),
            action_from_drives=lambda drives, baseline, amplitude:
            torch.full((1, 6), baseline))
        action = adaptive_antagonistic_action(environment, torch.tensor([[0., 1.]]))
        self.assertAlmostEqual(float(action[0, 0]), 0.4)
        self.assertAlmostEqual(float(action[0, 3]), 0.4)
        self.assertAlmostEqual(float(action[0, 2]), 0.)
        self.assertAlmostEqual(float(action[0, 5]), 1.)
        self.assertAlmostEqual(float(action[0, 1]), 0.4)
        opposite = adaptive_antagonistic_action(environment, torch.tensor([[0., -1.]]))
        self.assertAlmostEqual(float(opposite[0, 2]), 1.)
        self.assertAlmostEqual(float(opposite[0, 5]), 0.)
        with self.assertRaises(ValueError):
            adaptive_antagonistic_action(environment, torch.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
