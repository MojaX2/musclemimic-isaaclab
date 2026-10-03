"""Check controlled shoulder-lift sweep parameter grouping."""

import unittest
import math

import torch

from sweep_infant_crawl_shoulder_lift import lift_parameter_grid


class ShoulderLiftSweepTests(unittest.TestCase):
    def test_variants_share_fixed_gait_and_group_by_amplitude(self):
        base = torch.arange(7, dtype=torch.float32)
        grid = lift_parameter_grid(base, [0., 0.3, 0.6], 2)
        self.assertEqual(grid.shape, (6, 8))
        torch.testing.assert_close(grid[:, :7], base.expand(6, -1))
        torch.testing.assert_close(grid[:, 7], torch.tensor([0., 0., 0.3, 0.3, 0.6, 0.6]))

    def test_rejects_amplitude_outside_search_range(self):
        with self.assertRaises(ValueError):
            lift_parameter_grid(torch.zeros(7), [-0.1], 2)
        with self.assertRaises(ValueError):
            lift_parameter_grid(torch.zeros(7), [0.9], 2)

    def test_phase_offset_adds_ninth_parameter(self):
        grid = lift_parameter_grid(torch.zeros(7), [0.2, 0.4], 2, math.pi)
        self.assertEqual(grid.shape, (4, 9))
        torch.testing.assert_close(grid[:, 7], torch.tensor([0.2, 0.2, 0.4, 0.4]))
        torch.testing.assert_close(grid[:, 8], torch.full((4,), math.pi))


if __name__ == "__main__":
    unittest.main()
