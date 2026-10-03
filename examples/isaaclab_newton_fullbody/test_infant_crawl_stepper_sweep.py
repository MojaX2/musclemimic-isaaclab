"""Validate matched parameter blocks for contact-driven stepper experiments."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from sweep_infant_crawl_stepper import BlendedSupportPolicy, checkpoint_variants, parameter_grid


class StepperSweepTests(unittest.TestCase):
    def test_policy_blend_interpolates_actor_drives(self):
        class ConstantPolicy:
            def __init__(self, value):
                self.value = value

            def actor(self, observation):
                return torch.full_like(observation, self.value)

        observation = torch.zeros((2, 3))
        blended = BlendedSupportPolicy(ConstantPolicy(-1.), ConstantPolicy(1.), 0.25)
        torch.testing.assert_close(blended.actor(observation),
                                   torch.full_like(observation, -0.5))

    def test_candidate_blocks_match_world_count(self):
        grid = parameter_grid(((0., 0., 0., 0., 0., 0.),
                               (0.5, 0.3, 0.4, 0.2, 0., 0.)), 3, "cpu")
        self.assertEqual(grid.shape, (6, 6))
        torch.testing.assert_close(grid[:3], torch.zeros((3, 6)))
        torch.testing.assert_close(grid[3:], torch.tensor([[0.5, 0.3, 0.4, 0.2, 0., 0.]]).expand(3, -1))

    def test_rejects_excessive_drives(self):
        with self.assertRaises(ValueError):
            parameter_grid(((1.1, 0., 0., 0., 0., 0.),), 1, "cpu")

    def test_checkpoint_variants_include_zero_drive_control(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stepper.npz"
            np.savez_compressed(path, parameters=np.array([0.5, 0.3, 0.4, 0.2, 0., 0.]))
            variants = checkpoint_variants(path)
            self.assertEqual(variants[0], (0.,) * 6)
            self.assertEqual(len(variants[1]), 6)


if __name__ == "__main__":
    unittest.main()
