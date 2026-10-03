"""Check seed-held-out splitting and tactile controls."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from train_infant_world_model import inputs, trajectory_split


class InfantWorldModelTests(unittest.TestCase):
    def test_holdout_is_not_mixed_into_training(self):
        with TemporaryDirectory() as directory:
            paths = []
            for seed in (1, 2, 3):
                path = Path(directory) / f"seed{seed}.npz"
                values = np.full((10, 2, 3), seed, dtype=np.float32)
                np.savez(path, current=values, action=values, tactile=values,
                         target=values)
                paths.append(path)
            training, validation, heldout = trajectory_split(paths, paths[-1])
        self.assertEqual(len(training["current"]), 32)
        self.assertEqual(len(validation["current"]), 8)
        self.assertEqual(len(heldout["current"]), 20)
        self.assertFalse(np.any(training["current"] == 3))
        self.assertTrue(np.all(heldout["current"] == 3))

    def test_touch_controls_preserve_other_inputs(self):
        split = {"current": torch.tensor([[1.0], [2.0]]),
                 "action": torch.tensor([[3.0], [4.0]]),
                 "tactile": torch.tensor([[5.0], [6.0]])}
        original = inputs(split, "touch")
        removed = inputs(split, "no_touch")
        shuffled = inputs(split, "shuffled_touch", torch.tensor([1, 0]))
        shuffled_action = inputs(split, "shuffled_action", torch.tensor([1, 0]))
        torch.testing.assert_close(original[:, :2], removed[:, :2])
        torch.testing.assert_close(original[:, :2], shuffled[:, :2])
        torch.testing.assert_close(removed[:, 2], torch.zeros(2))
        torch.testing.assert_close(shuffled[:, 2], torch.tensor([6.0, 5.0]))
        torch.testing.assert_close(shuffled_action[:, 1], torch.tensor([4.0, 3.0]))
        torch.testing.assert_close(shuffled_action[:, 2], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
