"""Check contact-prediction alignment across infant babbling resets."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from train_infant_tactile_model import (TactileModel, load_transitions, model_input,
                                        prepare_features, remap_input_normalization)


class InfantTactileModelTests(unittest.TestCase):
    def test_normalization_remap_preserves_pretrained_prediction(self):
        torch.manual_seed(31)
        original = TactileModel(4, 2)
        names = ("qpos", "current", "tactile", "action")
        old_statistics = {name: (torch.tensor([float(index)]),
                                 torch.tensor([float(index + 1)]))
                          for index, name in enumerate(names)}
        new_statistics = {name: (np.array([index * 0.3], dtype=np.float32),
                                 np.array([1.5 + index], dtype=np.float32))
                          for index, name in enumerate(names)}
        checkpoint = {"input_size": 4, "statistics": old_statistics,
                      "state_dict": original.state_dict()}
        remapped = TactileModel(4, 2)
        remapped.load_state_dict(remap_input_normalization(checkpoint, new_statistics))
        raw = torch.randn((5, 4))
        old_input = torch.cat([(raw[:, index:index + 1] - old_statistics[name][0]) /
                               old_statistics[name][1]
                               for index, name in enumerate(names)], dim=1)
        new_input = torch.cat([(raw[:, index:index + 1] -
                                torch.as_tensor(new_statistics[name][0])) /
                               torch.as_tensor(new_statistics[name][1])
                               for index, name in enumerate(names)], dim=1)
        torch.testing.assert_close(original(old_input), remapped(new_input),
                                   atol=1e-6, rtol=1e-6)

    def test_explicit_newton_transitions_keep_pre_action_alignment(self):
        steps = np.arange(4, dtype=np.float32)[:, None, None]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "newton.npz"
            np.savez_compressed(path, qpos_before=steps + 100,
                                current=steps + 200, action=steps + 300,
                                tactile=steps + 400, next_tactile=steps + 500)
            records = load_transitions(path, reset_interval=0)
        np.testing.assert_array_equal(records["step"], [0, 1, 2, 3])
        np.testing.assert_array_equal(records["qpos"][:, 0], [100, 101, 102, 103])
        np.testing.assert_array_equal(records["target"][:, 0], [500, 501, 502, 503])

    def test_transitions_exclude_resets_and_use_pre_action_qpos(self):
        steps = np.arange(8, dtype=np.float32)[:, None, None]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez_compressed(path, qpos=steps + 100, current=steps + 200,
                                action=steps + 300, tactile=steps + 400)
            records = load_transitions(path, reset_interval=4)
        np.testing.assert_array_equal(records["step"], [1, 2, 5, 6])
        np.testing.assert_array_equal(records["qpos"][:, 0], [100, 101, 104, 105])
        np.testing.assert_array_equal(records["action"][:, 0], [301, 302, 305, 306])
        np.testing.assert_array_equal(records["target"][:, 0], [402, 403, 406, 407])

    def test_multistep_targets_do_not_cross_reset(self):
        steps = np.arange(12, dtype=np.float32)[:, None, None]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez_compressed(path, qpos=steps + 100, current=steps + 200,
                                action=steps + 300, tactile=steps + 400)
            records = load_transitions(path, reset_interval=6, prediction_horizon=3)
        np.testing.assert_array_equal(records["step"], [1, 2, 7, 8])
        np.testing.assert_array_equal(records["qpos"][:, 0], [100, 101, 106, 107])
        np.testing.assert_array_equal(records["target"][:, 0], [404, 405, 410, 411])

    def test_explicit_multistep_target_uses_later_post_action_contact(self):
        steps = np.arange(4, dtype=np.float32)[:, None, None]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "explicit.npz"
            np.savez_compressed(path, qpos_before=steps + 100, current=steps,
                                action=steps, tactile=steps + 200,
                                next_tactile=steps + 300)
            records = load_transitions(path, reset_interval=0, prediction_horizon=2)
            np.testing.assert_array_equal(records["step"], [0, 1, 2])
            np.testing.assert_array_equal(records["target"][:, 0], [301, 302, 303])
            with self.assertRaises(ValueError):
                load_transitions(path, reset_interval=20, prediction_horizon=2)
            with self.assertRaises(ValueError):
                load_transitions(path, reset_interval=0, prediction_horizon=4)

    def test_action_ablation_keeps_feature_width(self):
        split = {"qpos": torch.ones((2, 1)), "current": torch.ones((2, 1)),
                 "tactile": torch.ones((2, 1)), "action": torch.tensor([[1., 2.], [3., 4.]])}
        action = model_input(split, "action")
        no_action = model_input(split, "no_action")
        shuffled = model_input(split, "shuffled_action", torch.tensor([1, 0]))
        self.assertEqual(action.shape, no_action.shape)
        torch.testing.assert_close(no_action[:, -2:], torch.zeros((2, 2)))
        torch.testing.assert_close(shuffled[:, -2:], action.flip(0)[:, -2:])

    def test_explicit_regions_keep_comparisons_on_same_skin_parts(self):
        records = {"qpos": np.zeros((4, 1)), "current": np.zeros((4, 1)),
                   "tactile": np.zeros((4, 15)), "action": np.zeros((4, 1)),
                   "target": np.zeros((4, 15))}
        prepared, _, active = prepare_features((records, records, records), "cpu",
                                               ["right_hand", "left_hand"])
        np.testing.assert_array_equal(active, [6, 7])
        self.assertEqual(prepared[0]["label"].shape, (4, 2))
        with self.assertRaises(ValueError):
            prepare_features((records, records, records), "cpu",
                             ["right_hand", "right_hand"])


if __name__ == "__main__":
    unittest.main()
