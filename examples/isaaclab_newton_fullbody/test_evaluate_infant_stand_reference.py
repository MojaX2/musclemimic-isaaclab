"""Check contact-aligned standing feedback references."""

from types import SimpleNamespace
import unittest

import torch

from evaluate_infant_stand_reference import reference_feedback_drives


class StandReferenceTests(unittest.TestCase):
    def test_reference_endpoints_and_midpoint(self):
        initial = torch.tensor([[0., 0., 0.]])
        positions = torch.tensor([[0.2, -0.4, 0.]])
        environment = SimpleNamespace(worlds=1, position=initial.clone(), initial=initial,
                                      velocity=torch.zeros((1, 3)),
                                      crawl_qpos_ids=torch.tensor([0, 1]),
                                      crawl_qvel_ids=torch.tensor([0, 1]))
        bias = torch.zeros((1, 2))
        torch.testing.assert_close(reference_feedback_drives(environment, bias, positions, 0),
                                   torch.zeros((1, 2)))
        torch.testing.assert_close(reference_feedback_drives(environment, bias, positions, 0.5),
                                   torch.tensor([[0.15, -0.3]]))
        torch.testing.assert_close(reference_feedback_drives(environment, bias, positions, 1),
                                   torch.tensor([[0.3, -0.6]]))

    def test_invalid_blend_and_position_shape(self):
        environment = SimpleNamespace(worlds=1, position=torch.zeros((1, 3)),
                                      initial=torch.zeros((1, 3)),
                                      velocity=torch.zeros((1, 3)),
                                      crawl_qpos_ids=torch.tensor([0, 1]),
                                      crawl_qvel_ids=torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            reference_feedback_drives(environment, torch.zeros((1, 2)),
                                      torch.zeros((1, 3)), 1.1)
        with self.assertRaises(ValueError):
            reference_feedback_drives(environment, torch.zeros((1, 2)),
                                      torch.zeros((2, 3)), 0.5)


if __name__ == "__main__":
    unittest.main()
