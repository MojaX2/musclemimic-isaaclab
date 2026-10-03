"""Check the action-shuffle control for infant tactile prediction."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from evaluate_infant_newton_crawl import BlendedActor, ShuffledActionPredictor


class NewtonTactileAuditTests(unittest.TestCase):
    def test_blended_actor_interpolates_muscle_drives(self):
        primary = SimpleNamespace(actor=Mock(return_value=torch.tensor([[0., 1.]])))
        secondary = SimpleNamespace(actor=Mock(return_value=torch.tensor([[1., -1.]])))
        observation = torch.zeros((1, 3))
        torch.testing.assert_close(BlendedActor(primary, secondary, 0.25).actor(observation),
                                   torch.tensor([[0.25, 0.5]]))
        with self.assertRaises(ValueError):
            BlendedActor(primary, secondary, 1.1)

    def test_action_shuffle_preserves_values_but_changes_world_pairing(self):
        encoder = SimpleNamespace(active_regions=[0, 2], body_weight=10.,
                                  predict=Mock(return_value=torch.ones((3, 2))))
        predictor = ShuffledActionPredictor(encoder)
        action = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        environment = object()
        measure = object()
        torch.testing.assert_close(predictor.predict(environment, measure, action),
                                   torch.ones((3, 2)))
        arguments = encoder.predict.call_args.args
        self.assertIs(arguments[0], environment)
        self.assertIs(arguments[1], measure)
        torch.testing.assert_close(arguments[2], action.roll(1, dims=0))
        self.assertEqual(predictor.active_regions, [0, 2])
        self.assertEqual(predictor.body_weight, 10.)


if __name__ == "__main__":
    unittest.main()
