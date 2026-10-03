"""Check that the babbling encoder can be replayed in a policy task."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from infant_world_features import WorldFeatureEncoder
from train_infant_world_model import WorldModel


class InfantWorldFeatureTests(unittest.TestCase):
    def test_snapshot_replays_frozen_features(self):
        torch.manual_seed(3)
        model = WorldModel(8, 3)
        checkpoint = {
            "state_dict": model.state_dict(), "input_size": 8, "output_size": 3,
            "mode": "touch", "statistics": {
                "current": (torch.zeros(5), torch.ones(5)),
                "action": (torch.zeros(2), torch.ones(2)),
                "tactile": (torch.zeros(1), torch.ones(1)),
            },
        }
        muscles = SimpleNamespace(qpos_ids=torch.tensor([0]), spring=torch.zeros(1),
                                  moment_negative=torch.ones(1),
                                  moment_positive=-torch.ones(1),
                                  reference_negative=torch.ones(1),
                                  reference_positive=torch.ones(1),
                                  activity=torch.tensor([[0.2, 0.3]]))
        env = SimpleNamespace(velocity=torch.tensor([[0.1]]),
                              position=torch.tensor([[0.4]]), muscles=muscles,
                              source=SimpleNamespace(body_mass=np.array([1.0])))
        measure = {"touch": torch.tensor([[9.81]])}
        action = torch.tensor([[0.5, 0.6]])
        encoder = WorldFeatureEncoder(checkpoint, "cpu")
        expected = encoder.encode(env, measure, action)
        restored = WorldFeatureEncoder.restore_snapshot(encoder.snapshot(), "cpu")
        torch.testing.assert_close(restored.encode(env, measure, action), expected)
        self.assertEqual(expected.shape, (1, 256))
        random_encoder = WorldFeatureEncoder(checkpoint, "cpu", random_features=True)
        random_snapshot = random_encoder.snapshot()
        random_restored = WorldFeatureEncoder.restore_snapshot(random_snapshot, "cpu")
        self.assertTrue(random_restored.random_features)
        torch.testing.assert_close(random_restored.encode(env, measure, action),
                                   random_encoder.encode(env, measure, action))


if __name__ == "__main__":
    unittest.main()
