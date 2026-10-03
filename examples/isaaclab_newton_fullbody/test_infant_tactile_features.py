"""Check frozen babbling contact features and checkpoint restoration."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from infant_tactile_features import TactileFeatureEncoder
from train_infant_tactile_model import TactileModel


class InfantTactileFeaturesTests(unittest.TestCase):
    def test_encoder_snapshot_preserves_pretrained_and_random_features(self):
        input_size = 100 + 459 + 15 + 180
        model = TactileModel(input_size, 9)
        statistics = {name: (torch.zeros(width), torch.ones(width))
                      for name, width in (("qpos", 100), ("current", 459),
                                          ("tactile", 15), ("action", 180))}
        checkpoint = {"input_size": input_size, "output_size": 9,
                      "active_regions": list(range(9)),
                      "prediction_horizon_control_steps": 2,
                      "state_dict": model.state_dict(), "statistics": statistics}
        muscles = SimpleNamespace(
            qpos_ids=torch.arange(7, 97), spring=torch.zeros(90),
            moment_negative=torch.zeros(90), moment_positive=torch.zeros(90),
            reference_negative=torch.ones(90), reference_positive=torch.ones(90),
            activity=torch.zeros((2, 180)))
        environment = SimpleNamespace(position=torch.zeros((2, 100)),
                                      velocity=torch.zeros((2, 99)), muscles=muscles,
                                      source=SimpleNamespace(body_mass=np.array([1.])))
        measure = {"touch": torch.zeros((2, 15))}
        action = torch.zeros((2, 180))
        for random_features in (False, True):
            for normalize_features in (False, True):
                for representation in ("hidden", "predicted_contact", "blended_contact"):
                    encoder = TactileFeatureEncoder(checkpoint, "cpu", random_features,
                                                    normalize_features, representation)
                    encoded = encoder.encode(environment, measure, action)
                    predicted = encoder.predict(environment, measure, action)
                    restored = TactileFeatureEncoder.restore_snapshot(encoder.snapshot(), "cpu")
                    self.assertEqual(encoded.shape, (2, encoder.feature_size))
                    self.assertEqual(predicted.shape, (2, 9))
                    self.assertEqual(restored.random_features, random_features)
                    self.assertEqual(restored.normalize_features, normalize_features)
                    self.assertEqual(restored.representation, representation)
                    self.assertEqual(restored.contact_blend, 0.5)
                    self.assertEqual(restored.prediction_horizon_control_steps, 2)
                    if representation == "predicted_contact":
                        torch.testing.assert_close(encoded, predicted)
                    if representation == "blended_contact":
                        torch.testing.assert_close(encoded, predicted * 0.5)
                    torch.testing.assert_close(encoded, restored.encode(
                        environment, measure, action))
                    torch.testing.assert_close(predicted, restored.predict(
                        environment, measure, action))

    def test_blended_contact_preserves_measured_touch_at_zero_weight(self):
        model = TactileModel(8, 2)
        checkpoint = {"input_size": 8, "output_size": 2,
                      "active_regions": [0, 1], "state_dict": model.state_dict(),
                      "statistics": {name: (torch.zeros(width), torch.ones(width))
                                     for name, width in (("qpos", 2), ("current", 2),
                                                         ("tactile", 2), ("action", 2))}}
        muscles = SimpleNamespace(qpos_ids=torch.zeros(0, dtype=torch.long),
                                  spring=torch.zeros(0), moment_negative=torch.zeros(0),
                                  moment_positive=torch.zeros(0),
                                  reference_negative=torch.zeros(0),
                                  reference_positive=torch.zeros(0),
                                  activity=torch.zeros((1, 0)))
        environment = SimpleNamespace(position=torch.zeros((1, 2)),
                                      velocity=torch.zeros((1, 2)), muscles=muscles,
                                      source=SimpleNamespace(body_mass=np.array([1.])))
        measure = {"touch": torch.tensor([[1., 0.]])}
        action = torch.zeros((1, 2))
        current = torch.tensor([[1., 0.]])
        for blend in (0., 0.5, 1.):
            encoder = TactileFeatureEncoder(checkpoint, "cpu",
                                            representation="blended_contact",
                                            contact_blend=blend)
            prediction = encoder.predict(environment, measure, action)
            expected = current + blend * (prediction - current)
            torch.testing.assert_close(encoder.encode(environment, measure, action), expected)

    def test_feature_action_shuffle_uses_another_world_without_changing_predictor(self):
        model = TactileModel(8, 2)
        checkpoint = {"input_size": 8, "output_size": 2,
                      "active_regions": [0, 1], "state_dict": model.state_dict(),
                      "statistics": {name: (torch.zeros(width), torch.ones(width))
                                     for name, width in (("qpos", 2), ("current", 2),
                                                         ("tactile", 2), ("action", 2))}}
        muscles = SimpleNamespace(qpos_ids=torch.zeros(0, dtype=torch.long),
                                  spring=torch.zeros(0), moment_negative=torch.zeros(0),
                                  moment_positive=torch.zeros(0),
                                  reference_negative=torch.zeros(0),
                                  reference_positive=torch.zeros(0),
                                  activity=torch.zeros((2, 0)))
        environment = SimpleNamespace(position=torch.zeros((2, 2)),
                                      velocity=torch.zeros((2, 2)), muscles=muscles,
                                      source=SimpleNamespace(body_mass=np.array([1.])))
        measure = {"touch": torch.zeros((2, 2))}
        action = torch.tensor([[1., 0.], [0., 1.]])
        encoder = TactileFeatureEncoder(checkpoint, "cpu",
                                        representation="predicted_contact")
        original_prediction = encoder.predict(environment, measure, action)
        encoder.shuffle_actions = True
        torch.testing.assert_close(encoder.encode(environment, measure, action),
                                   encoder.predict(environment, measure,
                                                   action.roll(1, dims=0)))
        torch.testing.assert_close(encoder.predict(environment, measure, action),
                                   original_prediction)


if __name__ == "__main__":
    unittest.main()
