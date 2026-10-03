"""Check the calibrated contact prediction audit."""

import unittest

import torch

from audit_infant_contact_blend import blend_brier


class ContactBlendAuditTests(unittest.TestCase):
    def test_blend_uses_persistence_and_prediction_endpoints(self):
        current = torch.tensor([[1., 0.]])
        target = torch.tensor([[0., 0.]])
        predicted = torch.tensor([[0.2, 0.4]])
        self.assertAlmostEqual(blend_brier(current, target, predicted, 0)["brier"], 0.5)
        self.assertAlmostEqual(blend_brier(current, target, predicted, 1)["brier"], 0.1)
        self.assertAlmostEqual(blend_brier(current, target, predicted, 0.5)["brier"],
                               0.2)

    def test_invalid_blend_rejected(self):
        tensor = torch.zeros((1, 2))
        with self.assertRaisesRegex(ValueError, "Contact blend"):
            blend_brier(tensor, tensor, tensor, -0.1)


if __name__ == "__main__":
    unittest.main()
