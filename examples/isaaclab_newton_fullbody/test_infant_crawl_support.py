"""Check that the support objective requires elevation and hand-knee contact."""

import unittest

import torch

from developmental_skin import REGIONS
from train_infant_crawl_support import support_score


class InfantSupportRewardTests(unittest.TestCase):
    def test_collapsed_or_no_contact_pose_scores_lower(self):
        supported = {
            "chest_height": torch.tensor([0.19]),
            "head_height": torch.tensor([0.18]),
            "support_force": torch.tensor([[5.0, 5.0, 5.0, 5.0]]),
            "ground_touch": torch.zeros((1, 15)),
        }
        collapsed = {**supported, "chest_height": torch.tensor([0.05]),
                     "head_height": torch.tensor([0.03])}
        hovering = {**supported, "support_force": torch.zeros((1, 4))}
        dragging = {**supported, "ground_touch": torch.zeros((1, 15))}
        dragging["ground_touch"][0, REGIONS.index("pelvis")] = 5.0
        self.assertGreater(float(support_score(supported)), float(support_score(collapsed)))
        self.assertGreater(float(support_score(supported)), float(support_score(hovering)))
        self.assertGreater(float(support_score(supported)), float(support_score(dragging)))

    def test_low_posture_cannot_gain_from_contacts(self):
        contact = {"chest_height": torch.tensor([0.11]),
                   "head_height": torch.tensor([0.11]),
                   "support_force": torch.tensor([[5., 5., 5., 5.]]),
                   "ground_touch": torch.zeros((1, 15))}
        hovering = {**contact, "support_force": torch.zeros((1, 4))}
        self.assertAlmostEqual(float(support_score(contact)), float(support_score(hovering)))

    def test_palm_goal_does_not_count_fingertip_only_contacts(self):
        contact = {"chest_height": torch.tensor([0.19]),
                   "head_height": torch.tensor([0.18]),
                   "support_force": torch.tensor([[5., 5., 5., 5.]]),
                   "palm_shin_force": torch.tensor([[0., 0., 5., 5.]]),
                   "ground_touch": torch.zeros((1, 15))}
        palm_contact = {**contact, "palm_shin_force": torch.tensor([[5., 5., 5., 5.]])}
        self.assertGreater(float(support_score(palm_contact, require_palms=True)),
                           float(support_score(contact, require_palms=True)))
        self.assertEqual(float(support_score(palm_contact)), float(support_score(contact)))


if __name__ == "__main__":
    unittest.main()
