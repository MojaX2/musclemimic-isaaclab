"""Verify vestibular reflex routing to left and right antagonistic muscles."""

from types import SimpleNamespace
import unittest

import torch

from infant_crawl_env import CRAWL_JOINTS
from train_infant_balance_reflex import balance_reflex_drives


class InfantBalanceReflexTests(unittest.TestCase):
    def test_pitch_is_symmetric_and_roll_is_antisymmetric(self):
        position = torch.zeros((1, 7))
        position[0, 3] = 1
        position[0, 4] = 0.1
        position[0, 5] = 0.1
        environment = SimpleNamespace(worlds=1, root_address=0,
                                      position=position, velocity=torch.zeros((1, 6)))
        gains = torch.zeros((1, 12))
        gains[0, 0] = 2
        gains[0, 4] = 1
        drives = balance_reflex_drives(environment, torch.zeros((1, len(CRAWL_JOINTS))), gains)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index("robot:right_hip1")]), 0.4)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index("robot:left_hip1")]), 0.4)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index("robot:right_hip2")]), 0.2)
        self.assertAlmostEqual(float(drives[0, CRAWL_JOINTS.index("robot:left_hip2")]), -0.2)

    def test_gain_shape_is_checked(self):
        environment = SimpleNamespace(worlds=1, root_address=0,
                                      position=torch.zeros((1, 7)),
                                      velocity=torch.zeros((1, 6)))
        with self.assertRaises(ValueError):
            balance_reflex_drives(environment, torch.zeros((1, len(CRAWL_JOINTS))),
                                   torch.zeros((1, 10)))


if __name__ == "__main__":
    unittest.main()
