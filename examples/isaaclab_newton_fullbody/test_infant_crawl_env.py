"""Test the reduced antagonist action parameterization for crawling."""

from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch

from infant_crawl_env import (CRAWL_JOINTS, PALM_JOINTS, InfantCrawlEnv,
                              antagonistic_muscle_action, crawl_actuator_ids)


MIMO_SCENE = Path(__file__).resolve().parents[3] / "MIMo/mimoEnv/assets/benchmarkv2_scene.xml"


@unittest.skipUnless(MIMO_SCENE.exists(), "MIMo reference checkout unavailable")
class InfantCrawlActionTests(unittest.TestCase):
    def test_crawl_joint_actuators_are_unique(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        indices = crawl_actuator_ids(model)
        self.assertEqual(len(indices), len(CRAWL_JOINTS))
        self.assertEqual(len(np.unique(indices)), len(indices))

    def test_palm_joint_set_includes_wrist_and_toe_actuators(self):
        model = mujoco.MjModel.from_xml_path(str(MIMO_SCENE))
        indices = crawl_actuator_ids(model, PALM_JOINTS)
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(np.unique(indices)), len(indices))
        self.assertEqual(PALM_JOINTS[:len(CRAWL_JOINTS)], CRAWL_JOINTS)

    def test_signed_drive_excites_opposing_muscles(self):
        env = InfantCrawlEnv.__new__(InfantCrawlEnv)
        env.worlds, env.device = 2, "cpu"
        env.source = type("Model", (), {"nu": 90})()
        env.crawl_actuators = torch.arange(len(CRAWL_JOINTS))
        drive = torch.ones((2, len(CRAWL_JOINTS)))
        action = env.action_from_drives(drive)
        self.assertTrue(torch.all(action[:, :len(CRAWL_JOINTS)] == 0))
        self.assertTrue(torch.all(action[:, 90:90 + len(CRAWL_JOINTS)] == 0.5))

    def test_babbling_action_matches_crawl_action(self):
        env = InfantCrawlEnv.__new__(InfantCrawlEnv)
        env.worlds, env.device = 2, "cpu"
        env.source = type("Model", (), {"nu": 90})()
        env.crawl_actuators = torch.tensor([2, 7])
        drives = torch.tensor([[1.0, -1.0], [0.25, 0.0]])
        torch.testing.assert_close(
            antagonistic_muscle_action(drives, env.crawl_actuators, 90),
            env.action_from_drives(drives))
        action = env.action_from_drives(drives)
        self.assertEqual(float(action[0, 2]), 0.0)
        self.assertEqual(float(action[0, 92]), 0.5)
        self.assertAlmostEqual(float(action[0, 7]), 0.5)
        self.assertAlmostEqual(float(action[0, 97]), 0.0)
        self.assertAlmostEqual(float(action[0, 0]), 0.2)

    def test_feedback_drive_reacts_to_position_error(self):
        env = InfantCrawlEnv.__new__(InfantCrawlEnv)
        env.worlds = 1
        env.crawl_qpos_ids = torch.arange(len(CRAWL_JOINTS))
        env.crawl_qvel_ids = torch.arange(len(CRAWL_JOINTS))
        env.initial = torch.zeros((1, len(CRAWL_JOINTS)))
        env.position = torch.full_like(env.initial, -0.1)
        env.velocity = torch.zeros_like(env.initial)
        drive = env.feedback_drives(torch.zeros_like(env.initial), 2.0, 0.5)
        torch.testing.assert_close(drive, torch.full_like(drive, 0.2))


if __name__ == "__main__":
    unittest.main()
