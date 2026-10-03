"""Validate the two-foot static start against the checked-out MIMo scene."""

from pathlib import Path
import unittest

import mujoco

from prepare_infant_stand_pose import find_stand_pose


SCENE = Path(__file__).resolve().parents[3] / "infant_only_scene.xml"


@unittest.skipUnless(SCENE.exists(), "Infant scene unavailable")
class InfantStandPoseTests(unittest.TestCase):
    def test_static_pose_contacts_both_feet_without_other_bodies(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        position, report = find_stand_pose(model)
        self.assertTrue(report["qualified"])
        self.assertEqual(position.shape, (model.nq,))
        self.assertEqual({contact["body"] for contact in report["floor_contacts"]},
                         {"right_foot", "left_foot"})
        self.assertGreaterEqual(report["worst_contact_penetration_m"], -0.005)


if __name__ == "__main__":
    unittest.main()
