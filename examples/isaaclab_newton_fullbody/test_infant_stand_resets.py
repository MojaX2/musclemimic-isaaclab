"""Check floor projection of infant standing reset positions."""

from pathlib import Path
import unittest

import mujoco
import torch

from developmental_skin import REGIONS
from infant_skin import infant_geom_region_ids
from infant_stand_resets import align_stand_feet, project_foot_contact


SCENE = Path(__file__).resolve().parents[3] / "infant_only_scene.xml"


@unittest.skipUnless(SCENE.exists(), "Infant scene unavailable")
class InfantStandResetTests(unittest.TestCase):
    def test_alignment_places_both_feet_without_changing_hip_joints(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "mimo_orientation")
        root_address = int(model.jnt_qposadr[root_id])
        reference = torch.as_tensor(model.qpos0.copy(), dtype=torch.float32)
        reference[root_address + 2] = 0.329
        candidate = reference[None].clone()
        candidate[:, root_address + 2] += 0.007
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   "robot:right_hip1")
        hip_address = int(model.jnt_qposadr[hip_id])
        candidate[:, hip_address] -= 0.04
        aligned, diagnostics = align_stand_feet(model, candidate, reference, root_address)
        self.assertAlmostEqual(float(aligned[0, hip_address]),
                               float(candidate[0, hip_address]), places=6)
        self.assertEqual(len(diagnostics), 1)
        for distance in diagnostics[0]["foot_distance_m"]:
            self.assertAlmostEqual(distance, -0.0001, places=4)
        for error in diagnostics[0]["foot_orientation_error_rad"]:
            self.assertLess(error, 0.01)

    def test_projection_sets_shallow_foot_contact_without_changing_joints(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    "mimo_orientation")
        root_address = int(model.jnt_qposadr[root_id])
        positions = torch.as_tensor(model.qpos0.copy(), dtype=torch.float32)[None].repeat(2, 1)
        positions[:, root_address + 2] = torch.tensor([0.319, 0.339])
        projected, offsets = project_foot_contact(model, positions, root_address)
        self.assertEqual(offsets.shape, (2,))
        torch.testing.assert_close(projected[:, :root_address + 2],
                                   positions[:, :root_address + 2])
        torch.testing.assert_close(projected[:, root_address + 3:],
                                   positions[:, root_address + 3:])
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        regions = infant_geom_region_ids(model)
        feet = [geom_id for geom_id, region in enumerate(regions)
                if region in (REGIONS.index("right_foot"), REGIONS.index("left_foot"))]
        data = mujoco.MjData(model)
        for position in projected.numpy():
            data.qpos[:] = position
            mujoco.mj_forward(model, data)
            minimum = min(mujoco.mj_geomDistance(model, data, floor_id, geom_id, 1.0,
                                                  None) for geom_id in feet)
            self.assertAlmostEqual(minimum, -0.0001, places=5)


if __name__ == "__main__":
    unittest.main()
