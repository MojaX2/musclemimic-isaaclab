"""Project randomized infant standing starts onto a shallow two-foot contact."""

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from developmental_skin import REGIONS
from infant_skin import infant_geom_region_ids


def project_foot_contact(model, positions, root_address, target_penetration_m=0.0001):
    if positions.ndim != 2 or positions.shape[1] != model.nq:
        raise ValueError("Standing positions must match the MuJoCo model")
    if target_penetration_m < 0:
        raise ValueError("Target penetration must be nonnegative")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("Standing projection requires a named floor")
    regions = infant_geom_region_ids(model)
    foot_regions = {REGIONS.index("right_foot"), REGIONS.index("left_foot")}
    feet = [geom_id for geom_id, region in enumerate(regions) if region in foot_regions]
    if not feet:
        raise ValueError("Standing projection requires foot geoms")
    data = mujoco.MjData(model)
    projected = positions.detach().cpu().numpy().copy()
    offsets = np.empty(len(projected), dtype=np.float64)
    for world_id, position in enumerate(projected):
        data.qpos[:] = position
        mujoco.mj_forward(model, data)
        distance = min(mujoco.mj_geomDistance(model, data, floor_id, geom_id,
                                               1.0, None) for geom_id in feet)
        offsets[world_id] = -target_penetration_m - distance
        projected[world_id, root_address + 2] += offsets[world_id]
    return torch.as_tensor(projected, device=positions.device,
                           dtype=positions.dtype), offsets


def align_stand_feet(model, positions, reference, root_address,
                     target_penetration_m=0.0001):
    if positions.ndim != 2 or positions.shape[1] != model.nq or reference.shape != (model.nq,):
        raise ValueError("Standing alignment positions do not match the model")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    regions = infant_geom_region_ids(model)
    sides = ("right", "left")
    foot_geoms = [[geom_id for geom_id, region in enumerate(regions)
                   if region == REGIONS.index(f"{side}_foot")] for side in sides]
    foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      f"{side}_foot") for side in sides]
    ankle_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"robot:{side}_foot{axis}")
                 for side in sides for axis in (1, 2, 3)]
    if floor_id < 0 or min(foot_bodies + ankle_ids) < 0 or any(not group for group in foot_geoms):
        raise ValueError("Standing alignment requires both feet and all ankle joints")
    ankle_addresses = model.jnt_qposadr[ankle_ids]
    data = mujoco.MjData(model)
    data.qpos[:] = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference
    mujoco.mj_forward(model, data)
    desired_orientation = [Rotation.from_matrix(data.xmat[body_id].reshape(3, 3))
                           for body_id in foot_bodies]
    aligned = positions.detach().cpu().numpy().astype(np.float64, copy=True)
    diagnostics = []
    for position in aligned:
        base_quaternion = Rotation.from_quat(position[root_address + 3:root_address + 7][[1, 2, 3, 0]])
        original_ankles = position[ankle_addresses].copy()
        original_height = float(position[root_address + 2])
        initial = np.r_[original_height, 0.0, original_ankles]
        lower = np.r_[original_height - 0.04, -0.2,
                       model.jnt_range[ankle_ids, 0] + 1e-4]
        upper = np.r_[original_height + 0.04, 0.2,
                       model.jnt_range[ankle_ids, 1] - 1e-4]

        def residual(variables):
            position[root_address + 2] = variables[0]
            rotation = Rotation.from_rotvec([variables[1], 0, 0]) * base_quaternion
            quaternion = rotation.as_quat()
            position[root_address + 3:root_address + 7] = quaternion[[3, 0, 1, 2]]
            position[ankle_addresses] = variables[2:]
            data.qpos[:] = position
            mujoco.mj_forward(model, data)
            orientation_error = [
                (desired_orientation[side].inv() *
                 Rotation.from_matrix(data.xmat[foot_bodies[side]].reshape(3, 3)))
                .as_rotvec() for side in range(2)]
            distances = [min(mujoco.mj_geomDistance(model, data, floor_id, geom_id,
                                                    1.0, None) for geom_id in group)
                         for group in foot_geoms]
            return np.r_[np.concatenate(orientation_error),
                         100 * (np.asarray(distances) + target_penetration_m),
                         0.2 * variables[1], 0.05 * (variables[2:] - original_ankles),
                         0.5 * (variables[0] - original_height)]

        solution = least_squares(residual, initial, bounds=(lower, upper),
                                 max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        final_residual = residual(solution.x)
        diagnostics.append({"success": bool(solution.success), "evaluations": solution.nfev,
                            "root_height_correction_m": float(solution.x[0] - original_height),
                            "root_roll_rad": float(solution.x[1]),
                            "foot_distance_m": (final_residual[6:8] / 100 -
                                                 target_penetration_m).tolist(),
                            "foot_orientation_error_rad": [float(np.linalg.norm(
                                final_residual[side * 3:side * 3 + 3])) for side in range(2)]})
    return torch.as_tensor(aligned, device=positions.device,
                           dtype=positions.dtype), diagnostics
