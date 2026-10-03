"""Fit and qualify a hand-knee support pose for the MIMo infant."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from developmental_skin import REGIONS
from infant_skin import infant_geom_region_ids
from run_infant_babbling import initial_pose


SUPPORT_BODIES = ("right_hand", "left_hand", "right_lower_leg", "left_lower_leg")
TARGETS = np.array([[0.20, -0.23, 0.065], [0.20, 0.23, 0.065],
                    [-0.03, -0.06, 0.025], [-0.03, 0.06, 0.025]])
JOINT_NAMES = tuple(
    f"robot:{side}_{joint}"
    for side in ("right", "left")
    for joint in ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation",
                  "elbow", "hip1", "hip2", "hip3", "knee",
                  "foot1", "foot2", "foot3")
)
HAND_JOINT_NAMES = tuple(f"robot:{side}_hand{axis}"
                         for side in ("right", "left") for axis in (1, 2, 3))


def support_clearance(model, data, geom_id):
    vertical_axes = data.geom_xmat[geom_id].reshape(3, 3)[2]
    size = model.geom_size[geom_id]
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.dot(np.abs(vertical_axes), size)
    elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + abs(vertical_axes[2]) * size[1]
    else:
        raise ValueError("Support clearance expects a box or capsule geom")
    return float(data.geom_xpos[geom_id, 2] - extent)


def fit_pose(model, palm_support=False, reference_pose=None):
    pose = initial_pose(model, "prone") if reference_pose is None else reference_pose.copy()
    targets = TARGETS.copy()
    if palm_support:
        targets[:2, 2] = 0.03
        for side in ("right", "left"):
            for finger in ("ff", "mf", "rf", "lf"):
                for part, angle in (("knuckle", 1.4), ("middle", 0.95), ("distal", 0.48)):
                    finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                    f"robot:{side}_{finger}_{part}")
                    pose[model.jnt_qposadr[finger_id]] = angle
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
    root_address = int(model.jnt_qposadr[root_id])
    joint_names = JOINT_NAMES + HAND_JOINT_NAMES if palm_support else JOINT_NAMES
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    if min(joint_ids) < 0:
        raise ValueError("The MIMo infant is missing a required crawl joint")
    addresses = model.jnt_qposadr[joint_ids]
    lower = np.r_[0.10, np.deg2rad(65), model.jnt_range[joint_ids, 0] + 1e-5]
    upper = np.r_[0.28, np.deg2rad(115), model.jnt_range[joint_ids, 1] - 1e-5]
    center = np.r_[0.18 if reference_pose is None else pose[root_address + 2],
                   np.pi / 2, pose[addresses]]
    if reference_pose is None:
        for side in ("right", "left"):
            for joint, angle in (("shoulder_horizontal", 0.4), ("shoulder_ad_ab", 0.8),
                                 ("shoulder_rotation", -0.5), ("elbow", -0.9),
                                 ("hip1", -1.3), ("knee", -1.7)):
                center[2 + joint_names.index(f"robot:{side}_{joint}")] = angle
    center = np.clip(center, lower + 1e-4, upper - 1e-4)
    data = mujoco.MjData(model)
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (*SUPPORT_BODIES, "chest", "head")]
    palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"geom:{side}_hand1") for side in ("right", "left")]
    thigh_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"geom:{side}_upper_leg1") for side in ("right", "left")]
    if palm_support and min(*palm_ids, *thigh_ids) < 0:
        raise ValueError("Palm support geoms are missing")

    def forward(parameters):
        pose[root_address + 2] = parameters[0]
        quaternion = np.empty(4)
        mujoco.mju_axisAngle2Quat(quaternion, np.array([0.0, 1.0, 0.0]), parameters[1])
        pose[root_address + 3:root_address + 7] = quaternion
        pose[addresses] = parameters[2:]
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)

    def residual(parameters):
        forward(parameters)
        positions = data.xpos[body_ids[:4]]
        support_error = ((positions - targets) * np.array([5.0, 2.0, 8.0])).ravel()
        head_height = data.xpos[body_ids[5], 2]
        torso_height = data.xpos[body_ids[4], 2]
        penetration = min((contact.dist for contact in data.contact), default=0.0)
        tendon_violation = np.maximum(data.ten_length - model.tendon_range[:, 1], 0)
        tendon_violation += np.minimum(data.ten_length - model.tendon_range[:, 0], 0)
        contact_clearance = []
        if palm_support:
            contact_clearance = [80 * support_clearance(model, data, geom_id)
                                 for geom_id in palm_ids]
            contact_clearance += [15 * min(0.0, support_clearance(model, data, geom_id) - 0.005)
                                  for geom_id in thigh_ids]
        return np.r_[support_error,
                     10 * max(0.0, 0.16 - torso_height),
                     10 * max(0.0, 0.14 - head_height),
                     80 * min(0.0, penetration + 0.003),
                     contact_clearance,
                     10 * tendon_violation,
                     0.03 * (parameters - center)]

    result = least_squares(residual, center, bounds=(lower, upper),
                           max_nfev=200, ftol=1e-7, xtol=1e-7)
    forward(result.x)
    support_error = np.linalg.norm(data.xpos[body_ids[:4]] - targets, axis=1)
    penetration = min((contact.dist for contact in data.contact), default=0.0)
    tendon_violation = np.maximum(data.ten_length - model.tendon_range[:, 1], 0)
    tendon_violation += np.minimum(data.ten_length - model.tendon_range[:, 0], 0)
    contact_bodies = set()
    contact_regions = set()
    palm_contacts = set()
    region_ids = infant_geom_region_ids(model)
    for contact in data.contact:
        if contact.geom1 == 0 or contact.geom2 == 0:
            other = contact.geom2 if contact.geom1 == 0 else contact.geom1
            contact_bodies.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(model.geom_bodyid[other])))
            if other in palm_ids:
                palm_contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                    int(other)))
            if region_ids[other] >= 0:
                contact_regions.add(REGIONS[region_ids[other]])
    qualified = (bool(result.success) and max(support_error) < 0.06
                 and data.xpos[body_ids[4], 2] >= 0.16
                 and data.xpos[body_ids[5], 2] >= 0.14
                 and penetration >= -0.005
                 and max(np.abs(tendon_violation)) < 0.01
                 and {"right_hand", "left_hand", "right_shin", "left_shin"}.issubset(contact_regions)
                 and (not palm_support or
                      (len(palm_contacts) == 2 and
                       all(support_clearance(model, data, geom_id) >= 0.002
                           for geom_id in thigh_ids))))
    report = {
        "optimizer_converged": bool(result.success), "qualified": bool(qualified),
        "cost": float(result.cost), "evaluations": int(result.nfev),
        "support_body_position_m": dict(zip(SUPPORT_BODIES,
                                             data.xpos[body_ids[:4]].tolist())),
        "support_target_m": dict(zip(SUPPORT_BODIES, targets.tolist())),
        "support_error_m": dict(zip(SUPPORT_BODIES, support_error.tolist())),
        "chest_height_m": float(data.xpos[body_ids[4], 2]),
        "head_height_m": float(data.xpos[body_ids[5], 2]),
        "worst_contact_penetration_m": float(penetration),
        "max_tendon_limit_violation": float(max(np.abs(tendon_violation))),
        "ground_contact_bodies": sorted(contact_bodies),
        "ground_contact_regions": sorted(contact_regions),
        "palm_support_required": bool(palm_support),
        "palm_ground_contacts": sorted(palm_contacts),
        "palm_clearance_m": [support_clearance(model, data, geom_id) for geom_id in palm_ids],
        "thigh_clearance_m": [support_clearance(model, data, geom_id) for geom_id in thigh_ids],
        "joint_angles_rad": dict(zip(joint_names, result.x[2:].tolist())),
    }
    return pose.copy(), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palm-support", action="store_true")
    parser.add_argument("--initial-pose", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output pose or report already exists")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    reference_pose = None
    if args.initial_pose is not None:
        with np.load(args.initial_pose) as state:
            reference_pose = state["qpos"].copy()
    pose, report = fit_pose(model, palm_support=args.palm_support,
                            reference_pose=reference_pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["qualified"]:
        np.savez_compressed(args.output, qpos=pose, qvel=np.zeros(model.nv))
    print(json.dumps(report, indent=2))
    if not report["qualified"]:
        raise SystemExit("Crawling pose failed physical qualification; state was not saved")


if __name__ == "__main__":
    main()
