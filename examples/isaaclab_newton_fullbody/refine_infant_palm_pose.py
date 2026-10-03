"""Fit bilateral palm contacts while keeping the infant's shins on the floor."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from prepare_infant_crawl_pose import support_clearance


HAND_JOINTS = ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation",
               "elbow", "hand1", "hand2", "hand3")


def finger_geom_ids(model, side):
    result = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith(f"geom:{side}_"):
            continue
        if not any(part in name for part in ("ff", "mf", "rf", "lf", "th")):
            continue
        if model.geom_type[geom_id] in (mujoco.mjtGeom.mjGEOM_BOX.value,
                                        mujoco.mjtGeom.mjGEOM_CAPSULE.value):
            result.append(geom_id)
    if not result:
        raise ValueError(f"No finger collision geoms found for {side} hand")
    return result


def refine_pose(model, reference):
    if reference.shape != (model.nq,):
        raise ValueError("Reference pose does not match the infant model")
    pose = reference.copy()
    data = mujoco.MjData(model)
    optimizations = []
    for side in ("right", "left"):
        names = [f"robot:{side}_{joint}" for joint in HAND_JOINTS]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in names]
        if min(joint_ids) < 0:
            raise ValueError(f"Hand joint is missing for {side} hand")
        addresses = model.jnt_qposadr[joint_ids]
        lower = model.jnt_range[joint_ids, 0] + 0.001
        upper = model.jnt_range[joint_ids, 1] - 0.001
        palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"geom:{side}_hand1")
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand")
        if palm_id < 0 or hand_id < 0:
            raise ValueError(f"Palm geom or hand body is missing for {side} hand")
        fingers = finger_geom_ids(model, side)
        initial = pose[addresses].copy()
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        target_xy = data.xpos[hand_id, :2].copy()

        def residual(angles):
            data.qpos[:] = pose
            data.qpos[addresses] = angles
            mujoco.mj_forward(model, data)
            palm_clearance = support_clearance(model, data, palm_id)
            finger_clearance = np.asarray([support_clearance(model, data, geom_id)
                                           for geom_id in fingers])
            return np.r_[100 * (palm_clearance + 0.001),
                         100 * np.minimum(0, finger_clearance + 0.002),
                         5 * (data.xpos[hand_id, :2] - target_xy),
                         0.03 * (angles - initial)]

        start = initial.copy()
        start[0] += 0.6
        start[5] = 1.0
        result = least_squares(residual, np.clip(start, lower + 0.001, upper - 0.001),
                               bounds=(lower, upper), max_nfev=300, ftol=1e-6)
        pose[addresses] = result.x
        optimizations.append({"side": side, "converged": bool(result.success),
                              "evaluations": int(result.nfev), "cost": float(result.cost),
                              "joint_angles_rad": dict(zip(names, result.x.tolist()))})
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contacts = []
    for contact in data.contact:
        if contact.geom1 != floor_id and contact.geom2 != floor_id:
            continue
        geom_id = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        contacts.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                    int(geom_id)),
                         "penetration_m": float(contact.dist),
                         "position_m": contact.pos.tolist()})
    floor_geoms = {contact["geom"] for contact in contacts}
    palm_clearance = []
    thigh_clearance = []
    shin_clearance = []
    finger_clearance = []
    for side in ("right", "left"):
        for destination, part in ((palm_clearance, "hand1"),
                                  (thigh_clearance, "upper_leg1"),
                                  (shin_clearance, "lower_leg1")):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"geom:{side}_{part}")
            destination.append(support_clearance(model, data, geom_id))
        finger_clearance.append(min(support_clearance(model, data, geom_id)
                                    for geom_id in finger_geom_ids(model, side)))
    tendon_violation = np.maximum(data.ten_length - model.tendon_range[:, 1], 0)
    tendon_violation += np.minimum(data.ten_length - model.tendon_range[:, 0], 0)
    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    worst_contact = min((float(contact.dist) for contact in data.contact), default=0.0)
    has_supports = all(f"geom:{side}_{part}" in floor_geoms
                       for side in ("right", "left") for part in ("hand1", "lower_leg1"))
    only_hand_shin = all(any(name.startswith(f"geom:{side}_{part}")
                             for side in ("right", "left")
                             for part in ("hand", "ff", "mf", "rf", "lf", "th", "lower_leg"))
                         for name in floor_geoms)
    qualified = (has_supports and only_hand_shin and
                 all(-0.003 <= clearance <= 0.001 for clearance in palm_clearance) and
                 all(clearance >= 0.002 for clearance in thigh_clearance) and
                 all(abs(clearance) <= 0.002 for clearance in shin_clearance) and
                 min(finger_clearance) >= -0.003 and worst_contact >= -0.004 and
                 float(data.xpos[chest_id, 2]) >= 0.16 and
                 float(np.max(np.abs(tendon_violation))) < 0.01)
    report = {"qualified": bool(qualified), "optimizations": optimizations,
              "ground_contacts": contacts, "palm_clearance_m": palm_clearance,
              "finger_min_clearance_m": finger_clearance,
              "thigh_clearance_m": thigh_clearance, "shin_clearance_m": shin_clearance,
              "worst_contact_penetration_m": worst_contact,
              "chest_height_m": float(data.xpos[chest_id, 2]),
              "center_of_mass_m": data.subtree_com[0].tolist(),
              "max_tendon_limit_violation": float(np.max(np.abs(tendon_violation))),
              "scope": "Static palm-and-shin support; dynamic crawling unverified"}
    return pose, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        parser.error("Output pose or report already exists")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    with np.load(args.reference) as state:
        reference = state["qpos"].copy()
    pose, report = refine_pose(model, reference)
    report["scene"] = str(args.scene.resolve())
    report["reference"] = str(args.reference.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["qualified"]:
        raise SystemExit("Palm-and-shin pose failed physical qualification")
    np.savez_compressed(args.output, qpos=pose, qvel=np.zeros(model.nv))
    print(json.dumps({key: report[key] for key in
                      ("qualified", "palm_clearance_m", "finger_min_clearance_m",
                       "thigh_clearance_m", "shin_clearance_m")}, indent=2))


if __name__ == "__main__":
    main()
