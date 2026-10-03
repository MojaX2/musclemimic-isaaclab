"""Replace toe/thigh support with bilateral fingertip-and-shin floor support."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_infant_crawl_pose import support_clearance


SUPPORT_GEOMS = {f"geom:{side}_{part}" for side in ("right", "left")
                 for part in ("mfdistal1", "lower_leg1")}


def refine_pose(model, reference):
    if reference.shape != (model.nq,):
        raise ValueError("Reference pose does not match the infant model")
    pose = reference.copy()
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
    if root_id < 0:
        raise ValueError("Infant free root is missing")
    pose[model.jnt_qposadr[root_id] + 2] += 0.006
    changed_joints = []
    for side in ("right", "left"):
        for joint, offset in (("knee", 0.345), ("foot1", -1.09),
                              ("shoulder_horizontal", 0.02)):
            name = f"robot:{side}_{joint}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Required joint is missing: {name}")
            pose[model.jnt_qposadr[joint_id]] += offset
            changed_joints.append(joint_id)
        for joint in ("big_toe", "toes"):
            name = f"robot:{side}_{joint}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Required joint is missing: {name}")
            pose[model.jnt_qposadr[joint_id]] = -0.5
            changed_joints.append(joint_id)
    joint_angles = pose[model.jnt_qposadr[changed_joints]]
    joint_limits = model.jnt_range[changed_joints]
    joint_valid = bool(np.all(joint_angles >= joint_limits[:, 0]) and
                       np.all(joint_angles <= joint_limits[:, 1]))
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contacts = []
    for contact in data.contact:
        if contact.geom1 != floor_id and contact.geom2 != floor_id:
            continue
        other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        contacts.append({"geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                         "penetration_m": float(contact.dist),
                         "position_m": contact.pos.tolist()})
    contacted_geoms = {contact["geom"] for contact in contacts}
    thigh_clearance = []
    shin_clearance = []
    palm_clearance = []
    for side in ("right", "left"):
        for destination, part in ((thigh_clearance, "upper_leg1"),
                                  (shin_clearance, "lower_leg1"),
                                  (palm_clearance, "hand1")):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"geom:{side}_{part}")
            if geom_id < 0:
                raise ValueError(f"Required support geom is missing: {side}_{part}")
            destination.append(support_clearance(model, data, geom_id))
    tendon_violation = np.maximum(data.ten_length - model.tendon_range[:, 1], 0)
    tendon_violation += np.minimum(data.ten_length - model.tendon_range[:, 0], 0)
    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    worst_contact = min((float(contact.dist) for contact in data.contact), default=0.0)
    qualified = (joint_valid and contacted_geoms == SUPPORT_GEOMS and
                 all(clearance >= 0.002 for clearance in thigh_clearance) and
                 all(abs(clearance) <= 0.002 for clearance in shin_clearance) and
                 worst_contact >= -0.003 and
                 float(data.xpos[chest_id, 2]) >= 0.16 and
                 float(data.xpos[head_id, 2]) >= 0.14 and
                 float(np.max(np.abs(tendon_violation))) < 0.01)
    report = {"qualified": bool(qualified), "joint_limits_satisfied": joint_valid,
              "ground_contacts": contacts, "expected_support_geoms": sorted(SUPPORT_GEOMS),
              "thigh_clearance_m": thigh_clearance, "shin_clearance_m": shin_clearance,
              "palm_clearance_m": palm_clearance,
              "worst_contact_penetration_m": worst_contact,
              "chest_height_m": float(data.xpos[chest_id, 2]),
              "head_height_m": float(data.xpos[head_id, 2]),
              "center_of_mass_m": data.subtree_com[0].tolist(),
              "max_tendon_limit_violation": float(np.max(np.abs(tendon_violation))),
              "scope": "Static fingertip-and-shin support; not palm support or locomotion"}
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
        raise SystemExit("Refined crawl pose did not pass physical qualification")
    np.savez_compressed(args.output, qpos=pose, qvel=np.zeros(model.nv))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
