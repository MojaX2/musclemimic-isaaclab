"""Find a nonpenetrating two-foot standing start for the same MIMo infant."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def evaluate(model, data, floor_id):
    floor_contacts = []
    for contact in data.contact:
        if contact.geom1 != floor_id and contact.geom2 != floor_id:
            continue
        other = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[other]))
        floor_contacts.append((body_name, float(contact.dist)))
    return floor_contacts


def find_stand_pose(model):
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if root_id < 0 or floor_id < 0:
        raise ValueError("Infant standing pose needs a free root and named floor")
    root_height_id = int(model.jnt_qposadr[root_id]) + 2
    data = mujoco.MjData(model)
    candidates = []
    for height in np.linspace(0.315, 0.34, 251):
        data.qpos[:] = model.qpos0
        data.qpos[root_height_id] = height
        mujoco.mj_forward(model, data)
        contacts = evaluate(model, data, floor_id)
        bodies = {name for name, _ in contacts}
        if not {"right_foot", "left_foot"}.issubset(bodies):
            continue
        if any(name not in {"right_foot", "left_foot"} for name in bodies):
            continue
        worst = min(distance for _, distance in contacts)
        if worst < -0.005:
            continue
        candidates.append((abs(worst + 0.002), height, worst))
    if not candidates:
        return None, {"qualified": False, "reason": "No two-foot-only contact height passed"}
    _, height, penetration = min(candidates)
    data.qpos[:] = model.qpos0
    data.qpos[root_height_id] = height
    mujoco.mj_forward(model, data)
    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chest")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    violation = np.maximum(data.ten_length - model.tendon_range[:, 1], 0)
    violation += np.minimum(data.ten_length - model.tendon_range[:, 0], 0)
    report = {
        "qualified": bool(data.xpos[chest_id, 2] > 0.45 and data.xpos[head_id, 2] > 0.5
                          and np.max(np.abs(violation)) < 0.01),
        "root_height_m": float(height), "chest_height_m": float(data.xpos[chest_id, 2]),
        "head_height_m": float(data.xpos[head_id, 2]),
        "worst_contact_penetration_m": float(penetration),
        "floor_contacts": [{"body": name, "distance_m": distance}
                           for name, distance in evaluate(model, data, floor_id)],
        "max_tendon_limit_violation": float(np.max(np.abs(violation))),
        "scope": "Static infant two-foot contact; not balance or walking",
    }
    return data.qpos.copy(), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        parser.error("Standing pose or report already exists")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    position, report = find_stand_pose(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["qualified"]:
        raise SystemExit("Standing pose did not qualify; state was not saved")
    np.savez_compressed(args.output, qpos=position, qvel=np.zeros(model.nv))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
