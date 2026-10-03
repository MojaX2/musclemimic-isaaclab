"""Audit infant support surfaces and zero-velocity antagonist torque capacity."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from infant_crawl_env import CRAWL_JOINTS, crawl_actuator_ids
from infant_muscles import InfantMuscles
from prepare_infant_crawl_pose import support_clearance


def audit(scene, pose):
    model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    with np.load(pose) as state:
        position = state["qpos"].copy()
    data = mujoco.MjData(model)
    data.qpos[:] = position
    mujoco.mj_forward(model, data)
    actuator_ids = crawl_actuator_ids(model)
    muscles = InfantMuscles(model, 1)
    torque_limits = {}
    for direction in ("negative", "positive"):
        action = torch.zeros((1, 2 * model.nu))
        offset = model.nu if direction == "positive" else 0
        action[:, actuator_ids + offset] = 1
        muscles.reset()
        torque = muscles.step(torch.as_tensor(position[None], dtype=torch.float32),
                              torch.zeros((1, model.nv)), action, 0.01)[0]
        torque_limits[direction] = {name: float(torque[index])
                                    for name, index in zip(CRAWL_JOINTS, actuator_ids)}
    bias = {}
    for name, actuator_id in zip(CRAWL_JOINTS, actuator_ids):
        joint_id = model.actuator_trnid[actuator_id, 0]
        bias[name] = float(data.qfrc_bias[model.jnt_dofadr[joint_id]])
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ground_contacts = []
    for contact in data.contact:
        if contact.geom1 == floor_id or contact.geom2 == floor_id:
            other_id = contact.geom2 if contact.geom1 == floor_id else contact.geom1
            ground_contacts.append({
                "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id),
                "penetration_m": float(contact.dist),
            })
    clearance = {}
    for side in ("right", "left"):
        for part, geom in (("palm", f"geom:{side}_hand1"),
                           ("thigh", f"geom:{side}_upper_leg1")):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            clearance[f"{side}_{part}"] = support_clearance(model, data, geom_id)
    return {
        "scene": str(scene.resolve()), "pose": str(pose.resolve()),
        "body_mass_kg": float(model.body_mass.sum()),
        "ground_contacts": ground_contacts,
        "surface_clearance_m": clearance,
        "static_zero_velocity_torque_capacity_nm": torque_limits,
        "gravity_bias_without_contact_nm": bias,
        "interpretation": "Torque capacity is at zero speed and full excitation; contact dynamics and stability are not proven",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    result = audit(args.scene, args.pose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("body_mass_kg", "surface_clearance_m", "ground_contacts")}, indent=2))


if __name__ == "__main__":
    main()
