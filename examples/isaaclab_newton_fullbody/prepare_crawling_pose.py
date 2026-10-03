"""Fit a four-point support pose on the anatomical muscle model with MuJoCo FK."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from project_joint_equalities import project_joint_equalities


SUPPORT_BODIES = ("thirdmc_r", "thirdmc_l", "patella_r", "patella_l")
JOINT_NAMES = tuple(
    f"{joint}_{side}"
    for side in ("r", "l")
    for joint in ("hip_flexion", "hip_adduction", "knee_angle",
                  "elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex")
)


def fit_pose(model, seed):
    if seed.shape != (model.nq,):
        raise ValueError("Seed pose does not match the model")
    body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in (*SUPPORT_BODIES, "torso", "head")}
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in JOINT_NAMES]
    addresses = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    lower = np.r_[0.16, model.jnt_range[joint_ids, 0] + 1e-4]
    upper = np.r_[0.55, model.jnt_range[joint_ids, 1] - 1e-4]
    base = seed.astype(np.float64, copy=True)
    rotation = np.array([np.cos(np.pi / 4), 0., np.sin(np.pi / 4), 0.])
    mujoco.mju_mulQuat(base[3:7], rotation, seed[3:7])
    base[2] = 0.30
    center = np.r_[base[2], base[addresses]]
    center = np.clip(center, lower + 1e-3, upper - 1e-3)
    data = mujoco.MjData(model)
    targets = np.array([
        [0.38, 0.20, 0.055], [0.38, -0.20, 0.055],
        [-0.25, 0.16, 0.065], [-0.25, -0.16, 0.065],
    ])

    def forward(parameters):
        pose = base.copy()
        pose[2] = parameters[0]
        pose[addresses] = parameters[1:]
        pose, _ = project_joint_equalities(model, pose, np.zeros(model.nv))
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        return pose

    def residual(parameters):
        forward(parameters)
        positions = np.array([data.xpos[body_ids[name]] for name in SUPPORT_BODIES])
        support_error = (positions - targets) * np.array([1.5, 1.0, 4.0])
        torso_height = data.xpos[body_ids["torso"], 2]
        head_height = data.xpos[body_ids["head"], 2]
        min_contact_distance = min((contact.dist for contact in data.contact), default=0.)
        return np.r_[support_error.ravel(),
                     4.0 * max(0., 0.15 - torso_height),
                     4.0 * max(0., 0.13 - head_height),
                     8.0 * min(0., min_contact_distance + 0.003),
                     0.12 * (parameters - center)]

    result = least_squares(residual, center, bounds=(lower, upper),
                           max_nfev=150, diff_step=1e-4, ftol=1e-6)
    pose = forward(result.x)
    positions = {name: data.xpos[body_ids[name]].tolist() for name in body_ids}
    contact_distances = [float(contact.dist) for contact in data.contact]
    support_error = max(np.linalg.norm(np.array(positions[name]) - target)
                        for name, target in zip(SUPPORT_BODIES, targets))
    worst_penetration = min(contact_distances, default=0.)
    qualified = (bool(result.success) and support_error < 0.08
                 and positions["torso"][2] >= 0.15
                 and positions["head"][2] >= 0.13
                 and worst_penetration >= -0.005)
    report = {
        "optimizer_converged": bool(result.success), "qualified": qualified,
        "solver_message": result.message,
        "evaluations": int(result.nfev), "cost": float(result.cost),
        "max_support_error_m": float(support_error),
        "support_positions_m": positions,
        "support_target_positions_m": dict(zip(SUPPORT_BODIES, targets.tolist())),
        "worst_contact_penetration_m": worst_penetration,
        "contact_count": len(contact_distances),
        "joint_values_rad": dict(zip(JOINT_NAMES, result.x[1:].tolist())),
    }
    return pose, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--seed-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output already exists")
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    with np.load(args.seed_state) as source:
        seed = source["qpos"]
    pose, report = fit_pose(model, seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if report["qualified"]:
        np.savez_compressed(args.output, qpos=pose, qvel=np.zeros(model.nv))
    print(json.dumps(report, indent=2))
    if not report["qualified"]:
        raise SystemExit("Pose failed physical qualification; state was not saved")


if __name__ == "__main__":
    main()
