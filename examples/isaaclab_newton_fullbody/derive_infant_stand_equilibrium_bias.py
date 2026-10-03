"""Fit antagonist drives to static inverse-dynamics torques at infant stance."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from infant_crawl_env import CRAWL_JOINTS, antagonistic_muscle_action, crawl_actuator_ids
from infant_muscles import InfantMuscles
from train_infant_stand_ppo import load_bias


def balanced_static_pose(model, pose):
    if pose.shape != (model.nq,):
        raise ValueError("Standing pose does not match the model")
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
    root = int(model.jnt_qposadr[root_id])
    data = mujoco.MjData(model)

    def residual(variables):
        state = pose.copy()
        state[root + 2] += variables[0]
        state[root + 3:root + 7] = [np.cos(variables[1] / 2), 0,
                                     np.sin(variables[1] / 2), 0]
        data.qpos[:] = state
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        return np.array([data.qfrc_inverse[root + 2] / 100,
                         data.qfrc_inverse[root + 4] / 10])

    solution = least_squares(residual, [-0.0001, -0.002],
                             bounds=([-0.001, -0.01], [0.001, 0.01]),
                             max_nfev=100, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    residual(solution.x)
    return data.qpos.copy(), data.qfrc_inverse.copy(), {
        "root_height_correction_m": float(solution.x[0]),
        "root_pitch_rad": float(solution.x[1]),
        "root_inverse_force_n": float(data.qfrc_inverse[root + 2]),
        "root_inverse_pitch_torque_nm": float(data.qfrc_inverse[root + 4]),
        "contact_count": int(data.ncon),
    }


def torque_grid(model, pose, actuator_ids, baseline, amplitude, points=401):
    drives = torch.linspace(-1, 1, points)
    all_drives = drives[:, None].expand(-1, len(actuator_ids))
    action = antagonistic_muscle_action(all_drives, actuator_ids, model.nu,
                                        baseline, amplitude)
    muscles = InfantMuscles(model, points, device="cpu")
    positions = torch.as_tensor(pose, dtype=torch.float32).expand(points, -1)
    velocities = torch.zeros((points, model.nv))
    torques = muscles.step(positions, velocities, action, 0.01)
    return drives, torques


def fit_bias(model, pose, required, baseline_bias, baseline=0.4, amplitude=0.3):
    actuator_ids = torch.as_tensor(crawl_actuator_ids(model, CRAWL_JOINTS))
    drive_grid, torques = torque_grid(model, pose, actuator_ids, baseline, amplitude)
    dof_ids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    target = torch.as_tensor(required[dof_ids[actuator_ids.numpy()]], dtype=torch.float32)
    error = (torques[:, actuator_ids] - target[None]).abs()
    indices = error.argmin(dim=0)
    selected = drive_grid[indices]
    original = baseline_bias
    original_action = antagonistic_muscle_action(original[None], actuator_ids,
                                                 model.nu, baseline, amplitude)
    original_muscles = InfantMuscles(model, 1, device="cpu")
    original_torque = original_muscles.step(torch.as_tensor(pose[None], dtype=torch.float32),
                                            torch.zeros((1, model.nv)), original_action,
                                            0.01)[0, actuator_ids]
    selected_torque = torques[indices, actuator_ids]
    return selected, {
        "required_torque_nm": target.tolist(),
        "previous_torque_nm": original_torque.tolist(),
        "fitted_torque_nm": selected_torque.tolist(),
        "previous_rms_torque_error_nm": float((original_torque - target).square().mean().sqrt()),
        "fitted_rms_torque_error_nm": float((selected_torque - target).square().mean().sqrt()),
        "fitted_drive_saturation_count": int((selected.abs() >= 0.999).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--previous-bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".npz")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or bias checkpoint already exists")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    with np.load(args.pose) as saved:
        pose = saved["qpos"].astype(np.float64)
    robot_joints = [joint_id for joint_id in range(model.njnt)
                    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          joint_id) or "").startswith("robot:")]
    model.jnt_stiffness[robot_joints] = 0
    model.dof_damping[model.jnt_dofadr[robot_joints]] /= 20
    static_pose, required, equilibrium = balanced_static_pose(model, pose)
    previous = load_bias(args.previous_bias, "cpu")
    bias, torques = fit_bias(model, static_pose, required, previous)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "previous_bias": str(args.previous_bias.resolve()),
              "equilibrium": equilibrium, "muscle": torques,
              "joint_names": CRAWL_JOINTS,
              "scope": "Static torque initialization; not learned standing or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, bias=bias.numpy(), joint_names=np.asarray(CRAWL_JOINTS))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"equilibrium": equilibrium,
                      "previous_rms_torque_error_nm": torques["previous_rms_torque_error_nm"],
                      "fitted_rms_torque_error_nm": torques["fitted_rms_torque_error_nm"]}))


if __name__ == "__main__":
    main()
