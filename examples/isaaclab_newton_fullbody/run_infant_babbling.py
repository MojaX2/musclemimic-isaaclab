"""Collect infant muscle-babbling trajectories with direct GPU MuJoCo Warp.

This bypasses Newton's current MIMo MJCF conversion while retaining the same
MuJoCo Warp backend installed with Isaac Lab. It is a measurement experiment,
not a crawling or walking learner.
"""

import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from developmental_skin import REGIONS, SkinContactSensor
from infant_crawl_env import PALM_JOINTS, antagonistic_muscle_action, crawl_actuator_ids
from infant_muscles import InfantMuscles
from infant_skin import infant_geom_region_ids


def initial_pose(model, posture):
    pose = model.qpos0.copy()
    if posture == "upright":
        return pose
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
    if root_id < 0 or model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("MIMo free root joint is required for lying postures")
    address = int(model.jnt_qposadr[root_id])
    angle = -np.pi / 2 if posture == "supine" else np.pi / 2
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(quaternion, np.array([0.0, 1.0, 0.0]), angle)
    pose[address + 2] = 0.07 if posture == "supine" else 0.09
    pose[address + 3:address + 7] = quaternion
    return pose


def load_initial_pose(model, posture, pose_path=None):
    if pose_path is None:
        return initial_pose(model, posture)
    with np.load(pose_path) as state:
        pose = state["qpos"].copy()
    if pose.shape != (model.nq,):
        raise ValueError("Initial pose does not match the infant model")
    return pose


def virtual_lengths(muscles, position):
    angle = position[:, muscles.qpos_ids] - muscles.spring
    return torch.cat((angle * muscles.moment_negative + muscles.reference_negative,
                      angle * muscles.moment_positive + muscles.reference_positive), dim=1)


def shuffled_action_error(records):
    from run_motor_babbling import ridge_error

    states = torch.from_numpy(np.concatenate([record["current"] for record in records])).to("cuda:0")
    actions = torch.from_numpy(np.concatenate([record["action"] for record in records])).to("cuda:0")
    targets = torch.from_numpy(np.concatenate([record["target"] for record in records])).to("cuda:0")
    training_count = int(len(states) * 0.8)
    shuffle = torch.Generator().manual_seed(1731)
    shuffled_actions = torch.cat((
        actions[:training_count][torch.randperm(training_count, generator=shuffle).to("cuda:0")],
        actions[training_count:][torch.randperm(len(states) - training_count,
                                                generator=shuffle).to("cuda:0")],
    ))
    ablated = torch.cat((states, shuffled_actions), dim=1)
    return ridge_error(ablated[:training_count], targets[:training_count],
                       ablated[training_count:], targets[training_count:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--posture", choices=("upright", "supine", "prone"), default="supine")
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--control-steps", type=int, default=200)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--reset-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--excitation-max", type=float, default=0.3)
    parser.add_argument("--action-smoothing", type=float, default=0.85)
    parser.add_argument("--antagonistic-drives", action="store_true")
    parser.add_argument("--drive-hold-steps", type=int, default=4)
    args = parser.parse_args()
    if args.worlds < 2 or args.control_steps < 125 or args.physics_per_control < 1:
        parser.error("Use at least two worlds, 125 control steps, and one physics step per control")
    if args.reset_interval < 0 or not 0 < args.excitation_max <= 1:
        parser.error("Reset interval must be nonnegative and excitation maximum in (0, 1]")
    if not 0 <= args.action_smoothing < 1:
        parser.error("Action smoothing must be in [0, 1)")
    if args.drive_hold_steps < 1:
        parser.error("Drive hold steps must be positive")
    trajectory = args.output.with_suffix(".npz")
    if args.output.exists() or trajectory.exists():
        parser.error("Output JSON or trajectory already exists")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    robot_joints = [joint_id for joint_id in range(source.njnt)
                    if (mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT,
                                          joint_id) or "").startswith("robot:")]
    source.jnt_stiffness[robot_joints] = 0
    source.dof_damping[source.jnt_dofadr[robot_joints]] /= 20
    source.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    source_data = mujoco.MjData(source)
    source_data.qpos[:] = load_initial_pose(source, args.posture, args.pose)
    mujoco.mj_forward(source, source_data)
    initial_penetration = min((contact.dist for contact in source_data.contact), default=0.0)
    if initial_penetration < -0.01:
        raise ValueError(f"Initial infant posture penetrates the floor by {-initial_penetration:.3f} m")
    wp.set_device("cuda:0")
    warp_model = mjw.put_model(source)
    data = mjw.put_data(source, source_data, nworld=args.worlds, nconmax=4096, njmax=8192)
    position = wp.to_torch(data.qpos)
    velocity = wp.to_torch(data.qvel)
    applied_force = wp.to_torch(data.qfrc_applied)
    initial = position.clone()
    muscles = InfantMuscles(source, args.worlds, device="cuda:0")
    dof_ids = muscles.qvel_ids
    skin = SkinContactSensor(source, warp_model, data, "cuda:0",
                             region_ids=infant_geom_region_ids(source))
    total_mass = float(source.body_mass.sum())
    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    excitation = torch.full((args.worlds, 2 * source.nu), 0.02, device="cuda:0")
    actuator_ids = (torch.as_tensor(crawl_actuator_ids(source, PALM_JOINTS),
                                    device="cuda:0", dtype=torch.long)
                    if args.antagonistic_drives else None)
    drives = (torch.zeros((args.worlds, len(actuator_ids)), device="cuda:0")
              if actuator_ids is not None else None)
    target_drives = torch.zeros_like(drives) if drives is not None else None
    records = []
    max_contact_force = 0.0
    solver_limit_steps = 0
    max_joint_speed = 0.0

    def reset_worlds():
        position.copy_(initial)
        velocity.zero_()
        applied_force.zero_()
        data.ctrl.zero_()
        data.qacc_warmstart.zero_()
        muscles.reset()
        if drives is not None:
            drives.zero_()
        mjw.forward(warp_model, data)

    for control_step in range(args.control_steps):
        if control_step and args.reset_interval and control_step % args.reset_interval == 0:
            reset_worlds()
        if args.antagonistic_drives:
            if control_step % args.drive_hold_steps == 0:
                target_drives = 2 * torch.rand(drives.shape, device="cuda:0", generator=generator) - 1
            drives = args.action_smoothing * drives + (1 - args.action_smoothing) * target_drives
            excitation = antagonistic_muscle_action(drives, actuator_ids, source.nu)
        else:
            excitation = (args.action_smoothing * excitation + (1 - args.action_smoothing) * torch.rand(
                excitation.shape, device="cuda:0", generator=generator) * args.excitation_max).clamp(0, 1)
        before = torch.cat((velocity, virtual_lengths(muscles, position), muscles.activity), dim=1).clone()
        touch = skin.normalized(total_mass * 9.81).clone()
        max_contact_force = max(max_contact_force, float(skin.tensor.max()))
        for _ in range(args.physics_per_control):
            torque = muscles.step(position, velocity, excitation, float(source.opt.timestep))
            applied_force.zero_()
            applied_force[:, dof_ids] = torque
            mjw.step(warp_model, data)
            if not (torch.isfinite(position).all() and torch.isfinite(velocity).all()):
                raise RuntimeError(f"Nonfinite infant state at control step {control_step}")
            flags = data.overflow.numpy().astype(np.int64)
            if np.any(flags & 511):
                raise RuntimeError(f"MuJoCo Warp capacity overflow at control step {control_step}: {flags.tolist()}")
            solver_limit_steps += int(np.any(flags & 1536))
        max_joint_speed = max(max_joint_speed, float(velocity.abs().max()))
        records.append({
            "current": before.cpu().numpy(),
            "action": excitation.cpu().numpy(),
            "tactile": touch.cpu().numpy(),
            "target": torch.cat((velocity, virtual_lengths(muscles, position)), dim=1).cpu().numpy(),
            "qpos": position.clone().cpu().numpy(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trajectory, **{
        key: np.stack([record[key] for record in records])
        for key in ("current", "action", "tactile", "target", "qpos")
    })
    from run_motor_babbling import analyze
    result = analyze(records)
    result["mse_state_shuffled_action"] = shuffled_action_error(records)
    touches = np.concatenate([record["tactile"] for record in records])
    result.update({
        "scene": str(args.scene.resolve()), "trajectory": str(trajectory.resolve()),
        "posture": args.posture, "worlds": args.worlds, "seed": args.seed,
        "pose": str(args.pose.resolve()) if args.pose else None,
        "control_steps": args.control_steps, "physics_per_control": args.physics_per_control,
        "reset_interval": args.reset_interval, "physics_dt_s": float(source.opt.timestep),
        "action_smoothing": args.action_smoothing,
        "action_scheme": "antagonistic_palm" if args.antagonistic_drives else "independent",
        "drive_hold_steps": args.drive_hold_steps if args.antagonistic_drives else None,
        "initial_contact_penetration_m": float(initial_penetration),
        "skin_regions": list(REGIONS), "touch_nonzero_fraction": float(np.mean(touches > 0)),
        "touch_fraction_by_region": dict(zip(REGIONS, np.mean(touches > 0, axis=0).tolist())),
        "peak_contact_force_n": max_contact_force,
        "max_joint_speed_rad_s": max_joint_speed,
        "solver_limit_physics_steps": solver_limit_steps,
        "scope": "MIMo infant antagonistic muscle babbling with coarse rigid-contact touch; no learned crawl or walk",
    })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
