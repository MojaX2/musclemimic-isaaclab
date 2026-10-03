"""Audit infant muscle torque envelopes along falling standing trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from developmental_skin import REGIONS
from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_muscles import InfantMuscles
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def envelope_violations(required, lower, upper):
    if required.shape != lower.shape or required.shape != upper.shape:
        raise ValueError("Torque arrays must have matching shapes")
    if np.any(lower > upper):
        raise ValueError("Torque envelope lower bound exceeds upper bound")
    return np.maximum(np.maximum(lower - required, required - upper), 0)


def inverse_torque(model, positions, velocities, dof_ids):
    data = mujoco.MjData(model)
    required = np.empty((len(positions), len(dof_ids)), dtype=np.float64)
    root = np.empty((len(positions), 6), dtype=np.float64)
    contacts = np.empty(len(positions), dtype=np.int32)
    for index, (position, velocity) in enumerate(zip(positions, velocities)):
        data.qpos[:] = position
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        required[index] = data.qfrc_inverse[dof_ids]
        root[index] = data.qfrc_inverse[:6]
        contacts[index] = data.ncon
    return required, root, contacts


def muscle_envelopes(model, positions, velocities, actuator_ids):
    worlds = len(positions)
    count = model.nu
    action = torch.full((worlds, 4, 2 * count), 0.4)
    for branch, negative, positive in ((0, 0.7, 0.1), (1, 0.1, 0.7),
                                       (2, 1.0, 0.0), (3, 0.0, 1.0)):
        action[:, branch, actuator_ids] = negative
        action[:, branch, actuator_ids + count] = positive
    muscles = InfantMuscles(model, worlds * 4, device="cpu")
    position_batch = torch.as_tensor(positions, dtype=torch.float32).repeat_interleave(4, 0)
    velocity_batch = torch.as_tensor(velocities, dtype=torch.float32).repeat_interleave(4, 0)
    torque = muscles.step(position_batch, velocity_batch, action.reshape(worlds * 4, -1),
                          0.01).reshape(worlds, 4, count)[:, :, actuator_ids]
    return ((torch.minimum(torque[:, 0], torque[:, 1]).numpy(),
             torch.maximum(torque[:, 0], torque[:, 1]).numpy()),
            (torch.minimum(torque[:, 2], torque[:, 3]).numpy(),
             torch.maximum(torque[:, 2], torque[:, 3]).numpy()))


def analyze(model, env, positions, velocities, upright, commanded):
    actuator_ids = env.crawl_actuators.cpu().numpy()
    dof_ids = env.crawl_qvel_ids.cpu().numpy()
    required, root, contacts = inverse_torque(model, positions, velocities, dof_ids)
    control, full = muscle_envelopes(model, positions, velocities, actuator_ids)
    result = {"upright_both_feet_count": int(upright.sum()),
              "mean_inverse_root_vertical_n": float(root[:, 2].mean()),
              "mean_inverse_root_pitch_nm": float(root[:, 4].mean()),
              "mean_contact_count": float(contacts.mean()),
              "mean_commanded_torque_error_nm": float(np.abs(required - commanded).mean())}
    for label, (lower, upper) in (("control", control), ("full_activation", full)):
        violation = envelope_violations(required, lower, upper)
        result[label] = {
            "outside_joint_fraction": float((violation > 1e-5).mean()),
            "any_outside_world_count": int((violation > 1e-5).any(axis=1).sum()),
            "mean_violation_nm": float(violation.mean()),
            "upright_outside_joint_fraction": (float((violation[upright] > 1e-5).mean())
                                               if upright.any() else None),
            "most_violated_joints": [
                {"joint": CRAWL_JOINTS[index],
                 "outside_world_count": int((violation[:, index] > 1e-5).sum()),
                 "mean_violation_nm": float(violation[:, index].mean())}
                for index in np.argsort(violation.mean(axis=0))[::-1][:6]
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--previous-bias", type=Path, required=True)
    parser.add_argument("--equilibrium-bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7301, 7303, 7307])
    parser.add_argument("--snapshot-times", nargs="+", type=float,
                        default=[0., 0.5, 1., 1.5, 2.])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.worlds < 1 or not args.seeds or not args.snapshot_times:
        parser.error("Worlds, seeds, and snapshot times must be positive and nonempty")
    if min(args.snapshot_times) < 0:
        parser.error("Snapshot times must be nonnegative")
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds)
    previous = load_bias(args.previous_bias, env.device)
    equilibrium = load_bias(args.equilibrium_bias, env.device)
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(gain_values) != 6:
        parser.error("Expected six standing pressure-reflex gains")
    gains = torch.tensor(gain_values, device=env.device).expand(args.worlds, -1)
    timestep = float(env.source.opt.timestep)
    sample_steps = {round(time / timestep) for time in args.snapshot_times}
    if len(sample_steps) != len(args.snapshot_times) or any(abs(step * timestep - time) > 1e-8
           for step, time in zip(sorted(sample_steps), sorted(args.snapshot_times))):
        parser.error("Snapshot times must be unique and align with physics steps")
    records = []
    for seed in args.seeds:
        positions = aligned_positions(env, seed, args.worlds)
        pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
        for label, bias in (("previous", previous), ("equilibrium", equilibrium)):
            env.reset(positions)
            measure = env.measure()
            for step in range(max(sample_steps) + 1):
                if step in sample_steps:
                    upright = ((measure["chest_height"] > 0.45) &
                               (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).cpu().numpy()
                    state = analyze(env.source, env, env.position.cpu().numpy().copy(),
                                    env.velocity.cpu().numpy().copy(), upright,
                                    env.applied_force[:, env.crawl_qvel_ids].cpu().numpy().copy())
                    records.append({"seed": seed, "condition": label,
                                    "pose_sha256": pose_hash, "time_s": step * timestep,
                                    **state})
                if step == max(sample_steps):
                    break
                drives = env.feedback_drives(bias.expand(args.worlds, -1), 1.5, 1.0)
                drives = cop_reflex_drives(env, measure, drives, gains)
                measure = env.step(env.action_from_drives(drives, baseline=0.4,
                                                          amplitude=0.3), 1)
            print(json.dumps({"seed": seed, "condition": label}), flush=True)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "previous_bias": str(args.previous_bias.resolve()),
              "equilibrium_bias": str(args.equilibrium_bias.resolve()),
              "evolution": str(args.evolution.resolve()),
              "seeds": args.seeds, "worlds": args.worlds,
              "snapshot_times_s": sorted(step * timestep for step in sample_steps),
              "records": records,
              "scope": "Inverse dynamics at zero acceleration and steady muscle activity; "
                       "not a proof of balance recoverability or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
