"""Compare infant standing with muscle feedback and idealized joint torques."""

import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def robot_joint_addresses(model):
    joint_ids = [joint_id for joint_id in range(model.njnt)
                 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                       joint_id) or "").startswith("robot:")]
    position = torch.as_tensor(model.jnt_qposadr[joint_ids].copy(), device="cuda:0")
    velocity = torch.as_tensor(model.jnt_dofadr[joint_ids].copy(), device="cuda:0")
    return position, velocity


def record(env, measure, elapsed):
    forces = measure["foot_force"]
    total_force = forces.sum(-1)
    cop_x = ((measure["foot_center_xy"][:, :, 0] * forces).sum(-1) /
             total_force.clamp_min(1e-8))
    center_of_mass_x = wp.to_torch(env.data.subtree_com)[:, 0, 0]
    standing = ((measure["chest_height"] > 0.45) &
                (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1))
    joint_error = (env.position[:, env.crawl_qpos_ids] -
                   env.initial[:, env.crawl_qpos_ids]).abs().mean(0)
    leading_joints = torch.topk(joint_error, 5)
    return {"time_s": elapsed,
            "standing_fraction": float(standing.float().mean()),
            "mean_chest_height_m": float(measure["chest_height"].mean()),
            "mean_head_height_m": float(measure["head_height"].mean()),
            "mean_foot_force_n": float(total_force.mean()),
            "mean_com_minus_cop_x_m": float((center_of_mass_x - cop_x).mean()),
            "mean_root_x_m": float(env.position[:, env.root_address].mean()),
            "mean_root_height_m": float(env.position[:, env.root_address + 2].mean()),
            "mean_joint_error_rad": float(joint_error.mean()),
            "largest_joint_errors": [
                {"joint": CRAWL_JOINTS[int(index)], "mean_error_rad": float(error)}
                for error, index in zip(leading_joints.values, leading_joints.indices)],
            "mean_abs_controlled_torque_nm": float(env.applied_force[:, env.crawl_qvel_ids]
                                                    .abs().mean())}


def evaluate(scene, pose, bias_path, mode, seeds, duration_s, torque_gain,
             torque_damping, torque_limit):
    worlds = len(seeds) * 8
    env = InfantCrawlEnv(scene, pose, worlds)
    positions = []
    for seed in seeds:
        torch.manual_seed(seed)
        block = jittered_positions(env, 0.05, 0.01)
        positions.append(block[:8].clone())
    env.reset(torch.cat(positions, dim=0))
    bias = load_bias(bias_path, env.device)
    joint_position, joint_velocity = robot_joint_addresses(env.source)
    timestep = float(env.source.opt.timestep)
    steps = round(duration_s / timestep)
    history = [record(env, env.measure(), 0.0)]
    standing_steps = torch.zeros(worlds, device=env.device)
    for step in range(steps):
        if mode == "muscle":
            drives = env.feedback_drives(bias.expand(worlds, -1), 1.5, 1.0)
            measure = env.step(env.action_from_drives(drives, baseline=0.4,
                                                       amplitude=0.3), 1)
        else:
            error = env.initial[:, joint_position] - env.position[:, joint_position]
            torque = (torque_gain * error -
                      torque_damping * env.velocity[:, joint_velocity]).clamp(
                          -torque_limit, torque_limit)
            env.applied_force.zero_()
            if mode == "hybrid":
                drives = env.feedback_drives(bias.expand(worlds, -1), 1.5, 1.0)
                action = env.action_from_drives(drives, baseline=0.4, amplitude=0.3)
                muscle_torque = env.muscles.step(env.position, env.velocity, action,
                                                  timestep)
                env.applied_force[:, env.muscles.qvel_ids] = muscle_torque
            env.applied_force[:, joint_velocity] += torque
            mjw.step(env.model, env.data)
            measure = env.measure()
        standing_steps += ((measure["chest_height"] > 0.45) &
                           (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
        if (step + 1) % max(1, round(0.25 / timestep)) == 0 or step + 1 == steps:
            history.append(record(env, measure, (step + 1) * timestep))
    return {"mode": mode, "seeds": list(seeds), "worlds": worlds,
            "duration_s": steps * timestep, "torque_gain": torque_gain,
            "torque_damping": torque_damping, "torque_limit": torque_limit,
            "mean_standing_fraction": float((standing_steps / steps).mean()),
            "final_standing_count": int(((measure["chest_height"] > 0.45) &
                                          (measure["ground_touch"][:, FOOT_IDS] > 2)
                                          .all(-1)).sum()), "history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 43, 47])
    parser.add_argument("--torque-gain", type=float, default=100.0)
    parser.add_argument("--torque-damping", type=float, default=10.0)
    parser.add_argument("--torque-limit", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.duration_s <= 0 or min(args.torque_gain, args.torque_damping,
                                   args.torque_limit) <= 0:
        parser.error("Duration and torque parameters must be positive")
    results = [evaluate(args.scene, args.pose, args.bias, mode, args.seeds,
                        args.duration_s, args.torque_gain, args.torque_damping,
                        args.torque_limit) for mode in ("muscle", "torque", "hybrid")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "results": results,
                                       "scope": "Ideal torque is an actuation diagnostic, not an infant muscle or walking policy"},
                                      indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in
                       ("mode", "mean_standing_fraction", "final_standing_count")}
                      for row in results]))


if __name__ == "__main__":
    main()
