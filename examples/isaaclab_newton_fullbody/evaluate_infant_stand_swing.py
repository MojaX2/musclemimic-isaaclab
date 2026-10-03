"""Audit whether a unilateral muscle command produces a supported infant step."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import mujoco
import torch
import warp as wp

from evaluate_infant_newton_crawl import newton_body_id, newton_crawl_environment
from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_stand_ppo import load_bias


RIGHT_FOOT = 0
LEFT_FOOT = 1
HIP = CRAWL_JOINTS.index("robot:right_hip1")
KNEE = CRAWL_JOINTS.index("robot:right_knee")
ANKLE = CRAWL_JOINTS.index("robot:right_foot1")


def swing_envelope(time_s, start_s=0.15, end_s=0.9):
    if not start_s < end_s:
        raise ValueError("Swing interval must have positive duration")
    if time_s <= start_s or time_s >= end_s:
        return 0.0
    return math.sin(math.pi * (time_s - start_s) / (end_s - start_s))


def swing_drives(drives, condition, envelope):
    if condition not in ("control", "hip", "hip_knee"):
        raise ValueError("Unknown swing condition")
    if condition == "control" or envelope == 0:
        return drives
    result = drives.clone()
    result[:, HIP] = torch.minimum(result[:, HIP],
                                   torch.full_like(result[:, HIP], -envelope))
    if condition == "hip_knee":
        result[:, KNEE] = torch.minimum(result[:, KNEE],
                                        torch.full_like(result[:, KNEE], -envelope))
        result[:, ANKLE] = torch.maximum(result[:, ANKLE],
                                         torch.full_like(result[:, ANKLE], 0.5 * envelope))
    return result


def evaluate(env, positions, bias, gains, condition, steps, foot_body_id):
    env.reset(positions)
    measure = env.measure()
    initial_foot = wp.to_torch(env.data.xpos)[:, foot_body_id, :].clone()
    initial_root_x = env.position[:, env.root_address].clone()
    initial_hip = env.position[:, env.crawl_qpos_ids[HIP]].clone()
    standing = torch.zeros(env.worlds, device=env.device)
    late = torch.zeros_like(standing)
    liftoff = torch.zeros(env.worlds, device=env.device, dtype=torch.bool)
    planted = torch.zeros_like(liftoff)
    supported_liftoff = torch.zeros_like(liftoff)
    supported_plant = torch.zeros_like(liftoff)
    max_relative_advance = torch.full_like(initial_root_x, -float("inf"))
    max_airborne_advance = torch.full_like(initial_root_x, -float("inf"))
    max_foot_rise = torch.full_like(initial_root_x, -float("inf"))
    max_hip_flexion = torch.full_like(initial_root_x, -float("inf"))
    late_start = steps - max(1, steps // 4)
    dt = float(env.source.opt.timestep)
    for step in range(steps):
        drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        drives = cop_reflex_drives(env, measure, drives, gains)
        drives = swing_drives(drives, condition, swing_envelope(step * dt))
        measure = env.step(env.action_from_drives(drives, baseline=0.4,
                                                 amplitude=0.3), 1)
        foot = wp.to_torch(env.data.xpos)[:, foot_body_id, :]
        relative_advance = (foot[:, 0] - env.position[:, env.root_address] -
                            initial_foot[:, 0] + initial_root_x)
        rise = foot[:, 2] - initial_foot[:, 2]
        hip_flexion = initial_hip - env.position[:, env.crawl_qpos_ids[HIP]]
        max_relative_advance = torch.maximum(max_relative_advance, relative_advance)
        max_foot_rise = torch.maximum(max_foot_rise, rise)
        max_hip_flexion = torch.maximum(max_hip_flexion, hip_flexion)
        support = ((measure["chest_height"] > 0.45) &
                   (measure["foot_force"][:, LEFT_FOOT] > 2))
        airborne = measure["foot_force"][:, RIGHT_FOOT] <= 2
        max_airborne_advance = torch.where(airborne,
                                          torch.maximum(max_airborne_advance,
                                                        relative_advance),
                                          max_airborne_advance)
        new_liftoff = (step * dt >= 0.15) & airborne & ~liftoff
        supported_liftoff |= new_liftoff & support
        liftoff |= new_liftoff
        new_plant = (step * dt >= 0.15) & liftoff & ~airborne & ~planted
        supported_plant |= new_plant & support & (relative_advance > 0.03)
        planted |= new_plant
        standing_now = support & ~airborne
        standing += standing_now.float()
        if step >= late_start:
            late += standing_now.float()
    max_airborne_advance = torch.where(torch.isfinite(max_airborne_advance),
                                       max_airborne_advance, float("nan"))
    return {
        "standing_fraction": float((standing / steps).mean()),
        "late_standing_fraction": float((late / (steps - late_start)).mean()),
        "liftoff_count": int(liftoff.sum()),
        "supported_liftoff_count": int(supported_liftoff.sum()),
        "forward_supported_plant_count": int(supported_plant.sum()),
        "max_relative_advance_m": max_relative_advance.cpu().tolist(),
        "max_airborne_advance_m": [None if math.isnan(value) else value
                                    for value in max_airborne_advance.cpu().tolist()],
        "max_foot_rise_m": max_foot_rise.cpu().tolist(),
        "max_hip_flexion_rad": max_hip_flexion.cpu().tolist(),
        "solver_limit_steps": getattr(env, "solver_limit_steps", None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7401, 7403, 7407])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps) < 1 or not args.seeds:
        parser.error("Worlds, steps, and seeds must be positive and nonempty")
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(gain_values) != 6:
        parser.error("Expected six standing pressure-reflex gains")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    context = (newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                        controlled_joint_names=CRAWL_JOINTS)
               if args.backend == "newton" else
               nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds)))
    with context as env:
        body_id = (newton_body_id(env.collision_model, "right_foot")
                   if args.backend == "newton" else
                   mujoco.mj_name2id(env.source, mujoco.mjtObj.mjOBJ_BODY, "right_foot"))
        bias = load_bias(args.bias, env.device)
        gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
        records = []
        for seed in args.seeds:
            positions = aligned_positions(env, seed, env.worlds)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for condition in ("control", "hip", "hip_knee"):
                result = evaluate(env, positions, bias, gains, condition,
                                  args.control_steps, body_id)
                record = {"seed": seed, "condition": condition,
                          "pose_sha256": pose_hash, **result}
                records.append(record)
                print(json.dumps({key: record[key] for key in
                                  ("seed", "condition", "standing_fraction",
                                   "liftoff_count", "supported_liftoff_count",
                                   "forward_supported_plant_count")}), flush=True)
        report = {"scene": str(args.scene.resolve()),
                  "pose": str(args.pose.resolve()), "bias": str(args.bias.resolve()),
                  "evolution": str(args.evolution.resolve()),
                  "backend": args.backend, "worlds": args.worlds,
                  "seeds": args.seeds,
                  "duration_s": args.control_steps * float(source.opt.timestep),
                  "records": records,
                  "scope": "Open-loop unilateral swing feasibility audit, not learned walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
