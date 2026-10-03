"""Test selective full-range antagonist excitation during infant standing."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import mujoco
import torch

from developmental_skin import REGIONS
from evaluate_infant_newton_crawl import newton_crawl_environment
from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def adaptive_antagonistic_action(env, drives, onset=0.5):
    if not 0 <= onset < 1:
        raise ValueError("Adaptive excitation onset must be in [0, 1)")
    if drives.shape != (env.worlds, len(env.crawl_actuators)):
        raise ValueError("Adaptive excitation drives must match controlled joints")
    action = env.action_from_drives(drives, baseline=0.4, amplitude=0.3)
    blend = ((drives.abs() - onset) / (1 - onset)).clamp(0, 1)
    baseline = 0.4 + 0.1 * blend
    amplitude = 0.3 + 0.2 * blend
    actuator_ids = env.crawl_actuators
    action[:, actuator_ids] = (baseline - amplitude * drives).clamp(0, 1)
    action[:, actuator_ids + env.source.nu] = (baseline + amplitude * drives).clamp(0, 1)
    return action


def evaluate(env, positions, bias, gains, condition, steps):
    env.reset(positions)
    measure = env.measure()
    standing = torch.zeros(env.worlds, device=env.device)
    late = torch.zeros_like(standing)
    late_start = steps - max(1, steps // 4)
    for step in range(steps):
        drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        drives = cop_reflex_drives(env, measure, drives, gains)
        if condition == "standard":
            action = env.action_from_drives(drives, baseline=0.4, amplitude=0.3)
        elif condition == "adaptive":
            action = adaptive_antagonistic_action(env, drives)
        elif condition == "full_range":
            action = env.action_from_drives(drives, baseline=0.5, amplitude=0.5)
        else:
            raise ValueError("Unknown excitation condition")
        measure = env.step(action, 1)
        supported = ((measure["chest_height"] > 0.45) &
                     (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
        standing += supported
        if step >= late_start:
            late += supported
    return {"standing_fraction": float((standing / steps).mean()),
            "late_standing_fraction": float((late / (steps - late_start)).mean()),
            "final_standing_count": int(supported.sum()),
            "per_world_standing_fraction": (standing / steps).cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=800)
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
        bias = load_bias(args.bias, env.device)
        gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
        records = []
        for seed in args.seeds:
            positions = aligned_positions(env, seed, env.worlds)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for condition in ("standard", "adaptive", "full_range"):
                result = evaluate(env, positions, bias, gains, condition, args.control_steps)
                row = {"seed": seed, "condition": condition,
                       "pose_sha256": pose_hash, **result}
                records.append(row)
                print(json.dumps({key: row[key] for key in
                                  ("seed", "condition", "standing_fraction",
                                   "late_standing_fraction", "final_standing_count")}),
                      flush=True)
        report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
                  "bias": str(args.bias.resolve()),
                  "evolution": str(args.evolution.resolve()),
                  "backend": args.backend, "worlds": args.worlds, "seeds": args.seeds,
                  "duration_s": args.control_steps * float(env.source.opt.timestep),
                  "solver_limit_steps": getattr(env, "solver_limit_steps", None),
                  "records": records,
                  "scope": "Fixed standing reflex excitation audit, not acquired walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
