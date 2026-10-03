"""Compare learned infant tonic support against a co-contraction baseline."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_crawl_metrics import crawl_locomotion_metrics


def rollout(env, drives, positions, control_steps, physics_per_control):
    env.reset(positions)
    action = env.action_from_drives(drives)
    samples = []
    for _ in range(control_steps):
        sample = env.step(action, physics_steps=physics_per_control)
        samples.append({key: value.cpu().numpy() for key, value in sample.items()})
    chest = np.stack([sample["chest_height"] for sample in samples])
    head = np.stack([sample["head_height"] for sample in samples])
    support = np.stack([sample["support_force"] for sample in samples])
    palm_shin = np.stack([sample["palm_shin_force"] for sample in samples])
    trunk = np.stack([sample["ground_touch"][:, REGIONS.index("trunk")] for sample in samples])
    pelvis = np.stack([sample["ground_touch"][:, REGIONS.index("pelvis")] for sample in samples])
    settled_step = math.ceil(0.25 / (physics_per_control * float(env.source.opt.timestep)))
    stable_support = support[settled_step:]
    four_supports = np.all(support > 2.0, axis=2)
    elevated_four_supports = four_supports & (chest > 0.16) & (head > 0.14)
    elevated_palm_shin = np.all(palm_shin > 2.0, axis=2) & (chest > 0.16) & (head > 0.14)
    locomotion = crawl_locomotion_metrics(palm_shin, chest, head, pelvis,
                                          samples[-1]["forward_displacement"])
    return {
        "final_chest_height_m": chest[-1].tolist(),
        "final_head_height_m": head[-1].tolist(),
        "final_forward_displacement_m": samples[-1]["forward_displacement"].tolist(),
        "fraction_four_supports": float(np.mean(four_supports)),
        "fraction_four_supports_after_0p25s": float(np.mean(np.all(stable_support > 2.0, axis=2)))
        if len(stable_support) else None,
        "fraction_elevated_four_supports": float(np.mean(elevated_four_supports)),
        "fraction_elevated_four_supports_after_0p25s": float(np.mean(elevated_four_supports[settled_step:]))
        if len(stable_support) else None,
        "fraction_elevated_palm_shin_supports": float(np.mean(elevated_palm_shin)),
        "elevated_palm_shin_fraction_per_world": elevated_palm_shin.mean(axis=0).tolist(),
        "fraction_elevated_palm_shin_supports_after_0p25s": float(np.mean(elevated_palm_shin[settled_step:]))
        if len(stable_support) else None,
        "elevated_palm_shin_fraction_after_0p25s_per_world":
        elevated_palm_shin[settled_step:].mean(axis=0).tolist() if len(stable_support) else None,
        "fraction_chest_above_0p10_m": float(np.mean(chest > 0.10)),
        "fraction_chest_above_0p16_m": float(np.mean(chest > 0.16)),
        "fraction_head_above_0p14_m": float(np.mean(head > 0.14)),
        "fraction_trunk_contact": float(np.mean(trunk > 2.0)),
        "fraction_pelvis_contact": float(np.mean(pelvis > 2.0)),
        "final_support_force_n": support[-1].tolist(),
        "final_palm_shin_force_n": palm_shin[-1].tolist(),
        **locomotion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--joint-jitter-rad", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.worlds < 2 or args.control_steps < 1 or args.joint_jitter_rad < 0:
        parser.error("Use at least two worlds, one control step, and nonnegative jitter")
    with np.load(args.checkpoint) as checkpoint:
        drives = checkpoint["drives"]
        joint_names = checkpoint["joint_names"].tolist()
    if joint_names != list(CRAWL_JOINTS) or drives.shape != (len(CRAWL_JOINTS),):
        raise ValueError("Crawl checkpoint joint order does not match the model")
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds)
    random = np.random.default_rng(args.seed)
    positions = env.initial.clone()
    positions[:, env.root_address + 2] += torch.as_tensor(
        random.uniform(-0.01, 0.01, args.worlds), device=env.device, dtype=torch.float32)
    for name in CRAWL_JOINTS:
        joint_id = mujoco.mj_name2id(env.source, mujoco.mjtObj.mjOBJ_JOINT, name)
        address = int(env.source.jnt_qposadr[joint_id])
        jitter = torch.as_tensor(random.uniform(-args.joint_jitter_rad, args.joint_jitter_rad,
                                                args.worlds), device=env.device, dtype=torch.float32)
        positions[:, address] = (positions[:, address] + jitter).clamp(
            float(env.source.jnt_range[joint_id, 0]) + 1e-4,
            float(env.source.jnt_range[joint_id, 1]) - 1e-4)
    baseline = rollout(env, torch.zeros((args.worlds, len(CRAWL_JOINTS)), device=env.device),
                       positions, args.control_steps, args.physics_per_control)
    learned = rollout(env, torch.as_tensor(drives, device=env.device,
                                            dtype=torch.float32).expand(args.worlds, -1),
                      positions, args.control_steps, args.physics_per_control)
    report = {
        "scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
        "checkpoint": str(args.checkpoint.resolve()), "worlds": args.worlds,
        "seed": args.seed, "joint_jitter_rad": args.joint_jitter_rad,
        "root_height_jitter_m": 0.01,
        "duration_s": args.control_steps * args.physics_per_control * env.source.opt.timestep,
        "baseline": baseline, "learned": learned,
        "scope": "Held-out initial-pose test of tonic support; not forward crawling or walking",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
