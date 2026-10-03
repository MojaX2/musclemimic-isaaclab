"""Time-resolve foot support and torso collapse in infant standing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))
LOWER_JOINTS = [CRAWL_JOINTS.index(f"robot:{side}_{joint}")
                for side in ("right", "left")
                for joint in ("hip1", "hip2", "hip3", "knee", "foot1", "foot2", "foot3")]


def inspect(env, measure, step, timestep, drives):
    feet = measure["foot_force"] > 2
    total_force = measure["foot_force"].sum(-1)
    center_x = ((measure["foot_center_xy"][:, :, 0] * measure["foot_force"]).sum(-1) /
                total_force.clamp_min(1e-8))
    center_y = ((measure["foot_center_xy"][:, :, 1] * measure["foot_force"]).sum(-1) /
                total_force.clamp_min(1e-8))
    center_of_mass = wp.to_torch(env.data.subtree_com)[:, 0, :2]
    supported = feet.all(-1)
    displacement = torch.stack((center_of_mass[:, 0] - center_x,
                                center_of_mass[:, 1] - center_y), dim=-1)
    mean_displacement = (torch.where(supported[:, None], displacement, 0).sum(0) /
                         supported.sum().clamp_min(1))
    lower_error = (env.position[:, env.crawl_qpos_ids[LOWER_JOINTS]] -
                   env.initial[:, env.crawl_qpos_ids[LOWER_JOINTS]]).abs()
    lower_torque = env.applied_force[:, env.crawl_qvel_ids[LOWER_JOINTS]].abs()
    return {"time_s": step * timestep,
            "both_feet_count": int(supported.sum()),
            "upright_chest_count": int((measure["chest_height"] > 0.45).sum()),
            "standing_count": int((supported & (measure["chest_height"] > 0.45)).sum()),
            "mean_chest_height_m": float(measure["chest_height"].mean()),
            "mean_root_height_m": float(env.position[:, env.root_address + 2].mean()),
            "mean_foot_force_n": float(total_force.mean()),
            "supported_mean_com_minus_cop_xy_m": mean_displacement.cpu().tolist(),
            "mean_lower_joint_error_rad": float(lower_error.mean()),
            "mean_lower_joint_torque_nm": float(lower_torque.mean()),
            "lower_drive_saturation_fraction": float((drives[:, LOWER_JOINTS].abs() > 0.99)
                                                      .float().mean())}


def evaluate(env, positions, bias, baseline, amplitude, duration_s, gains=None):
    env.reset(positions)
    timestep = float(env.source.opt.timestep)
    steps = round(duration_s / timestep)
    measure = env.measure()
    first_foot_loss = torch.full((env.worlds,), -1., device=env.device)
    first_chest_loss = torch.full_like(first_foot_loss, -1.)
    foot_gap = torch.zeros(env.worlds, device=env.device, dtype=torch.int32)
    chest_gap = torch.zeros_like(foot_gap)
    trace = []
    drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
    if gains is not None:
        drives = cop_reflex_drives(env, measure, drives, gains)
    trace.append(inspect(env, measure, 0, timestep, drives))
    for step in range(1, steps + 1):
        drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        if gains is not None:
            drives = cop_reflex_drives(env, measure, drives, gains)
        measure = env.step(env.action_from_drives(drives, baseline, amplitude), 1)
        supported = (measure["foot_force"] > 2).all(-1)
        upright = measure["chest_height"] > 0.45
        foot_gap = torch.where(supported, 0, foot_gap + 1)
        chest_gap = torch.where(upright, 0, chest_gap + 1)
        first_foot_loss = torch.where((first_foot_loss < 0) & (foot_gap >= 3),
                                      (step - 2) * timestep, first_foot_loss)
        first_chest_loss = torch.where((first_chest_loss < 0) & (chest_gap >= 3),
                                       (step - 2) * timestep, first_chest_loss)
        if step % max(1, round(0.1 / timestep)) == 0 or step == steps:
            trace.append(inspect(env, measure, step, timestep, drives))
    both = (first_foot_loss >= 0) & (first_chest_loss >= 0)
    return {"baseline": baseline, "amplitude": amplitude,
            "gains": gains[0].cpu().tolist() if gains is not None else None,
            "worlds": env.worlds,
            "duration_s": steps * timestep,
            "initial_foot_force_n": trace[0]["mean_foot_force_n"],
            "first_foot_loss_s": first_foot_loss.cpu().tolist(),
            "first_chest_loss_s": first_chest_loss.cpu().tolist(),
            "foot_loss_before_chest_count": int((both &
                                                 (first_foot_loss < first_chest_loss)).sum()),
            "chest_loss_before_foot_count": int((both &
                                                 (first_chest_loss < first_foot_loss)).sum()),
            "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[53, 59, 61])
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--evolution", type=Path)
    parser.add_argument("--ankle-evolution", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.duration_s <= 0:
        parser.error("Duration must be positive")
    env = InfantCrawlEnv(args.scene, args.pose, len(args.seeds) * 8)
    bias = load_bias(args.bias, env.device)
    blocks = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        blocks.append(jittered_positions(env, 0.05, 0.01)[:8].clone())
    positions, diagnostics = align_stand_feet(env.source, torch.cat(blocks),
                                               env.initial[0], env.root_address)
    if args.evolution is None:
        conditions = [evaluate(env, positions, bias, baseline, amplitude,
                               args.duration_s) for baseline, amplitude in
                      ((0.4, 0.3), (0.4, 0.4))]
    else:
        if args.ankle_evolution is None:
            parser.error("Reflex comparison requires --ankle-evolution")
        selected = json.loads(args.evolution.read_text())["selected"]["gains"]
        ankle = json.loads(args.ankle_evolution.read_text())["selected"]["gains"]
        if len(selected) not in (4, 6) or len(ankle) != len(selected) - 2:
            parser.error("Expected four/six gains and a two-gain-smaller comparison")
        selected_name = ("hip_ankle" if len(selected) == 4 else "hip_ankle_roll")
        comparison_name = ("ankle_only" if len(selected) == 4 else "sagittal_only")
        conditions = []
        for name, gains in ((selected_name, selected), (comparison_name, ankle),
                            ("no_reflex", [0., 0.])):
            expanded = torch.tensor(gains, device=env.device).expand(env.worlds, -1)
            result = evaluate(env, positions, bias, 0.4, 0.3, args.duration_s,
                              expanded)
            conditions.append({"condition": name, **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "seeds": args.seeds,
                                       "alignment_count": len(diagnostics),
                                       "evolution": (str(args.evolution.resolve())
                                                     if args.evolution else None),
                                       "conditions": conditions,
                                       "scope": "Temporal association, not proof of causal fall mechanism or walking"},
                                      indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in
                       ("baseline", "amplitude", "foot_loss_before_chest_count",
                        "chest_loss_before_foot_count")}
                      for row in conditions]))


if __name__ == "__main__":
    main()
