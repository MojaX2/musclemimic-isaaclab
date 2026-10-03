"""Test plantar pressure-center feedback in the infant standing controller."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))
ANKLE_IDS = [CRAWL_JOINTS.index(f"robot:{side}_foot1") for side in ("right", "left")]
HIP_IDS = [CRAWL_JOINTS.index(f"robot:{side}_hip1") for side in ("right", "left")]
ROLL_ANKLE_IDS = [CRAWL_JOINTS.index(f"robot:{side}_foot2") for side in ("right", "left")]
ROLL_HIP_IDS = [CRAWL_JOINTS.index(f"robot:{side}_hip2") for side in ("right", "left")]


def trunk_reflex_drives(env, base_drives, gains):
    if gains.shape != (env.worlds, 4):
        raise ValueError("Trunk reflex needs four gains per infant")
    root = env.root_address
    quaternion = env.position[:, root + 3:root + 7]
    pitch = 2 * quaternion[:, 0] * quaternion[:, 2]
    roll = 2 * quaternion[:, 0] * quaternion[:, 1]
    pitch_velocity = env.velocity[:, root + 4]
    roll_velocity = env.velocity[:, root + 3]
    pitch_correction = gains[:, 0] * pitch + gains[:, 1] * pitch_velocity
    roll_correction = gains[:, 2] * roll + gains[:, 3] * roll_velocity
    drives = base_drives.clone()
    drives[:, HIP_IDS] += pitch_correction[:, None]
    drives[:, ROLL_HIP_IDS[0]] += roll_correction
    drives[:, ROLL_HIP_IDS[1]] -= roll_correction
    return drives.clamp(-1, 1)


def cop_reflex_drives(env, measure, base_drives, gains):
    if gains.shape not in ((env.worlds, 2), (env.worlds, 4), (env.worlds, 6)):
        raise ValueError("Pressure-center reflex needs two, four, or six gains per infant")
    forces = measure["foot_force"]
    total_force = forces.sum(-1)
    center_x = ((measure["foot_center_xy"][:, :, 0] * forces).sum(-1) /
                total_force.clamp_min(1e-8))
    center_of_mass_x = wp.to_torch(env.data.subtree_com)[:, 0, 0]
    displacement = center_of_mass_x - center_x
    forward_velocity = env.velocity[:, env.root_address]
    correction = (gains[:, 0] * displacement + gains[:, 1] * forward_velocity)
    correction = torch.where(total_force > 2, correction, 0)
    drives = base_drives.clone()
    drives[:, ANKLE_IDS] += correction[:, None]
    if gains.shape[1] >= 4:
        hip_correction = gains[:, 2] * displacement + gains[:, 3] * forward_velocity
        hip_correction = torch.where(total_force > 2, hip_correction, 0)
        drives[:, HIP_IDS] += hip_correction[:, None]
    if gains.shape[1] == 6:
        center_y = ((measure["foot_center_xy"][:, :, 1] * forces).sum(-1) /
                    total_force.clamp_min(1e-8))
        lateral_displacement = wp.to_torch(env.data.subtree_com)[:, 0, 1] - center_y
        lateral_velocity = env.velocity[:, env.root_address + 1]
        roll_correction = (gains[:, 4] * lateral_displacement +
                           gains[:, 5] * lateral_velocity)
        roll_correction = torch.where(total_force > 2, roll_correction, 0)
        drives[:, ROLL_ANKLE_IDS[0]] += roll_correction
        drives[:, ROLL_ANKLE_IDS[1]] -= roll_correction
    return drives.clamp(-1, 1)


def evaluate(env, bias, gains, positions, control_steps, return_late=False,
             baseline=0.4, amplitude=0.3, trunk_gains=None,
             root_vertical_unload=0.0, root_planar_stiffness=0.0,
             root_planar_damping=0.0):
    env.reset(positions)
    measure = env.measure()
    supported = torch.zeros(env.worlds, device=env.device)
    late_supported = torch.zeros_like(supported)
    late_start = control_steps - max(1, control_steps // 4)
    for step in range(control_steps):
        base = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        drives = cop_reflex_drives(env, measure, base, gains)
        if trunk_gains is not None:
            drives = trunk_reflex_drives(env, drives, trunk_gains)
        measure = env.step(env.action_from_drives(drives, baseline=baseline,
                                                  amplitude=amplitude), 1,
                           root_vertical_unload=root_vertical_unload,
                           root_planar_stiffness=root_planar_stiffness,
                           root_planar_damping=root_planar_damping)
        standing = ((measure["chest_height"] > 0.45) &
                    (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
        supported += standing
        if step >= late_start:
            late_supported += standing
    final = ((measure["chest_height"] > 0.45) &
             (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    if return_late:
        return (supported / control_steps, final, measure["chest_height"],
                late_supported / (control_steps - late_start))
    return supported / control_steps, final, measure["chest_height"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--balanced-resets", action="store_true")
    parser.add_argument("--heldout-seeds", nargs="+", type=int,
                        default=[41, 43, 47])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.control_steps < 1:
        parser.error("Control steps must be positive")
    position_grid, velocity_grid = np.meshgrid([-20., -10., 0., 10., 20.],
                                               [-4., -2., 0., 2., 4.], indexing="ij")
    gains = torch.as_tensor(np.column_stack((position_grid.ravel(), velocity_grid.ravel())),
                            device="cuda:0", dtype=torch.float32)
    env = InfantCrawlEnv(args.scene, args.pose, gains.shape[0])
    bias = load_bias(args.bias, env.device)
    fraction, final, height = evaluate(env, bias, gains, None, args.control_steps)
    nominal = [{"position_gain": float(gains[index, 0]),
                "velocity_gain": float(gains[index, 1]),
                "standing_fraction": float(fraction[index]),
                "final_standing": bool(final[index]),
                "final_chest_height_m": float(height[index])}
               for index in range(env.worlds)]
    selected = sorted(nominal, key=lambda row: row["standing_fraction"], reverse=True)[:4]
    if not any(row["position_gain"] == 0 and row["velocity_gain"] == 0 for row in selected):
        selected.append(next(row for row in nominal if row["position_gain"] == 0 and
                             row["velocity_gain"] == 0))
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    validation_bias = load_bias(args.bias, validation.device)
    heldout = []
    for seed in args.heldout_seeds:
        torch.manual_seed(seed)
        positions = jittered_positions(validation, 0.05, 0.01)
        if args.balanced_resets:
            positions, _ = align_stand_feet(validation.source, positions,
                                            validation.initial[0],
                                            validation.root_address)
        for candidate in selected:
            candidate_gains = torch.tensor([[candidate["position_gain"],
                                             candidate["velocity_gain"]]],
                                           device=validation.device).expand(8, -1)
            fraction, final, height = evaluate(validation, validation_bias,
                                               candidate_gains, positions, args.control_steps)
            heldout.append({"seed": seed, "position_gain": candidate["position_gain"],
                            "velocity_gain": candidate["velocity_gain"],
                            "standing_fraction": float(fraction.mean()),
                            "final_standing_fraction": float(final.mean()),
                            "final_chest_height_m": height.cpu().tolist()})
        print(json.dumps({"heldout_seed": seed}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "duration_s": args.control_steps * float(env.source.opt.timestep),
                                       "balanced_resets": args.balanced_resets,
                                       "heldout_seeds": args.heldout_seeds,
                                       "nominal": nominal, "heldout": heldout}, indent=2) + "\n")


if __name__ == "__main__":
    main()
