"""Compare raw versus floor-projected infant standing reset perturbations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from developmental_skin import REGIONS
from infant_crawl_env import InfantCrawlEnv
from infant_stand_resets import align_stand_feet, project_foot_contact
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def evaluate(env, positions, bias, control_steps, reset_target=False):
    env.reset(positions)
    initial = env.measure()
    initial_foot_force = initial["foot_force"].sum(-1).detach().cpu().tolist()
    standing_steps = torch.zeros(env.worlds, device=env.device)
    for _ in range(control_steps):
        if reset_target:
            drives = (bias.expand(env.worlds, -1) +
                      1.5 * (positions[:, env.crawl_qpos_ids] -
                             env.position[:, env.crawl_qpos_ids]) -
                      env.velocity[:, env.crawl_qvel_ids]).clamp(-1, 1)
        else:
            drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        measure = env.step(env.action_from_drives(drives, baseline=0.4,
                                                   amplitude=0.3), 1)
        standing_steps += ((measure["chest_height"] > 0.45) &
                           (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    final_standing = ((measure["chest_height"] > 0.45) &
                      (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1))
    return {"initial_foot_force_n": initial_foot_force,
            "mean_initial_foot_force_n": sum(initial_foot_force) / env.worlds,
            "standing_fraction": (standing_steps / control_steps).cpu().tolist(),
            "mean_standing_fraction": float((standing_steps / control_steps).mean()),
            "final_standing": final_standing.cpu().tolist(),
            "final_standing_count": int(final_standing.sum()),
            "final_chest_height_m": measure["chest_height"].cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 43, 47])
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--bilateral", action="store_true")
    parser.add_argument("--reset-target", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if args.duration_s <= 0:
        parser.error("Duration must be positive")
    env = InfantCrawlEnv(args.scene, args.pose, 8)
    bias = load_bias(args.bias, env.device)
    control_steps = round(args.duration_s / float(env.source.opt.timestep))
    comparisons = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        raw = jittered_positions(env, 0.05, 0.01)
        projected, offsets = project_foot_contact(env.source, raw, env.root_address)
        bilateral, alignment = (align_stand_feet(env.source, raw, env.initial[0],
                                                  env.root_address)
                                 if args.bilateral else (None, None))
        raw_result = evaluate(env, raw, bias, control_steps, args.reset_target)
        projected_result = evaluate(env, projected, bias, control_steps,
                                    args.reset_target)
        bilateral_result = (evaluate(env, bilateral, bias, control_steps,
                                     args.reset_target)
                            if args.bilateral else None)
        comparison = {"seed": seed,
                            "raw_pose_sha256": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest(),
                            "projected_pose_sha256": hashlib.sha256(
                                projected.cpu().numpy().tobytes()).hexdigest(),
                            "root_height_correction_m": offsets.tolist(),
                            "raw": raw_result, "projected": projected_result}
        if args.bilateral:
            comparison.update({"bilateral_pose_sha256": hashlib.sha256(
                                   bilateral.cpu().numpy().tobytes()).hexdigest(),
                               "bilateral_alignment": alignment,
                               "bilateral": bilateral_result})
        comparisons.append(comparison)
        print(json.dumps({"seed": seed,
                          "raw_standing": raw_result["mean_standing_fraction"],
                          "projected_standing": projected_result["mean_standing_fraction"],
                          "bilateral_standing": (bilateral_result["mean_standing_fraction"]
                                                 if args.bilateral else None)}),
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "duration_s": control_steps * float(env.source.opt.timestep),
                                       "bilateral": args.bilateral,
                                       "reset_target": args.reset_target,
                                       "comparisons": comparisons,
                                       "scope": "Same joint perturbations; floor-height reset ablation, not walking"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
