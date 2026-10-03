"""Compare learned infant standing feedback against its evolved PD baseline."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch
import mujoco

from developmental_skin import REGIONS
from evaluate_infant_newton_crawl import newton_crawl_environment
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe
from train_infant_stand_ppo import step_stand_control
from train_infant_stand_support import stand_score


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def rollout(env, positions, bias, policy, residual_scale, use_touch,
            control_steps, physics_per_control, cop_gains=None,
            muscle_baseline=0.2, muscle_amplitude=0.3):
    env.reset(positions)
    measure = env.measure()
    initial_foot_force = float(measure["foot_force"].sum(-1).mean())
    standing, chest, scores, both_feet = [], [], [], []
    with torch.no_grad():
        for _ in range(control_steps):
            residual = (policy.actor(observe(env, measure, use_touch))
                        if policy is not None else None)
            measure = step_stand_control(env, measure, bias, residual, residual_scale,
                                         physics_per_control, cop_gains,
                                         muscle_baseline, muscle_amplitude)
            feet = (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)
            standing.append(((measure["chest_height"] > 0.45) & feet).float())
            both_feet.append(feet.float())
            chest.append(measure["chest_height"].clone())
            scores.append(stand_score(env, measure))
    standing = torch.stack(standing)
    chest = torch.stack(chest)
    return {"initial_mean_foot_force_n": initial_foot_force,
            "standing_fraction": float(standing.mean()),
            "late_standing_fraction": float(standing[-max(1, control_steps // 4):].mean()),
            "final_standing_fraction": float(standing[-1].mean()),
            "both_feet_fraction": float(torch.stack(both_feet).mean()),
            "mean_reward": float(torch.stack(scores).mean()),
            "final_chest_height_m": chest[-1].cpu().tolist(),
            "final_forward_displacement_m": measure["forward_displacement"].cpu().tolist()}


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
    parser.add_argument("--balanced-resets", action="store_true")
    parser.add_argument("--backend", choices=("direct", "newton"), default="direct")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    context = (newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                       controlled_joint_names=CRAWL_JOINTS)
               if args.backend == "newton" else
               nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds)))
    with context as env:
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        if list(checkpoint["joint_names"]) != list(CRAWL_JOINTS):
            raise ValueError("Standing policy joint order does not match the infant")
        policy = SupportPolicy(checkpoint["observation_size"], len(CRAWL_JOINTS)).to(env.device)
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()
        cop_values = checkpoint.get("cop_gains")
        if cop_values is not None and len(cop_values) != 6:
            raise ValueError("Standing checkpoint has invalid COP gains")
        if cop_values is not None and checkpoint.get("cop_update_physics_steps", 1) != 1:
            raise ValueError("Standing COP checkpoint used an incompatible update period")
        cop_gains = (torch.tensor(cop_values, device=env.device,
                                  dtype=torch.float32).expand(env.worlds, -1)
                     if cop_values is not None else None)
        muscle_baseline = checkpoint.get("muscle_baseline", 0.2)
        muscle_amplitude = checkpoint.get("muscle_amplitude", 0.3)
        torch.manual_seed(args.seed)
        positions = jittered_positions(env, 0.05, 0.01)
        if args.balanced_resets:
            positions, _ = align_stand_feet(env.source, positions, env.initial[0],
                                            env.root_address)
        baseline = rollout(env, positions, checkpoint["bias"], None, 0,
                           checkpoint["use_touch"], args.control_steps,
                           args.physics_per_control, cop_gains, muscle_baseline,
                           muscle_amplitude)
        learned = rollout(env, positions, checkpoint["bias"], policy,
                          checkpoint["residual_scale"], checkpoint["use_touch"],
                          args.control_steps, args.physics_per_control,
                          cop_gains, muscle_baseline, muscle_amplitude)
        report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
                  "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
                  "backend": args.backend,
                  "initial_pose_sha256": hashlib.sha256(
                      positions.cpu().numpy().tobytes()).hexdigest(),
                  "worlds": args.worlds,
                  "balanced_resets": args.balanced_resets,
                  "cop_gains": cop_values,
                  "duration_s": args.control_steps * args.physics_per_control *
                                env.source.opt.timestep,
                  "use_touch": checkpoint["use_touch"], "baseline": baseline,
                  "learned": learned,
                  "solver_limit_steps": getattr(env, "solver_limit_steps", None),
                  "scope": "Held-out standing posture; not stepping or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in ("baseline", "learned")}, indent=2))


if __name__ == "__main__":
    main()
