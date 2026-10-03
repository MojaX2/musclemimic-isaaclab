"""Evolve a tactile stance/swing controller around the infant support policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evolve_infant_crawl_cpg import evaluate, summarize
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from infant_crawl_stepper import TactileStepper
from train_infant_crawl_ppo import SupportPolicy, jittered_positions


def stepper_score(result, parameters):
    return (7 * result["forward_displacement_m"].clamp(-0.15, 0.3) +
            1.5 * result["minimum_support_fraction"] +
            result["final_quarter_support_fraction"] +
            0.8 * result["hand_switches"].clamp(max=2) +
            1.5 * result["crawl_success"].float() -
            0.01 * parameters.square().mean(-1))


def rollout(env, policy, parameters, positions, control_steps, physics_per_control, use_touch):
    controller = TactileStepper(parameters, absolute_targets=True,
                                forward_reach=True, strong_plant=True, leg_push=True)
    result = evaluate(env, policy, None, positions, control_steps, physics_per_control,
                      use_touch, drive_controller=controller)
    result["stepper_score"] = stepper_score(result, parameters)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--initialize-from", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        parser.error("Output already exists")
    if min(args.population, args.generations, args.control_steps, args.physics_per_control) < 1:
        parser.error("Training dimensions must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None):
        raise ValueError("Stepper evolution requires a compatible palm-control policy")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.population,
                         controlled_joint_names=PALM_JOINTS)
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    mean = torch.tensor([0.5, 0.3, 0.4, 0.2, -0.2, 0.2], device=env.device)
    if args.initialize_from is not None:
        with np.load(args.initialize_from) as previous:
            initial = np.asarray(previous["parameters"], dtype=np.float32)
        if initial.shape != (6,) or not np.isfinite(initial).all():
            raise ValueError("Initial stepper parameters must be six finite values")
        mean = torch.as_tensor(initial, device=env.device)
    deviation = torch.tensor([0.2, 0.2, 0.2, 0.15, 0.35, 0.35], device=env.device)
    lower = torch.tensor([0., 0., 0., 0., -1., -1.], device=env.device)
    upper = torch.tensor([1., 1., 1., 1., 1., 1.], device=env.device)
    candidate_pool = []
    history = []
    for generation in range(args.generations):
        candidates = (mean + torch.randn((args.population, 6), device=env.device) *
                      deviation).clamp(lower, upper)
        candidates[0] = mean.clamp(lower, upper)
        candidates[1] = torch.tensor([0.5, 0.3, 0.4, 0.2, 0., 0.], device=env.device)
        positions = jittered_positions(env, 0.02, 0.005)
        result = rollout(env, policy, candidates, positions, args.control_steps,
                         args.physics_per_control, checkpoint["use_touch"])
        score = result["stepper_score"]
        elite = candidates[torch.topk(score, max(4, args.population // 4)).indices]
        mean = 0.6 * mean + 0.4 * elite.mean(0)
        deviation = (0.6 * deviation + 0.4 * elite.std(0, unbiased=False)).clamp(0.05, 0.7)
        winner = int(score.argmax())
        candidate_pool.append((float(score[winner]), candidates[winner].clone()))
        history.append({"generation": generation + 1, "best_score": float(score[winner]),
                        "winner_forward_m": float(result["forward_displacement_m"][winner]),
                        "winner_hand_switches": float(result["hand_switches"][winner]),
                        "winner_crawl_success": bool(result["crawl_success"][winner])})
        print(json.dumps(history[-1]), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8,
                                controlled_joint_names=PALM_JOINTS)
    torch.manual_seed(17)
    validation_positions = jittered_positions(validation, 0.05, 0.01)
    contenders = sorted(candidate_pool, key=lambda item: item[0], reverse=True)[:10]
    contenders.append((None, mean.clamp(lower, upper)))
    validation_records = []
    selected = None
    selected_score = -float("inf")
    for training_score, parameters in contenders:
        result = rollout(validation, policy, parameters.expand(8, -1),
                         validation_positions, args.control_steps,
                         args.physics_per_control, checkpoint["use_touch"])
        score = float(result["stepper_score"].mean())
        validation_records.append({"training_score": training_score, "score": score,
                                   "forward_m": float(result["forward_displacement_m"].mean()),
                                   "hand_switches": float(result["hand_switches"].mean()),
                                   "crawl_success_count": int(result["crawl_success"].sum())})
        if score > selected_score:
            selected_score = score
            selected = parameters.clone()
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
              "initialize_from": (str(args.initialize_from.resolve())
                                  if args.initialize_from is not None else None),
              "population": args.population, "generations": args.generations,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "parameter_order": ["lift_shoulder_horizontal", "lift_shoulder_ad_ab",
                                  "reach_shoulder_horizontal", "wrist_pitch",
                                  "diagonal_hip_swing", "diagonal_knee_swing"],
              "selected_parameters": selected.cpu().tolist(),
              "selected_validation_score": selected_score,
              "validation_candidates": validation_records, "history": history,
              "scope": "GA for tactile stance/swing muscle targets; crawling not assumed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=selected.cpu().numpy())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
