"""Evolve antagonist muscle biases around infant standing posture feedback."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from train_infant_crawl_ppo import jittered_positions


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def stand_score(env, measure):
    chest = ((measure["chest_height"] - 0.4) / 0.18).clamp(0, 1)
    head = ((measure["head_height"] - 0.45) / 0.18).clamp(0, 1)
    supported = (measure["ground_touch"][:, FOOT_IDS] > 2).float()
    floor = measure["ground_touch"]
    grounded = ((floor[:, REGIONS.index("pelvis")] > 2) |
                (floor[:, REGIONS.index("trunk")] > 2)).float()
    root = env.root_address
    alignment = (env.position[:, root + 3:root + 7] *
                 env.initial[:, root + 3:root + 7]).sum(-1).abs().clamp(0, 1)
    elevation = torch.minimum(chest, head)
    return (elevation * (0.5 + 0.2 * supported.mean(-1) +
                         0.2 * supported.all(-1).float() + 0.1 * alignment)
            - 0.3 * grounded)


def evaluate(env, biases, positions, control_steps, physics_per_control, kp, kd):
    env.reset(positions)
    total = torch.zeros(env.worlds, device=env.device)
    successful = torch.zeros_like(total)
    last = None
    for _ in range(control_steps):
        drives = env.feedback_drives(biases, kp, kd)
        last = env.step(env.action_from_drives(drives), physics_per_control)
        total += stand_score(env, last)
        successful += ((last["chest_height"] > 0.45) &
                       (last["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    return (total / control_steps + 2 * stand_score(env, last) -
            0.01 * biases.square().mean(-1)), last, successful / control_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".npz")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or checkpoint already exists")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.population)
    mean = torch.zeros(len(CRAWL_JOINTS), device=env.device)
    deviation = torch.full_like(mean, 0.4)
    best_score = -float("inf")
    best_bias = None
    history = []
    candidate_pool = []
    for generation in range(args.generations):
        candidates = (mean + torch.randn((args.population, len(CRAWL_JOINTS)),
                                          device=env.device) * deviation).clamp(-1, 1)
        candidates[0] = mean.clamp(-1, 1)
        candidates[1] = 0
        positions = jittered_positions(env, 0.02, 0.005)
        scores, last, successful = evaluate(env, candidates, positions,
                                             args.control_steps, args.physics_per_control,
                                             args.kp, args.kd)
        elite = candidates[torch.topk(scores, max(4, args.population // 4)).indices]
        mean = 0.6 * mean + 0.4 * elite.mean(0)
        deviation = (0.6 * deviation + 0.4 * elite.std(0, unbiased=False)).clamp(0.05, 0.8)
        winner = int(scores.argmax())
        if float(scores[winner]) > best_score:
            best_score = float(scores[winner])
            best_bias = candidates[winner].clone()
        candidate_pool.append((float(scores[winner]), candidates[winner].clone()))
        record = {"generation": generation + 1, "best_batch_score": float(scores[winner]),
                  "best_so_far": best_score,
                  "winner_success_fraction": float(successful[winner]),
                  "winner_final_chest_height_m": float(last["chest_height"][winner])}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    torch.manual_seed(17)
    validation_positions = jittered_positions(validation, 0.05, 0.01)
    contenders = sorted(candidate_pool, key=lambda pair: pair[0], reverse=True)[:10]
    contenders.append((None, mean.clamp(-1, 1)))
    validation_records = []
    selected_bias = None
    selected_score = -float("inf")
    for training_score, contender in contenders:
        scores, final, success = evaluate(
            validation, contender.expand(8, -1), validation_positions,
            args.control_steps, args.physics_per_control, args.kp, args.kd)
        final_success = ((final["chest_height"] > 0.45) &
                         (final["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
        selection_score = float(scores.mean() + 2 * final_success.mean() + 0.2 * success.mean())
        validation_records.append({"training_score": training_score,
                                   "validation_score": selection_score,
                                   "validation_success_fraction": float(success.mean()),
                                   "validation_final_success_fraction": float(final_success.mean()),
                                   "validation_final_chest_m": final["chest_height"].cpu().tolist()})
        if selection_score > selected_score:
            selected_score = selection_score
            selected_bias = contender.clone()
    best_bias = selected_bias
    heldout = InfantCrawlEnv(args.scene, args.pose, 8)
    torch.manual_seed(41)
    positions = jittered_positions(heldout, 0.05, 0.01)
    baseline_score, baseline_last, baseline_success = evaluate(
        heldout, torch.zeros((8, len(CRAWL_JOINTS)), device=heldout.device), positions,
        args.control_steps, args.physics_per_control, args.kp, args.kd)
    learned_score, learned_last, learned_success = evaluate(
        heldout, best_bias.expand(8, -1), positions,
        args.control_steps, args.physics_per_control, args.kp, args.kd)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "seed": args.seed, "population": args.population,
              "generations": args.generations, "kp": args.kp, "kd": args.kd,
              "best_training_score": best_score, "history": history,
              "validation_seed": 17, "validation_candidates": validation_records,
              "selected_validation_score": selected_score,
              "heldout_seed": 41, "heldout_joint_jitter_rad": 0.05,
              "heldout_root_height_jitter_m": 0.01,
              "heldout_baseline_mean_score": float(baseline_score.mean()),
              "heldout_learned_mean_score": float(learned_score.mean()),
              "heldout_baseline_success_fraction": float(baseline_success.mean()),
              "heldout_learned_success_fraction": float(learned_success.mean()),
              "heldout_baseline_final_chest_m": baseline_last["chest_height"].cpu().tolist(),
              "heldout_learned_final_chest_m": learned_last["chest_height"].cpu().tolist(),
              "best_bias": best_bias.cpu().tolist(),
              "scope": "Infant standing support optimization; not stepping or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, bias=best_bias.cpu().numpy(),
                        joint_names=np.asarray(CRAWL_JOINTS))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("best_training_score", "heldout_baseline_success_fraction",
                       "heldout_learned_success_fraction")}, indent=2))


if __name__ == "__main__":
    main()
