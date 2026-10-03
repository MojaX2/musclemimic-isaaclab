"""Learn tonic antagonist muscle synergies for MIMo hand-knee support.

This is the first crawling-curriculum stage. It optimizes posture support, not
forward crawling or walking, using batched CEM on direct GPU MuJoCo Warp.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv


def support_score(measure, require_palms=False):
    torso = ((measure["chest_height"] - 0.06) / 0.12).clamp(0, 1)
    head = ((measure["head_height"] - 0.06) / 0.12).clamp(0, 1)
    height_gate = (((measure["chest_height"] - 0.14) / 0.04).clamp(0, 1) *
                   ((measure["head_height"] - 0.12) / 0.04).clamp(0, 1))
    supports = measure["palm_shin_force" if require_palms else "support_force"] > 2.0
    contacts = supports.float().mean(dim=1)
    full_support = supports.all(dim=1).float()
    pelvis = (measure["ground_touch"][:, REGIONS.index("pelvis")] > 2.0).float()
    trunk = (measure["ground_touch"][:, REGIONS.index("trunk")] > 2.0).float()
    return (0.35 * torso + 0.15 * head + height_gate *
            (0.35 * contacts + 0.15 * full_support)
            - 0.2 * pelvis - 0.2 * trunk)


def evaluate(env, drives, control_steps, physics_per_control):
    env.reset()
    action = env.action_from_drives(drives)
    total = torch.zeros(env.worlds, device=env.device)
    last = None
    for _ in range(control_steps):
        last = env.step(action, physics_steps=physics_per_control)
        total += support_score(last)
    total /= control_steps
    total += 0.5 * support_score(last)
    total -= 0.01 * drives.square().mean(dim=1)
    return total, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".npz")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or checkpoint already exists")
    if args.population < 8 or args.generations < 1 or args.control_steps < 2:
        parser.error("Use at least eight candidates, one generation, and two control steps")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.population)
    mean = torch.zeros(len(CRAWL_JOINTS), device=env.device)
    deviation = torch.full_like(mean, 0.5)
    best_score = -float("inf")
    best_drives = None
    history = []
    baseline_drives = torch.zeros((args.population, len(CRAWL_JOINTS)), device=env.device)
    baseline_scores, baseline_state = evaluate(env, baseline_drives,
                                                args.control_steps, args.physics_per_control)
    baseline_score = float(baseline_scores[0])
    baseline_final_height = float(baseline_state["chest_height"][0])
    for generation in range(args.generations):
        candidates = (mean + torch.randn((args.population, len(CRAWL_JOINTS)),
                                          device=env.device) * deviation).clamp(-1, 1)
        candidates[0] = mean.clamp(-1, 1)
        candidates[1] = 0
        scores, final_state = evaluate(env, candidates, args.control_steps,
                                       args.physics_per_control)
        elite_ids = torch.topk(scores, max(4, args.population // 4)).indices
        elites = candidates[elite_ids]
        mean = 0.6 * mean + 0.4 * elites.mean(dim=0)
        deviation = (0.6 * deviation + 0.4 * elites.std(dim=0,
                                                       unbiased=False)).clamp(0.05, 0.8)
        winner = int(torch.argmax(scores))
        winner_score = float(scores[winner])
        if winner_score > best_score:
            best_score = winner_score
            best_drives = candidates[winner].clone()
        history.append({
            "generation": generation,
            "best_batch_score": winner_score,
            "mean_batch_score": float(scores.mean()),
            "best_so_far": best_score,
            "winner_final_chest_height_m": float(final_state["chest_height"][winner]),
            "winner_final_head_height_m": float(final_state["head_height"][winner]),
            "winner_final_support_regions": int((final_state["support_force"][winner] > 2).sum()),
        })
        print(json.dumps(history[-1]), flush=True)
    final_drives = best_drives.expand(args.population, -1)
    final_scores, final_state = evaluate(env, final_drives,
                                         args.control_steps, args.physics_per_control)
    report = {
        "scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
        "seed": args.seed, "population": args.population,
        "generations": args.generations, "control_steps": args.control_steps,
        "physics_per_control": args.physics_per_control,
        "baseline_score": baseline_score, "best_score": best_score,
        "replayed_best_score": float(final_scores[0]),
        "baseline_final_chest_height_m": baseline_final_height,
        "best_final_chest_height_m": float(final_state["chest_height"][0]),
        "best_final_head_height_m": float(final_state["head_height"][0]),
        "best_final_support_force_n": final_state["support_force"][0].cpu().tolist(),
        "best_forward_displacement_m": float(final_state["forward_displacement"][0]),
        "crawl_joint_names": list(CRAWL_JOINTS),
        "best_drives": best_drives.cpu().tolist(),
        "history": history,
        "scope": "Tonic hand-knee support optimization, not learned crawling locomotion or walking",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, drives=best_drives.cpu().numpy(),
                        joint_names=np.asarray(CRAWL_JOINTS))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "baseline_score", "best_score", "replayed_best_score",
        "baseline_final_chest_height_m", "best_final_chest_height_m",
        "best_final_support_force_n")}, indent=2))


if __name__ == "__main__":
    main()
