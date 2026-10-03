"""Evolve vestibular muscle reflexes for unsupported infant standing."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias
from train_infant_stand_support import stand_score


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))
REFLEX_JOINTS = (
    ("hip1", "foot1", "pitch"),
    ("hip2", "foot2", "roll"),
    ("hip3", "foot3", "yaw"),
)


def balance_reflex_drives(env, base_drives, gains):
    if gains.shape != (env.worlds, 12):
        raise ValueError("Vestibular reflex needs 12 gains per infant")
    root = env.root_address
    quaternion = env.position[:, root + 3:root + 7]
    rotation_signals = torch.stack((2 * quaternion[:, 0] * quaternion[:, 2],
                                    2 * quaternion[:, 0] * quaternion[:, 1],
                                    2 * quaternion[:, 0] * quaternion[:, 3]), dim=1)
    velocity_signals = torch.stack((env.velocity[:, root + 4],
                                    env.velocity[:, root + 3],
                                    env.velocity[:, root + 5]), dim=1) * 0.2
    signals = torch.stack((rotation_signals, velocity_signals), dim=-1)
    corrections = (gains.reshape(env.worlds, 3, 2, 2) * signals[:, :, None, :]).sum(-1)
    drives = base_drives.clone()
    for axis_id, (hip, ankle, _) in enumerate(REFLEX_JOINTS):
        side_signs = (1, 1) if axis_id == 0 else (1, -1)
        for side, sign in zip(("right", "left"), side_signs):
            hip_index = CRAWL_JOINTS.index(f"robot:{side}_{hip}")
            ankle_index = CRAWL_JOINTS.index(f"robot:{side}_{ankle}")
            drives[:, hip_index] += sign * corrections[:, axis_id, 0]
            drives[:, ankle_index] += sign * corrections[:, axis_id, 1]
    return drives.clamp(-1, 1)


def evaluate(env, bias, gains, positions, control_steps, physics_per_control):
    env.reset(positions)
    total = torch.zeros(env.worlds, device=env.device)
    standing = torch.zeros_like(total)
    last = None
    for _ in range(control_steps):
        base = env.feedback_drives(bias.expand(env.worlds, -1), 3.0, 1.0)
        drives = balance_reflex_drives(env, base, gains)
        last = env.step(env.action_from_drives(drives), physics_per_control)
        total += stand_score(env, last)
        standing += ((last["chest_height"] > 0.45) &
                     (last["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    final_standing = ((last["chest_height"] > 0.45) &
                      (last["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
    objective = total / control_steps + 3 * stand_score(env, last) + 2 * final_standing
    objective -= 0.001 * gains.square().mean(-1)
    return objective, standing / control_steps, final_standing, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--control-steps", type=int, default=40)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    checkpoint = args.output.with_suffix(".npz")
    if args.output.exists() or checkpoint.exists():
        parser.error("Output or checkpoint already exists")
    if args.population < 8 or args.generations < 1 or args.control_steps < 2:
        parser.error("Use at least eight candidates, one generation, two control steps")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.population)
    bias = load_bias(args.bias, env.device)
    mean = torch.zeros(12, device=env.device)
    mean[:4] = torch.tensor([2.0, 2.0, -0.5, -0.5], device=env.device)
    deviation = torch.full_like(mean, 1.5)
    candidate_pool = []
    history = []
    for generation in range(args.generations):
        candidates = (mean + torch.randn((args.population, 12), device=env.device) *
                      deviation).clamp(-6, 6)
        candidates[0] = mean.clamp(-6, 6)
        candidates[1] = 0
        positions = jittered_positions(env, 0.02, 0.005)
        score, fraction, final, last = evaluate(env, bias, candidates, positions,
                                                 args.control_steps, args.physics_per_control)
        elite = candidates[torch.topk(score, max(4, args.population // 4)).indices]
        mean = 0.6 * mean + 0.4 * elite.mean(0)
        deviation = (0.6 * deviation + 0.4 * elite.std(0, unbiased=False)).clamp(0.1, 3)
        winner = int(score.argmax())
        candidate_pool.append((float(score[winner]), candidates[winner].clone()))
        record = {"generation": generation + 1, "winner_score": float(score[winner]),
                  "winner_standing_fraction": float(fraction[winner]),
                  "winner_final_standing": bool(final[winner]),
                  "winner_final_chest_m": float(last["chest_height"][winner])}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    torch.manual_seed(17)
    validation_positions = jittered_positions(validation, 0.05, 0.01)
    contenders = sorted(candidate_pool, key=lambda item: item[0], reverse=True)[:10]
    contenders += [(None, mean.clamp(-6, 6)), (None, torch.zeros_like(mean))]
    validation_records = []
    selected_score = -float("inf")
    selected_gains = None
    for training_score, contender in contenders:
        score, fraction, final, _ = evaluate(
            validation, bias, contender.expand(8, -1), validation_positions,
            args.control_steps, args.physics_per_control)
        selection_score = float(score.mean() + fraction.mean())
        validation_records.append({"training_score": training_score,
                                   "validation_score": selection_score,
                                   "standing_fraction": float(fraction.mean()),
                                   "final_standing_fraction": float(final.mean())})
        if selection_score > selected_score:
            selected_score = selection_score
            selected_gains = contender.clone()
    heldout = InfantCrawlEnv(args.scene, args.pose, 8)
    torch.manual_seed(41)
    heldout_positions = jittered_positions(heldout, 0.05, 0.01)
    baseline = evaluate(heldout, bias, torch.zeros((8, 12), device=heldout.device),
                        heldout_positions, args.control_steps, args.physics_per_control)
    learned = evaluate(heldout, bias, selected_gains.expand(8, -1),
                       heldout_positions, args.control_steps, args.physics_per_control)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "bias": str(args.bias.resolve()), "seed": args.seed,
              "population": args.population, "generations": args.generations,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "history": history, "validation_seed": 17,
              "validation_candidates": validation_records,
              "heldout_seed": 41, "heldout_joint_jitter_rad": 0.05,
              "heldout_root_height_jitter_m": 0.01,
              "heldout_baseline_score": float(baseline[0].mean()),
              "heldout_learned_score": float(learned[0].mean()),
              "heldout_baseline_standing_fraction": float(baseline[1].mean()),
              "heldout_learned_standing_fraction": float(learned[1].mean()),
              "heldout_baseline_final_standing_fraction": float(baseline[2].mean()),
              "heldout_learned_final_standing_fraction": float(learned[2].mean()),
              "heldout_baseline_final_chest_m": baseline[3]["chest_height"].cpu().tolist(),
              "heldout_learned_final_chest_m": learned[3]["chest_height"].cpu().tolist(),
              "selected_gains": selected_gains.cpu().tolist(),
              "gain_order": [f"{axis}_{joint}_{signal}" for hip, ankle, axis in REFLEX_JOINTS
                             for joint in (hip, ankle) for signal in ("angle", "velocity")],
              "scope": "Vestibular standing reflex; not stepping or walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, gains=selected_gains.cpu().numpy(),
                        gain_order=np.asarray(report["gain_order"]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("heldout_baseline_standing_fraction", "heldout_learned_standing_fraction",
                       "heldout_baseline_final_standing_fraction",
                       "heldout_learned_final_standing_fraction")}, indent=2))


if __name__ == "__main__":
    main()
