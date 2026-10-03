"""Evolve mirrored ankle-roll feedback around the infant sagittal balance reflex."""

import argparse
import json
from pathlib import Path

import torch

from evolve_infant_stand_cop_reflex import aligned_positions
from evolve_infant_stand_hip_reflex import score_candidate, score_population
from infant_crawl_env import InfantCrawlEnv
from train_infant_stand_ppo import load_bias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--baseline-evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--worlds-per-candidate", type=int, default=4)
    parser.add_argument("--control-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--validation-seed", type=int, default=2201)
    parser.add_argument("--heldout-seeds", nargs="+", type=int,
                        default=[2301, 2303, 2307])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if (args.generations < 1 or args.population < 4 or
            args.worlds_per_candidate < 1 or args.control_steps < 1 or
            not args.heldout_seeds):
        parser.error("Evolution dimensions must be positive")
    training_seeds = {args.seed + 100 + generation * 2 + offset
                      for generation in range(args.generations) for offset in range(2)}
    if (args.validation_seed in training_seeds or
            args.validation_seed in args.heldout_seeds or
            training_seeds.intersection(args.heldout_seeds)):
        parser.error("Training, validation, and held-out seeds must be distinct")
    sagittal = json.loads(args.baseline_evolution.read_text())["selected"]["gains"]
    if len(sagittal) != 4:
        parser.error("Baseline evolution must contain four sagittal gains")
    baseline = torch.tensor([*sagittal, 0., 0.], device="cuda:0")
    mean = torch.zeros(2, device="cuda:0")
    deviation = torch.tensor([25., 4.], device="cuda:0")
    lower = torch.tensor([-80., -12.], device="cuda:0")
    upper = -lower
    environment = InfantCrawlEnv(args.scene, args.pose,
                                 args.population * args.worlds_per_candidate)
    bias = load_bias(args.bias, environment.device)
    history = []
    pool = []
    for generation in range(args.generations):
        torch.manual_seed(args.seed + generation)
        lateral = (mean + torch.randn((args.population, 2), device=environment.device) *
                   deviation).clamp(lower, upper)
        lateral[0] = mean
        lateral[1] = 0
        lateral[2] = torch.tensor([-20., -2.], device=environment.device)
        candidates = baseline.expand(args.population, -1).clone()
        candidates[:, 4:] = lateral
        fractions = torch.zeros(args.population, device=environment.device)
        late_fractions = torch.zeros_like(fractions)
        finals = torch.zeros_like(fractions)
        seeds = [args.seed + 100 + generation * 2 + offset for offset in range(2)]
        for seed in seeds:
            fraction, late, final = score_population(
                environment, bias, candidates, seed, args.worlds_per_candidate,
                args.control_steps)
            fractions += fraction / len(seeds)
            late_fractions += late / len(seeds)
            finals += final / len(seeds)
        scores = 0.5 * fractions + 1.5 * late_fractions + 2 * finals
        elite = lateral[torch.topk(scores, max(2, args.population // 4)).indices]
        mean = 0.5 * mean + 0.5 * elite.mean(0)
        deviation = (0.7 * deviation +
                     0.3 * elite.std(0, unbiased=False)).clamp(
                         torch.tensor([2., 0.4], device=environment.device),
                         torch.tensor([35., 6.], device=environment.device))
        for index in range(args.population):
            pool.append({"generation": generation + 1,
                         "gains": candidates[index].cpu().tolist(),
                         "training_fraction": float(fractions[index]),
                         "training_late_fraction": float(late_fractions[index]),
                         "training_final_fraction": float(finals[index]),
                         "training_score": float(scores[index])})
        record = {"generation": generation + 1, "seeds": seeds,
                  "best_score": float(scores.max()),
                  "best_final_fraction": float(finals[scores.argmax()]),
                  "mean_lateral_gains": mean.cpu().tolist()}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    validation_bias = load_bias(args.bias, validation.device)
    validation_positions = aligned_positions(validation, args.validation_seed, 8)
    contenders = sorted(pool, key=lambda row: row["training_score"], reverse=True)[:8]
    contenders.append({"gains": baseline.cpu().tolist()})
    validation_results = []
    seen = set()
    for contender in contenders:
        gains = tuple(round(value, 5) for value in contender["gains"])
        if gains in seen:
            continue
        seen.add(gains)
        validation_results.append(score_candidate(
            validation, validation_bias,
            torch.tensor(gains, device=validation.device),
            validation_positions, args.control_steps))
    selected = max(validation_results, key=lambda row: row["score"])
    heldout = []
    for seed in args.heldout_seeds:
        positions = aligned_positions(validation, seed, 8)
        for name, gains in (("selected", selected["gains"]),
                            ("sagittal_only", baseline.cpu().tolist()),
                            ("no_reflex", [0., 0., 0., 0., 0., 0.])):
            result = score_candidate(validation, validation_bias,
                                     torch.tensor(gains, device=validation.device),
                                     positions, args.control_steps)
            heldout.append({"seed": seed, "condition": name, **result})
        print(json.dumps({"heldout_seed": seed}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
        "bias": str(args.bias.resolve()),
        "baseline_evolution": str(args.baseline_evolution.resolve()),
        "duration_s": args.control_steps * float(environment.source.opt.timestep),
        "seed": args.seed, "validation_seed": args.validation_seed,
        "heldout_seeds": args.heldout_seeds, "generations": args.generations,
        "population": args.population,
        "worlds_per_candidate": args.worlds_per_candidate,
        "gain_order": ["ankle_position", "ankle_velocity",
                       "hip_position", "hip_velocity",
                       "ankle_roll_position", "ankle_roll_velocity"],
        "history": history, "pool": pool, "validation": validation_results,
        "selected": selected, "heldout": heldout,
        "scope": "Six-gain evolved balance reflex; not acquired walking"},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
