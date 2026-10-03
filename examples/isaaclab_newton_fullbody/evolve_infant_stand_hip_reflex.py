"""Evolve ankle and hip pressure reflexes for sustained infant standing."""

import argparse
import json
from pathlib import Path

import torch

from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import InfantCrawlEnv
from sweep_infant_cop_reflex import evaluate
from train_infant_stand_ppo import load_bias


def score_population(env, bias, gains, seed, worlds_per_candidate, control_steps):
    population = gains.shape[0]
    positions = aligned_positions(env, seed, worlds_per_candidate).repeat(population, 1)
    expanded = gains.repeat_interleave(worlds_per_candidate, dim=0)
    fraction, final, _, late = evaluate(env, bias, expanded, positions,
                                        control_steps, return_late=True)
    return tuple(values.reshape(population, worlds_per_candidate).mean(-1)
                 for values in (fraction, late, final))


def score_candidate(env, bias, gains, positions, control_steps):
    expanded = gains[None].expand(env.worlds, -1)
    fraction, final, chest, late = evaluate(env, bias, expanded, positions,
                                            control_steps, return_late=True)
    score = 0.5 * fraction.mean() + 1.5 * late.mean() + 2 * final.mean()
    return {"gains": gains.detach().cpu().tolist(),
            "standing_fraction": float(fraction.mean()),
            "late_standing_fraction": float(late.mean()),
            "final_standing_count": int(final.sum()),
            "score": float(score),
            "per_world_standing_fraction": fraction.cpu().tolist(),
            "per_world_late_fraction": late.cpu().tolist(),
            "final_chest_height_m": chest.cpu().tolist()}


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
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--validation-seed", type=int, default=1701)
    parser.add_argument("--heldout-seeds", nargs="+", type=int,
                        default=[1801, 1803, 1807])
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
    saved = json.loads(args.baseline_evolution.read_text())
    ankle_gains = saved["selected"]["gains"]
    if len(ankle_gains) != 2:
        parser.error("Baseline evolution must have two ankle gains")
    baseline = torch.tensor([*ankle_gains, 0., 0.], device="cuda:0")
    mean = baseline.clone()
    deviation = torch.tensor([10., 2., 15., 3.], device="cuda:0")
    lower = torch.tensor([-60., -12., -50., -10.], device="cuda:0")
    upper = -lower
    environment = InfantCrawlEnv(args.scene, args.pose,
                                 args.population * args.worlds_per_candidate)
    bias = load_bias(args.bias, environment.device)
    history = []
    pool = []
    for generation in range(args.generations):
        torch.manual_seed(args.seed + generation)
        candidates = (mean + torch.randn((args.population, 4), device=environment.device) *
                      deviation).clamp(lower, upper)
        candidates[0] = mean
        candidates[1] = baseline
        candidates[2] = 0
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
        elite = candidates[torch.topk(scores, max(2, args.population // 4)).indices]
        mean = 0.5 * mean + 0.5 * elite.mean(0)
        deviation = (0.7 * deviation +
                     0.3 * elite.std(0, unbiased=False)).clamp(
                         torch.tensor([2., 0.4, 2., 0.4], device=environment.device),
                         torch.tensor([25., 5., 25., 5.], device=environment.device))
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
                  "mean_gains": mean.cpu().tolist()}
        history.append(record)
        print(json.dumps(record), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8)
    validation_bias = load_bias(args.bias, validation.device)
    validation_positions = aligned_positions(validation, args.validation_seed, 8)
    contenders = sorted(pool, key=lambda row: row["training_score"], reverse=True)[:8]
    contenders.extend({"gains": gains} for gains in (baseline.cpu().tolist(),
                                                      [0., 0., 0., 0.]))
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
                            ("ankle_only", baseline.cpu().tolist()),
                            ("no_reflex", [0., 0., 0., 0.])):
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
                       "hip_position", "hip_velocity"],
        "history": history, "pool": pool, "validation": validation_results,
        "selected": selected, "heldout": heldout,
        "scope": "Four-gain evolved balance reflex; not acquired walking"},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
