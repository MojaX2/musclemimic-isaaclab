"""Evolve pressure-center reflex gains on balanced infant standing starts."""

import argparse
import json
from pathlib import Path

import torch

from infant_crawl_env import InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from sweep_infant_cop_reflex import evaluate
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


def aligned_positions(env, seed, count):
    torch.manual_seed(seed)
    raw = jittered_positions(env, 0.05, 0.01)[:count].clone()
    positions, _ = align_stand_feet(env.source, raw, env.initial[0],
                                    env.root_address)
    return positions


def score_population(env, bias, gains, seed, worlds_per_candidate, control_steps):
    population = gains.shape[0]
    positions = aligned_positions(env, seed, worlds_per_candidate)
    positions = positions.repeat(population, 1)
    expanded = gains.repeat_interleave(worlds_per_candidate, dim=0)
    fraction, final, _ = evaluate(env, bias, expanded, positions, control_steps)
    fraction = fraction.reshape(population, worlds_per_candidate).mean(-1)
    final = final.reshape(population, worlds_per_candidate).mean(-1)
    return fraction, final


def score_candidate(env, bias, gains, positions, control_steps):
    expanded = gains[None].expand(env.worlds, -1)
    fraction, final, chest = evaluate(env, bias, expanded, positions, control_steps)
    return {"gains": gains.detach().cpu().tolist(),
            "standing_fraction": float(fraction.mean()),
            "final_standing_count": int(final.sum()),
            "score": float(fraction.mean() + 2 * final.mean()),
            "final_chest_height_m": chest.detach().cpu().tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--worlds-per-candidate", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--validation-seed", type=int, default=83)
    parser.add_argument("--heldout-seeds", nargs="+", type=int,
                        default=[89, 97, 101])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if (args.generations < 1 or args.population < 4 or args.worlds_per_candidate < 1 or
            args.control_steps < 1 or not args.heldout_seeds):
        parser.error("Evolution and evaluation dimensions must be positive")
    if args.validation_seed in args.heldout_seeds:
        parser.error("Validation seed must differ from held-out seeds")
    training_seeds = set(range(args.seed + 100,
                               args.seed + 100 + 2 * args.generations))
    if args.validation_seed in training_seeds or training_seeds.intersection(args.heldout_seeds):
        parser.error("Validation and held-out seeds must differ from training seeds")
    env = InfantCrawlEnv(args.scene, args.pose,
                         args.population * args.worlds_per_candidate)
    bias = load_bias(args.bias, env.device)
    mean = torch.tensor([-20., -2.], device=env.device)
    deviation = torch.tensor([15., 3.], device=env.device)
    lower = torch.tensor([-60., -12.], device=env.device)
    upper = -lower
    history = []
    pool = []
    for generation in range(args.generations):
        torch.manual_seed(args.seed + generation)
        candidates = (mean + torch.randn((args.population, 2), device=env.device) *
                      deviation).clamp(lower, upper)
        candidates[0] = mean
        candidates[1] = 0
        candidates[2] = torch.tensor([-20., -2.], device=env.device)
        scores = torch.zeros(args.population, device=env.device)
        fractions = torch.zeros_like(scores)
        finals = torch.zeros_like(scores)
        seeds = [args.seed + 100 + generation * 2 + offset for offset in range(2)]
        for seed in seeds:
            fraction, final = score_population(env, bias, candidates, seed,
                                               args.worlds_per_candidate,
                                               args.control_steps)
            fractions += fraction / len(seeds)
            finals += final / len(seeds)
            scores += (fraction + 2 * final) / len(seeds)
        elite = candidates[torch.topk(scores, max(2, args.population // 4)).indices]
        mean = 0.5 * mean + 0.5 * elite.mean(0)
        deviation = (0.7 * deviation +
                     0.3 * elite.std(0, unbiased=False)).clamp(
                         torch.tensor([2., 0.4], device=env.device),
                         torch.tensor([25., 5.], device=env.device))
        for index in range(args.population):
            pool.append({"generation": generation + 1,
                         "gains": candidates[index].cpu().tolist(),
                         "training_fraction": float(fractions[index]),
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
    candidates = sorted(pool, key=lambda row: row["training_score"], reverse=True)[:8]
    candidates.extend({"gains": gains} for gains in ([0., 0.], [-20., -2.]))
    validation_results = []
    seen = set()
    for candidate in candidates:
        gains = tuple(round(value, 5) for value in candidate["gains"])
        if gains in seen:
            continue
        seen.add(gains)
        result = score_candidate(validation, validation_bias,
                                 torch.tensor(gains, device=validation.device),
                                 validation_positions, args.control_steps)
        validation_results.append(result)
    selected = max(validation_results, key=lambda row: row["score"])
    comparisons = []
    for seed in args.heldout_seeds:
        positions = aligned_positions(validation, seed, 8)
        for label, gains in (("selected", selected["gains"]),
                             ("previous", [-20., -2.]), ("no_reflex", [0., 0.])):
            result = score_candidate(validation, validation_bias,
                                     torch.tensor(gains, device=validation.device),
                                     positions, args.control_steps)
            comparisons.append({"seed": seed, "condition": label, **result})
        print(json.dumps({"heldout_seed": seed}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scene": str(args.scene.resolve()),
                                       "pose": str(args.pose.resolve()),
                                       "bias": str(args.bias.resolve()),
                                       "seed": args.seed,
                                       "validation_seed": args.validation_seed,
                                       "heldout_seeds": args.heldout_seeds,
                                       "generations": args.generations,
                                       "population": args.population,
                                       "worlds_per_candidate": args.worlds_per_candidate,
                                       "duration_s": args.control_steps *
                                                     float(env.source.opt.timestep),
                                       "history": history, "pool": pool,
                                       "validation": validation_results,
                                       "selected": selected,
                                       "heldout": comparisons,
                                       "scope": "Two-gain evolved reflex; not neural development or walking"},
                                      indent=2) + "\n")


if __name__ == "__main__":
    main()
