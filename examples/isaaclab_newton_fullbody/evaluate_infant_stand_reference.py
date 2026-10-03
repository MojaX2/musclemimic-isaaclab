"""Compare nominal and contact-aligned joint targets for standing feedback."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import mujoco
import torch

from developmental_skin import REGIONS
from evaluate_infant_newton_crawl import newton_crawl_environment
from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_stand_ppo import load_bias


FOOT_IDS = (REGIONS.index("right_foot"), REGIONS.index("left_foot"))


def reference_feedback_drives(env, bias, positions, blend):
    if not 0 <= blend <= 1:
        raise ValueError("Reference blend must be between zero and one")
    if positions.shape != env.position.shape:
        raise ValueError("Reference positions must match environment coordinates")
    if bias.shape != (env.worlds, len(env.crawl_qpos_ids)):
        raise ValueError("Bias must match controlled joints")
    nominal = env.initial[:, env.crawl_qpos_ids]
    aligned = positions[:, env.crawl_qpos_ids]
    target = nominal + blend * (aligned - nominal)
    return (bias + 1.5 * (target - env.position[:, env.crawl_qpos_ids]) -
            env.velocity[:, env.crawl_qvel_ids]).clamp(-1, 1)


def evaluate(env, bias, gains, positions, blend, control_steps):
    env.reset(positions)
    measure = env.measure()
    supported = torch.zeros(env.worlds, device=env.device)
    late_supported = torch.zeros_like(supported)
    late_start = control_steps - max(1, control_steps // 4)
    for step in range(control_steps):
        base = reference_feedback_drives(env, bias, positions, blend)
        drives = cop_reflex_drives(env, measure, base, gains)
        measure = env.step(env.action_from_drives(drives, baseline=0.4,
                                                  amplitude=0.3), 1)
        standing = ((measure["chest_height"] > 0.45) &
                    (measure["ground_touch"][:, FOOT_IDS] > 2).all(-1)).float()
        supported += standing
        if step >= late_start:
            late_supported += standing
    return {
        "standing_fraction": float((supported / control_steps).mean()),
        "late_standing_fraction": float((late_supported / (control_steps - late_start)).mean()),
        "final_standing_count": int(standing.sum()),
        "per_world_standing_fraction": (supported / control_steps).cpu().tolist(),
        "final_chest_height_m": measure["chest_height"].cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=800)
    parser.add_argument("--seeds", nargs="+", type=int, default=[6501, 6503, 6507])
    parser.add_argument("--blends", nargs="+", type=float, default=[0., 0.5, 1.])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps) < 1 or not args.seeds or not args.blends:
        parser.error("Worlds, control steps, seeds, and blends must be nonempty and positive")
    if any(not 0 <= value <= 1 for value in args.blends):
        parser.error("Reference blends must be between zero and one")
    gains_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(gains_values) != 6:
        parser.error("Standing evolution must contain six pressure-center gains")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if args.backend == "newton":
        context = newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                           controlled_joint_names=CRAWL_JOINTS)
    else:
        context = nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds))
    with context as env:
        bias = load_bias(args.bias, env.device).expand(env.worlds, -1)
        gains = torch.tensor(gains_values, device=env.device,
                             dtype=torch.float32).expand(env.worlds, -1)
        records = []
        for seed in args.seeds:
            positions = aligned_positions(env, seed, env.worlds)
            pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
            for blend in args.blends:
                result = evaluate(env, bias, gains, positions, blend, args.control_steps)
                records.append({"seed": seed, "reference_blend": blend,
                                "pose_sha256": pose_hash, **result})
                print(json.dumps({"seed": seed, "reference_blend": blend,
                                  "standing_fraction": result["standing_fraction"],
                                  "late_standing_fraction": result["late_standing_fraction"],
                                  "final_standing_count": result["final_standing_count"]}),
                      flush=True)
        report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
                  "bias": str(args.bias.resolve()), "evolution": str(args.evolution.resolve()),
                  "backend": args.backend, "worlds": args.worlds,
                  "duration_s": args.control_steps * float(env.source.opt.timestep),
                  "seeds": args.seeds, "blends": args.blends, "records": records,
                  "scope": "Paired standing feedback reference audit; no walking policy"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
