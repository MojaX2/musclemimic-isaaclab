"""Compare the same infant standing pressure reflex in direct Warp and Isaac Lab/Newton."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import mujoco
import torch
import warp as wp

from evaluate_infant_newton_crawl import newton_crawl_environment
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from infant_stand_resets import align_stand_feet
from sweep_infant_cop_reflex import evaluate
from train_infant_crawl_ppo import jittered_positions
from train_infant_stand_ppo import load_bias


def evaluate_backend(environment, bias, selected, comparison, seeds, control_steps,
                     baseline, amplitude, trunk_gains=None):
    records = []
    for seed in seeds:
        torch.manual_seed(seed)
        raw = jittered_positions(environment, 0.05, 0.01)
        positions, alignment = align_stand_feet(environment.source, raw,
                                                environment.initial[0],
                                                environment.root_address)
        environment.reset(positions)
        initial = environment.measure()
        mass_center = wp.to_torch(environment.data.subtree_com)[:, 0, 0]
        initial_diagnostics = {
            "pose_sha256": hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
            "foot_force_n": initial["foot_force"].mean(0).cpu().tolist(),
            "chest_height_m": float(initial["chest_height"].mean()),
            "center_of_mass_x_m": float(mass_center.mean()),
            "alignment_foot_distance_m": [item["foot_distance_m"] for item in alignment],
        }
        conditions = (("evolved", selected, trunk_gains),
                      ("no_trunk", selected, None),
                      ("no_reflex", [0., 0.], None)) if trunk_gains is not None else (
                          ("evolved", selected, None),
                          (comparison[0], comparison[1], None),
                          ("no_reflex", [0., 0.], None))
        for name, values, vestibular in conditions:
            gains = torch.tensor(values, device=environment.device,
                                 dtype=torch.float32).expand(environment.worlds, -1)
            trunk = (torch.tensor(vestibular, device=environment.device,
                                  dtype=torch.float32).expand(environment.worlds, -1)
                     if vestibular is not None else None)
            standing, final, chest, late = evaluate(environment, bias, gains, positions,
                                                    control_steps, return_late=True,
                                                    baseline=baseline, amplitude=amplitude,
                                                    trunk_gains=trunk)
            records.append({"seed": seed, "condition": name, "gains": values,
                            "initial": initial_diagnostics,
                            "standing_fraction": float(standing.mean()),
                            "late_standing_fraction": float(late.mean()),
                            "final_standing_count": int(final.sum()),
                            "per_world_standing_fraction": standing.cpu().tolist(),
                            "final_chest_height_m": chest.cpu().tolist()})
        print(json.dumps({"seed": seed}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--ankle-evolution", type=Path)
    parser.add_argument("--trunk-evolution", type=Path)
    parser.add_argument("--backend", choices=("direct", "newton"), required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--muscle-baseline", type=float, default=0.4)
    parser.add_argument("--muscle-amplitude", type=float, default=0.3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[89, 97, 101])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.worlds, args.control_steps) < 1 or not args.seeds:
        parser.error("Worlds, control steps, and seeds must be positive")
    if (args.muscle_baseline < args.muscle_amplitude or
            args.muscle_baseline + args.muscle_amplitude > 1 or
            args.muscle_amplitude < 0):
        parser.error("Muscle excitation range must stay within zero and one")
    selected = json.loads(args.evolution.read_text())["selected"]["gains"]
    if len(selected) not in (2, 4, 6):
        parser.error("Standing evolution must contain two, four, or six gains")
    trunk_gains = None
    if args.trunk_evolution is not None:
        if len(selected) != 6:
            parser.error("Trunk reflex comparison requires six foot gains")
        trunk_gains = json.loads(args.trunk_evolution.read_text())["selected"]["gains"]
        if len(trunk_gains) != 4:
            parser.error("Trunk evolution must contain four gains")
        comparison = ("no_trunk", selected)
    elif len(selected) in (4, 6):
        if args.ankle_evolution is None:
            parser.error("Extended-reflex comparison requires --ankle-evolution")
        ankle = json.loads(args.ankle_evolution.read_text())["selected"]["gains"]
        if len(ankle) != len(selected) - 2:
            parser.error("Comparison evolution must contain two fewer gains")
        comparison = (("ankle_only" if len(selected) == 4 else "sagittal_only"), ankle)
    else:
        comparison = ("previous", [-20., -2.])
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if args.backend == "newton":
        context = newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                           controlled_joint_names=CRAWL_JOINTS)
    else:
        context = nullcontext(InfantCrawlEnv(args.scene, args.pose, args.worlds,
                                             controlled_joint_names=CRAWL_JOINTS))
    with context as environment:
        bias = load_bias(args.bias, environment.device)
        records = evaluate_backend(environment, bias, selected, comparison, args.seeds,
                                   args.control_steps, args.muscle_baseline,
                                   args.muscle_amplitude, trunk_gains)
        report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
                  "backend": args.backend, "bias": str(args.bias.resolve()),
                  "evolution": str(args.evolution.resolve()), "worlds": args.worlds,
                  "ankle_evolution": (str(args.ankle_evolution.resolve())
                                       if args.ankle_evolution else None),
                  "trunk_evolution": (str(args.trunk_evolution.resolve())
                                      if args.trunk_evolution else None),
                  "duration_s": args.control_steps * float(environment.source.opt.timestep),
                  "muscle_baseline": args.muscle_baseline,
                  "muscle_amplitude": args.muscle_amplitude,
                  "seeds": args.seeds, "records": records,
                  "solver_limit_steps": getattr(environment, "solver_limit_steps", None),
                  "scope": "Standing pressure reflex comparison; no walking controller"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
