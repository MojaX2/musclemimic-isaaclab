"""Evaluate deterministic tactile-feedback support on held-out infant poses."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_infant_crawl_support import rollout as tonic_rollout
from developmental_skin import REGIONS
from infant_crawl_env import CRAWL_JOINTS, PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from infant_crawl_oscillator import cpg_drives, fixed_limb_drives, oscillator_phase
from infant_crawl_metrics import crawl_locomotion_metrics
from infant_tactile_features import TactileFeatureEncoder
from infant_world_features import WorldFeatureEncoder
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe


def policy_rollout(env, policy, positions, control_steps, physics_per_control, use_touch,
                   world_features=None, oscillator_parameters=None, reference_policy=None):
    env.reset(positions)
    measure = env.measure()
    latent = (torch.zeros((env.worlds, getattr(world_features, "feature_size", 256)), device=env.device)
              if world_features is not None else None)
    samples = []
    with torch.no_grad():
        interval = physics_per_control * float(env.source.opt.timestep)
        for step in range(control_steps):
            phase = (oscillator_phase(oscillator_parameters, step * interval)
                     if oscillator_parameters is not None else None)
            drives = policy.actor(observe(env, measure, use_touch, latent, phase)).clamp(-1, 1)
            if oscillator_parameters is not None:
                if reference_policy is not None:
                    reference = reference_policy.actor(
                        observe(env, measure, use_touch, latent)).clamp(-1, 1)
                    drives = fixed_limb_drives(drives, reference, oscillator_parameters,
                                               step * interval)
                else:
                    drives = cpg_drives(drives, oscillator_parameters, step * interval)
            muscle_action = env.action_from_drives(drives)
            if world_features is not None:
                latent = world_features.encode(env, measure, muscle_action)
            measure = env.step(muscle_action, physics_per_control)
            samples.append({key: value.cpu().numpy() for key, value in measure.items()})
    support = np.stack([sample["support_force"] for sample in samples])
    palm_shin = np.stack([sample["palm_shin_force"] for sample in samples])
    chest = np.stack([sample["chest_height"] for sample in samples])
    head = np.stack([sample["head_height"] for sample in samples])
    ground = np.stack([sample["ground_touch"] for sample in samples])
    settle_steps = math.ceil(0.25 / (physics_per_control * env.source.opt.timestep))
    elevated_support = np.all(support > 2, axis=2) & (chest > 0.16) & (head > 0.14)
    elevated_palm_shin = np.all(palm_shin > 2, axis=2) & (chest > 0.16) & (head > 0.14)
    locomotion = crawl_locomotion_metrics(
        palm_shin, chest, head, ground[:, :, REGIONS.index("pelvis")],
        samples[-1]["forward_displacement"])
    return {
        "fraction_four_supports": float(np.mean(np.all(support > 2, axis=2))),
        "fraction_four_supports_after_0p25s": float(np.mean(np.all(support[settle_steps:] > 2, axis=2)))
        if len(samples) > settle_steps else None,
        "fraction_elevated_four_supports": float(np.mean(elevated_support)),
        "fraction_elevated_four_supports_after_0p25s": float(np.mean(elevated_support[settle_steps:]))
        if len(samples) > settle_steps else None,
        "fraction_elevated_palm_shin_supports": float(np.mean(elevated_palm_shin)),
        "elevated_palm_shin_fraction_per_world": elevated_palm_shin.mean(axis=0).tolist(),
        "fraction_elevated_palm_shin_supports_after_0p25s": float(np.mean(elevated_palm_shin[settle_steps:]))
        if len(samples) > settle_steps else None,
        "elevated_palm_shin_fraction_after_0p25s_per_world":
        elevated_palm_shin[settle_steps:].mean(axis=0).tolist()
        if len(samples) > settle_steps else None,
        "fraction_chest_above_0p16_m": float(np.mean(chest > 0.16)),
        "fraction_pelvis_contact": float(np.mean(ground[:, :, REGIONS.index("pelvis")] > 2)),
        "final_chest_height_m": chest[-1].tolist(),
        "final_head_height_m": head[-1].tolist(),
        "final_forward_displacement_m": samples[-1]["forward_displacement"].tolist(),
        "final_support_force_n": support[-1].tolist(),
        "final_palm_shin_force_n": palm_shin[-1].tolist(),
        **locomotion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=20)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--joint-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    joint_names = tuple(checkpoint["joint_names"])
    if joint_names not in (CRAWL_JOINTS, PALM_JOINTS):
        raise ValueError("Crawl checkpoint joint order does not match a supported action set")
    if joint_names == PALM_JOINTS and checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION:
        raise ValueError("Palm checkpoint uses a different tactile sensor mapping")
    oscillator_values = checkpoint.get("oscillator_parameters")
    if oscillator_values is not None and (joint_names != PALM_JOINTS or
            checkpoint.get("phase_observation_version") != "sin_cos_v1" or
            (checkpoint.get("world_features") is not None and
             checkpoint["world_features"].get("kind") != "tactile_next_contact_v1")):
        raise ValueError("Oscillator checkpoint has incompatible joints or phase observations")
    env = InfantCrawlEnv(args.scene, args.pose, args.worlds,
                         controlled_joint_names=joint_names)
    if (checkpoint.get("world_features") is not None and
            checkpoint.get("feature_timing") != "current_state_current_action_to_next_observation"):
        raise ValueError("Checkpoint uses a rejected, time-misaligned world-model feature path")
    policy = SupportPolicy(checkpoint["observation_size"], len(joint_names)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    feature_snapshot = checkpoint.get("world_features")
    if feature_snapshot is None:
        world_features = None
    elif feature_snapshot.get("kind") == "tactile_next_contact_v1":
        world_features = TactileFeatureEncoder.restore_snapshot(feature_snapshot, env.device)
    else:
        world_features = WorldFeatureEncoder.restore_snapshot(feature_snapshot, env.device)
    oscillator_parameters = (torch.tensor(oscillator_values, device=env.device).expand(args.worlds, -1)
                             if oscillator_values is not None else None)
    frozen_reference = checkpoint.get("frozen_limb_reference")
    reference_policy = None
    if frozen_reference is not None:
        if oscillator_parameters is None or frozen_reference["observation_size"] + 2 != checkpoint["observation_size"]:
            raise ValueError("Frozen oscillator reference does not match the phase-augmented policy")
        reference_policy = SupportPolicy(frozen_reference["observation_size"], len(joint_names)).to(env.device)
        reference_policy.load_state_dict(frozen_reference["policy"])
        reference_policy.eval()
    torch.manual_seed(args.seed)
    positions = jittered_positions(env, args.joint_jitter, 0.01)
    baseline = tonic_rollout(env, torch.zeros((args.worlds, len(joint_names)), device=env.device),
                             positions, args.control_steps, args.physics_per_control)
    learned = policy_rollout(env, policy, positions, args.control_steps,
                             args.physics_per_control, checkpoint["use_touch"], world_features,
                             oscillator_parameters, reference_policy)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "checkpoint": str(args.checkpoint.resolve()), "worlds": args.worlds,
              "seed": args.seed, "joint_jitter_rad": args.joint_jitter,
              "initial_pose_sha256": hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
              "duration_s": args.control_steps * args.physics_per_control * env.source.opt.timestep,
              "use_touch": checkpoint["use_touch"], "baseline": baseline,
              "oscillator_parameters": oscillator_values,
              "fixed_oscillator_limbs": frozen_reference is not None,
              "world_model_features": ("random" if world_features.random_features else "pretrained")
              if world_features is not None else None,
              "feature_kind": feature_snapshot.get("kind", "proprio_forward_v1")
              if feature_snapshot is not None else None,
              "learned": learned,
              "scope": "Held-out support and strict crawl-gait metrics; walking not tested"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"baseline_four_supports": baseline["fraction_four_supports"],
                      "learned_four_supports": learned["fraction_four_supports"],
                      "learned_final_chest_m": learned["final_chest_height_m"]}, indent=2))


if __name__ == "__main__":
    main()
