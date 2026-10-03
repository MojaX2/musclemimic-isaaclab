"""Evolve diagonal-limb oscillators around the tactile infant support policy."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from developmental_skin import REGIONS
from infant_crawl_env import PALM_JOINTS, PALM_SENSOR_VERSION, InfantCrawlEnv
from run_infant_babbling import virtual_lengths
from infant_crawl_oscillator import (LIMB_IDS, SHOULDER_LIFT_IDS, WRIST_IDS,
                                    cpg_drives, exclusive_contact_state)
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe


def height_gait_gain(chest_height):
    return ((chest_height - 0.16) / 0.04).clamp(0, 1)


def shin_gait_gain(palm_shin_force):
    return ((palm_shin_force[:, 2:].sum(-1) - 10) / 20).clamp(0, 1)


def apply_gait_gain(base, drives, gain, leg_only=False):
    if leg_only:
        adjusted = drives.clone()
        adjusted[:, LIMB_IDS[4:]] = (base[:, LIMB_IDS[4:]] +
                                     gain[:, None] * (drives[:, LIMB_IDS[4:]] -
                                                      base[:, LIMB_IDS[4:]]))
        return adjusted.clamp(-1, 1)
    return (base + gain[:, None] * (drives - base)).clamp(-1, 1)


def guard_arm_oscillation(base, drives, palm_shin_force, shoulder_lift=False):
    if base.shape != drives.shape or palm_shin_force.shape != (base.shape[0], 4):
        raise ValueError("Contact guard requires matched drives and four contact forces")
    guarded = drives.clone()
    right_arm = (LIMB_IDS[0], LIMB_IDS[2], WRIST_IDS[0])
    left_arm = (LIMB_IDS[1], LIMB_IDS[3], WRIST_IDS[1])
    if shoulder_lift:
        right_arm += (SHOULDER_LIFT_IDS[0],)
        left_arm += (SHOULDER_LIFT_IDS[1],)
    right_allowed = palm_shin_force[:, 1] > 2
    left_allowed = palm_shin_force[:, 0] > 2
    guarded[:, right_arm] = torch.where(right_allowed[:, None],
                                        drives[:, right_arm], base[:, right_arm])
    guarded[:, left_arm] = torch.where(left_allowed[:, None],
                                       drives[:, left_arm], base[:, left_arm])
    return guarded, torch.stack((right_allowed, left_allowed), dim=1)


def gait_objective_score(forward, support_fraction, elevated_fraction, four_fraction,
                         pelvis_fraction, hand_switches, shin_switches, crawl_success,
                         amplitudes):
    return (4 * forward.clamp(-0.1, 0.3) + 2 * support_fraction +
            0.5 * elevated_fraction + 0.2 * four_fraction +
            0.4 * torch.minimum(hand_switches, shin_switches).clamp(max=3) -
            pelvis_fraction + 2 * crawl_success.float() -
            0.01 * amplitudes.square().mean(-1))


def sustained_gait_bonus(late_forward, final_quarter_support, final_quarter_elevated):
    return (4 * late_forward.clamp(-0.2, 0.2) +
            2 * final_quarter_support + final_quarter_elevated)


def coupled_sustained_gait_bonus(late_forward, final_quarter_support,
                                 final_quarter_elevated, hand_switches):
    positive_progress = late_forward.clamp(0, 0.2)
    return (30 * positive_progress * final_quarter_support +
            15 * positive_progress * final_quarter_elevated +
            0.6 * hand_switches.clamp(max=2) * final_quarter_support -
            4 * (-late_forward).clamp_min(0))


def crawl_success_gate(forward, elevated_fraction, support_fraction,
                       late_support_fraction, pelvis_fraction, hand_switches,
                       shin_switches):
    legacy = ((forward >= 0.05) & (elevated_fraction >= 0.75) &
              (support_fraction >= 0.5) & (pelvis_fraction < 0.1) &
              (hand_switches >= 2) & (shin_switches >= 2))
    return legacy, legacy & (late_support_fraction >= 0.5)


def evaluate(env, policy, parameters, positions, control_steps, physics_per_control,
             use_touch, gait_objective=False, sustained_gait_objective=False,
             coupled_sustained_gait_objective=False,
             record_trace=False, height_gate=False,
             shin_force_gate=False, shin_leg_gate=False, contact_guard=False,
             drive_controller=None, observation_augment=None, world_features=None,
             audit_predictors=None, transition_records=None, reference_policy=None,
             recorded_positions=None):
    if sum((height_gate, shin_force_gate, shin_leg_gate)) > 1:
        raise ValueError("Select only one oscillator amplitude gate")
    if sustained_gait_objective and coupled_sustained_gait_objective:
        raise ValueError("Select only one sustained gait objective")
    if drive_controller is not None and (contact_guard or height_gate or
                                         shin_force_gate or shin_leg_gate):
        raise ValueError("Stateful drive controller cannot use CPG gates")
    env.reset(positions)
    if drive_controller is not None:
        drive_controller.reset()
    measure = env.measure()
    latent = (torch.zeros((env.worlds, world_features.feature_size), device=env.device)
              if world_features is not None else None)
    tactile_error = (torch.zeros(env.worlds, device=env.device)
                     if world_features is not None and hasattr(world_features, "predict") and
                     getattr(world_features, "prediction_horizon_control_steps", 1) == 1 else None)
    if any(getattr(predictor.encoder if hasattr(predictor, "encoder") else predictor,
                   "prediction_horizon_control_steps", 1) != 1
           for predictor in (audit_predictors or {}).values()):
        raise ValueError("One-step tactile audit cannot score multistep predictors")
    persistence_error = torch.zeros_like(tactile_error) if tactile_error is not None else None
    audit_errors = {name: torch.zeros(env.worlds, device=env.device)
                    for name in (audit_predictors or {})}
    audit_persistence_error = (torch.zeros(env.worlds, device=env.device)
                               if audit_errors else None)
    audit_transition_errors = {name: torch.zeros(env.worlds, device=env.device)
                               for name in audit_errors}
    audit_transition_count = (torch.zeros(env.worlds, device=env.device)
                              if audit_errors else None)
    elevated_steps = torch.zeros(env.worlds, device=env.device)
    supported_steps = torch.zeros_like(elevated_steps)
    final_quarter_elevated_steps = torch.zeros_like(elevated_steps)
    final_quarter_supported_steps = torch.zeros_like(elevated_steps)
    four_steps = torch.zeros_like(elevated_steps)
    pelvis_steps = torch.zeros_like(elevated_steps)
    hand_switches = torch.zeros_like(elevated_steps)
    shin_switches = torch.zeros_like(elevated_steps)
    guarded_arm_steps = torch.zeros_like(elevated_steps)
    last_hand = torch.zeros(env.worlds, device=env.device, dtype=torch.int32)
    last_shin = torch.zeros_like(last_hand)
    trace = {name: [] for name in ("chest_height_m", "head_height_m", "forward_m",
                                   "palm_shin_force_n", "pelvis_force_n",
                                   "oscillator_gain")} if record_trace else None
    interval = physics_per_control * float(env.source.opt.timestep)
    half_step = max(0, control_steps // 2 - 1)
    final_quarter_start = control_steps - max(1, control_steps // 4)
    half_forward = None
    with torch.no_grad():
        for step in range(control_steps):
            observation = observe(env, measure, use_touch, latent_features=latent)
            if observation_augment is not None:
                observation = observation_augment(observation, drive_controller)
            base = policy.actor(observation).clamp(-1, 1)
            if drive_controller is not None:
                reference_drives = (reference_policy.actor(observation).clamp(-1, 1)
                                    if reference_policy is not None else None)
                drives = (drive_controller(base, measure, reference_drives)
                          if reference_drives is not None else drive_controller(base, measure))
            else:
                drives = cpg_drives(base, parameters, step * interval)
            if contact_guard:
                drives, allowed_arms = guard_arm_oscillation(
                    base, drives, measure["palm_shin_force"], parameters.shape[1] >= 8)
                guarded_arm_steps += (~allowed_arms).float().mean(-1)
            if height_gate:
                gain = height_gait_gain(measure["chest_height"])
            elif shin_force_gate or shin_leg_gate:
                gain = shin_gait_gain(measure["palm_shin_force"])
            else:
                gain = torch.ones_like(measure["chest_height"])
            drives = apply_gait_gain(base, drives, gain, leg_only=shin_leg_gate)
            muscle_action = env.action_from_drives(drives)
            if transition_records is not None:
                body_weight = float(env.source.body_mass.sum()) * 9.81
                transition_records.append({
                    "qpos_before": env.position.detach().cpu().numpy().copy(),
                    "current": torch.cat((env.velocity,
                                           virtual_lengths(env.muscles, env.position),
                                           env.muscles.activity), dim=1).detach().cpu().numpy().copy(),
                    "tactile": torch.log1p(measure["touch"].clamp_min(0) /
                                            body_weight).detach().cpu().numpy().copy(),
                    "action": muscle_action.detach().cpu().numpy().copy(),
                })
            if world_features is not None:
                latent = world_features.encode(env, measure, muscle_action)
            if tactile_error is not None:
                predicted_contact = world_features.predict(env, measure, muscle_action)
                current_contact = (torch.log1p(measure["touch"].clamp_min(0) /
                                                  world_features.body_weight)
                                   [:, world_features.active_regions] > 1e-4).float()
            if audit_errors:
                audit_predictions = {name: predictor.predict(env, measure, muscle_action)
                                     for name, predictor in audit_predictors.items()}
                audit_reference = next(iter(audit_predictors.values()))
                audit_current_contact = (torch.log1p(measure["touch"].clamp_min(0) /
                                                        audit_reference.body_weight)
                                         [:, audit_reference.active_regions] > 1e-4).float()
            measure = env.step(muscle_action, physics_per_control)
            if recorded_positions is not None:
                recorded_positions.append(env.position.detach().cpu().numpy().copy())
            if transition_records is not None:
                transition_records[-1]["next_tactile"] = torch.log1p(
                    measure["touch"].clamp_min(0) /
                    body_weight).detach().cpu().numpy().copy()
            if tactile_error is not None:
                next_contact = (torch.log1p(measure["touch"].clamp_min(0) /
                                               world_features.body_weight)
                                [:, world_features.active_regions] > 1e-4).float()
                tactile_error += (predicted_contact - next_contact).square().mean(-1)
                persistence_error += (current_contact - next_contact).square().mean(-1)
            if audit_errors:
                audit_next_contact = (torch.log1p(measure["touch"].clamp_min(0) /
                                                     audit_reference.body_weight)
                                      [:, audit_reference.active_regions] > 1e-4).float()
                transitions = audit_next_contact != audit_current_contact
                audit_transition_count += transitions.sum(-1)
                for name, prediction in audit_predictions.items():
                    squared_error = (prediction - audit_next_contact).square()
                    audit_errors[name] += squared_error.mean(-1)
                    audit_transition_errors[name] += (squared_error * transitions).sum(-1)
                audit_persistence_error += (audit_current_contact -
                                            audit_next_contact).square().mean(-1)
            contact = measure["palm_shin_force"] > 2
            elevated = ((measure["chest_height"] > 0.16) &
                        (measure["head_height"] > 0.14))
            minimum_support = contact[:, :2].any(-1) & contact[:, 2:].any(-1)
            pelvis = measure["ground_touch"][:, REGIONS.index("pelvis")] > 2
            if record_trace:
                trace["oscillator_gain"].append(gain.clone())
                trace["chest_height_m"].append(measure["chest_height"].clone())
                trace["head_height_m"].append(measure["head_height"].clone())
                trace["forward_m"].append(measure["forward_displacement"].clone())
                trace["palm_shin_force_n"].append(measure["palm_shin_force"].clone())
                trace["pelvis_force_n"].append(
                    measure["ground_touch"][:, REGIONS.index("pelvis")].clone())
            hand_state = exclusive_contact_state(measure["palm_shin_force"][:, :2])
            shin_state = exclusive_contact_state(measure["palm_shin_force"][:, 2:])
            hand_switches += ((hand_state != 0) & (last_hand != 0) &
                              (hand_state != last_hand) & elevated).float()
            shin_switches += ((shin_state != 0) & (last_shin != 0) &
                              (shin_state != last_shin) & elevated).float()
            last_hand = torch.where(hand_state != 0, hand_state, last_hand)
            last_shin = torch.where(shin_state != 0, shin_state, last_shin)
            elevated_steps += elevated.float()
            supported_steps += (elevated & minimum_support).float()
            if step >= final_quarter_start:
                final_quarter_elevated_steps += elevated.float()
                final_quarter_supported_steps += (elevated & minimum_support).float()
            four_steps += (elevated & contact.all(-1)).float()
            pelvis_steps += pelvis.float()
            if step == half_step:
                half_forward = measure["forward_displacement"].clone()
    forward = measure["forward_displacement"]
    elevated_fraction = elevated_steps / control_steps
    support_fraction = supported_steps / control_steps
    final_quarter_steps = control_steps - final_quarter_start
    final_quarter_elevated_fraction = final_quarter_elevated_steps / final_quarter_steps
    final_quarter_support_fraction = final_quarter_supported_steps / final_quarter_steps
    four_fraction = four_steps / control_steps
    pelvis_fraction = pelvis_steps / control_steps
    amplitudes = (drive_controller.parameters if drive_controller is not None else
                  torch.cat((parameters[:, 1:5], parameters[:, 6:8]), dim=1))
    legacy_crawl_success, crawl_success = crawl_success_gate(
        forward, elevated_fraction, support_fraction, final_quarter_support_fraction,
        pelvis_fraction, hand_switches, shin_switches)
    if gait_objective:
        score = gait_objective_score(forward, support_fraction, elevated_fraction,
                                     four_fraction, pelvis_fraction, hand_switches,
                                     shin_switches, crawl_success, amplitudes)
        if sustained_gait_objective:
            score += sustained_gait_bonus(forward - half_forward,
                                          final_quarter_support_fraction,
                                          final_quarter_elevated_fraction)
        if coupled_sustained_gait_objective:
            score += coupled_sustained_gait_bonus(forward - half_forward,
                                                  final_quarter_support_fraction,
                                                  final_quarter_elevated_fraction,
                                                  hand_switches)
    else:
        score = (5 * forward + support_fraction + 0.2 * four_fraction +
                 0.2 * hand_switches.clamp(max=3) + 0.1 * shin_switches.clamp(max=3) -
                 0.5 * pelvis_fraction - 0.01 * amplitudes.square().mean(-1))
    result = {"score": score, "forward_displacement_m": forward,
              "late_forward_progress_m": forward - half_forward,
              "elevated_fraction": elevated_fraction,
              "minimum_support_fraction": support_fraction,
              "final_quarter_elevated_fraction": final_quarter_elevated_fraction,
              "final_quarter_support_fraction": final_quarter_support_fraction,
              "four_support_fraction": four_fraction,
              "guarded_arm_fraction": guarded_arm_steps / control_steps,
              "pelvis_fraction": pelvis_fraction,
              "hand_switches": hand_switches, "shin_switches": shin_switches,
              "legacy_crawl_success": legacy_crawl_success,
              "crawl_success": crawl_success}
    if record_trace:
        result.update({"trace_" + name: torch.stack(values) for name, values in trace.items()})
    if drive_controller is not None:
        result.update(drive_controller.metrics(control_steps))
    if tactile_error is not None:
        result["tactile_next_contact_brier"] = tactile_error / control_steps
        result["tactile_persistence_brier"] = persistence_error / control_steps
    if audit_errors:
        for name, error in audit_errors.items():
            result[f"audit_tactile_brier_{name}"] = error / control_steps
            result[f"audit_tactile_transition_brier_{name}"] = (
                audit_transition_errors[name] / audit_transition_count.clamp_min(1))
        result["audit_tactile_persistence_brier"] = audit_persistence_error / control_steps
        result["audit_tactile_transition_count"] = audit_transition_count
        result["audit_tactile_transition_fraction"] = (
            audit_transition_count /
            (control_steps * len(audit_reference.active_regions)))
    return result


def summarize(result):
    return {key: value.detach().cpu().tolist() for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--control-steps", type=int, default=40)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--wrist-oscillation", action="store_true")
    parser.add_argument("--shoulder-lift-oscillation", action="store_true")
    parser.add_argument("--shoulder-lift-phase-oscillation", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--gait-objective", action="store_true")
    parser.add_argument("--sustained-gait-objective", action="store_true")
    parser.add_argument("--coupled-sustained-gait-objective", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        parser.error("Output already exists")
    if min(args.population, args.generations, args.control_steps, args.physics_per_control) < 1:
        parser.error("Training dimensions must be positive")
    if args.shoulder_lift_oscillation and not args.wrist_oscillation:
        parser.error("Shoulder lift requires the wrist-oscillation parameter set")
    if args.shoulder_lift_phase_oscillation and not args.shoulder_lift_oscillation:
        parser.error("Shoulder-lift phase requires the shoulder-lift parameter set")
    if args.sustained_gait_objective and not args.gait_objective:
        parser.error("Sustained gait objective requires the gait objective")
    if args.coupled_sustained_gait_objective and not args.gait_objective:
        parser.error("Coupled sustained gait objective requires the gait objective")
    if args.sustained_gait_objective and args.coupled_sustained_gait_objective:
        parser.error("Select only one sustained gait objective")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("world_features") is not None):
        raise ValueError("CPG requires a compatible palm-control support checkpoint")
    torch.manual_seed(args.seed)
    env = InfantCrawlEnv(args.scene, args.pose, args.population,
                         controlled_joint_names=PALM_JOINTS)
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to(env.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    mean = torch.tensor([1.5, 0., 0., 0., 0., 0.], device=env.device)
    deviation = torch.tensor([0.5, 0.2, 0.2, 0.2, 0.2, 1.0], device=env.device)
    lower = torch.tensor([0.5, -0.6, -0.6, -0.6, -0.6, -math.pi], device=env.device)
    upper = torch.tensor([3.0, 0.6, 0.6, 0.6, 0.6, math.pi], device=env.device)
    if args.wrist_oscillation:
        mean = torch.cat((mean, torch.zeros(1, device=env.device)))
        deviation = torch.cat((deviation, torch.full((1,), 0.2, device=env.device)))
        lower = torch.cat((lower, torch.full((1,), -0.6, device=env.device)))
        upper = torch.cat((upper, torch.full((1,), 0.6, device=env.device)))
    if args.shoulder_lift_oscillation:
        mean = torch.cat((mean, torch.zeros(1, device=env.device)))
        deviation = torch.cat((deviation, torch.full((1,), 0.2, device=env.device)))
        lower = torch.cat((lower, torch.zeros(1, device=env.device)))
        upper = torch.cat((upper, torch.full((1,), 0.8, device=env.device)))
    if args.shoulder_lift_phase_oscillation:
        mean = torch.cat((mean, torch.full((1,), math.pi, device=env.device)))
        deviation = torch.cat((deviation, torch.full((1,), 0.5, device=env.device)))
        lower = torch.cat((lower, torch.full((1,), -math.pi, device=env.device)))
        upper = torch.cat((upper, torch.full((1,), math.pi, device=env.device)))
    if args.gait_objective:
        lower[1], upper[1] = -1.0, 1.0
    if args.initialize_from is not None:
        with np.load(args.initialize_from) as previous:
            prior = previous["parameters"].copy()
        if args.wrist_oscillation and prior.shape == (6,):
            prior = np.r_[prior, 0.]
        if args.shoulder_lift_oscillation and prior.shape == (7,):
            prior = np.r_[prior, 0.]
        if args.shoulder_lift_phase_oscillation and prior.shape == (8,):
            prior = np.r_[prior, math.pi]
        if prior.shape != (mean.numel(),):
            raise ValueError("CPG initial parameters do not match the selected oscillator")
        mean = torch.as_tensor(prior, device=env.device, dtype=torch.float32).clamp(lower, upper)
    baseline_parameters = mean.clone()
    candidate_pool = []
    history = []
    for generation in range(args.generations):
        candidates = (mean + torch.randn((args.population, mean.numel()), device=env.device) *
                      deviation).clamp(lower, upper)
        candidates[0] = mean.clamp(lower, upper)
        candidates[1] = baseline_parameters
        positions = jittered_positions(env, 0.02, 0.005)
        result = evaluate(env, policy, candidates, positions, args.control_steps,
                          args.physics_per_control, checkpoint["use_touch"], args.gait_objective,
                          args.sustained_gait_objective, args.coupled_sustained_gait_objective)
        score = result["score"]
        elite = candidates[torch.topk(score, max(4, args.population // 4)).indices]
        mean = 0.6 * mean + 0.4 * elite.mean(0)
        deviation = (0.6 * deviation + 0.4 * elite.std(0, unbiased=False)).clamp(0.05, 1.0)
        winner = int(score.argmax())
        candidate_pool.append((float(score[winner]), candidates[winner].clone()))
        history.append({"generation": generation + 1,
                        "best_score": float(score[winner]),
                        "winner_forward_m": float(result["forward_displacement_m"][winner]),
                        "winner_hand_switches": float(result["hand_switches"][winner]),
                        "winner_crawl_success": bool(result["crawl_success"][winner])})
        print(json.dumps(history[-1]), flush=True)
    validation = InfantCrawlEnv(args.scene, args.pose, 8,
                                controlled_joint_names=PALM_JOINTS)
    torch.manual_seed(17)
    validation_positions = jittered_positions(validation, 0.05, 0.01)
    contenders = sorted(candidate_pool, key=lambda item: item[0], reverse=True)[:10]
    contenders += [(None, mean.clamp(lower, upper)), (None, baseline_parameters)]
    validation_records = []
    selected = None
    selected_score = -float("inf")
    for training_score, parameters in contenders:
        result = evaluate(validation, policy, parameters.expand(8, -1), validation_positions,
                          args.control_steps, args.physics_per_control, checkpoint["use_touch"],
                          args.gait_objective, args.sustained_gait_objective,
                          args.coupled_sustained_gait_objective)
        value = float(result["score"].mean())
        validation_records.append({"training_score": training_score,
                                   "score": value,
                                   "forward_m": float(result["forward_displacement_m"].mean()),
                                   "crawl_success_fraction": float(result["crawl_success"].float().mean())})
        if value > selected_score:
            selected_score = value
            selected = parameters.clone()
    heldout = []
    for seed in (41, 43, 47):
        torch.manual_seed(seed)
        positions = jittered_positions(validation, 0.05, 0.01)
        baseline = evaluate(validation, policy, baseline_parameters.expand(8, -1), positions,
                            args.control_steps, args.physics_per_control, checkpoint["use_touch"],
                            args.gait_objective, args.sustained_gait_objective,
                            args.coupled_sustained_gait_objective)
        learned = evaluate(validation, policy, selected.expand(8, -1), positions,
                           args.control_steps, args.physics_per_control, checkpoint["use_touch"],
                           args.gait_objective, args.sustained_gait_objective,
                           args.coupled_sustained_gait_objective)
        heldout.append({"seed": seed, "baseline": summarize(baseline),
                        "learned": summarize(learned)})
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
              "population": args.population, "generations": args.generations,
              "duration_s": args.control_steps * args.physics_per_control *
                            float(env.source.opt.timestep),
              "parameter_order": ["frequency_hz", "shoulder_amplitude", "elbow_amplitude",
                                  "hip_amplitude", "knee_amplitude", "leg_phase_rad"] +
                                 (["wrist_pitch_amplitude"] if args.wrist_oscillation else []) +
                                 (["alternating_shoulder_lift_amplitude"]
                                  if args.shoulder_lift_oscillation else []) +
                                 (["shoulder_lift_phase_offset_rad"]
                                  if args.shoulder_lift_phase_oscillation else []),
              "initialize_from": str(args.initialize_from.resolve())
              if args.initialize_from else None,
              "gait_objective": args.gait_objective,
              "sustained_gait_objective": args.sustained_gait_objective,
              "coupled_sustained_gait_objective": args.coupled_sustained_gait_objective,
              "baseline_parameters": baseline_parameters.cpu().tolist(),
              "selected_parameters": selected.cpu().tolist(),
              "selected_validation_score": selected_score,
              "validation_candidates": validation_records,
              "history": history, "heldout": heldout,
              "scope": "Diagonal-limb CPG around support PPO; crawl gait not assumed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), parameters=selected.cpu().numpy())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
