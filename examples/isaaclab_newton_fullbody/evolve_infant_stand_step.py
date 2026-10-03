"""Evolve tactile-triggered weight shift and right-leg swing on infant standing."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import torch
import warp as wp

from evolve_infant_stand_cop_reflex import aligned_positions
from infant_crawl_env import CRAWL_JOINTS, InfantCrawlEnv
from sweep_infant_cop_reflex import cop_reflex_drives
from train_infant_stand_ppo import load_bias


JOINTS = tuple(CRAWL_JOINTS.index(f"robot:{name}") for name in (
    "left_hip2", "left_foot2", "right_hip2", "right_hip1", "right_knee",
    "right_foot1", "left_hip1", "left_foot1"))
PARAMETERS = ("left_hip_roll", "left_ankle_roll", "right_hip_roll",
              "right_hip_flex", "right_knee_flex", "right_ankle_pitch",
              "left_hip_pitch", "left_ankle_pitch")
LOWER = (-1., -1., -1., 0., 0., -1., -1., -1.)
UPPER = (1., 1., 1., 2., 2., 1., 1., 1.)


def phase_window(elapsed, start, end):
    phase = ((elapsed - start) / (end - start)).clamp(0, 1)
    return torch.sin(torch.pi * phase)


def step_drives(base, parameters, shift, swing, stance_preload=False):
    if parameters.shape != (base.shape[0], len(PARAMETERS)):
        raise ValueError("Step parameters must match the world batch")
    result = base.clone()
    for index, joint in enumerate(JOINTS):
        envelope = (shift if index < 3 else
                    torch.maximum(shift, swing) if stance_preload and index >= 6 else
                    swing)
        direction = -1 if index in (3, 4) else 1
        result[:, joint] += direction * parameters[:, index] * envelope
    return result.clamp(-1, 1)


def selective_swing_action(env, drives, swing, excitation):
    if not 0 <= excitation <= 1:
        raise ValueError("Swing excitation must be in [0, 1]")
    action = env.action_from_drives(drives, baseline=0.4, amplitude=0.3)
    if excitation == 0:
        return action
    blend = excitation * swing
    baseline = 0.4 + 0.1 * blend[:, None]
    amplitude = 0.3 + 0.2 * blend[:, None]
    joint_ids = list(JOINTS[3:6])
    actuator_ids = env.crawl_actuators[joint_ids]
    selected = drives[:, joint_ids]
    action[:, actuator_ids] = (baseline - amplitude * selected).clamp(0, 1)
    action[:, actuator_ids + env.source.nu] = (baseline + amplitude * selected).clamp(0, 1)
    return action


def landing_reflex_drives(base, parameters, elapsed, duration):
    if parameters.shape != (base.shape[0], 2):
        raise ValueError("Landing reflex parameters must match the world batch")
    if duration <= 0:
        raise ValueError("Landing reflex duration must be positive")
    active = ((elapsed >= 0) & (elapsed < duration)).to(base.dtype)
    result = base.clone()
    result[:, [JOINTS[3], JOINTS[6]]] += parameters[:, :1] * active[:, None]
    result[:, [JOINTS[5], JOINTS[7]]] += parameters[:, 1:] * active[:, None]
    return result.clamp(-1, 1)


def step_events(qualified, contact_streak, completed, active, left_support,
                airborne, advance, rise):
    qualified = (qualified | (active & left_support & airborne &
                              (advance > 0.03) & (rise > 0.02)))
    landing = qualified & active & left_support & ~airborne & (advance > 0.03)
    contact_streak = torch.where(landing, contact_streak + 1, 0)
    completed = completed | (contact_streak >= 10)
    return qualified, contact_streak, completed


def sustained_gait_success(gait_precursor, longest_support_steps, timestep,
                           minimum_duration=0.5):
    if gait_precursor.shape != longest_support_steps.shape:
        raise ValueError("Gait and support events must match by world")
    return gait_precursor & (longest_support_steps * timestep >= minimum_duration)


def evaluate(env, positions, bias, gains, parameters, steps, foot_body_id,
             shift_end=0.4, swing_start=0.2, swing_end=0.95, trace_every=0,
             stance_preload=False, swing_excitation=0.0, fitness_mode="world",
             landing_parameters=None, landing_duration=0.5):
    if not (0 < shift_end and 0 <= swing_start < swing_end):
        raise ValueError("Invalid step timing")
    if trace_every < 0:
        raise ValueError("Trace interval must be nonnegative")
    if not 0 <= swing_excitation <= 1:
        raise ValueError("Swing excitation must be in [0, 1]")
    if fitness_mode not in ("world", "gait"):
        raise ValueError("Unknown step fitness mode")
    if landing_duration <= 0:
        raise ValueError("Landing reflex duration must be positive")
    if landing_parameters is not None and landing_parameters.shape != (env.worlds, 2):
        raise ValueError("Landing reflex parameters must match the world batch")
    env.reset(positions)
    measure = env.measure()
    initial_foot = wp.to_torch(env.data.xpos)[:, foot_body_id, :].clone()
    initial_root = env.position[:, env.root_address].clone()
    onset = torch.full((env.worlds,), float("inf"), device=env.device)
    landing_onset = torch.full_like(onset, float("inf"))
    liftoff = torch.zeros(env.worlds, dtype=torch.bool, device=env.device)
    relative_qualified = torch.zeros_like(liftoff)
    relative_completed = torch.zeros_like(liftoff)
    world_qualified = torch.zeros_like(liftoff)
    world_completed = torch.zeros_like(liftoff)
    root_forward_step = torch.zeros_like(liftoff)
    combined_qualified = torch.zeros_like(liftoff)
    combined_completed = torch.zeros_like(liftoff)
    gait_precursor = torch.zeros_like(liftoff)
    relative_contact_streak = torch.zeros(env.worlds, dtype=torch.int32,
                                          device=env.device)
    world_contact_streak = torch.zeros_like(relative_contact_streak)
    combined_contact_streak = torch.zeros_like(relative_contact_streak)
    standing = torch.zeros(env.worlds, device=env.device)
    late = torch.zeros_like(standing)
    post_step_support_streak = torch.zeros(env.worlds, dtype=torch.int32,
                                           device=env.device)
    longest_post_step_support = torch.zeros_like(post_step_support_streak)
    best_supported_airborne_advance = torch.zeros_like(standing)
    best_supported_world_advance = torch.zeros_like(standing)
    best_supported_root_advance = torch.zeros_like(standing)
    best_supported_joint_advance = torch.zeros_like(standing)
    best_supported_rise = torch.zeros_like(standing)
    trace = []
    dt = float(env.source.opt.timestep)
    late_start = steps - max(1, steps // 4)
    for step in range(steps):
        time_s = step * dt
        both_feet = (measure["foot_force"] > 2).all(-1)
        ready = (time_s >= 0.1) & both_feet & (measure["chest_height"] > 0.45)
        onset = torch.where(torch.isinf(onset) & ready, time_s, onset)
        elapsed = time_s - onset
        shift = phase_window(elapsed, 0., shift_end)
        swing = phase_window(elapsed, swing_start, swing_end)
        drives = env.feedback_drives(bias.expand(env.worlds, -1), 1.5, 1.0)
        drives = cop_reflex_drives(env, measure, drives, gains)
        drives = step_drives(drives, parameters, shift, swing, stance_preload)
        if landing_parameters is not None:
            drives = landing_reflex_drives(drives, landing_parameters,
                                           time_s - landing_onset, landing_duration)
        previous_chest_height = measure["chest_height"]
        measure = env.step(selective_swing_action(env, drives, swing,
                                                  swing_excitation), 1)
        chest_velocity = (measure["chest_height"] - previous_chest_height) / dt
        foot = wp.to_torch(env.data.xpos)[:, foot_body_id, :]
        world_advance = foot[:, 0] - initial_foot[:, 0]
        root_advance = env.position[:, env.root_address] - initial_root
        advance = world_advance - root_advance
        rise = foot[:, 2] - initial_foot[:, 2]
        left_support = ((measure["chest_height"] > 0.45) &
                        (measure["foot_force"][:, 1] > 2))
        airborne = measure["foot_force"][:, 0] <= 2
        active = (elapsed >= swing_start) & (elapsed <= swing_end + 0.25)
        supported_airborne = active & left_support & airborne
        liftoff |= supported_airborne
        best_supported_airborne_advance = torch.where(
            supported_airborne,
            torch.maximum(best_supported_airborne_advance, advance.clamp_min(0)),
            best_supported_airborne_advance)
        best_supported_world_advance = torch.where(
            supported_airborne,
            torch.maximum(best_supported_world_advance, world_advance.clamp_min(0)),
            best_supported_world_advance)
        joint_advance = torch.minimum(torch.minimum(advance, world_advance), root_advance)
        best_supported_joint_advance = torch.where(
            supported_airborne,
            torch.maximum(best_supported_joint_advance, joint_advance.clamp_min(0)),
            best_supported_joint_advance)
        best_supported_rise = torch.where(
            supported_airborne, torch.maximum(best_supported_rise, rise.clamp_min(0)),
            best_supported_rise)
        relative_qualified, relative_contact_streak, relative_completed = step_events(
            relative_qualified, relative_contact_streak, relative_completed,
            active, left_support,
            airborne, advance, rise)
        previous_world_completed = world_completed
        world_qualified, world_contact_streak, world_completed = step_events(
            world_qualified, world_contact_streak, world_completed,
            active, left_support, airborne, world_advance, rise)
        root_forward_step |= (world_completed & ~previous_world_completed &
                              (root_advance > 0.01))
        previous_combined_completed = combined_completed
        combined_qualified, combined_contact_streak, combined_completed = step_events(
            combined_qualified, combined_contact_streak, combined_completed,
            active, left_support, airborne,
            torch.minimum(advance, world_advance), rise)
        gait_precursor |= (combined_completed & ~previous_combined_completed &
                           (root_advance > 0.01))
        landing_ready = (combined_qualified & left_support & ~airborne &
                         (advance > 0.03) & (world_advance > 0.03))
        landing_onset = torch.where(torch.isinf(landing_onset) & landing_ready,
                                     (step + 1) * dt, landing_onset)
        standing_now = left_support & ~airborne
        standing += standing_now.float()
        best_supported_root_advance = torch.where(
            standing_now,
            torch.maximum(best_supported_root_advance, root_advance.clamp_min(0)),
            best_supported_root_advance)
        post_step_support_streak = torch.where(
            world_completed & standing_now, post_step_support_streak + 1, 0)
        longest_post_step_support = torch.maximum(longest_post_step_support,
                                                   post_step_support_streak)
        if step >= late_start:
            late += standing_now.float()
        if trace_every and step % trace_every == 0:
            trace.append({
                "time_s": round((step + 1) * dt, 4),
                "chest_height_m": measure["chest_height"].cpu().tolist(),
                "chest_vertical_velocity_mps": chest_velocity.cpu().tolist(),
                "foot_force_N": measure["foot_force"].cpu().tolist(),
                "foot_relative_advance_m": advance.cpu().tolist(),
                "foot_world_advance_m": world_advance.cpu().tolist(),
                "root_world_advance_m": root_advance.cpu().tolist(),
                "foot_rise_m": rise.cpu().tolist(),
                "root_velocity_xy_mps": env.velocity[:, env.root_dof_address:
                                                        env.root_dof_address + 2].cpu().tolist(),
                "relative_qualified_swing": relative_qualified.cpu().tolist(),
                "relative_placement": relative_completed.cpu().tolist(),
                "world_qualified_swing": world_qualified.cpu().tolist(),
                "world_forward_step": world_completed.cpu().tolist(),
                "combined_step": combined_completed.cpu().tolist(),
            })
    standing_fraction = standing / steps
    late_fraction = late / (steps - late_start)
    sustained_gait_precursor = sustained_gait_success(
        gait_precursor, longest_post_step_support, dt)
    if fitness_mode == "gait":
        score = (standing_fraction + 0.5 * late_fraction +
                 0.5 * world_completed.float() + 2. * combined_qualified.float() +
                 6. * combined_completed.float() + 12. * gait_precursor.float() +
                 2. * (longest_post_step_support * dt >= 0.5).float() +
                 10. * best_supported_joint_advance.clamp(max=0.05))
    else:
        score = (standing_fraction + 0.5 * late_fraction +
                 0.2 * relative_qualified.float() + world_qualified.float() +
                 5. * world_completed.float() + 2. * root_forward_step.float() +
                 5. * combined_completed.float() + 2. * gait_precursor.float() +
                 2. * best_supported_world_advance.clamp(max=0.15) +
                 best_supported_root_advance.clamp(max=0.15))
    return {"score": score, "standing_fraction": standing_fraction,
            "late_standing_fraction": late_fraction,
            "supported_liftoff": liftoff,
            "relative_qualified_swing": relative_qualified,
            "relative_placement": relative_completed,
            "world_qualified_swing": world_qualified,
            "world_forward_step": world_completed,
            "root_forward_step": root_forward_step,
            "combined_qualified_swing": combined_qualified,
            "combined_step": combined_completed,
            "gait_precursor": gait_precursor,
            "sustained_gait_precursor_0p5s": sustained_gait_precursor,
            "post_step_support_0p5s": longest_post_step_support * dt >= 0.5,
            "longest_post_step_support_s": longest_post_step_support * dt,
            "max_supported_relative_advance_m": best_supported_airborne_advance,
            "max_supported_world_advance_m": best_supported_world_advance,
            "max_supported_root_advance_m": best_supported_root_advance,
            "max_supported_joint_advance_m": best_supported_joint_advance,
            "max_supported_foot_rise_m": best_supported_rise,
            "final_root_advance_m": env.position[:, env.root_address] - initial_root,
            "trace": trace,
            "solver_limit_steps": getattr(env, "solver_limit_steps", None)}


def summarize(result):
    return {name: (int(value.sum()) if value.dtype == torch.bool else float(value.mean()))
            for name, value in result.items() if isinstance(value, torch.Tensor)} | {
                "solver_limit_steps": result["solver_limit_steps"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--evolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--worlds-per-candidate", type=int, default=2)
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--shift-end", type=float, default=0.4)
    parser.add_argument("--swing-start", type=float, default=0.2)
    parser.add_argument("--swing-end", type=float, default=0.95)
    parser.add_argument("--stance-preload", action="store_true")
    parser.add_argument("--swing-excitation", type=float, default=0.0)
    parser.add_argument("--fitness-mode", choices=("world", "gait"), default="world")
    parser.add_argument("--seed-candidate", nargs=len(PARAMETERS), type=float,
                        default=[1., 1., -1., 2., 2., 0.5, 0., 0.])
    parser.add_argument("--seed", type=int, default=7601)
    parser.add_argument("--selection-seed", type=int, default=7651)
    parser.add_argument("--heldout-seeds", nargs="+", type=int,
                        default=[7701, 7703, 7707])
    parser.add_argument("--heldout-worlds", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    if min(args.generations, args.population, args.worlds_per_candidate,
           args.control_steps, args.heldout_worlds) < 1 or not args.heldout_seeds:
        parser.error("Positive evolution and evaluation sizes are required")
    if not (0 < args.shift_end and 0 <= args.swing_start < args.swing_end):
        parser.error("Invalid weight-shift or swing timing")
    if not 0 <= args.swing_excitation <= 1:
        parser.error("Swing excitation must be in [0, 1]")
    training_seeds = {args.seed + generation * 2 + offset
                      for generation in range(args.generations) for offset in range(2)}
    if (training_seeds.intersection(args.heldout_seeds) or
            args.selection_seed in training_seeds or
            args.selection_seed in args.heldout_seeds):
        parser.error("Training, selection, and held-out seeds must differ")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    env = InfantCrawlEnv(args.scene, args.pose,
                         args.population * args.worlds_per_candidate)
    foot_body_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    bias = load_bias(args.bias, env.device)
    gain_values = json.loads(args.evolution.read_text())["selected"]["gains"]
    gains = torch.tensor(gain_values, device=env.device).expand(env.worlds, -1)
    lower = torch.tensor(LOWER, device=env.device)
    upper = torch.tensor(UPPER, device=env.device)
    seed_candidate = torch.tensor(args.seed_candidate, device=env.device)
    if not torch.all((seed_candidate >= lower) & (seed_candidate <= upper)):
        parser.error("Seed candidate must lie within the search bounds")
    mean = seed_candidate.clone()
    deviation = torch.tensor([0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.5], device=env.device)
    history = []
    pool = []
    for generation in range(args.generations):
        torch.manual_seed(args.seed + generation)
        candidates = (mean + torch.randn((args.population, len(PARAMETERS)),
                                          device=env.device) * deviation).clamp(lower, upper)
        candidates[0] = mean
        candidates[1] = 0
        if args.population > 2:
            candidates[2] = seed_candidate
        expanded = candidates.repeat_interleave(args.worlds_per_candidate, dim=0)
        scores = torch.zeros(args.population, device=env.device)
        plants = torch.zeros_like(scores)
        supports = torch.zeros_like(scores)
        for offset in range(2):
            seed = args.seed + generation * 2 + offset
            positions = aligned_positions(env, seed, args.worlds_per_candidate)
            result = evaluate(env, positions.repeat(args.population, 1), bias, gains,
                              expanded, args.control_steps, foot_body_id,
                              args.shift_end, args.swing_start, args.swing_end,
                              stance_preload=args.stance_preload,
                              swing_excitation=args.swing_excitation,
                              fitness_mode=args.fitness_mode)
            scores += result["score"].reshape(args.population, -1).mean(-1) / 2
            plants += result["gait_precursor"].float().reshape(
                args.population, -1).mean(-1) / 2
            supports += result["standing_fraction"].reshape(
                args.population, -1).mean(-1) / 2
        elite = candidates[torch.topk(scores, max(2, args.population // 4)).indices]
        mean = (0.4 * mean + 0.6 * elite.mean(0)).clamp(lower, upper)
        deviation = (0.75 * deviation + 0.25 * elite.std(0).nan_to_num(0)).clamp_min(0.08)
        best = int(scores.argmax())
        row = {"generation": generation + 1, "best_score": float(scores[best]),
               "best_gait_precursor_fraction": float(plants[best]),
               "best_standing_fraction": float(supports[best]),
               "best_parameters": candidates[best].cpu().tolist(),
               "mean_parameters": mean.cpu().tolist()}
        history.append(row)
        pool.extend((float(scores[index]), candidates[index].clone())
                    for index in range(args.population))
        print(json.dumps(row), flush=True)
    pool.sort(key=lambda item: item[0], reverse=True)
    validation = InfantCrawlEnv(args.scene, args.pose, args.heldout_worlds)
    validation_gains = torch.tensor(gain_values, device=validation.device).expand(
        validation.worlds, -1)
    selection_positions = aligned_positions(validation, args.selection_seed,
                                            validation.worlds)
    selection_rows = []
    for training_score, candidate in pool[:min(8, len(pool))]:
        result = evaluate(validation, selection_positions, bias, validation_gains,
                          candidate.expand(validation.worlds, -1),
                          args.control_steps, foot_body_id,
                          args.shift_end, args.swing_start, args.swing_end,
                          stance_preload=args.stance_preload,
                          swing_excitation=args.swing_excitation,
                          fitness_mode=args.fitness_mode)
        selection_rows.append({"training_score": training_score,
                               "parameters": candidate.cpu().tolist(),
                               **summarize(result)})
    selected_row = max(selection_rows, key=lambda row: row["score"])
    selected = torch.tensor(selected_row["parameters"], device=env.device)
    records = []
    for seed in args.heldout_seeds:
        positions = aligned_positions(validation, seed, validation.worlds)
        pose_hash = hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest()
        for label, parameters in (("control", torch.zeros_like(selected)),
                                  ("selected", selected)):
            result = evaluate(validation, positions, bias, validation_gains,
                              parameters.expand(validation.worlds, -1),
                              args.control_steps, foot_body_id,
                              args.shift_end, args.swing_start, args.swing_end,
                              stance_preload=args.stance_preload,
                              swing_excitation=args.swing_excitation,
                              fitness_mode=args.fitness_mode)
            row = {"seed": seed, "condition": label, "pose_sha256": pose_hash,
                   **summarize(result)}
            records.append(row)
            print(json.dumps(row), flush=True)
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
              "bias": str(args.bias.resolve()),
              "evolution": str(args.evolution.resolve()),
              "duration_s": args.control_steps * float(source.opt.timestep),
              "shift_end_s": args.shift_end,
              "swing_start_s": args.swing_start,
              "swing_end_s": args.swing_end,
              "stance_preload": args.stance_preload,
              "swing_excitation": args.swing_excitation,
              "fitness_mode": args.fitness_mode,
              "seed_candidate": args.seed_candidate,
              "parameter_names": PARAMETERS, "training_seeds": sorted(training_seeds),
              "selection_seed": args.selection_seed,
              "heldout_seeds": args.heldout_seeds, "heldout_worlds": args.heldout_worlds,
              "selected_parameters": selected.cpu().tolist(),
              "history": history, "selection_rows": selection_rows,
              "heldout_records": records,
              "scope": "Single tactile-triggered step evolution, not walking"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
