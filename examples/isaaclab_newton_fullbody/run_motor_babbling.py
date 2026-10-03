"""Collect spontaneous muscle activity and measure sensorimotor prediction.

Run with the pinned Isaac Lab Python after scripts/run.py doctor and asset prep.
This is an early body-schema assay, not a crawling or walking learner.
"""

import argparse
import json
import os
from pathlib import Path

import newton
import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonMJWarpManager, NewtonManager
from newton.solvers import SolverMuJoCo

from developmental_skin import REGIONS, SkinContactSensor, geom_region_ids


class MuscleManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return FullBodySolver(model, **cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg))


def ridge_error(train_input, train_target, test_input, test_target):
    """Predict state changes with a regularized linear sensorimotor model."""
    mean = train_input.mean(0)
    scale = train_input.std(0).clamp_min(1e-5)
    train = (train_input - mean) / scale
    test = (test_input - mean) / scale
    current_train = train_input[:, :train_target.shape[1]]
    current_test = test_input[:, :test_target.shape[1]]
    change = train_target - current_train
    target_mean = change.mean(0)
    target_scale = change.std(0).clamp_min(1e-5)
    target = (change - target_mean) / target_scale
    train = torch.cat((train, torch.ones(len(train), 1, device=train.device)), dim=1)
    test = torch.cat((test, torch.ones(len(test), 1, device=test.device)), dim=1)
    penalty = 10.0 * train.shape[1]
    if len(train) < train.shape[1]:
        gram = train @ train.T
        gram += penalty * torch.eye(len(train), device=train.device)
        predicted_change = test @ train.T @ torch.linalg.solve(gram, target)
    else:
        gram = train.T @ train
        gram += penalty * torch.eye(train.shape[1], device=train.device)
        predicted_change = test @ torch.linalg.solve(gram, train.T @ target)
    predicted = current_test + predicted_change * target_scale + target_mean
    return float((predicted - test_target).square().mean())


def analyze(records):
    current = torch.from_numpy(np.concatenate([record["current"] for record in records])).to("cuda:0")
    action = torch.from_numpy(np.concatenate([record["action"] for record in records])).to("cuda:0")
    tactile = torch.from_numpy(np.concatenate([record["tactile"] for record in records])).to("cuda:0")
    target = torch.from_numpy(np.concatenate([record["target"] for record in records])).to("cuda:0")
    cut = int(len(current) * 0.8)
    if cut < 100 or len(current) - cut < 20:
        raise ValueError("Collect at least 125 transitions for the prediction assay")
    baseline = float((target[cut:] - current[cut:, :target.shape[1]]).square().mean())
    with_action = torch.cat((current, action), dim=1)
    with_touch = torch.cat((with_action, tactile), dim=1)
    generator = torch.Generator().manual_seed(1729)
    shuffled_train = tactile[:cut][torch.randperm(cut, generator=generator).to(tactile.device)]
    shuffled_test = tactile[cut:][torch.randperm(len(tactile) - cut, generator=generator).to(tactile.device)]
    shuffled = torch.cat((with_action, torch.cat((shuffled_train, shuffled_test))), dim=1)
    return {
        "samples": len(current), "held_out_samples": len(current) - cut,
        "target": "next joint velocity and muscle length",
        "mse_persistence": baseline,
        "mse_state_only": ridge_error(current[:cut], target[:cut], current[cut:], target[cut:]),
        "mse_state_action": ridge_error(with_action[:cut], target[:cut], with_action[cut:], target[cut:]),
        "mse_state_action_touch": ridge_error(with_touch[:cut], target[:cut], with_touch[cut:], target[cut:]),
        "mse_state_action_shuffled_touch": ridge_error(shuffled[:cut], target[:cut], shuffled[cut:], target[cut:]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, default=Path(__file__).resolve().parents[2]
                        / "artifacts/best-policy/initial.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--reset-interval", type=int, default=100,
                        help="Control steps between upright resets; zero disables resets")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--debug-contacts", action="store_true")
    args = parser.parse_args()
    if args.worlds < 2 or args.steps < 125:
        parser.error("--worlds must be >=2 and --steps must be >=125")
    if args.reset_interval < 0:
        parser.error("--reset-interval must be nonnegative")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    os.environ["MM_MODEL_XML"] = str(args.model_xml.resolve())
    os.environ["MM_MATCH_PHYSICS"] = "1"
    global FullBodySolver, make_builder
    from newton_fullbody_adapter import FullBodySolver, JOINT_DOF, REFERENCE, make_builder
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    wp.config.kernel_cache_dir = str(args.output.parent / "warp_cache")
    cfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, update_data_interval=0,
                                   integrator="euler", iterations=20, ls_iterations=8,
                                   nconmax=512, njmax=2048),
        use_cuda_graph=True, num_substeps=1,
    )
    cfg.class_type = MuscleManager
    with build_simulation_context(sim_cfg=SimulationCfg(dt=0.002, device="cuda:0", physics=cfg)) as sim:
        template = make_builder()
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for _ in range(args.worlds):
            builder.add_world(template)
        NewtonManager.set_builder(builder)
        sim.reset()
        solver = NewtonManager._solver
        data = solver.mjw_data
        control = wp.to_torch(NewtonManager.get_control().mujoco.ctrl).view(args.worlds, -1)
        velocity = wp.to_torch(data.qvel)
        position = wp.to_torch(data.qpos)
        length = wp.to_torch(data.actuator_length)
        activation = wp.to_torch(data.act)
        skin = SkinContactSensor(solver.mj_model, solver.mjw_model, data, "cuda:0")
        source = list(range(7))
        destination = list(range(7))
        dofs = solver.mjc_jnt_to_newton_dof.numpy().reshape(args.worlds, -1)[0]
        for joint_id in range(REFERENCE.njnt):
            if REFERENCE.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(REFERENCE, mujoco.mjtObj.mjOBJ_JOINT, joint_id).replace("-", "_")
            matching = np.flatnonzero(dofs == JOINT_DOF[name])
            if len(matching) != 1:
                raise RuntimeError(f"Cannot map anatomical joint {name}")
            source.append(int(REFERENCE.jnt_qposadr[joint_id]))
            destination.append(int(solver.mj_model.jnt_qposadr[int(matching[0])]))
        initial = solver.mj_model.qpos0.copy()
        initial[destination] = np.load(args.initial_state)["qpos"][source]
        initial = torch.as_tensor(initial, dtype=position.dtype, device="cuda:0")

        def reset_worlds():
            state = NewtonManager.get_state_0()
            mask = wp.array([True] * args.worlds + [False], dtype=wp.bool, device="cuda:0")
            solver.reset(state, mask, flags=0)
            position.copy_(initial.expand_as(position))
            velocity.zero_()
            activation.zero_()
            control.zero_()
            mjw.forward(solver.mjw_model, data)
            solver._update_newton_state(solver.model, state, data, state_prev=state)

        reset_worlds()
        generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
        excitation = torch.full_like(control, 0.05)
        records = []
        for step in range(args.steps):
            if step and args.reset_interval and step % args.reset_interval == 0:
                reset_worlds()
            if step % 10 == 0:
                excitation = (0.8 * excitation + 0.2 * torch.rand(
                    control.shape, device="cuda:0", generator=generator)).clamp(0.02, 0.6)
            state = torch.cat((velocity, length, activation), dim=1).clone()
            touch = skin.normalized(float(np.sum(solver.mj_model.body_mass)) * 9.81).clone()
            if step == 0 and args.debug_contacts:
                count = int(data.nacon.numpy()[0])
                pairs = data.contact.geom.numpy()[:min(count, 20)]
                region_ids = geom_region_ids(solver.mj_model)
                print("INITIAL_CONTACTS", json.dumps([{
                    "geoms": [mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                              for geom in pair],
                    "regions": [REGIONS[region_ids[int(geom)]] if region_ids[int(geom)] >= 0 else None
                                for geom in pair],
                } for pair in pairs]), flush=True)
            control.copy_(excitation)
            for _ in range(10):
                sim.step(render=False)
            if bool((wp.to_torch(data.overflow).to(torch.int64) & 511).any()):
                raise RuntimeError(f"Contact capacity overflow at step {step}")
            next_state = torch.cat((velocity, length), dim=1).clone()
            if not (torch.isfinite(next_state).all() and torch.isfinite(touch).all()):
                raise RuntimeError(f"Nonfinite sensor data at step {step}")
            records.append({
                "current": state.cpu().numpy(), "action": excitation.cpu().numpy(),
                "tactile": touch.cpu().numpy(), "target": next_state.cpu().numpy(),
                "qpos": position.clone().cpu().numpy(),
            })
    trajectory = args.output.with_suffix(".npz")
    if trajectory.exists():
        raise RuntimeError(f"Trajectory already exists: {trajectory}")
    np.savez_compressed(trajectory, **{
        key: np.stack([record[key] for record in records])
        for key in ("current", "action", "tactile", "target", "qpos")
    })
    result = analyze(records)
    touch_samples = np.concatenate([record["tactile"] for record in records])
    result.update({"seed": args.seed, "worlds": args.worlds,
                   "reset_interval_steps": args.reset_interval, "regions": REGIONS,
                   "initial_state": str(args.initial_state.resolve()),
                   "nonzero_touch_fraction": float(np.mean(touch_samples > 0)),
                   "touch_fraction_by_region": dict(zip(REGIONS,
                       [float(value) for value in np.mean(touch_samples > 0, axis=0)])),
                   "model_xml": str(args.model_xml.resolve()),
                   "scope": "Adult muscle model, spontaneous excitation, coarse rigid-contact skin; no infant morphology or developmental progression"})
    result["trajectory"] = str(trajectory)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
