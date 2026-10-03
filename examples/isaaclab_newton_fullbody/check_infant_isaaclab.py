"""Check whether an infant MJCF runs in Isaac Lab's Newton/MuJoCo Warp backend."""

import argparse
import inspect
import json
from pathlib import Path
import re

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonMJWarpManager, NewtonManager
from newton.solvers import SolverMuJoCo

from developmental_skin import REGIONS, SkinContactSensor
from infant_muscles import InfantMuscles
from infant_skin import infant_geom_region_ids


@wp.kernel
def _write_muscle_torques(torques: wp.array2d(dtype=wp.float32),
                          dof_ids: wp.array(dtype=wp.int32),
                          dofs_per_world: int,
                          joint_forces: wp.array(dtype=wp.float32)):
    world, actuator = wp.tid()
    joint_forces[world * dofs_per_world + dof_ids[actuator]] = torques[world, actuator]


class InfantSolver(SolverMuJoCo):
    source_model = None

    def _init_tendons(self, *args, **kwargs):
        return [], []

    def _init_actuators(self, *args, **kwargs):
        arguments = inspect.signature(SolverMuJoCo._init_actuators).bind(self, *args, **kwargs).arguments
        count = super()._init_actuators(*args, **kwargs)
        source = self.source_model
        if source is None:
            raise RuntimeError("Infant source model was not configured")
        model = arguments["model"]
        spec = arguments["spec"]
        dof_to_joint = arguments["dof_to_mjc_joint"]
        joint_names = arguments["mjc_joint_names"]
        joint_starts = model.joint_qd_start.numpy()
        joint_dimensions = model.joint_dof_dim.numpy()
        for tendon_id in range(source.ntendon):
            name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
            tendon = spec.add_tendon(name=name)
            tendon.limited = int(source.tendon_limited[tendon_id])
            tendon.range = source.tendon_range[tendon_id].tolist()
            tendon.margin = float(source.tendon_margin[tendon_id])
            tendon.solref_limit = source.tendon_solref_lim[tendon_id].tolist()
            tendon.solimp_limit = source.tendon_solimp_lim[tendon_id].tolist()
            tendon.stiffness[0] = float(source.tendon_stiffness[tendon_id])
            tendon.damping[0] = float(source.tendon_damping[tendon_id])
            tendon.frictionloss = float(source.tendon_frictionloss[tendon_id])
            tendon.armature = float(source.tendon_armature[tendon_id])
            for wrap_id in range(source.tendon_adr[tendon_id],
                                 source.tendon_adr[tendon_id] + source.tendon_num[tendon_id]):
                if source.wrap_type[wrap_id] != mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError(f"Unsupported non-joint wrap in tendon {name}")
                source_joint_id = int(source.wrap_objid[wrap_id])
                source_joint_name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT,
                                                      source_joint_id)
                dof_id = int(source.jnt_dofadr[source_joint_id])
                matching_joints = [joint_id for joint_id, (start, dimension) in
                                   enumerate(zip(joint_starts, joint_dimensions))
                                   if start <= dof_id < start + sum(dimension)]
                if len(matching_joints) != 1 or source_joint_name not in model.joint_label[matching_joints[0]]:
                    raise ValueError(f"Cannot prove source-to-Newton DOF mapping for {source_joint_name}")
                target_joint_id = int(dof_to_joint[dof_id])
                if not 0 <= target_joint_id < len(joint_names):
                    raise ValueError(f"No MuJoCo joint mapping for {source_joint_name}")
                tendon.wrap_joint(joint_names[target_joint_id], float(source.wrap_prm[wrap_id]))
        return count


class InfantManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return InfantSolver(model, **cls._filter_solver_kwargs(InfantSolver, solver_cfg))


def audit_geom_parameters(source, restored):
    source_ids = {mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_GEOM, geom_id): geom_id
                  for geom_id in range(source.ngeom)}
    restored_names = [mujoco.mj_id2name(restored, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                      for geom_id in range(restored.ngeom)]
    restored_ids = {re.sub(r"_\d+$", "", name.rsplit("/", 1)[-1]): geom_id
                    for geom_id, name in enumerate(restored_names) if name is not None}
    if len(restored_ids) != sum(name is not None for name in restored_names):
        raise RuntimeError("Newton geom names are ambiguous after normalization")
    matching = sorted((name for name in source_ids.keys() & restored_ids.keys() if name is not None))
    fields = ("geom_condim", "geom_contype", "geom_conaffinity", "geom_solref",
              "geom_solimp", "geom_friction", "geom_size", "geom_pos", "geom_quat",
              "geom_margin", "geom_gap")
    differences = {}
    for field in fields:
        original = getattr(source, field)
        converted = getattr(restored, field)
        mismatches = []
        for name in matching:
            before = np.asarray(original[source_ids[name]])
            after = np.asarray(converted[restored_ids[name]])
            if not np.allclose(before, after, rtol=0, atol=1e-6):
                mismatches.append({"geom": name, "source": before.tolist(),
                                   "restored": after.tolist()})
        differences[field] = {"count": len(mismatches), "examples": mismatches[:8]}
    return {"source_geom_count": source.ngeom, "restored_geom_count": restored.ngeom,
            "restored_geom_names_sample": restored_names[:5],
            "matched_named_geoms": len(matching),
            "missing_from_restored": sorted(name for name in source_ids.keys() -
                                            restored_ids.keys() if name is not None)[:20],
            "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--muscle-drive", action="store_true")
    parser.add_argument("--identical-actions", action="store_true")
    parser.add_argument("--excitation-scale", type=float, default=0.05)
    parser.add_argument("--replay-world", type=int)
    parser.add_argument("--integrator", choices=("euler", "implicitfast"), default="implicitfast")
    parser.add_argument("--physics-dt", type=float)
    parser.add_argument("--audit-model", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=100)
    parser.add_argument("--line-search-iterations", type=int, default=40)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1:
        parser.error("Worlds and steps must be positive")
    if not 0 <= args.excitation_scale <= 1:
        parser.error("Excitation scale must be between zero and one")
    if args.replay_world is not None and (args.worlds != 1 or args.replay_world < 0):
        parser.error("Replay world requires one environment and a nonnegative index")
    if args.physics_dt is not None and args.physics_dt <= 0:
        parser.error("Physics timestep must be positive")
    if args.solver_iterations < 1 or args.line_search_iterations < 1:
        parser.error("Solver iteration limits must be positive")
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    physics_dt = args.physics_dt or float(source.opt.timestep)
    if source.eq_active0.any():
        raise ValueError("The infant bridge can skip only inactive model equalities")
    InfantSolver.source_model = source
    config = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True, update_data_interval=0,
                                   integrator=args.integrator, iterations=args.solver_iterations,
                                   ls_iterations=args.line_search_iterations,
                                   cone=("elliptic" if int(source.opt.cone) ==
                                         int(mujoco.mjtCone.mjCONE_ELLIPTIC) else "pyramidal"),
                                   enable_multiccd=not bool(int(source.opt.disableflags) &
                                       int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
                                   tolerance=float(source.opt.tolerance),
                                   nconmax=4096, njmax=8192),
        use_cuda_graph=True, num_substeps=1,
    )
    config.class_type = InfantManager
    with build_simulation_context(sim_cfg=SimulationCfg(dt=physics_dt,
                                                        device="cuda:0", physics=config)) as simulation:
        template = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template)
        template.add_mjcf(str(args.scene.resolve()), parse_mujoco_options=False,
                          skip_equality_constraints=True)
        muscle_joint_ids = [joint_id for joint_id in range(source.njnt)
                            if (mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT,
                                                  joint_id) or "").startswith("robot:")]
        muscle_dof_ids = np.array([source.jnt_dofadr[joint_id] for joint_id in muscle_joint_ids])
        if args.muscle_drive:
            passive_stiffness = template.custom_attributes["mujoco:dof_passive_stiffness"].values
            for joint_id, dof_id in zip(muscle_joint_ids, muscle_dof_ids):
                if not np.isclose(template.joint_damping[dof_id], source.dof_damping[dof_id]):
                    raise ValueError(f"Damping DOF mismatch for source joint {joint_id}")
                if not np.isclose(passive_stiffness[dof_id], source.jnt_stiffness[joint_id]):
                    raise ValueError(f"Stiffness DOF mismatch for source joint {joint_id}")
                template.joint_damping[dof_id] /= 20
                passive_stiffness[dof_id] = 0.0
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for _ in range(args.worlds):
            builder.add_world(template)
        NewtonManager.set_builder(builder)
        simulation.reset()
        solver = NewtonManager._solver
        solver.mjw_model.opt.timestep.fill_(physics_dt)
        solver.mj_model.opt.timestep = physics_dt
        data = solver.mjw_data
        newton_control = NewtonManager.get_control()
        control = newton_control.mujoco.ctrl
        skin = SkinContactSensor(solver.mj_model, solver.mjw_model, data, "cuda:0",
                                 region_ids=infant_geom_region_ids(solver.mj_model))
        touch_counts = torch.zeros(len(REGIONS), device="cuda:0", dtype=torch.int32)
        touch_peaks = torch.zeros(len(REGIONS), device="cuda:0")
        source_data = mujoco.MjData(source)
        restored_data = mujoco.MjData(solver.mj_model)
        mujoco.mj_forward(source, source_data)
        mujoco.mj_forward(solver.mj_model, restored_data)
        if solver.mj_model.ntendon != source.ntendon or solver.mj_model.nv != source.nv:
            raise RuntimeError("Restored infant tendon or DOF count does not match the source")
        tendon_jacobian_error = float(np.max(np.abs(source_data.ten_J - restored_data.ten_J)))
        tendon_range_error = float(np.max(np.abs(source.tendon_range - solver.mj_model.tendon_range)))
        if tendon_jacobian_error > 1e-6 or tendon_range_error > 1e-6:
            raise RuntimeError("Restored infant tendon Jacobian or range differs from the source")
        if args.muscle_drive:
            stiffness_error = float(np.max(np.abs(solver.mj_model.jnt_stiffness[muscle_joint_ids])))
            damping_error = float(np.max(np.abs(solver.mj_model.dof_damping[muscle_dof_ids] -
                                                source.dof_damping[muscle_dof_ids] / 20)))
            if stiffness_error > 1e-6 or damping_error > 1e-6:
                raise RuntimeError("MIMo muscle passive joint properties were not applied")
        if args.pose is not None:
            with np.load(args.pose) as saved:
                pose = np.asarray(saved["qpos"], dtype=np.float32)
                velocity = np.asarray(saved["qvel"], dtype=np.float32)
            if (pose.shape != (source.nq,) or velocity.shape != (source.nv,) or
                    not np.isfinite(pose).all() or not np.isfinite(velocity).all()):
                raise ValueError("Infant pose must contain finite source-sized qpos and qvel")
            wp.to_torch(data.qpos).copy_(torch.as_tensor(pose, device="cuda:0").expand(args.worlds, -1))
            wp.to_torch(data.qvel).copy_(torch.as_tensor(velocity, device="cuda:0").expand(args.worlds, -1))
            data.qacc_warmstart.zero_()
            mjw.forward(solver.mjw_model, data)
        initial = data.qpos.numpy().copy()
        root_joint_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT,
                                          "mimo_orientation")
        if root_joint_id < 0:
            raise ValueError("MIMo root joint is missing")
        root_height_id = int(source.jnt_qposadr[root_joint_id]) + 2
        first_step_root_height = None
        rng = np.random.default_rng(7)
        if args.muscle_drive:
            if source.nv != solver.mj_model.nv or source.nq != solver.mj_model.nq:
                raise RuntimeError("Infant muscle drive requires matching source and solver coordinates")
            muscles = InfantMuscles(source, args.worlds, device="cuda:0")
            dof_ids = wp.array(muscles.qvel_ids.cpu().numpy().astype(np.int32),
                               dtype=wp.int32, device="cuda:0")
            muscle_actions = torch.zeros((args.worlds, 2 * source.nu), device="cuda:0")
            control.zero_()
            max_muscle_torque = 0.0
            max_applied_joint_force = 0.0
        solver_limit_steps = 0
        for step in range(args.steps):
            if args.muscle_drive:
                if step % 20 == 0:
                    action_worlds = (args.replay_world + 1 if args.replay_world is not None
                                     else 1 if args.identical_actions else args.worlds)
                    action_shape = (action_worlds, 2 * source.nu)
                    sampled_actions = rng.uniform(0, args.excitation_scale,
                                                  size=action_shape).astype(np.float32)
                    if args.replay_world is not None:
                        sampled_actions = sampled_actions[args.replay_world:args.replay_world + 1]
                    muscle_actions.copy_(torch.from_numpy(np.broadcast_to(
                        sampled_actions, muscle_actions.shape).copy()).to("cuda:0"))
                torques = muscles.step(wp.to_torch(data.qpos), wp.to_torch(data.qvel),
                                       muscle_actions, physics_dt)
                newton_control.joint_f.zero_()
                wp.launch(_write_muscle_torques, dim=(args.worlds, source.nu),
                          inputs=[wp.from_torch(torques.contiguous()), dof_ids, source.nv],
                          outputs=[newton_control.joint_f], device="cuda:0")
                max_muscle_torque = max(max_muscle_torque, float(torques.abs().max()))
            elif step % 20 == 0:
                action_worlds = (args.replay_world + 1 if args.replay_world is not None
                                 else 1 if args.identical_actions else args.worlds)
                action_shape = (action_worlds, source.nu)
                sampled_actions = rng.uniform(-args.excitation_scale, args.excitation_scale,
                                              size=action_shape).astype(np.float32)
                if args.replay_world is not None:
                    sampled_actions = sampled_actions[args.replay_world:args.replay_world + 1]
                control.assign(np.broadcast_to(sampled_actions,
                                               (args.worlds, source.nu)).copy().reshape(control.shape))
            simulation.step(render=False)
            if step == 0:
                first_step_root_height = float(data.qpos.numpy()[:, root_height_id].mean())
            touch = skin.read()
            touch_counts += (touch > 1e-6).sum(dim=0).to(torch.int32)
            touch_peaks = torch.maximum(touch_peaks, touch.max(dim=0).values)
            if args.muscle_drive:
                max_applied_joint_force = max(max_applied_joint_force,
                                              float(wp.to_torch(data.qfrc_applied).abs().max()))
            position_finite = np.isfinite(data.qpos.numpy()).all(axis=1)
            velocity_finite = np.isfinite(data.qvel.numpy()).all(axis=1)
            if not (position_finite.all() and velocity_finite.all()):
                raise RuntimeError(f"Nonfinite infant state at step {step}: "
                                   f"qpos={position_finite.tolist()}, "
                                   f"qvel={velocity_finite.tolist()}, "
                                   f"overflow={data.overflow.numpy().tolist()}")
            flags = data.overflow.numpy().astype(np.int64)
            if np.any(flags & 511):
                raise RuntimeError(f"MuJoCo Warp capacity overflow at step {step}: {flags.tolist()}")
            solver_limit_steps += int(np.any(flags & 1536))
        report = {
            "source": str(args.scene.resolve()), "source_nq": source.nq,
            "pose": str(args.pose.resolve()) if args.pose is not None else None,
            "source_nv": source.nv, "source_nu": source.nu,
            "source_ntendon": source.ntendon,
            "solver_nq": solver.mj_model.nq, "solver_nv": solver.mj_model.nv,
            "solver_nu": solver.mj_model.nu,
            "solver_ntendon": solver.mj_model.ntendon,
            "tendon_jacobian_max_abs_error": tendon_jacobian_error,
            "tendon_range_max_abs_error": tendon_range_error,
            "worlds": args.worlds, "steps": args.steps,
            "physics_dt_s": physics_dt,
            "integrator": args.integrator,
            "solver_iterations": args.solver_iterations,
            "line_search_iterations": args.line_search_iterations,
            "max_qpos_change": float(np.max(np.abs(data.qpos.numpy() - initial))),
            "initial_root_height_m": float(initial[:, root_height_id].mean()),
            "first_step_root_height_m": first_step_root_height,
            "solver_limit_steps": solver_limit_steps,
            "muscle_model": ("antagonist muscle torque through Newton joint_f with MIMo "
                             "passive joint adjustment" if args.muscle_drive
                             else "joint motors only; infant_muscles.py torques are not wired into the solver"),
            "muscle_drive": args.muscle_drive,
            "identical_actions": args.identical_actions,
            "replay_world": args.replay_world,
            "excitation_scale": args.excitation_scale,
            "tactile_model": "15 coarse Newton contact-force regions; MIMo taxels are not ported",
            "tactile_regions": list(REGIONS),
            "tactile_nonzero_world_steps": touch_counts.cpu().tolist(),
            "tactile_peak_force_n": touch_peaks.cpu().tolist(),
            "all_finite": True,
        }
        if args.muscle_drive:
            report["max_muscle_torque_nm"] = max_muscle_torque
            report["max_applied_joint_force_nm"] = max_applied_joint_force
            report["mean_muscle_activity"] = float(muscles.activity.mean())
            report["passive_joint_stiffness_max_abs_error"] = stiffness_error
            report["passive_joint_damping_max_abs_error"] = damping_error
        if args.audit_model:
            report["geometry_audit"] = audit_geom_parameters(source, solver.mj_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
