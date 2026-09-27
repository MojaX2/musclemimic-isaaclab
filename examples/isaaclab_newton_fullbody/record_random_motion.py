"""Record actual Isaac Lab/Newton GPU motion for offline visualization."""
import json
from pathlib import Path

import mujoco
import newton
import numpy as np
import warp as wp

wp.config.kernel_cache_dir = '/tmp/muscle-feasibility/warp-cache'
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg, NewtonManager, NewtonMJWarpManager
from newton.solvers import SolverMuJoCo
from newton_fullbody_adapter import FullBodySolver, make_builder, REFERENCE, JOINT_DOF

class MuscleManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return FullBodySolver(model, **cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg))

OUT = Path('/tmp/muscle-feasibility/renders')
OUT.mkdir(parents=True, exist_ok=True)
DT, FPS, SECONDS, WORLDS = .002, 30, 6, 4
cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(
    use_mujoco_contacts=True, integrator='implicitfast', iterations=100,
    ls_iterations=50, nconmax=1024, njmax=2048), use_cuda_graph=True, num_substeps=1)
cfg.class_type = MuscleManager
with build_simulation_context(sim_cfg=SimulationCfg(dt=DT, device='cuda:0', physics=cfg)) as sim:
    template = make_builder()
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    for _ in range(WORLDS):
        builder.add_world(template)
    NewtonManager.set_builder(builder)
    sim.reset()
    solver = NewtonManager._solver
    model = solver.mj_model
    control = NewtonManager.get_control().mujoco.ctrl
    newton_dofs = solver.mjc_jnt_to_newton_dof.numpy().reshape(-1)
    # Translate the solver's scalar joints to the original anatomy used for rendering.
    source_q, target_q = list(range(7)), list(range(7))
    for j in range(REFERENCE.njnt):
        if REFERENCE.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(REFERENCE, mujoco.mjtObj.mjOBJ_JOINT, j)
        candidates = np.flatnonzero(newton_dofs == JOINT_DOF[name])
        assert len(candidates) == 1, (name, candidates)
        source_q.append(int(model.jnt_qposadr[candidates[0]]))
        target_q.append(int(REFERENCE.jnt_qposadr[j]))
    assert len(set(target_q)) == REFERENCE.nq
    rng = np.random.default_rng(42)
    knots = rng.uniform(.02, .30, size=(int(SECONDS/.25)+3, control.shape[0])).astype(np.float32)
    knots[0] = 0
    poses, acts, times = [], [], []
    step = 0
    max_force, max_constraints, max_contacts = 0., 0, 0
    for frame in range(FPS * SECONDS):
        while step * DT < frame / FPS - 1e-9:
            t = step * DT / .25
            i, f = int(t), t % 1
            blend = f*f*(3-2*f)
            control.assign((1-blend)*knots[i] + blend*knots[i+1])
            sim.step(render=False)
            flags = solver.mjw_data.overflow.numpy()
            assert not flags.any(), (step, flags.tolist())
            step += 1
        q = solver.mjw_data.qpos.numpy()[0]
        a = solver.mjw_data.act.numpy()[0]
        force = solver.mjw_data.actuator_force.numpy()
        assert np.isfinite(q).all() and np.isfinite(a).all() and np.isfinite(force).all()
        pose = REFERENCE.qpos0.copy()
        pose[target_q] = q[source_q]
        poses.append(pose)
        acts.append(a.copy())
        times.append(step*DT)
        max_force = max(max_force, float(np.abs(force).max()))
        max_constraints = max(max_constraints, int(solver.mjw_data.nefc.numpy().max()))
        max_contacts = max(max_contacts, int(solver.mjw_data.nacon.numpy().max()))
        if frame % 30 == 0:
            print('CAPTURE', frame, 'time', step*DT, flush=True)
    np.savez_compressed(OUT/'isaaclab_random_motion.npz', qpos=poses, act=acts, times=times, fps=FPS)
    (OUT/'isaaclab_random_motion.json').write_text(json.dumps({
        'physics': 'Isaac Lab SimulationContext -> Newton -> MuJoCo Warp, CUDA Graph',
        'num_envs': WORLDS, 'recorded_env': 0, 'dt': DT, 'fps': FPS, 'frames': len(poses),
        'duration_seconds': SECONDS, 'seed': 42, 'max_abs_force': max_force,
        'peak_constraints': max_constraints, 'peak_contacts_all_worlds': max_contacts,
        'overflow_detected': False, 'policy': 'smooth random muscle excitation; no balance controller',
        'rendering': 'offline from recorded GPU poses; no CPU dynamics integration'
    }, indent=2))
    print('SAVED', OUT/'isaaclab_random_motion.npz', flush=True)
