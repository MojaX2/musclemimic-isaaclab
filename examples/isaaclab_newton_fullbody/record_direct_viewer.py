"""Record actual Isaac Lab/Newton GPU motion for offline visualization."""
import json
import os
import subprocess
os.environ.pop("DISPLAY", None)
from PIL import Image
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

from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

class NewtonMuscleManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls, model, solver_cfg):
        return FullBodySolver(model, **cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg))

OUT = Path('/tmp/muscle-feasibility/renders')
OUT.mkdir(parents=True, exist_ok=True)
DT, FPS, SECONDS, WORLDS = .002, 30, 6, 1
cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(
    use_mujoco_contacts=True, integrator='implicitfast', iterations=100,
    ls_iterations=50, nconmax=1024, njmax=2048), use_cuda_graph=True, num_substeps=1)
cfg.class_type = NewtonMuscleManager
viewer_cfg = NewtonGLVisualizerCfg(headless=True, window_width=1280, window_height=720,
    eye=(3., -3., 1.8), lookat=(0., 0., .7), focal_length=24., max_visible_envs=1,
    background_color=(.12, .15, .18),
    show_collision=False, enable_picking=False, streaming_view=False)
ffmpeg = next(Path('/tmp/muscle-feasibility/mjlab-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux*'))
video = OUT/'isaaclab_newton_viewer_direct.mp4'
writer = None
with build_simulation_context(sim_cfg=SimulationCfg(dt=DT, device='cuda:0', physics=cfg, visualizer_cfgs=[viewer_cfg])) as sim:
    template = make_builder()
    # Display anatomy only; collision/wrapping geometries remain in the physics model.
    for shape, geo_type in enumerate(template.shape_type):
        if geo_type not in [newton.GeoType.MESH, newton.GeoType.PLANE]:
            template.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    for _ in range(WORLDS):
        builder.add_world(template)
    NewtonManager.set_builder(builder)
    sim.reset()
    assert sim._visualizers, 'Newton visualizer was not initialized'
    viewer = sim._visualizers[0]
    print('VIEWER', type(viewer).__name__, flush=True)
    writer = subprocess.Popen([str(ffmpeg), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', '1280x720', '-pix_fmt', 'rgb24', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
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
        # Direct live NewtonGLVisualizer framebuffer; no replay or MuJoCo renderer.
        sim.render()
        # Draw current GPU-computed muscle paths in the Newton viewer itself.
        points = solver.mjw_data.wrap_xpos.numpy()[0].reshape(-1, 3)
        adr = solver.mjw_data.ten_wrapadr.numpy()[0]
        nums = solver.mjw_data.ten_wrapnum.numpy()[0]
        starts, ends, colors = [], [], []
        for muscle, tendon in enumerate(model.actuator_trnid[:, 0]):
            path = points[int(adr[tendon]):int(adr[tendon]+nums[tendon])]
            color = np.array([.35, .08, .10]) + np.clip(a[muscle]/.35, 0, 1)*np.array([.60, .47, .02])
            for k in range(len(path)-1):
                starts.append(path[k]); ends.append(path[k+1]); colors.append(color)
        if starts:
            viewer._viewer.log_lines('muscles', wp.array(starts, dtype=wp.vec3, device='cuda:0'),
                wp.array(ends, dtype=wp.vec3, device='cuda:0'),
                wp.array(colors, dtype=wp.vec3, device='cuda:0'))
        pixels = np.ascontiguousarray(viewer.render_rgb_array())
        assert pixels.shape == (720, 1280, 3), pixels.shape
        writer.stdin.write(pixels.tobytes())
        if frame in [0, 30, 90, 150]:
            Image.fromarray(pixels).save(OUT/f'direct_viewer_{frame:03d}.png')
        poses.append(pose)
        acts.append(a.copy())
        times.append(step*DT)
        max_force = max(max_force, float(np.abs(force).max()))
        max_constraints = max(max_constraints, int(solver.mjw_data.nefc.numpy().max()))
        max_contacts = max(max_contacts, int(solver.mjw_data.nacon.numpy().max()))
        if frame % 30 == 0:
            print('CAPTURE', frame, 'time', step*DT, flush=True)
    writer.stdin.close()
    assert writer.wait() == 0
    print('VIDEO', video, flush=True)
    (OUT/'isaaclab_direct_viewer.json').write_text(json.dumps({
        'physics': 'Isaac Lab SimulationContext -> Newton -> MuJoCo Warp, CUDA Graph',
        'num_envs': WORLDS, 'recorded_env': 0, 'dt': DT, 'fps': FPS, 'frames': len(poses),
        'duration_seconds': SECONDS, 'seed': 42, 'max_abs_force': max_force,
        'peak_constraints': max_constraints, 'peak_contacts_all_worlds': max_contacts,
        'overflow_detected': False, 'policy': 'smooth random muscle excitation; no balance controller',
        'rendering': 'Isaac Lab NewtonGLVisualizer live framebuffer during simulation; live GPU tendon paths added as lines'
    }, indent=2))

