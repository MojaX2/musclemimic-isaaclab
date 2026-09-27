"""Small real GPU rollout through mjlab Simulation and Entity control APIs."""
import gc
import json
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg

wp.config.kernel_cache_dir = '/tmp/muscle-feasibility/warp-cache'
ROOT = Path('/home/k_miyazawa/musclemimic/.venv/lib/python3.11/site-packages/musclemimic_models/model')
OUT = Path('/tmp/muscle-feasibility/mjlab_gpu_results.json')
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
wp.init()
print('GPU', torch.cuda.get_device_name(0), flush=True)
results = dict(gpu=torch.cuda.get_device_name(0), versions={p: version(p) for p in
               ['mjlab', 'mujoco', 'mujoco-warp', 'torch', 'warp-lang']}, models={})

def test_model(name, relative):
    path = ROOT / relative
    source = mujoco.MjSpec.from_file(str(path))
    entity = Entity(EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(path)),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(
            target_names_expr=tuple(a.target for a in source.actuators),
            transmission_type=TransmissionType.TENDON,
        ),)),
    ))
    model = entity.compile()
    nworld, steps, dt = 4, 1000, 0.001
    print('BUILD', name, 'muscles', model.nu, 'worlds', nworld, flush=True)
    build_start = time.perf_counter()
    sim = Simulation(num_envs=nworld, model=model, device='cuda:0', cfg=SimulationCfg(
        nconmax=512, njmax=2048,
        mujoco=MujocoCfg(timestep=dt, integrator='implicitfast', iterations=50, ls_iterations=20),
    ))
    entity.initialize(model, sim.model, sim.data, 'cuda:0')
    sim.reset()
    entity.reset()
    sim.forward()
    wp.synchronize()
    build_seconds = time.perf_counter() - build_start
    print('READY', name, 'build_seconds', round(build_seconds, 2), flush=True)
    assert sim.wp_data.qpos.device.is_cuda and sim.wp_data.act.device.is_cuda
    initial_q = sim.wp_data.qpos.numpy().copy()
    rng = np.random.default_rng(42)
    targets = rng.beta(0.8, 2.0, size=(7, nworld, model.nu)).astype(np.float32) * 0.55
    targets[0] = 0
    targets = torch.tensor(targets, device='cuda:0')
    tendon_ids = torch.tensor(model.actuator_trnid[:, 0].copy(), device='cuda:0', dtype=torch.long)
    max_force = 0.0
    start = time.perf_counter()
    for step in range(steps):
        i, t = divmod(step, 200)
        fraction = t / 200.0
        blend = fraction * fraction * (3 - 2 * fraction)
        excitation = (1 - blend) * targets[i] + blend * targets[i + 1]
        entity.set_tendon_effort_target(excitation, tendon_ids=tendon_ids)
        entity.write_data_to_sim()
        sim.step()
        if (step + 1) % 250 == 0:
            wp.synchronize()
            for field in ['qpos', 'qvel', 'act', 'actuator_force']:
                assert np.isfinite(getattr(sim.wp_data, field).numpy()).all(), (name, field, step)
            np.testing.assert_allclose(sim.wp_data.ctrl.numpy(), excitation.cpu().numpy(), atol=1e-6)
            max_force = max(max_force, float(np.abs(sim.wp_data.actuator_force.numpy()).max()))
            print('STEP', name, step + 1, 'activation_mean', float(sim.wp_data.act.numpy().mean()), flush=True)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    final_q = sim.wp_data.qpos.numpy().copy()
    final_act = sim.wp_data.act.numpy().copy()
    times = sim.wp_data.time.numpy().copy()
    np.testing.assert_allclose(times, steps * dt, atol=1e-4)
    assert max_force > 0
    assert float(np.max(np.abs(final_q - initial_q))) > 1e-5
    assert float(np.max(np.abs(final_q[0] - final_q[1]))) > 1e-5
    assert final_act.min() >= -1e-6 and final_act.max() <= 1.000001
    warnings = sim.wp_data.warning.numpy().copy() if hasattr(sim.wp_data, 'warning') else None
    # Reset just one environment; the other worlds must retain their muscle states.
    ids = torch.tensor([1], device='cuda:0', dtype=torch.long)
    sim.reset(ids)
    entity.reset(ids)
    wp.synchronize()
    after_act = sim.wp_data.act.numpy()
    np.testing.assert_allclose(after_act[1], 0, atol=1e-7)
    np.testing.assert_array_equal(after_act[[0, 2, 3]], final_act[[0, 2, 3]])
    result = dict(status='PASS', num_envs=nworld, muscles=int(model.nu), steps_per_env=steps,
        simulated_seconds_per_env=steps * dt, gpu_state_device=str(sim.wp_data.qpos.device),
        cuda_graph=bool(sim.use_cuda_graph), build_seconds=build_seconds,
        rollout_wall_seconds=elapsed, finite_states=True, control_mapping_verified=True,
        activation_range=[float(final_act.min()), float(final_act.max())],
        max_abs_actuator_force=max_force, max_abs_qpos_change=float(np.max(np.abs(final_q-initial_q))),
        selective_reset_verified=True, warning_data=None if warnings is None else warnings.tolist())
    print('RESULT', name, json.dumps(result), flush=True)
    return result

for name, relative in [('bimanual', 'arm/myoarm_bimanual.xml'), ('fullbody', 'body/myofullbody.xml')]:
    try:
        results['models'][name] = test_model(name, relative)
    except Exception as exc:
        results['models'][name] = dict(status='FAIL', error=repr(exc))
        traceback.print_exc()
    OUT.write_text(json.dumps(results, indent=2))
    gc.collect()
    torch.cuda.empty_cache()
print('SAVED', OUT, flush=True)
if any(r['status'] != 'PASS' for r in results['models'].values()):
    raise SystemExit(1)
