"""Candidate leg-only muscle residual basis; no trained policy or gait claim.

Use equality-reduced moment arms at standing and all reset-bank poses to exclude
muscles that directly actuate upper-body independent joints. Dynamic coupling
and baseline feedback can still move the upper body.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from project_joint_equalities import project_joint_equalities


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'outputs/isaac_velocity_g1/reference_candidates'
    m = mujoco.MjModel.from_xml_path(str(root / 'outputs/isaac_velocity/assets/model.xml'))
    d = mujoco.MjData(m)
    initial = np.load(root / 'outputs/isaac_velocity/assets/initial.npz')['qpos']
    bank = np.load(folder / 'projected_cycle_walking_states.npz')
    dependent = {int(m.eq_obj1id[e]) for e in range(m.neq)
                 if m.eq_active0[e] and int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT)}
    independent = [j for j in range(1, m.njnt) if j not in dependent]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in independent]
    lower = np.array([name.startswith(('hip_', 'knee_', 'ankle_', 'subtalar_', 'mtp_')) for name in joints])
    assert lower.sum() == 14
    eye = np.zeros((len(independent), m.nv))
    eye[np.arange(len(independent)), m.jnt_dofadr[independent]] = 1.
    maximum = np.zeros((len(independent), m.nu))
    for pose in np.vstack([initial, bank['qpos']]):
        q, _ = project_joint_equalities(m, pose, np.zeros(m.nv))
        _, tangent = project_joint_equalities(m, np.tile(q, (len(independent), 1)), eye)
        d.qpos[:] = q
        d.qvel[:] = 0.
        mujoco.mj_forward(m, d)
        moment = np.zeros((m.nu, m.nv))
        for actuator in range(m.nu):
            start = int(d.moment_rowadr[actuator])
            end = start + int(d.moment_rownnz[actuator])
            moment[actuator, d.moment_colind[start:end]] = d.actuator_moment[start:end]
        maximum = np.maximum(maximum, np.abs(tangent @ moment.T))
    leg = maximum[lower].max(0)
    upper = maximum[~lower].max(0)
    selected = np.flatnonzero((leg > 1e-6) & (upper < 1e-8))
    assert 0 < len(selected) < m.nu
    basis = np.eye(m.nu, dtype=np.float32)[selected]
    np.testing.assert_array_equal(basis @ basis.T, np.eye(len(selected)))
    names = np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(m.nu)])
    output = folder / 'leg_only_muscle_basis.npz'
    np.savez_compressed(output, basis=basis, actuator_names=names)
    report = dict(path=str(output), selected_count=len(selected), total_muscles=m.nu,
                  checked_poses=1 + len(bank['qpos']), selected_actuators=names[selected].tolist(),
                  independent_leg_joints=np.array(joints)[lower].tolist(),
                  maximum_sampled_upper_joint_moment_arm=float(upper[selected].max()),
                  existing_transform_scale=float(np.sqrt(m.nu / len(selected))),
                  trained=False, walking_success=False,
                  limitations='Zero direct upper-joint moment arms at sampled poses only. Frozen baseline remains state dependent; dynamic coupling can still destabilize trunk. Must validate GPU mapping and PPO before training.')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'selected_actuators'}, indent=2))


if __name__ == '__main__':
    main()
