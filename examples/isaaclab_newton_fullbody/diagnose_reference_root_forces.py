"""Inverse-dynamics root residuals of a kinematic reference; not feasibility proof.

Native soft contacts determine inverse forces from acceleration and penetration.
Height sensitivity and a standing control are therefore reported explicitly.
No policy, simulator training state, or source reference is changed.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from project_joint_equalities import project_joint_equalities


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / 'outputs/isaac_velocity_g1/reference_candidates'
    model = mujoco.MjModel.from_xml_path(str(root / 'outputs/isaac_velocity/assets/model.xml'))
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    initial = np.load(root / 'outputs/isaac_velocity/assets/initial.npz')['qpos']
    source = Path('/home/k_miyazawa/.musclemimic/caches/AMASS/MyoFullBody/gmr/KIT/314/walking_medium09_poses.npz')
    raw = np.load(source)
    assert raw['joint_names'].tolist() == [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    q, v = project_joint_equalities(model, raw['qpos'], raw['qvel'])
    acceleration = np.gradient(v, 1. / float(raw['frequency']), axis=0)
    assert np.isfinite(acceleration).all()
    velocity_from_pose = []
    for frame in range(443, 555):
        estimate = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, estimate, 2. / float(raw['frequency']), q[frame - 1], q[frame + 1])
        velocity_from_pose.append(estimate)
    velocity_error = np.asarray(velocity_from_pose) - v[443:555]
    velocity_check = {
        name: dict(rmse=float(np.sqrt(np.mean(velocity_error[:, indices] ** 2))),
                   max_abs=float(np.abs(velocity_error[:, indices]).max()))
        for name, indices in [('root_linear_mps', slice(0, 3)),
                              ('root_angular_radps', slice(3, 6)),
                              ('internal_radps', slice(6, None))]
    }
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'head')
    if head < 0:
        raise ValueError('Missing native head body')
    bodyweight = float(model.body_mass.sum()) * 9.81
    controls = []
    for label, pose, velocity in [('standing', initial, np.zeros(model.nv)),
                                  ('reference_450', q[450], v[450])]:
        for support in [0., .3]:
            mujoco.mj_resetData(model, data)
            data.qpos[:] = pose
            data.qvel[:] = velocity
            data.act[:] = .12
            data.ctrl[:] = .12
            data.xfrc_applied[head, 2] = support * bodyweight
            mujoco.mj_forward(model, data)
            applied = np.zeros(model.nv)
            mujoco.mj_applyFT(model, data, data.xfrc_applied[head, :3], np.zeros(3),
                             data.xipos[head], head, applied)
            expected = data.qfrc_actuator.copy() + applied
            mujoco.mj_inverse(model, data)
            error = data.qfrc_inverse - expected
            controls.append(dict(state=label, head_support_bodyweight=support,
                                 root_max_abs_error=float(np.abs(error[:6]).max()),
                                 all_dof_max_abs_error=float(np.abs(error).max())))
    if max(c['root_max_abs_error'] for c in controls) > 1e-3:
        raise RuntimeError(f'Forward/inverse root consistency check failed: {controls}')
    records = []
    for label, poses, velocities, accelerations in [
        ('standing', initial[None], np.zeros((1, model.nv)), np.zeros((1, model.nv))),
        ('projected_reference_443_555', q[443:555], v[443:555], acceleration[443:555]),
    ]:
        for height_offset in [-.005, 0., .005]:
            residuals = {0.: [], .3: []}
            contacts = []
            for pose, velocity, accel in zip(poses, velocities, accelerations):
                mujoco.mj_resetData(model, data)
                data.qpos[:] = pose
                data.qpos[2] += height_offset
                data.qvel[:] = velocity
                data.qacc[:] = accel
                mujoco.mj_inverse(model, data)
                contacts.append(int(data.ncon))
                for support in residuals:
                    applied = np.zeros(model.nv)
                    mujoco.mj_applyFT(model, data, np.array([0., 0., support * bodyweight]),
                                     np.zeros(3), data.xipos[head], head, applied)
                    residuals[support].append((data.qfrc_inverse - applied)[:6].copy())
            for support, values in residuals.items():
                values = np.asarray(values)
                linear = np.linalg.norm(values[:, :3], axis=1) / bodyweight
                angular = np.linalg.norm(values[:, 3:], axis=1)
                records.append(dict(reference=label, height_offset_m=height_offset,
                                    head_support_bodyweight=support, frames=len(values),
                                    contacts_mean=float(np.mean(contacts)),
                                    linear_residual_bodyweights_median=float(np.median(linear)),
                                    linear_residual_bodyweights_p95=float(np.percentile(linear, 95)),
                                    angular_residual_Nm_median=float(np.median(angular))))
    result = dict(mujoco_version=mujoco.__version__, source=str(source), bodyweight_N=bodyweight,
                  recorded_vs_central_pose_difference_velocity=velocity_check,
                  forward_inverse_controls=controls,
                  records=records, limitations=[
                      'CPU native MuJoCo inverse dynamics, not an Isaac forward rollout.',
                      'Reference acceleration is the finite difference of equality-projected recorded velocity.',
                      'Soft-contact force is highly sensitive to reference penetration. Residuals do not prove that nearby tracking is impossible.',
                      'No bounded muscle allocation or activation-dynamics solution is performed.',
                      'This diagnostic alone cannot establish the cause of PPO failure.',
                  ])
    output = folder / 'reference_root_force_audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
