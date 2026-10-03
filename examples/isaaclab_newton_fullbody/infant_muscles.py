"""Batched antagonist muscle dynamics for MIMo joint actuators.

The equations and XML parameter interpretation follow MIMo's MIT-licensed
``mimoActuation/muscle.py``. This module computes torques; the Newton bridge
still needs an actuator path without MIMo's XML motor force limits.
"""

import torch


def _bump(length, lower, middle, upper):
    left = 0.5 * (lower + middle)
    right = 0.5 * (middle + upper)
    rising = 0.5 * (length - lower).square() / (left - lower) ** 2
    left_center = 1 - 0.5 * (length - middle).square() / (middle - left) ** 2
    right_center = 1 - 0.5 * (length - middle).square() / (right - middle) ** 2
    falling = 0.5 * (length - upper).square() / (upper - right) ** 2
    return torch.where(
        (length <= lower) | (length >= upper), 0.0,
        torch.where(length < left, rising,
                    torch.where(length < middle, left_center,
                                torch.where(length < right, right_center, falling))),
    )


class InfantMuscles:
    """Compute two-muscle activation, virtual fibers, and torque for each motor."""

    def __init__(self, model, worlds, device="cpu"):
        self.device = device
        self.worlds = worlds
        self.actuator_count = model.nu
        joint_ids = model.actuator_trnid[:, 0]
        self.qpos_ids = torch.as_tensor(model.jnt_qposadr[joint_ids].copy(), device=device)
        self.qvel_ids = torch.as_tensor(model.jnt_dofadr[joint_ids].copy(), device=device)
        self.spring = torch.as_tensor(model.qpos_spring[model.jnt_qposadr[joint_ids]].copy(),
                                      device=device, dtype=torch.float32)
        joint_range = torch.as_tensor(model.jnt_range[joint_ids].copy(), device=device,
                                      dtype=torch.float32)
        phi_min = joint_range[:, 0] - self.spring
        phi_max = joint_range[:, 1] - self.spring
        epsilon = 1e-8
        self.moment_negative = (0.3 + epsilon) / (phi_max - phi_min + epsilon)
        self.moment_positive = (0.3 + epsilon) / (phi_min - phi_max - epsilon)
        self.reference_negative = 0.75 - self.moment_negative * phi_min
        self.reference_positive = 0.75 - self.moment_positive * phi_max
        if model.nuser_actuator < 3:
            raise ValueError("MIMo actuator user fields must contain vmax and two fmax values")
        self.vmax = torch.as_tensor(model.actuator_user[:, 0].copy(), device=device,
                                    dtype=torch.float32)
        self.fmax_negative = torch.as_tensor(model.actuator_user[:, 1].copy(), device=device,
                                             dtype=torch.float32)
        self.fmax_positive = torch.as_tensor(model.actuator_user[:, 2].copy(), device=device,
                                             dtype=torch.float32)
        if torch.any(self.vmax <= 0):
            raise ValueError("MIMo vmax must be positive")
        self.activity = torch.zeros((worlds, 2 * self.actuator_count), device=device)

    def reset(self, world_ids=None):
        if world_ids is None:
            self.activity.zero_()
        else:
            self.activity[world_ids] = 0

    def step(self, qpos, qvel, action, timestep):
        if qpos.shape[0] != self.worlds or action.shape != self.activity.shape:
            raise ValueError("Muscle batch shapes do not match the number of worlds or actuators")
        self.activity = torch.clamp(self.activity + timestep *
                                    (torch.clamp(action, 0, 1) - self.activity) / 0.01, 0, 1)
        angle = qpos[:, self.qpos_ids] - self.spring
        velocity = qvel[:, self.qvel_ids]
        length_negative = angle * self.moment_negative + self.reference_negative
        length_positive = angle * self.moment_positive + self.reference_positive
        speed_negative = velocity * self.moment_negative
        speed_positive = velocity * self.moment_positive
        force_negative = self._force(length_negative, speed_negative,
                                     self.activity[:, :self.actuator_count])
        force_positive = self._force(length_positive, speed_positive,
                                     self.activity[:, self.actuator_count:])
        return -(self.moment_negative * force_negative * self.fmax_negative +
                 self.moment_positive * force_positive * self.fmax_positive)

    def _force(self, length, speed, activity):
        force_length = _bump(length, 0.5, 1.0, 1.6) + 0.15 * _bump(length, 0.5, 0.725, 0.95)
        effective_speed = speed / self.vmax
        force_velocity = torch.where(
            effective_speed < -1, 0.0,
            torch.where(effective_speed <= 0, (effective_speed + 1).square(),
                        torch.where(effective_speed <= 0.2,
                                    1.2 - (0.2 - effective_speed).square() / 0.2, 1.2)),
        )
        midpoint = 1.3
        passive = torch.where(
            length <= 1, 0.0,
            torch.where(length <= midpoint,
                        0.25 * 1.3 * ((length - 1) / (midpoint - 1)).pow(3),
                        0.25 * 1.3 * (1 + 3 * (length - midpoint) / (midpoint - 1))),
        )
        return force_length * force_velocity * activity + passive
