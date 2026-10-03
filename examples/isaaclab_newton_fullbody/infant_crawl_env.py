"""Direct MuJoCo Warp infant task with antagonist muscles and contact feedback."""

import math
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from developmental_skin import REGIONS, SkinContactSensor
from infant_muscles import InfantMuscles
from infant_skin import infant_geom_region_ids


CRAWL_JOINTS = tuple(
    f"robot:{side}_{joint}"
    for side in ("right", "left")
    for joint in ("shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation", "elbow",
                  "hip1", "hip2", "hip3", "knee", "foot1", "foot2", "foot3")
)
PALM_JOINTS = CRAWL_JOINTS + tuple(
    f"robot:{side}_{joint}"
    for side in ("right", "left")
    for joint in ("hand1", "hand2", "hand3", "big_toe", "toes")
)
PALM_SENSOR_VERSION = "palm_metacarpal_group_v1"


def crawl_actuator_ids(model, joint_names=CRAWL_JOINTS):
    actuator_by_joint = {int(model.actuator_trnid[actuator_id, 0]): actuator_id
                         for actuator_id in range(model.nu)}
    result = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or joint_id not in actuator_by_joint:
            raise ValueError(f"Missing crawl actuator for {name}")
        result.append(actuator_by_joint[joint_id])
    return np.asarray(result, dtype=np.int64)


def antagonistic_muscle_action(drives, actuator_ids, actuator_count, baseline=0.2, amplitude=0.3):
    action = torch.full((drives.shape[0], 2 * actuator_count), baseline,
                        device=drives.device, dtype=drives.dtype)
    action[:, actuator_ids] = torch.clamp(baseline - amplitude * drives, 0, 1)
    action[:, actuator_ids + actuator_count] = torch.clamp(baseline + amplitude * drives, 0, 1)
    return action


def planar_tether_force(position, velocity, origin, stiffness, damping):
    if position.shape != origin.shape or velocity.shape != position.shape or position.shape[-1] != 2:
        raise ValueError("Planar tether needs matching XY vectors")
    if not math.isfinite(stiffness) or not math.isfinite(damping) or stiffness < 0 or damping < 0:
        raise ValueError("Planar tether gains must be nonnegative")
    return -stiffness * (position - origin) - damping * velocity


class InfantCrawlEnv:
    """Batch infant physics for posture-support and locomotion experiments."""

    def __init__(self, scene, pose, worlds, device="cuda:0",
                 controlled_joint_names=CRAWL_JOINTS, solver_iterations=None,
                 solver_ls_iterations=None, integrator=None, jacobian=None):
        if worlds < 1:
            raise ValueError("World count must be positive")
        self.device = device
        self.worlds = worlds
        self.controlled_joint_names = tuple(controlled_joint_names)
        if self.controlled_joint_names not in (CRAWL_JOINTS, PALM_JOINTS):
            raise ValueError("Unrecognized infant crawl joint set")
        self.source = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        with np.load(pose) as state:
            initial = state["qpos"].copy()
        if initial.shape != (self.source.nq,):
            raise ValueError("Crawl pose does not match the infant model")
        robot_joints = [joint_id for joint_id in range(self.source.njnt)
                        if (mujoco.mj_id2name(self.source, mujoco.mjtObj.mjOBJ_JOINT,
                                              joint_id) or "").startswith("robot:")]
        self.source.jnt_stiffness[robot_joints] = 0
        self.source.dof_damping[self.source.jnt_dofadr[robot_joints]] /= 20
        self.source.opt.jacobian = (mujoco.mjtJacobian.mjJAC_SPARSE
                                    if jacobian is None else jacobian)
        if solver_iterations is not None:
            if solver_iterations < 1:
                raise ValueError("Solver iterations must be positive")
            self.source.opt.iterations = solver_iterations
        if solver_ls_iterations is not None:
            if solver_ls_iterations < 1:
                raise ValueError("Line-search iterations must be positive")
            self.source.opt.ls_iterations = solver_ls_iterations
        if integrator is not None:
            self.source.opt.integrator = integrator
        source_data = mujoco.MjData(self.source)
        source_data.qpos[:] = initial
        mujoco.mj_forward(self.source, source_data)
        wp.set_device(device)
        self.model = mjw.put_model(self.source)
        self.data = mjw.put_data(self.source, source_data, nworld=worlds,
                                 nconmax=4096, njmax=8192)
        self.position = wp.to_torch(self.data.qpos)
        self.velocity = wp.to_torch(self.data.qvel)
        self.applied_force = wp.to_torch(self.data.qfrc_applied)
        self.initial = self.position.clone()
        self.muscles = InfantMuscles(self.source, worlds, device=device)
        self.skin = SkinContactSensor(self.source, self.model, self.data, device,
                                      region_ids=infant_geom_region_ids(self.source))
        self.floor_geom_id = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom_id < 0:
            raise ValueError("Crawl scene requires a named floor geom")
        support_geom_groups = []
        for part_group in (("hand1", "hand2", "hand3", "hand4",
                            "lfmetacarpal1", "lfmetacarpal2"), ("lower_leg1",)):
            for side in ("right", "left"):
                support_geom_groups.append([
                    mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"geom:{side}_{part}") for part in part_group])
        if min(geom_id for group in support_geom_groups for geom_id in group) < 0:
            raise ValueError("Crawl scene requires palm and shin collision geoms")
        self.skin.configure_ground_geoms(support_geom_groups)
        self.chest_id = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_BODY, "chest")
        self.head_id = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.palm_body_ids = [mujoco.mj_name2id(
            self.source, mujoco.mjtObj.mjOBJ_BODY, f"{side}_lfmetacarpal")
            for side in ("right", "left")]
        if min(self.palm_body_ids) < 0:
            raise ValueError("Crawl scene requires metacarpal bodies")
        self.root_id = mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_JOINT,
                                          "mimo_orientation")
        self.root_address = int(self.source.jnt_qposadr[self.root_id])
        self.root_dof_address = int(self.source.jnt_dofadr[self.root_id])
        self.root_tether_origin = self.position[:, self.root_address:self.root_address + 2].clone()
        self.body_weight = float(self.source.body_mass.sum() * -self.source.opt.gravity[2])
        self.crawl_actuators = torch.as_tensor(crawl_actuator_ids(self.source,
                                                                   self.controlled_joint_names),
                                                device=device, dtype=torch.long)
        crawl_joints = [mujoco.mj_name2id(self.source, mujoco.mjtObj.mjOBJ_JOINT, name)
                        for name in self.controlled_joint_names]
        self.crawl_qpos_ids = torch.as_tensor(self.source.jnt_qposadr[crawl_joints].copy(),
                                               device=device, dtype=torch.long)
        self.crawl_qvel_ids = torch.as_tensor(self.source.jnt_dofadr[crawl_joints].copy(),
                                               device=device, dtype=torch.long)

    def reset(self, positions=None):
        if positions is not None and positions.shape != self.initial.shape:
            raise ValueError("Reset positions must match the batched infant coordinates")
        self.position.copy_(self.initial if positions is None else positions)
        self.root_tether_origin = self.position[:, self.root_address:self.root_address + 2].clone()
        self.velocity.zero_()
        self.applied_force.zero_()
        self.data.ctrl.zero_()
        self.data.qacc_warmstart.zero_()
        self.muscles.reset()
        mjw.forward(self.model, self.data)

    def action_from_drives(self, drives, baseline=0.2, amplitude=0.3):
        if drives.shape != (self.worlds, len(self.crawl_actuators)):
            raise ValueError("Crawl drive array has the wrong shape")
        return antagonistic_muscle_action(drives, self.crawl_actuators, self.source.nu,
                                          baseline, amplitude)

    def feedback_drives(self, bias, position_gain, velocity_gain):
        if bias.shape != (self.worlds, len(self.crawl_qpos_ids)):
            raise ValueError("Crawl bias array has the wrong shape")
        position_error = (self.initial[:, self.crawl_qpos_ids] -
                          self.position[:, self.crawl_qpos_ids])
        return (bias + position_gain * position_error -
                velocity_gain * self.velocity[:, self.crawl_qvel_ids]).clamp(-1, 1)

    def step(self, action, physics_steps=10, root_vertical_unload=0.0,
             root_planar_stiffness=0.0, root_planar_damping=0.0):
        if action.shape != (self.worlds, 2 * self.source.nu) or physics_steps < 1:
            raise ValueError("Invalid infant muscle action or physics step count")
        if not 0 <= root_vertical_unload <= 1:
            raise ValueError("Root vertical unload must be between zero and one")
        if (not math.isfinite(root_planar_stiffness) or
                not math.isfinite(root_planar_damping) or
                root_planar_stiffness < 0 or root_planar_damping < 0):
            raise ValueError("Planar tether gains must be nonnegative")
        for _ in range(physics_steps):
            torque = self.muscles.step(self.position, self.velocity, action,
                                       float(self.source.opt.timestep))
            self.applied_force.zero_()
            self.applied_force[:, self.muscles.qvel_ids] = torque
            self.applied_force[:, self.root_dof_address + 2] = (
                root_vertical_unload * self.body_weight)
            if root_planar_stiffness or root_planar_damping:
                self.applied_force[:, self.root_dof_address:self.root_dof_address + 2] = (
                    planar_tether_force(
                        self.position[:, self.root_address:self.root_address + 2],
                        self.velocity[:, self.root_dof_address:self.root_dof_address + 2],
                        self.root_tether_origin, root_planar_stiffness,
                        root_planar_damping))
            mjw.step(self.model, self.data)
            if not (torch.isfinite(self.position).all() and torch.isfinite(self.velocity).all()):
                raise RuntimeError("Nonfinite infant crawl state")
            if np.any(data_flags := self.data.overflow.numpy().astype(np.int64) & 511):
                raise RuntimeError(f"MuJoCo Warp capacity overflow: {data_flags.tolist()}")
        return self.measure()

    def measure(self):
        touch = self.skin.read().clone()
        ground_touch = self.skin.read_ground(self.floor_geom_id).clone()
        foot_force, foot_center_xy = self.skin.read_foot_pressure(self.floor_geom_id)
        palm_shin_force = self.skin.read_ground_geoms(self.floor_geom_id).clone()
        positions = wp.to_torch(self.data.xpos)
        support_ids = [REGIONS.index(name) for name in
                       ("right_hand", "left_hand", "right_shin", "left_shin")]
        return {
            "chest_height": positions[:, self.chest_id, 2].clone(),
            "head_height": positions[:, self.head_id, 2].clone(),
            "forward_displacement": (self.position[:, self.root_address] -
                                     self.initial[:, self.root_address]).clone(),
            "support_force": ground_touch[:, support_ids],
            "touch": touch,
            "ground_touch": ground_touch,
            "foot_force": foot_force,
            "foot_center_xy": foot_center_xy,
            "palm_shin_force": palm_shin_force,
            "palm_relative_x": (positions[:, self.palm_body_ids, 0] -
                                self.position[:, self.root_address, None]).clone(),
        }
