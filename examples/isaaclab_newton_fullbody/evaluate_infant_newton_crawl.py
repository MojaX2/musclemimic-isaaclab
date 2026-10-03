"""Evaluate an infant crawl policy with Isaac Lab's Newton/MuJoCo Warp backend."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from newton.solvers import SolverMuJoCo

from check_infant_isaaclab import (InfantManager, InfantSolver, _write_muscle_torques,
                                  audit_geom_parameters)
from developmental_skin import REGIONS, SkinContactSensor
from evolve_infant_crawl_cpg import evaluate
from infant_crawl_env import (CRAWL_JOINTS, PALM_JOINTS, PALM_SENSOR_VERSION,
                              InfantCrawlEnv, antagonistic_muscle_action,
                              crawl_actuator_ids, planar_tether_force)
from infant_crawl_stepper import TactileStepper
from infant_muscles import InfantMuscles
from infant_skin import infant_geom_region_ids
from infant_tactile_features import TactileFeatureEncoder
from train_infant_crawl_ppo import SupportPolicy, jittered_positions, observe


def mapped_names(model, object_type, count):
    names = {}
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name is None:
            continue
        original = re.sub(r"_\d+$", "", name.rsplit("/", 1)[-1])
        if original in names:
            raise ValueError(f"Ambiguous Newton object name: {original}")
        names[original] = object_id
    return names


def newton_body_id(model, source_name):
    matches = [body_id for body_id in range(model.nbody)
               if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                              body_id)) is not None and
               (name == source_name or name.endswith("_" + source_name))]
    if len(matches) != 1:
        raise ValueError(f"Expected one Newton body for {source_name}, found {matches}")
    return matches[0]


def joint_anchor_audit(source, converted):
    converted_by_address = {int(converted.jnt_qposadr[joint_id]): joint_id
                            for joint_id in range(converted.njnt)}
    mismatches = []
    for joint_id in range(source.njnt):
        if int(source.jnt_type[joint_id]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                 int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        address = int(source.jnt_qposadr[joint_id])
        converted_id = converted_by_address.get(address)
        if converted_id is None:
            raise ValueError(f"Converted model is missing joint coordinate {address}")
        error = float(np.linalg.norm(source.jnt_pos[joint_id] -
                                     converted.jnt_pos[converted_id]))
        if error > 1e-5:
            mismatches.append({"joint": mujoco.mj_id2name(source,
                mujoco.mjtObj.mjOBJ_JOINT, joint_id), "anchor_error_m": error})
    mismatches.sort(key=lambda item: item["anchor_error_m"], reverse=True)
    return {"mismatch_count": len(mismatches), "max_error_m":
            mismatches[0]["anchor_error_m"] if mismatches else 0.0,
            "largest_mismatches": mismatches[:20]}


def dynamics_audit(source, converted):
    if source.nv != converted.nv:
        raise ValueError("Converted model has a different velocity dimension")
    result = {}
    for field in ("dof_armature", "dof_frictionloss", "dof_damping",
                  "jnt_stiffness", "jnt_range", "qpos_spring"):
        original = np.array(getattr(source, field), copy=True)
        restored = np.asarray(getattr(converted, field))
        if field in ("dof_damping", "jnt_stiffness"):
            for joint_id in range(source.njnt):
                name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if name and name.startswith("robot:"):
                    if field == "dof_damping":
                        original[source.jnt_dofadr[joint_id]] /= 20
                    else:
                        original[joint_id] = 0
        if original.shape != restored.shape:
            result[field] = {"source_shape": original.shape,
                             "converted_shape": restored.shape}
            continue
        difference = np.abs(original - restored)
        result[field] = {"mismatch_count": int(np.count_nonzero(difference > 1e-6)),
                         "max_absolute_error": float(difference.max(initial=0))}
    body_errors = []
    for source_id in range(1, source.nbody):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, source_id)
        if name is None:
            continue
        converted_id = newton_body_id(converted, name)
        errors = {field: float(np.max(np.abs(getattr(source, field)[source_id] -
                                           getattr(converted, field)[converted_id])))
                  for field in ("body_mass", "body_inertia", "body_ipos",
                                "body_iquat", "body_pos", "body_quat")}
        if max(errors.values()) > 1e-6:
            body_errors.append({"body": name, "errors": errors})
    result["body_mass_inertia"] = {"mismatch_count": len(body_errors),
                                   "largest_mismatches": sorted(body_errors,
                                       key=lambda item: max(item["errors"].values()),
                                       reverse=True)[:12]}
    return result


def cpu_contact_audit(scene, converted, positions):
    source = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    for joint_id in range(source.njnt):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name and name.startswith("robot:"):
            source.jnt_stiffness[joint_id] = 0
            source.dof_damping[source.jnt_dofadr[joint_id]] /= 20
    for field in ("integrator", "cone", "jacobian", "solver", "iterations",
                  "ls_iterations", "tolerance", "disableflags", "enableflags"):
        setattr(source.opt, field, getattr(converted.opt, field))
    models = (source, converted)
    option_differences = {}
    for name in dir(source.opt):
        if name.startswith("_"):
            continue
        try:
            original = np.asarray(getattr(source.opt, name))
            restored = np.asarray(getattr(converted.opt, name))
            if (original.dtype.kind in "biuf" and original.shape == restored.shape and
                    not np.allclose(original, restored, rtol=0, atol=1e-9)):
                option_differences[name] = {"source": original.tolist(),
                                            "converted": restored.tolist()}
        except (AttributeError, TypeError, ValueError):
            continue
    data = []
    matrices = []
    floor_force = []
    selected_contacts = []
    for model in models:
        state = mujoco.MjData(model)
        state.qpos[:] = positions[0].cpu().numpy()
        mujoco.mj_forward(model, state)
        data.append(state)
        matrix = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, state, matrix)
        matrices.append(matrix)
        floor_id = mapped_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)["floor"]
        hand_id = mapped_names(model, mujoco.mjtObj.mjOBJ_GEOM,
                               model.ngeom)["geom:right_hand2"]
        total = 0.0
        selected = []
        for contact_id in range(state.ncon):
            contact = state.contact[contact_id]
            if floor_id not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, state, contact_id, force)
            total += max(float(force[0]), 0.0)
            if hand_id in (contact.geom1, contact.geom2):
                address = int(contact.efc_address)
                selected.append({"distance_m": float(contact.dist),
                                 "position_m": contact.pos.tolist(),
                                 "solref": contact.solref.tolist(),
                                 "solimp": contact.solimp.tolist(),
                                 "friction": contact.friction.tolist(),
                                 "efc_address": address,
                                 "efc_D": (state.efc_D[address:address + contact.dim].tolist()
                                           if address >= 0 else []),
                                 "efc_KBIP": (state.efc_KBIP[address:address +
                                                                contact.dim].tolist()
                                              if address >= 0 else []),
                                 "normal_force_n": float(force[0])})
        floor_force.append(total)
        selected_contacts.append(selected)
    return {"mass_matrix_max_absolute_error": float(np.max(np.abs(matrices[0] - matrices[1]))),
            "bias_force_max_absolute_error": float(np.max(np.abs(data[0].qfrc_bias -
                                                               data[1].qfrc_bias))),
            "tendon_jacobian_max_absolute_error": float(np.max(np.abs(data[0].ten_J -
                                                                   data[1].ten_J))),
            "option_differences": option_differences,
            "floor_contact_force_n": floor_force,
            "contact_count": [state.ncon for state in data],
            "right_hand2_contacts": selected_contacts}


class SupportOnlyController:
    def __init__(self, worlds, device):
        self.parameters = torch.zeros((worlds, 6), device=device)

    def reset(self):
        pass

    def __call__(self, drives, measure):
        return drives

    def metrics(self, control_steps):
        return {}


class ShuffledActionPredictor:
    def __init__(self, encoder):
        self.encoder = encoder
        self.active_regions = encoder.active_regions

    @property
    def body_weight(self):
        return self.encoder.body_weight

    def predict(self, environment, measure, muscle_action):
        return self.encoder.predict(environment, measure, muscle_action.roll(1, dims=0))


class BlendedActor:
    def __init__(self, primary, secondary, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Policy blend fraction must be between zero and one")
        self.primary = primary
        self.secondary = secondary
        self.fraction = fraction

    def actor(self, observation):
        return ((1 - self.fraction) * self.primary.actor(observation) +
                self.fraction * self.secondary.actor(observation))


class InfantNewtonCrawlEnv:
    def __init__(self, source, pose, worlds, simulation,
                 controlled_joint_names=PALM_JOINTS):
        self.source = source
        self.worlds = worlds
        self.device = "cuda:0"
        self.simulation = simulation
        self.controlled_joint_names = tuple(controlled_joint_names)
        if self.controlled_joint_names not in (CRAWL_JOINTS, PALM_JOINTS):
            raise ValueError("Unrecognized infant Newton joint set")
        solver = NewtonManager._solver
        self.model = solver.mjw_model
        self.collision_model = solver.mj_model
        self.data = solver.mjw_data
        self.position = wp.to_torch(self.data.qpos)
        self.velocity = wp.to_torch(self.data.qvel)
        with np.load(pose) as saved:
            initial = np.asarray(saved["qpos"], dtype=np.float32)
        if initial.shape != (source.nq,) or not np.isfinite(initial).all():
            raise ValueError("Initial crawl pose does not match the infant model")
        self.initial = torch.as_tensor(initial, device=self.device).expand(worlds, -1).clone()
        self.muscles = InfantMuscles(source, worlds, device=self.device)
        self.joint_force = NewtonManager.get_control().joint_f
        self.joint_force_torch = wp.to_torch(self.joint_force).reshape(worlds, source.nv)
        self.muscle_dof_ids = wp.array(self.muscles.qvel_ids.cpu().numpy().astype(np.int32),
                                       dtype=wp.int32, device=self.device)
        self.crawl_actuators = torch.as_tensor(crawl_actuator_ids(source,
                                                                  self.controlled_joint_names),
                                               device=self.device, dtype=torch.long)
        source_joint_ids = [mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name)
                            for name in self.controlled_joint_names]
        self.crawl_qpos_ids = torch.as_tensor(source.jnt_qposadr[source_joint_ids].copy(),
                                              device=self.device, dtype=torch.long)
        self.crawl_qvel_ids = torch.as_tensor(source.jnt_dofadr[source_joint_ids].copy(),
                                              device=self.device, dtype=torch.long)
        root_id = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, "mimo_orientation")
        self.root_address = int(source.jnt_qposadr[root_id])
        self.root_dof_address = int(source.jnt_dofadr[root_id])
        self.body_weight = float(source.body_mass.sum() * -source.opt.gravity[2])
        geom_ids = mapped_names(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                solver.mj_model.ngeom)
        self.floor_geom_id = geom_ids["floor"]
        self.chest_id = newton_body_id(solver.mj_model, "chest")
        self.head_id = newton_body_id(solver.mj_model, "head")
        self.palm_body_ids = [newton_body_id(solver.mj_model, f"{side}_lfmetacarpal")
                              for side in ("right", "left")]
        self.skin = SkinContactSensor(solver.mj_model, self.model, self.data, self.device,
                                      region_ids=infant_geom_region_ids(solver.mj_model))
        groups = []
        for parts in (("hand1", "hand2", "hand3", "hand4", "lfmetacarpal1",
                       "lfmetacarpal2"), ("lower_leg1",)):
            for side in ("right", "left"):
                groups.append([geom_ids[f"geom:{side}_{part}"] for part in parts])
        self.skin.configure_ground_geoms(groups)
        self.reset()

    def reset(self, positions=None):
        target = self.initial if positions is None else positions
        if target.shape != self.initial.shape:
            raise ValueError("Reset positions must match the batched infant coordinates")
        self.position.copy_(target)
        self.root_tether_origin = self.position[:, self.root_address:self.root_address + 2].clone()
        self.velocity.zero_()
        self.data.qacc_warmstart.zero_()
        self.data.ctrl.zero_()
        self.joint_force.zero_()
        self.muscles.reset()
        self.solver_limit_steps = 0
        self.model.opt.timestep.fill_(float(self.source.opt.timestep))
        self.collision_model.opt.timestep = float(self.source.opt.timestep)
        mjw.forward(self.model, self.data)

    def action_from_drives(self, drives, baseline=0.2, amplitude=0.3):
        if drives.shape != (self.worlds, len(self.controlled_joint_names)):
            raise ValueError("Infant drive array has the wrong shape")
        return antagonistic_muscle_action(drives, self.crawl_actuators, self.source.nu,
                                          baseline, amplitude)

    def feedback_drives(self, bias, position_gain, velocity_gain):
        if bias.shape != (self.worlds, len(self.controlled_joint_names)):
            raise ValueError("Infant feedback bias has the wrong shape")
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
            self.joint_force.zero_()
            wp.launch(_write_muscle_torques, dim=(self.worlds, self.source.nu),
                      inputs=[wp.from_torch(torque.contiguous()), self.muscle_dof_ids,
                              self.source.nv], outputs=[self.joint_force], device=self.device)
            self.joint_force_torch[:, self.root_dof_address + 2] = (
                root_vertical_unload * self.body_weight)
            if root_planar_stiffness or root_planar_damping:
                self.joint_force_torch[:, self.root_dof_address:self.root_dof_address + 2] = (
                    planar_tether_force(
                        self.position[:, self.root_address:self.root_address + 2],
                        self.velocity[:, self.root_dof_address:self.root_dof_address + 2],
                        self.root_tether_origin, root_planar_stiffness,
                        root_planar_damping))
            self.simulation.step(render=False)
            if not (torch.isfinite(self.position).all() and torch.isfinite(self.velocity).all()):
                raise RuntimeError("Nonfinite Newton infant crawl state")
            overflow = self.data.overflow.numpy().astype(np.int64)
            if np.any(overflow & 511):
                raise RuntimeError(f"Newton MuJoCo Warp capacity overflow: {overflow.tolist()}")
            self.solver_limit_steps += int(np.any(overflow & 1536))
        return self.measure()

    def measure(self):
        touch = self.skin.read().clone()
        ground_touch = self.skin.read_ground(self.floor_geom_id).clone()
        foot_force, foot_center_xy = self.skin.read_foot_pressure(self.floor_geom_id)
        palm_shin_force = self.skin.read_ground_geoms(self.floor_geom_id).clone()
        positions = wp.to_torch(self.data.xpos)
        support_ids = [REGIONS.index(name) for name in
                       ("right_hand", "left_hand", "right_shin", "left_shin")]
        return {"chest_height": positions[:, self.chest_id, 2].clone(),
                "head_height": positions[:, self.head_id, 2].clone(),
                "forward_displacement": (self.position[:, self.root_address] -
                                         self.initial[:, self.root_address]).clone(),
                "support_force": ground_touch[:, support_ids],
                "touch": touch,
                "ground_touch": ground_touch,
                "foot_force": foot_force.clone(),
                "foot_center_xy": foot_center_xy.clone(),
                "palm_shin_force": palm_shin_force,
                "palm_relative_x": (positions[:, self.palm_body_ids, 0] -
                                    self.position[:, self.root_address, None]).clone()}


def evaluate_policy(environment, checkpoint, args):
    policy = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to("cuda:0")
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    if args.blend_checkpoint is not None:
        secondary_checkpoint = torch.load(args.blend_checkpoint, map_location="cpu",
                                          weights_only=True)
        for name in ("observation_size", "joint_names", "palm_sensor_version",
                     "use_touch", "stepper_observation_version",
                     "stepper_residual_strength", "stepper_parameters"):
            if secondary_checkpoint.get(name) != checkpoint.get(name):
                raise ValueError(f"Blended policy differs in {name}")
        for name in ("world_features", "stepper_reference_policy"):
            original = checkpoint.get(name)
            comparison = secondary_checkpoint.get(name)
            if (original is None) != (comparison is None):
                raise ValueError(f"Blended policy differs in {name}")
            if original is not None:
                original_state = original["state_dict"] if name == "world_features" else original
                compared_state = comparison["state_dict"] if name == "world_features" else comparison
                if (original_state.keys() != compared_state.keys() or
                        any(not torch.equal(value, compared_state[key])
                            for key, value in original_state.items())):
                    raise ValueError(f"Blended policy differs in {name}")
        secondary = SupportPolicy(checkpoint["observation_size"], len(PALM_JOINTS)).to("cuda:0")
        secondary.load_state_dict(secondary_checkpoint["policy"])
        secondary.eval()
        policy = BlendedActor(policy, secondary, args.blend_fraction)
    controller = SupportOnlyController(args.worlds, "cuda:0")
    if args.stepper_parameters is not None:
        with np.load(args.stepper_parameters) as saved:
            parameters = np.asarray(saved["parameters"], dtype=np.float32)
        if parameters.shape != (6,) or not np.isfinite(parameters).all():
            raise ValueError("Stepper parameters must be six finite values")
        saved_parameters = checkpoint.get("stepper_parameters")
        if saved_parameters is not None and not np.allclose(parameters, saved_parameters,
                                                             rtol=0, atol=1e-6):
            raise ValueError("Stepper parameters do not match the checkpoint")
        if args.stepper_parameter_override is not None:
            parameters = np.asarray(args.stepper_parameter_override, dtype=np.float32)
            if (not np.isfinite(parameters).all() or
                    (parameters[:4] < 0).any() or (parameters[:4] > 1).any() or
                    (np.abs(parameters[4:]) > 1).any()):
                raise ValueError("Stepper parameter override exceeds the drive range")
        residual_strength = (checkpoint.get("stepper_residual_strength", 0.0)
                             if args.stepper_residual_strength is None else
                             args.stepper_residual_strength)
        if not 0 <= residual_strength <= 1:
            raise ValueError("Stepper residual strength must be between zero and one")
        brace_strength = (checkpoint.get("stance_brace_strength", 0.0)
                          if args.stance_brace_strength is None else args.stance_brace_strength)
        controller = TactileStepper(torch.as_tensor(parameters, device="cuda:0").expand(
            args.worlds, -1), absolute_targets=True, forward_reach=True,
            strong_plant=True, leg_push=True,
            stance_loss_grace_steps=checkpoint.get("stance_loss_grace_steps", 2),
            max_plant_steps=args.max_plant_steps,
            stance_brace_strength=brace_strength,
            alternate_on_abort=args.alternate_on_abort,
            residual_strength=residual_strength,
            min_reach_m=(checkpoint.get("stepper_min_reach_m", 0.0)
                         if args.stepper_min_reach_m is None else args.stepper_min_reach_m))
    reference_policy = None
    if getattr(controller, "residual_strength", 0) > 0:
        reference_policy = SupportPolicy(checkpoint["observation_size"],
                                         len(PALM_JOINTS)).to("cuda:0")
        reference_policy.load_state_dict(
            checkpoint.get("stepper_reference_policy") or checkpoint["policy"])
        reference_policy.eval().requires_grad_(False)
    observation_version = checkpoint.get("stepper_observation_version")
    if observation_version not in (None, 1):
        raise ValueError("Unsupported stepper observation version")
    if observation_version == 1 and args.stepper_parameters is None:
        raise ValueError("Stepper observations require --stepper-parameters")
    feature_snapshot = checkpoint.get("world_features")
    if feature_snapshot is not None and feature_snapshot.get("kind") != "tactile_next_contact_v1":
        raise ValueError("Unsupported Newton crawl feature encoder")
    world_features = (TactileFeatureEncoder.restore_snapshot(feature_snapshot,
                                                               environment.device)
                      if feature_snapshot is not None else None)
    if args.shuffle_tactile_feature_actions:
        if world_features is None or args.worlds < 2:
            raise ValueError("Shuffled tactile feature actions need two worlds and an encoder")
        world_features.shuffle_actions = True
    audit_predictors = None
    if args.tactile_audit_checkpoint is not None:
        if args.worlds < 2:
            raise ValueError("Action-shuffle audit requires at least two worlds")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 4000)
            learned_predictor = TactileFeatureEncoder.from_path(
                args.tactile_audit_checkpoint, environment.device,
                normalize_features=True)
            audit_predictors = {"learned": learned_predictor,
                                "learned_shuffled_action": ShuffledActionPredictor(
                                    learned_predictor),
                                "random": TactileFeatureEncoder.from_path(
                                    args.tactile_audit_checkpoint, environment.device,
                                    random_features=True, normalize_features=True)}
    def augment_observation(observation, stepper):
        return torch.cat((observation, stepper.observation()), dim=-1)
    observation_augment = augment_observation if observation_version == 1 else None
    torch.manual_seed(args.seed)
    positions = jittered_positions(environment, 0.05, 0.01)
    environment.reset(positions)
    initial_measure = environment.measure()
    base_observation_size = observe(environment, initial_measure,
                                    checkpoint["use_touch"]).shape[-1]
    expected_size = base_observation_size + (controller.observation_size
                                             if observation_version == 1 else 0)
    expected_size += world_features.feature_size if world_features is not None else 0
    if checkpoint["observation_size"] != expected_size:
        raise ValueError("Checkpoint observation size does not match stepper features")
    with torch.no_grad():
        initial_observation = observe(environment, initial_measure, checkpoint["use_touch"])
        if world_features is not None:
            initial_observation = torch.cat((initial_observation,
                                             torch.zeros((args.worlds, world_features.feature_size),
                                                         device=environment.device)), dim=-1)
        if observation_augment is not None:
            controller.reset()
            initial_observation = observation_augment(initial_observation, controller)
        initial_drive = policy.actor(initial_observation)
    initial_diagnostics = {
        "chest_height_m": float(initial_measure["chest_height"].mean()),
        "head_height_m": float(initial_measure["head_height"].mean()),
        "palm_shin_force_n": initial_measure["palm_shin_force"].mean(0).cpu().tolist(),
        "ground_touch_n": initial_measure["ground_touch"].mean(0).cpu().tolist(),
        "actor_drive_mean": float(initial_drive.mean()),
    }
    collision_model = getattr(environment, "collision_model", environment.source)
    geom_ids = mapped_names(collision_model, mujoco.mjtObj.mjOBJ_GEOM,
                            collision_model.ngeom)
    geom_position = wp.to_torch(environment.data.geom_xpos)
    initial_diagnostics["geom_center_height_m"] = {
        name: float(geom_position[:, geom_ids[name], 2].mean())
        for name in ("geom:right_hand1", "geom:left_hand1",
                     "geom:right_lower_leg1", "geom:left_lower_leg1")}
    body_position = wp.to_torch(environment.data.xpos)
    initial_diagnostics["body_center_height_m"] = {
        name: float(body_position[:, newton_body_id(collision_model, name), 2].mean())
        for name in ("right_hand", "left_hand", "right_lower_leg", "left_lower_leg")}
    initial_diagnostics["geom_local_position_m"] = {
        name: collision_model.geom_pos[geom_ids[name]].tolist()
        for name in ("geom:right_hand1", "geom:left_hand1",
                     "geom:right_lower_leg1", "geom:left_lower_leg1")}
    initial_diagnostics["body_local_position_m"] = {
        name: collision_model.body_pos[newton_body_id(collision_model, name)].tolist()
        for name in ("right_hand", "left_hand", "right_lower_arm", "left_lower_arm")}
    initial_diagnostics["joint_anchor_audit"] = joint_anchor_audit(
        environment.source, collision_model)
    initial_diagnostics["dynamics_audit"] = dynamics_audit(
        environment.source, collision_model)
    initial_diagnostics["geom_parameter_audit"] = audit_geom_parameters(
        environment.source, collision_model)
    initial_diagnostics["selected_geom_parameters"] = {}
    for name in ("floor", "geom:right_hand1", "geom:left_hand1",
                 "geom:right_hand2", "geom:left_hand2",
                 "geom:right_lower_leg1", "geom:left_lower_leg1"):
        source_id = mujoco.mj_name2id(environment.source,
                                      mujoco.mjtObj.mjOBJ_GEOM, name)
        converted_id = geom_ids[name]
        initial_diagnostics["selected_geom_parameters"][name] = {}
        for field in ("geom_type", "geom_size", "geom_rbound", "geom_margin",
                      "geom_gap", "geom_priority", "geom_solmix"):
            original = np.asarray(getattr(environment.source, field)[source_id])
            restored = np.asarray(getattr(collision_model, field)[converted_id])
            initial_diagnostics["selected_geom_parameters"][name][field] = {
                "source": original.tolist(), "converted": restored.tolist()}
    initial_diagnostics["total_mass_kg"] = {
        "source": float(mujoco.mj_getTotalmass(environment.source)),
        "converted": float(mujoco.mj_getTotalmass(collision_model))}
    initial_diagnostics["solver_options"] = {
        name: int(getattr(collision_model.opt, name))
        for name in ("integrator", "cone", "jacobian", "solver", "iterations",
                     "ls_iterations", "disableflags", "enableflags")}
    if args.cpu_contact_audit:
        initial_diagnostics["cpu_contact_audit"] = cpu_contact_audit(
            args.scene, collision_model, positions)
    initial_diagnostics["wrist_joint_metadata"] = {}
    joint_ids = {int(collision_model.jnt_qposadr[joint_id]): joint_id
                 for joint_id in range(collision_model.njnt)}
    for side in ("right", "left"):
        for index in (1, 2, 3):
            joint_name = f"robot:{side}_hand{index}"
            source_id = mujoco.mj_name2id(environment.source,
                                          mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_id = joint_ids[int(environment.source.jnt_qposadr[source_id])]
            initial_diagnostics["wrist_joint_metadata"][joint_name] = {
                "qpos_address": int(collision_model.jnt_qposadr[joint_id]),
                "position": collision_model.jnt_pos[joint_id].tolist(),
                "axis": collision_model.jnt_axis[joint_id].tolist(),
            }
    active = int(environment.data.nacon.numpy()[0])
    contact_geoms = environment.data.contact.geom.numpy()[:active]
    floor_id = geom_ids["floor"]
    region_ids = infant_geom_region_ids(collision_model)
    ground_contact_counts = np.zeros(len(REGIONS), dtype=np.int32)
    for first, second in contact_geoms:
        if first == floor_id:
            region = region_ids[second]
        elif second == floor_id:
            region = region_ids[first]
        else:
            continue
        if region >= 0:
            ground_contact_counts[region] += 1
    initial_diagnostics["ground_contact_counts"] = ground_contact_counts.tolist()
    contact_force = environment.skin.force.numpy()[:active, 0]
    contact_distance = environment.data.contact.dist.numpy()[:active]
    contact_world = environment.data.contact.worldid.numpy()[:active]
    contact_position = environment.data.contact.pos.numpy()[:active]
    contact_frame = environment.data.contact.frame.numpy()[:active]
    contact_solref = environment.data.contact.solref.numpy()[:active]
    contact_friction = environment.data.contact.friction.numpy()[:active]
    initial_diagnostics["ground_contact_details_world0"] = {}
    for name in ("geom:right_hand1", "geom:left_hand1",
                 "geom:right_hand2", "geom:left_hand2",
                 "geom:right_lfmetacarpal1", "geom:left_lfmetacarpal1",
                 "geom:right_lower_leg1", "geom:left_lower_leg1"):
        geom_id = geom_ids[name]
        matching = (((contact_geoms[:, 0] == floor_id) &
                     (contact_geoms[:, 1] == geom_id)) |
                    ((contact_geoms[:, 1] == floor_id) &
                     (contact_geoms[:, 0] == geom_id))) & (contact_world == 0)
        positive = matching & (contact_force > 1e-6)
        initial_diagnostics["ground_contact_details_world0"][name] = {
            "candidate_count": int(matching.sum()),
            "force_positive_count": int(positive.sum()),
            "normal_force_n": float(contact_force[positive].sum()),
            "minimum_distance_m": (float(contact_distance[matching].min())
                                   if matching.any() else None),
            "points": [{"distance_m": float(contact_distance[contact_id]),
                        "position_m": contact_position[contact_id].tolist(),
                        "normal": contact_frame[contact_id, 0].tolist(),
                        "solref": contact_solref[contact_id].tolist(),
                        "friction": contact_friction[contact_id].tolist(),
                        "force_n": float(contact_force[contact_id])}
                       for contact_id in np.flatnonzero(matching)[:12]]}
    transition_records = [] if args.save_tactile_transitions else None
    recorded_positions = [] if args.save_qpos else None
    result = evaluate(environment, policy, None, positions, args.control_steps,
                      args.physics_per_control, checkpoint["use_touch"],
                      gait_objective=True, record_trace=args.record_trace,
                      drive_controller=controller,
                      observation_augment=observation_augment,
                      world_features=world_features,
                      reference_policy=reference_policy,
                      audit_predictors=audit_predictors,
                      transition_records=transition_records,
                      recorded_positions=recorded_positions)
    if recorded_positions is not None:
        args.save_qpos.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_qpos,
                            qpos=np.stack(recorded_positions),
                            initial_qpos=positions.cpu().numpy(),
                            dt=args.physics_per_control * float(environment.source.opt.timestep))
    if transition_records is not None:
        args.save_tactile_transitions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_tactile_transitions, **{
            name: np.stack([record[name] for record in transition_records])
            for name in ("qpos_before", "current", "tactile", "action", "next_tactile")})
    summary = {name: float(value.float().mean()) for name, value in result.items()
               if not name.startswith("trace_")}
    per_world_names = ("forward_displacement_m", "elevated_fraction",
                       "minimum_support_fraction", "final_quarter_support_fraction",
                       "pelvis_fraction", "hand_switches", "shin_switches",
                       "legacy_crawl_success", "crawl_success")
    report = {"scene": str(args.scene.resolve()), "pose": str(args.pose.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "blend_checkpoint": (str(args.blend_checkpoint.resolve())
                                 if args.blend_checkpoint else None),
            "blend_fraction": args.blend_fraction,
            "stepper_observation_version": observation_version,
            "world_features": (feature_snapshot.get("kind") if feature_snapshot else None),
            "random_tactile_features": (feature_snapshot.get("random_features")
                                        if feature_snapshot else None),
            "tactile_representation": (feature_snapshot.get("representation", "hidden")
                                        if feature_snapshot else None),
            "contact_blend": (feature_snapshot.get("contact_blend")
                               if feature_snapshot else None),
            "tactile_prediction_horizon_control_steps": (
                feature_snapshot.get("prediction_horizon_control_steps", 1)
                if feature_snapshot else None),
            "tactile_feature_actions_shuffled": args.shuffle_tactile_feature_actions,
            "tactile_audit_checkpoint": (str(args.tactile_audit_checkpoint.resolve())
                                         if args.tactile_audit_checkpoint else None),
            "tactile_transitions": (str(args.save_tactile_transitions.resolve())
                                    if args.save_tactile_transitions else None),
            "stepper_parameters": (str(args.stepper_parameters.resolve())
                                   if args.stepper_parameters else None),
            "stepper_parameter_override": args.stepper_parameter_override,
            "stepper_residual_strength": (controller.residual_strength
                                           if args.stepper_parameters is not None else None),
            "stepper_min_reach_m": (controller.min_reach_m
                                    if args.stepper_parameters is not None else None),
            "max_plant_steps": args.max_plant_steps,
            "stance_brace_strength": (controller.stance_brace_strength
                                      if args.stepper_parameters is not None else None),
            "alternate_on_abort": args.alternate_on_abort,
            "backend": args.backend, "worlds": args.worlds, "seed": args.seed,
            "direct_match_newton_solver": args.direct_match_newton_solver,
            "newton_cone": args.newton_cone,
            "newton_multiccd": args.newton_multiccd,
            "newton_tolerance": args.newton_tolerance,
            "deterministic": args.deterministic,
            "duration_s": args.control_steps * args.physics_per_control *
                          float(environment.source.opt.timestep),
            "initial_pose_sha256": hashlib.sha256(positions.cpu().numpy().tobytes()).hexdigest(),
            "initial_diagnostics": initial_diagnostics,
            "mean": summary,
            "per_world": {name: result[name].detach().cpu().tolist()
                          for name in per_world_names},
            "crawl_success_count": int(result["crawl_success"].sum()),
            "solver_limit_steps": getattr(environment, "solver_limit_steps", None),
            "scope": "Crawl criteria are heuristic; isolated success "
                     "does not establish robust crawling or walking"}
    if args.record_trace:
        report["trace"] = {name.removeprefix("trace_"): value.detach().cpu().tolist()
                           for name, value in result.items() if name.startswith("trace_")}
    return report


@contextmanager
def newton_crawl_environment(source, scene, pose, worlds, cone="elliptic",
                             enable_multiccd=True, tolerance=1e-10,
                             controlled_joint_names=PALM_JOINTS):
    InfantSolver.source_model = source
    config = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True,
                       update_data_interval=0, integrator="implicitfast", iterations=100,
                       ls_iterations=40, nconmax=4096, njmax=8192, cone=cone,
                       enable_multiccd=enable_multiccd, tolerance=tolerance),
                       use_cuda_graph=True, num_substeps=1)
    config.class_type = InfantManager
    with build_simulation_context(sim_cfg=SimulationCfg(dt=float(source.opt.timestep),
                                                        device="cuda:0", physics=config)) as simulation:
        template = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template)
        template.add_mjcf(str(scene.resolve()), parse_mujoco_options=False,
                          skip_equality_constraints=True)
        stiffness = template.custom_attributes["mujoco:dof_passive_stiffness"].values
        robot_joint_ids = [joint_id for joint_id in range(source.njnt)
                           if (mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT,
                                                 joint_id) or "").startswith("robot:")]
        for joint_id in robot_joint_ids:
            dof_id = int(source.jnt_dofadr[joint_id])
            if not np.isclose(template.joint_damping[dof_id], source.dof_damping[dof_id]):
                raise ValueError("Newton joint damping does not match the source model")
            if not np.isclose(stiffness[dof_id], source.jnt_stiffness[joint_id]):
                raise ValueError("Newton joint stiffness does not match the source model")
            template.joint_damping[dof_id] /= 20
            stiffness[dof_id] = 0.0
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for _ in range(worlds):
            builder.add_world(template)
        NewtonManager.set_builder(builder)
        simulation.reset()
        yield InfantNewtonCrawlEnv(source, pose, worlds, simulation,
                                   controlled_joint_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blend-checkpoint", type=Path)
    parser.add_argument("--blend-fraction", type=float, default=0.0)
    parser.add_argument("--stepper-parameters", type=Path)
    parser.add_argument("--stepper-parameter-override", nargs=6, type=float)
    parser.add_argument("--stepper-residual-strength", type=float)
    parser.add_argument("--stepper-min-reach-m", type=float)
    parser.add_argument("--max-plant-steps", type=int, default=8)
    parser.add_argument("--stance-brace-strength", type=float)
    parser.add_argument("--alternate-on-abort", action="store_true")
    parser.add_argument("--tactile-audit-checkpoint", type=Path)
    parser.add_argument("--shuffle-tactile-feature-actions", action="store_true")
    parser.add_argument("--save-tactile-transitions", type=Path)
    parser.add_argument("--save-qpos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--control-steps", type=int, default=80)
    parser.add_argument("--physics-per-control", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--backend", choices=("newton", "direct"), default="newton")
    parser.add_argument("--direct-match-newton-solver", action="store_true")
    parser.add_argument("--newton-cone", choices=("pyramidal", "elliptic"),
                        default="elliptic")
    parser.add_argument("--newton-multiccd", dest="newton_multiccd", action="store_true",
                        default=True)
    parser.add_argument("--newton-no-multiccd", dest="newton_multiccd", action="store_false")
    parser.add_argument("--newton-tolerance", type=float, default=1e-10)
    parser.add_argument("--cpu-contact-audit", action="store_true")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if min(args.worlds, args.control_steps, args.physics_per_control) < 1:
        parser.error("Evaluation dimensions must be positive")
    if (args.max_plant_steps < 2 or
            (args.stance_brace_strength is not None and
             not 0 <= args.stance_brace_strength <= 1)):
        parser.error("Stepper timing or brace strength is invalid")
    if args.stepper_parameter_override is not None and args.stepper_parameters is None:
        parser.error("Parameter override requires a stepper parameter file")
    if (args.stepper_residual_strength is not None and
            (args.stepper_parameters is None or
             not 0 <= args.stepper_residual_strength <= 1)):
        parser.error("Stepper residual strength requires a stepper in the valid range")
    if (args.stepper_min_reach_m is not None and
            (args.stepper_parameters is None or
             not 0 <= args.stepper_min_reach_m <= 0.2)):
        parser.error("Minimum hand reach requires a stepper and a value in [0, 0.2] m")
    if args.newton_tolerance <= 0:
        parser.error("Newton tolerance must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    if not 0 <= args.blend_fraction <= 1 or (args.blend_fraction > 0 and
                                             args.blend_checkpoint is None):
        parser.error("Policy blend needs a checkpoint and fraction in [0, 1]")
    if args.save_tactile_transitions and args.save_tactile_transitions.exists():
        parser.error("Tactile transition output already exists")
    if args.save_qpos and args.save_qpos.exists():
        parser.error("Qpos output already exists")
    if args.save_qpos and args.save_qpos.suffix != ".npz":
        parser.error("Qpos output must use .npz")
    if (args.save_tactile_transitions and
            args.save_tactile_transitions.suffix != ".npz"):
        parser.error("Tactile transition output must use .npz")
    source = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    if source.eq_active0.any():
        raise ValueError("The infant bridge can skip only inactive model equalities")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (tuple(checkpoint["joint_names"]) != PALM_JOINTS or
            checkpoint.get("palm_sensor_version") != PALM_SENSOR_VERSION or
            checkpoint.get("oscillator_parameters") is not None):
        raise ValueError("Newton crawl evaluation requires a compatible palm policy")
    if args.backend == "direct":
        environment = InfantCrawlEnv(args.scene, args.pose, args.worlds,
                                     controlled_joint_names=PALM_JOINTS,
                                     solver_iterations=(100 if args.direct_match_newton_solver else None),
                                     solver_ls_iterations=(40 if args.direct_match_newton_solver else None),
                                     integrator=(mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                                                 if args.direct_match_newton_solver else None),
                                     jacobian=(mujoco.mjtJacobian.mjJAC_AUTO
                                               if args.direct_match_newton_solver else None))
        report = evaluate_policy(environment, checkpoint, args)
    else:
        if args.deterministic:
            wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
        with newton_crawl_environment(source, args.scene, args.pose, args.worlds,
                                     args.newton_cone, args.newton_multiccd,
                                     args.newton_tolerance) as environment:
            report = evaluate_policy(environment, checkpoint, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
