"""Coarse body skin from the collision contacts of the MuscleMimic model.

The model already contains skin-colored collision geoms. This module reads actual
MuJoCo Warp contacts on those geoms; it does not add a deformable skin layer.
"""

import re

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


REGIONS = (
    "trunk", "head", "right_upper_arm", "left_upper_arm",
    "right_forearm", "left_forearm", "right_hand", "left_hand",
    "pelvis", "right_thigh", "left_thigh", "right_shin", "left_shin",
    "right_foot", "left_foot",
)


def body_region(name):
    name = (name or "").lower()
    side = "right" if name.endswith("_r") else "left" if name.endswith("_l") else None
    if name in {"head", "neck", "head_attach"}:
        return "head"
    if name in {"sacrum", "pelvis"}:
        return "pelvis"
    if side:
        if any(part in name for part in ("humerus", "scapula", "clavicle")):
            return f"{side}_upper_arm"
        if any(part in name for part in ("ulna", "radius")):
            return f"{side}_forearm"
        if any(part in name for part in ("femur", "patella")):
            return f"{side}_thigh"
        if "tibia" in name:
            return f"{side}_shin"
        if any(part in name for part in ("talus", "calcn", "toes")):
            return f"{side}_foot"
        if re.search(r"(?:hand|wrist|mc|ph|thumb|lunate|scaphoid|pisiform|triquetrum|capitate|trapez|hamate)", name):
            return f"{side}_hand"
    if side:
        return None
    return "trunk"


def geom_region_name(geom_name, body_name):
    if geom_name and "/" in geom_name:
        return body_region(geom_name.split("/")[-2])
    return body_region(body_name)


def geom_region_ids(model):
    """Return one region index per collision geom, or -1 for non-skin geoms."""
    result = np.full(model.ngeom, -1, dtype=np.int32)
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        region = geom_region_name(geom_name, body_name)
        if region in REGIONS:
            result[geom_id] = REGIONS.index(region)
    if not np.any(result >= 0):
        raise ValueError("No collidable body geoms found for skin sensing")
    return result


@wp.kernel
def accumulate_skin_contacts(
    active: wp.array(dtype=int), world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i), region: wp.array(dtype=int),
    force: wp.array(dtype=wp.spatial_vector), out: wp.array2d(dtype=float),
):
    contact_id = wp.tid()
    if contact_id >= active[0]:
        return
    first = geom[contact_id][0]
    second = geom[contact_id][1]
    if first < 0 or second < 0:
        return
    normal = wp.max(force[contact_id][0], 0.0)
    first_region = region[first]
    second_region = region[second]
    if first_region >= 0:
        wp.atomic_add(out, world[contact_id], first_region, normal)
    if second_region >= 0:
        wp.atomic_add(out, world[contact_id], second_region, normal)


@wp.kernel
def accumulate_ground_contacts(
    active: wp.array(dtype=int), world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i), region: wp.array(dtype=int),
    force: wp.array(dtype=wp.spatial_vector), floor_geom: int,
    out: wp.array2d(dtype=float),
):
    contact_id = wp.tid()
    if contact_id >= active[0]:
        return
    first = geom[contact_id][0]
    second = geom[contact_id][1]
    if first < 0 or second < 0:
        return
    normal = wp.max(force[contact_id][0], 0.0)
    if first == floor_geom and region[second] >= 0:
        wp.atomic_add(out, world[contact_id], region[second], normal)
    if second == floor_geom and region[first] >= 0:
        wp.atomic_add(out, world[contact_id], region[first], normal)


@wp.kernel
def accumulate_foot_pressure(
    active: wp.array(dtype=int), world: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i), position: wp.array(dtype=wp.vec3),
    region: wp.array(dtype=int), force: wp.array(dtype=wp.spatial_vector),
    floor_geom: int, right_foot: int, left_foot: int,
    out: wp.array3d(dtype=float),
):
    contact_id = wp.tid()
    if contact_id >= active[0]:
        return
    first = geom[contact_id][0]
    second = geom[contact_id][1]
    if first < 0 or second < 0:
        return
    foot_region = -1
    if first == floor_geom:
        foot_region = region[second]
    elif second == floor_geom:
        foot_region = region[first]
    foot_index = -1
    if foot_region == right_foot:
        foot_index = 0
    elif foot_region == left_foot:
        foot_index = 1
    if foot_index < 0:
        return
    normal = wp.max(force[contact_id][0], 0.0)
    world_id = world[contact_id]
    wp.atomic_add(out, world_id, foot_index, 0, normal)
    wp.atomic_add(out, world_id, foot_index, 1, normal * position[contact_id][0])
    wp.atomic_add(out, world_id, foot_index, 2, normal * position[contact_id][1])


class SkinContactSensor:
    """Region-level contact force in newtons, shape (worlds, regions)."""

    def __init__(self, model, warp_model, data, device, region_ids=None):
        self.model = warp_model
        self.data = data
        self.device = device
        if region_ids is None:
            region_ids = geom_region_ids(model)
        if region_ids.shape != (model.ngeom,) or np.any((region_ids < -1) | (region_ids >= len(REGIONS))):
            raise ValueError("Skin region map must match the model geoms and known regions")
        self.region = wp.array(region_ids, device=device)
        self.indices = wp.array(np.arange(data.naconmax, dtype=np.int32), device=device)
        self.force = wp.zeros(data.naconmax, dtype=wp.spatial_vector, device=device)
        self.buffer = wp.zeros((data.qpos.shape[0], len(REGIONS)), dtype=float, device=device)
        self.tensor = wp.to_torch(self.buffer)
        self.ground_buffer = wp.zeros((data.qpos.shape[0], len(REGIONS)), dtype=float, device=device)
        self.ground_tensor = wp.to_torch(self.ground_buffer)
        self.foot_pressure_buffer = wp.zeros((data.qpos.shape[0], 2, 3), dtype=float, device=device)
        self.foot_pressure_tensor = wp.to_torch(self.foot_pressure_buffer)
        self.selected_ground_region = None
        self.selected_ground_buffer = None
        self.selected_ground_tensor = None

    def configure_ground_geoms(self, geom_groups):
        groups = [[group] if isinstance(group, (int, np.integer)) else list(group)
                  for group in geom_groups]
        flat_ids = [geom_id for group in groups for geom_id in group]
        if not groups or any(not group for group in groups) or len(set(flat_ids)) != len(flat_ids):
            raise ValueError("Selected ground geom groups must be nonempty and disjoint")
        if any(geom_id < 0 or geom_id >= self.region.shape[0] for geom_id in flat_ids):
            raise ValueError("Selected ground geom ID is outside the model")
        region = np.full(self.region.shape[0], -1, dtype=np.int32)
        for group_id, group in enumerate(groups):
            region[np.asarray(group, dtype=np.int64)] = group_id
        self.selected_ground_region = wp.array(region, device=self.device)
        self.selected_ground_buffer = wp.zeros((self.data.qpos.shape[0], len(groups)),
                                                dtype=float, device=self.device)
        self.selected_ground_tensor = wp.to_torch(self.selected_ground_buffer)

    def read(self):
        self.tensor.zero_()
        mjw.contact_force(self.model, self.data, self.indices, False, self.force)
        contact = self.data.contact
        wp.launch(
            accumulate_skin_contacts,
            dim=self.data.naconmax,
            inputs=[self.data.nacon, contact.worldid, contact.geom,
                    self.region, self.force, self.buffer],
            device=self.device,
        )
        return self.tensor

    def read_ground(self, floor_geom_id):
        self.ground_tensor.zero_()
        mjw.contact_force(self.model, self.data, self.indices, False, self.force)
        contact = self.data.contact
        wp.launch(
            accumulate_ground_contacts,
            dim=self.data.naconmax,
            inputs=[self.data.nacon, contact.worldid, contact.geom,
                    self.region, self.force, floor_geom_id, self.ground_buffer],
            device=self.device,
        )
        return self.ground_tensor

    def read_ground_geoms(self, floor_geom_id):
        if self.selected_ground_region is None:
            raise ValueError("Selected ground geoms have not been configured")
        self.selected_ground_tensor.zero_()
        mjw.contact_force(self.model, self.data, self.indices, False, self.force)
        contact = self.data.contact
        wp.launch(
            accumulate_ground_contacts,
            dim=self.data.naconmax,
            inputs=[self.data.nacon, contact.worldid, contact.geom,
                    self.selected_ground_region, self.force, floor_geom_id,
                    self.selected_ground_buffer],
            device=self.device,
        )
        return self.selected_ground_tensor

    def normalized(self, body_weight_newtons):
        return torch.log1p(self.read().clamp_min(0) / body_weight_newtons)

    def read_foot_pressure(self, floor_geom_id):
        """Return foot normal force and world-frame pressure centroid in metres."""
        self.foot_pressure_tensor.zero_()
        mjw.contact_force(self.model, self.data, self.indices, False, self.force)
        contact = self.data.contact
        wp.launch(
            accumulate_foot_pressure,
            dim=self.data.naconmax,
            inputs=[self.data.nacon, contact.worldid, contact.geom, contact.pos,
                    self.region, self.force, floor_geom_id,
                    REGIONS.index("right_foot"), REGIONS.index("left_foot"),
                    self.foot_pressure_buffer],
            device=self.device,
        )
        force = self.foot_pressure_tensor[:, :, 0].clone()
        centroid = self.foot_pressure_tensor[:, :, 1:3] / force.clamp_min(1e-8).unsqueeze(-1)
        return force, torch.where(force.unsqueeze(-1) > 0, centroid, 0)
