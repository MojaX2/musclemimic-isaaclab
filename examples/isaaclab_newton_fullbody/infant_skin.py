"""Map MIMo collision geoms to coarse body-contact regions."""

import mujoco
import numpy as np

from developmental_skin import REGIONS


def _body_region(name):
    if name in {"hip", "lower_body"}:
        return "pelvis"
    if name in {"upper_body", "chest"}:
        return "trunk"
    if name in {"head", "neck", "left_eye", "right_eye"}:
        return "head"
    for side in ("right", "left"):
        if not name.startswith(side + "_"):
            continue
        part = name[len(side) + 1:]
        if part == "upper_arm":
            return f"{side}_upper_arm"
        if part == "lower_arm":
            return f"{side}_forearm"
        if part in {"upper_leg"}:
            return f"{side}_thigh"
        if part in {"lower_leg"}:
            return f"{side}_shin"
        if part in {"foot", "big_toe", "toes"}:
            return f"{side}_foot"
        return f"{side}_hand"
    return None


def infant_geom_region_ids(model):
    """Return a 15-region index for each MIMo collidable geom, excluding props."""
    result = np.full(model.ngeom, -1, dtype=np.int32)
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.geom_bodyid[geom_id])) or ""
        if "/" in geom_name:
            body_name = geom_name.split("/")[-2]
        region = _body_region(body_name)
        if region is not None:
            result[geom_id] = REGIONS.index(region)
    if len(np.unique(result[result >= 0])) != len(REGIONS):
        raise ValueError("MIMo collision geoms do not cover every skin region")
    return result
