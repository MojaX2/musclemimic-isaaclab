"""Diagonal muscle oscillator and phase observation for infant crawl control."""

import math

import torch

from infant_crawl_env import PALM_JOINTS


LIMB_JOINTS = ("right_shoulder_horizontal", "left_shoulder_horizontal",
               "right_elbow", "left_elbow", "right_hip1", "left_hip1",
               "right_knee", "left_knee")
LIMB_IDS = tuple(PALM_JOINTS.index(f"robot:{name}") for name in LIMB_JOINTS)
WRIST_IDS = tuple(PALM_JOINTS.index(f"robot:{side}_hand2") for side in ("right", "left"))
SHOULDER_LIFT_IDS = tuple(PALM_JOINTS.index(f"robot:{side}_shoulder_ad_ab")
                          for side in ("right", "left"))
OSCILLATOR_IDS = LIMB_IDS + WRIST_IDS


def exclusive_contact_state(force_pair):
    right = force_pair[:, 0] > 2
    left = force_pair[:, 1] > 2
    return torch.where(right & ~left, 1, torch.where(left & ~right, -1, 0))


def oscillator_phase(parameters, elapsed_seconds):
    if parameters.ndim != 2 or parameters.shape[1] not in (6, 7, 8, 9):
        raise ValueError("Oscillator parameters must have six to nine values per infant")
    angle = 2 * math.pi * parameters[:, 0].clamp(0.5, 3.0) * elapsed_seconds
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def cpg_drives(base_drives, parameters, elapsed_seconds):
    if base_drives.ndim != 2 or base_drives.shape[1] != len(PALM_JOINTS):
        raise ValueError("CPG base drives must cover all palm-control joints")
    if parameters.ndim != 2 or parameters.shape[0] != base_drives.shape[0] or parameters.shape[1] not in (6, 7, 8, 9):
        raise ValueError("CPG parameters must have six to nine values per infant")
    arm_phase = oscillator_phase(parameters, elapsed_seconds)[:, 0]
    leg_phase = torch.sin(2 * math.pi * parameters[:, 0].clamp(0.5, 3.0) *
                          elapsed_seconds + parameters[:, 5])
    drives = base_drives.clone()
    for index, signal in enumerate((arm_phase, -arm_phase, arm_phase, -arm_phase,
                                    -leg_phase, leg_phase, -leg_phase, leg_phase)):
        amplitude = parameters[:, 1 + index // 2]
        drives[:, LIMB_IDS[index]] += amplitude * signal
    if parameters.shape[1] >= 7:
        drives[:, WRIST_IDS[0]] += parameters[:, 6] * arm_phase
        drives[:, WRIST_IDS[1]] -= parameters[:, 6] * arm_phase
    if parameters.shape[1] >= 8:
        lift_phase = arm_phase
        if parameters.shape[1] == 9:
            lift_phase = torch.sin(2 * math.pi * parameters[:, 0].clamp(0.5, 3.0) *
                                   elapsed_seconds + parameters[:, 8])
        drives[:, SHOULDER_LIFT_IDS[0]] -= parameters[:, 7] * lift_phase.clamp_min(0)
        drives[:, SHOULDER_LIFT_IDS[1]] -= parameters[:, 7] * (-lift_phase).clamp_min(0)
    return drives.clamp(-1, 1)


def fixed_limb_drives(learned_drives, reference_drives, parameters, elapsed_seconds):
    if learned_drives.shape != reference_drives.shape:
        raise ValueError("Learned and frozen reference drives must have matching shapes")
    combined = learned_drives.clone()
    oscillated = cpg_drives(reference_drives, parameters, elapsed_seconds)
    oscillated_ids = OSCILLATOR_IDS + (SHOULDER_LIFT_IDS if parameters.shape[1] >= 8 else ())
    combined[:, oscillated_ids] = oscillated[:, oscillated_ids]
    return combined.clamp(-1, 1)
