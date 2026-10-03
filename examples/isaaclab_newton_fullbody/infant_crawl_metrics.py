"""Conservative locomotion gate for hand-shin infant crawl rollouts."""

import numpy as np


def contact_switch_count(pair_force, valid_mask=None):
    if pair_force.ndim != 3 or pair_force.shape[-1] != 2:
        raise ValueError("Contact pair must have shape (time, worlds, 2)")
    if valid_mask is None:
        valid_mask = np.ones(pair_force.shape[:2], dtype=bool)
    if valid_mask.shape != pair_force.shape[:2]:
        raise ValueError("Contact validity mask must have shape (time, worlds)")
    right = pair_force[:, :, 0] > 2
    left = pair_force[:, :, 1] > 2
    exclusive = np.where(right & ~left, 1, np.where(left & ~right, -1, 0))
    counts = np.zeros(pair_force.shape[1], dtype=np.int64)
    previous = np.zeros_like(counts)
    for step in range(pair_force.shape[0]):
        current = exclusive[step]
        counts += ((current != 0) & (previous != 0) &
                   (current != previous) & valid_mask[step])
        previous = np.where(current != 0, current, previous)
    return counts


def crawl_locomotion_metrics(palm_shin_force, chest_height, head_height,
                             pelvis_force, final_forward_displacement):
    if palm_shin_force.ndim != 3 or palm_shin_force.shape[-1] != 4:
        raise ValueError("Palm-shin force must have shape (time, worlds, 4)")
    if (chest_height.shape != palm_shin_force.shape[:2] or
            head_height.shape != palm_shin_force.shape[:2] or
            pelvis_force.shape != palm_shin_force.shape[:2] or
            final_forward_displacement.shape != (palm_shin_force.shape[1],)):
        raise ValueError("Crawl trajectory arrays have inconsistent dimensions")
    elevated = (chest_height > 0.16) & (head_height > 0.14)
    supported = ((palm_shin_force[:, :, :2] > 2).any(-1) &
                 (palm_shin_force[:, :, 2:] > 2).any(-1))
    supported_fraction = (elevated & supported).mean(axis=0)
    late_start = len(supported) - max(1, len(supported) // 4)
    late_supported_fraction = (elevated[late_start:] & supported[late_start:]).mean(axis=0)
    elevated_fraction = elevated.mean(axis=0)
    pelvis_fraction = (pelvis_force > 2).mean(axis=0)
    hand_switches = contact_switch_count(palm_shin_force[:, :, :2], elevated)
    shin_switches = contact_switch_count(palm_shin_force[:, :, 2:], elevated)
    legacy_success = ((final_forward_displacement >= 0.05) &
                      (elevated_fraction >= 0.75) & (supported_fraction >= 0.5) &
                      (pelvis_fraction < 0.1) & (hand_switches >= 2) & (shin_switches >= 2))
    success = legacy_success & (late_supported_fraction >= 0.5)
    return {"forward_crawl_success_fraction": float(success.mean()),
            "forward_crawl_success_per_world": success.tolist(),
            "legacy_forward_crawl_success_per_world": legacy_success.tolist(),
            "hand_support_switches_per_world": hand_switches.tolist(),
            "shin_support_switches_per_world": shin_switches.tolist(),
            "elevated_minimum_support_fraction_per_world": supported_fraction.tolist(),
            "late_minimum_support_fraction_per_world": late_supported_fraction.tolist()}
