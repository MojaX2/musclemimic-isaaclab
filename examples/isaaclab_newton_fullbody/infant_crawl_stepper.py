"""Contact-triggered alternating stance and swing for the infant crawl model."""

import torch
import torch.nn.functional as functional

from infant_crawl_env import PALM_JOINTS
from infant_crawl_oscillator import LIMB_IDS, SHOULDER_LIFT_IDS, WRIST_IDS


class TactileStepper:
    observation_size = 10

    def __init__(self, parameters, lift_steps=3, reach_steps=3, plant_steps=2,
                 max_plant_steps=8, cooldown_steps=3, absolute_targets=False,
                 forward_reach=False, strong_plant=False, leg_push=False,
                 stance_loss_grace_steps=0, stance_brace_strength=0.0,
                 alternate_on_abort=False, residual_strength=0.0,
                 min_reach_m=0.0):
        if parameters.ndim != 2 or parameters.shape[1] != 6:
            raise ValueError("Stepper parameters must contain six values per infant")
        if min(lift_steps, reach_steps, plant_steps, cooldown_steps) < 1 or max_plant_steps < plant_steps:
            raise ValueError("Stepper stage durations are invalid")
        if stance_loss_grace_steps < 0:
            raise ValueError("Stance loss grace steps must be nonnegative")
        if not 0 <= stance_brace_strength <= 1:
            raise ValueError("Stance brace strength must be between zero and one")
        if not 0 <= residual_strength <= 1:
            raise ValueError("Stepper residual strength must be between zero and one")
        if min_reach_m < 0:
            raise ValueError("Minimum hand reach must be nonnegative")
        self.parameters = parameters
        self.lift_steps = lift_steps
        self.reach_steps = reach_steps
        self.plant_steps = plant_steps
        self.max_plant_steps = max_plant_steps
        self.cooldown_steps = cooldown_steps
        self.absolute_targets = absolute_targets
        self.forward_reach = forward_reach
        self.strong_plant = strong_plant
        self.leg_push = leg_push
        self.stance_loss_grace_steps = stance_loss_grace_steps
        self.stance_brace_strength = stance_brace_strength
        self.alternate_on_abort = alternate_on_abort
        self.residual_strength = residual_strength
        self.min_reach_m = min_reach_m
        self.reset()

    def reset(self):
        worlds = self.parameters.shape[0]
        device = self.parameters.device
        self.stage = torch.zeros(worlds, device=device, dtype=torch.long)
        self.age = torch.zeros_like(self.stage)
        self.swing_side = torch.zeros_like(self.stage)
        self.cooldown = torch.full_like(self.stage, self.cooldown_steps)
        self.attempts = torch.zeros(worlds, device=device)
        self.completed = torch.zeros_like(self.attempts)
        self.aborted = torch.zeros_like(self.attempts)
        self.active_steps = torch.zeros_like(self.attempts)
        self.unload_events = torch.zeros_like(self.attempts)
        self.swing_unloaded = torch.zeros(worlds, device=device, dtype=torch.bool)
        self.stance_gap = torch.zeros_like(self.stage)
        self.stance_loss_aborts = torch.zeros_like(self.attempts)
        self.plant_timeouts = torch.zeros_like(self.attempts)
        self.override_mask = torch.zeros((worlds, len(PALM_JOINTS)), device=device,
                                         dtype=torch.bool)
        self.swing_origin_x = torch.zeros(worlds, device=device)
        self.forward_qualified_completions = torch.zeros(worlds, device=device)
        self.completed_reach_m = torch.zeros(worlds, device=device)

    def observation(self):
        durations = torch.tensor((1, self.lift_steps, self.reach_steps,
                                  self.max_plant_steps), device=self.parameters.device)
        return torch.cat((functional.one_hot(self.stage, 4).float(),
                          functional.one_hot(self.swing_side, 2).float(),
                          (self.age / durations[self.stage]).clamp(0, 1)[:, None],
                          (self.cooldown / self.cooldown_steps)[:, None],
                          self.swing_unloaded.float()[:, None],
                          (self.stance_gap / max(1, self.stance_loss_grace_steps + 1))
                          .clamp(0, 1)[:, None]), dim=-1)

    def __call__(self, base, measure, reference_base=None):
        if base.shape != (self.parameters.shape[0], len(PALM_JOINTS)):
            raise ValueError("Stepper base drives do not match the infant action space")
        if self.residual_strength > 0 and (reference_base is None or
                                           reference_base.shape != base.shape):
            raise ValueError("Residual stepper requires matching reference drives")
        contact_force = measure["palm_shin_force"]
        if contact_force.shape != (base.shape[0], 4):
            raise ValueError("Stepper requires palm and shin contact forces")
        hand_x = measure.get("palm_relative_x")
        if self.min_reach_m > 0 and hand_x is None:
            raise ValueError("Reach-qualified stepper needs relative hand positions")
        if hand_x is not None and hand_x.shape != (base.shape[0], 2):
            raise ValueError("Relative hand positions must match both palms")
        hands = contact_force[:, :2] > 2
        shins = contact_force[:, 2:] > 2
        swing_contact = torch.where(self.swing_side == 0, hands[:, 0], hands[:, 1])
        stance_contact = torch.where(self.swing_side == 0, hands[:, 1], hands[:, 0])
        newly_unloaded = (self.stage != 0) & ~swing_contact & ~self.swing_unloaded
        self.unload_events += newly_unloaded.float()
        self.swing_unloaded |= newly_unloaded
        self.cooldown = (self.cooldown - 1).clamp_min(0)
        self.stance_gap = torch.where((self.stage != 0) & ~stance_contact,
                                      self.stance_gap + 1, 0)
        abort = (self.stage != 0) & (self.stance_gap > self.stance_loss_grace_steps)
        lift_done = (self.stage == 1) & (self.age >= self.lift_steps)
        reach_done = (self.stage == 2) & (self.age >= self.reach_steps)
        plant_done = ((self.stage == 3) & (self.age >= self.plant_steps) &
                      swing_contact & self.swing_unloaded)
        reach = (hand_x.gather(1, self.swing_side[:, None]).squeeze(1) -
                 self.swing_origin_x if hand_x is not None else
                 torch.zeros_like(self.swing_origin_x))
        if self.min_reach_m > 0:
            plant_done &= reach >= self.min_reach_m
        plant_timeout = (self.stage == 3) & (self.age >= self.max_plant_steps)
        done = plant_done & ~abort
        failed = (abort | plant_timeout) & ~done
        finished = done | failed
        self.completed += done.float()
        self.forward_qualified_completions += (done & (reach >= 0.03)).float()
        self.completed_reach_m += torch.where(done, reach, 0.)
        self.aborted += failed.float()
        self.stance_loss_aborts += (failed & abort).float()
        self.plant_timeouts += (failed & ~abort & plant_timeout).float()
        self.swing_side = torch.where(done | (failed & self.alternate_on_abort),
                                      1 - self.swing_side, self.swing_side)
        self.stage = torch.where(finished, 0, self.stage)
        self.age = torch.where(finished, 0, self.age)
        self.cooldown = torch.where(finished, self.cooldown_steps, self.cooldown)
        self.swing_unloaded = torch.where(finished, False, self.swing_unloaded)
        self.stance_gap = torch.where(finished, 0, self.stance_gap)
        self.stage = torch.where(lift_done & ~finished, 2, self.stage)
        self.stage = torch.where(reach_done & ~finished, 3, self.stage)
        self.age = torch.where((lift_done | reach_done) & ~finished, 0, self.age)
        ready = ((self.stage == 0) & (self.cooldown == 0) & hands.all(-1) &
                 shins.any(-1) & ~finished)
        self.stage = torch.where(ready, 1, self.stage)
        self.age = torch.where(ready, 0, self.age)
        self.attempts += ready.float()
        if hand_x is not None:
            starting_x = hand_x.gather(1, self.swing_side[:, None]).squeeze(1)
            self.swing_origin_x = torch.where(ready, starting_x, self.swing_origin_x)
        self.swing_unloaded = torch.where(ready, False, self.swing_unloaded)
        self.active_steps += (self.stage != 0).float()
        drive = base.clone()
        self.override_mask.zero_()
        for side in (0, 1):
            swing = self.swing_side == side
            active = swing & (self.stage != 0)
            lift = swing & (self.stage == 1)
            reach = swing & (self.stage == 2)
            plant = swing & (self.stage == 3)
            arm = LIMB_IDS[side]
            shoulder_lift = SHOULDER_LIFT_IDS[side]
            wrist = WRIST_IDS[side]
            if self.stance_brace_strength > 0:
                stance_side = 1 - side
                for joint_id, target in ((LIMB_IDS[stance_side], 1.),
                                         (SHOULDER_LIFT_IDS[stance_side], 1.),
                                         (WRIST_IDS[stance_side], -0.5)):
                    drive[:, joint_id] = torch.where(
                        active,
                        drive[:, joint_id] + self.stance_brace_strength *
                        (target - drive[:, joint_id]),
                        drive[:, joint_id])
                    if self.stance_brace_strength == 1:
                        self.override_mask[:, joint_id] |= active
            opposite_leg = 1 - side
            hip = LIMB_IDS[4 + opposite_leg]
            knee = LIMB_IDS[6 + opposite_leg]
            if self.absolute_targets:
                for joint_id in (arm, shoulder_lift, wrist):
                    self.override_mask[:, joint_id] |= active
                drive[:, arm] = torch.where(lift | (reach & self.forward_reach),
                                            -self.parameters[:, 0],
                                            torch.where(reach | plant, self.parameters[:, 2],
                                                        drive[:, arm]))
                drive[:, shoulder_lift] = torch.where(lift, -self.parameters[:, 1],
                                                       torch.where((reach | plant) & self.forward_reach,
                                                                   self.parameters[:, 1],
                                                                   torch.where(reach, -self.parameters[:, 1],
                                                                               drive[:, shoulder_lift])))
                drive[:, wrist] = torch.where(lift | reach, self.parameters[:, 3],
                                              torch.where(plant, -0.5 * self.parameters[:, 3],
                                                          drive[:, wrist]))
            else:
                drive[:, arm] += torch.where(lift | (reach & self.forward_reach),
                                             -self.parameters[:, 0],
                                             torch.where(reach | plant, self.parameters[:, 2], 0.))
                drive[:, shoulder_lift] += torch.where(lift, -self.parameters[:, 1],
                                                       torch.where((reach | plant) & self.forward_reach,
                                                                   self.parameters[:, 1],
                                                                   torch.where(reach, -self.parameters[:, 1], 0.)))
                drive[:, wrist] += torch.where(lift | reach, self.parameters[:, 3],
                                              torch.where(plant, -0.5 * self.parameters[:, 3], 0.))
            drive[:, hip] += torch.where(lift | reach, self.parameters[:, 4], 0.)
            drive[:, knee] += torch.where(lift | reach, self.parameters[:, 5], 0.)
            if self.leg_push:
                drive[:, hip] -= torch.where(plant, self.parameters[:, 4], 0.)
                drive[:, knee] -= torch.where(plant, self.parameters[:, 5], 0.)
            if self.strong_plant:
                for joint_id in (arm, shoulder_lift, wrist):
                    self.override_mask[:, joint_id] |= plant
                drive[:, arm] = torch.where(plant, 1., drive[:, arm])
                drive[:, shoulder_lift] = torch.where(plant, 1., drive[:, shoulder_lift])
                drive[:, wrist] = torch.where(plant, -0.5, drive[:, wrist])
        if self.residual_strength > 0:
            drive = torch.where(self.override_mask,
                                drive + self.residual_strength *
                                (base - reference_base), drive)
            self.override_mask.zero_()
        self.age += (self.stage != 0).long()
        return drive.clamp(-1, 1)

    def metrics(self, control_steps):
        return {"step_attempts": self.attempts,
                "swing_unload_events": self.unload_events,
                "step_completions": self.completed,
                "step_aborts": self.aborted,
                "forward_qualified_completions": self.forward_qualified_completions,
                "mean_completed_reach_m": (self.completed_reach_m /
                                           self.completed.clamp_min(1)),
                "stance_loss_aborts": self.stance_loss_aborts,
                "plant_timeouts": self.plant_timeouts,
                "swing_active_fraction": self.active_steps / control_steps}
