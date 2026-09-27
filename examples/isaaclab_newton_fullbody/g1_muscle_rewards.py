"""G1 Flat reward structure adapted to muscle excitations; engine-independent Torch.

References and deliberate differences are documented in G1_REWARD_DESIGN.md.
All rate terms are multiplied by control dt; touchdown is a discrete event.
"""
import torch

WEIGHTS = dict(track_lin_vel=2., track_yaw_rate=1., lin_vel_z=-.2,
               ang_vel_xy=-.05, orientation=-1., termination=-200.,
               air_time=.75, feet_slide=-.1, ankle_limits=-1.,
               hip_deviation=-.1, arm_deviation=-.1, torso_deviation=-.1,
               joint_acceleration=-1e-7, joint_torque=-2e-6,
               excitation_rate=-.005, excitation_effort=-.05,
               swing_clearance=.5, scuff=-.5, flight=-.5, long_swing=-1.,
               prolonged_double_support=-.5, phase_clearance=1.)
TOUCHDOWN_WEIGHT = .15


def forward_velocity(root_velocity, heading):
    # Anatomical forward is -world X at reset, irrespective of root joint yaw.
    c, s = heading.cos(), heading.sin()
    return torch.stack((-c*root_velocity[:,0]-s*root_velocity[:,1],
                        s*root_velocity[:,0]-c*root_velocity[:,1]), dim=-1)


class G1MuscleRewards:
    def __init__(self, n, device, dt=.02):
        self.dt = dt
        self.air = torch.zeros(n, 2, device=device)
        self.stance = torch.zeros_like(self.air)
        self.peak = torch.zeros_like(self.air)
        self.double_support = torch.zeros(n, device=device)
        self.last_touchdown = torch.full((n,), -1, device=device, dtype=torch.long)

    def reset(self, mask):
        self.air[mask] = 0
        self.stance[mask] = 0
        self.peak[mask] = 0
        self.double_support[mask] = 0
        self.last_touchdown[mask] = -1

    def gait_terms(self, contact, clearance, foot_velocity_xy, command, phase=None):
        moving = command.abs() > .1
        landed = contact & (self.air > 0)
        good_step = landed & (self.air >= .12) & (self.air <= .6) & (self.peak >= .04)
        side = torch.arange(2, device=contact.device)[None,:]
        alternating = (side != self.last_touchdown[:,None])
        # Simultaneous landings do not count as an alternating walking step.
        touchdown = (good_step & alternating).any(1) & (landed.sum(1)==1) & moving
        self.last_touchdown.copy_(torch.where(touchdown, landed.long().argmax(1), self.last_touchdown))
        self.air.copy_(torch.where(contact, 0., self.air+self.dt))
        self.stance.copy_(torch.where(contact, self.stance+self.dt, 0.))
        self.peak.copy_(torch.where(contact, 0., torch.maximum(self.peak, clearance)))
        single = contact.sum(1)==1
        # Permit a normal transfer between feet, but discourage indefinite
        # standing under a moving command. Flight/hover remain penalized below.
        self.double_support.copy_(torch.where(contact.all(1) & moving,
                                             self.double_support+self.dt, 0.))
        stalled = ((self.double_support-.3)/.3).clamp(0.,1.)
        mode_time = torch.where(contact, self.stance, self.air)
        air_reward = mode_time.min(1).values.clamp(max=.4)*single*moving
        # G1's capped air-time by itself can reward a indefinitely held-up foot.
        prolonged = self.air.max(1).values > .6
        air_reward *= ~prolonged
        swing = (~contact) & single[:,None] & moving[:,None] & (self.air <= .5)
        clearance_reward = (torch.exp(-((clearance-.05)/.025).square())*swing).sum(1)
        mid_swing = swing & (self.air>=.10) & (self.air<=.35)
        scuff = (((.025-clearance)/.025).clamp(0,1)*mid_swing).sum(1)
        # Dense progress from grounded feet toward alternating 5 cm swings.
        # Subtract the grounded-foot baseline: standing earns exactly zero.
        # No motion dataset or force is used; phase is already observed by PPO.
        phase_clearance=torch.zeros_like(command)
        if phase is not None:
            wave=phase.sin()
            target=.05*torch.stack([wave.clamp(min=0),(-wave).clamp(min=0)],dim=1)
            height=clearance.clamp(min=0)
            progress=((target.square()-(height-target).square())/.05**2).clamp(-4.,1.)
            # Require the intended stance foot to support the body. This does
            # not gate on swing-foot liftoff, unlike the existing swing bonus.
            stance_index=(wave>=0).long()
            supported=contact.gather(1,stance_index[:,None]).squeeze(1)
            score=progress.sum(1)
            phase_clearance=torch.where(score>0,score*supported,score)*moving
        return dict(air_time=air_reward,
                    feet_slide=(foot_velocity_xy.norm(dim=-1)*contact).sum(1),
                    swing_clearance=clearance_reward, scuff=scuff,
                    flight=(~contact).all(1).float(),
                    long_swing=prolonged.float()*moving,
                    prolonged_double_support=stalled,
                    phase_clearance=phase_clearance,
                    touchdown=touchdown.float())

    def __call__(self, *, velocity_xy, command, root_velocity, gravity_xy,
                 fallen, contact, clearance, foot_velocity_xy, ankle_violation,
                 hip_deviation, arm_deviation, torso_deviation,
                 joint_acceleration, joint_torque, excitation, previous, phase=None):
        terms=self.gait_terms(contact, clearance, foot_velocity_xy, command,phase)
        target=torch.stack((command, torch.zeros_like(command)),dim=1)
        terms.update(track_lin_vel=torch.exp(-(velocity_xy-target).square().sum(1)/.25),
                     track_yaw_rate=torch.exp(-root_velocity[:,5].square()/.25),
                     lin_vel_z=root_velocity[:,2].square(),
                     ang_vel_xy=root_velocity[:,3:5].square().sum(1),
                     orientation=gravity_xy.square().sum(1), termination=fallen.float(),
                     ankle_limits=ankle_violation.sum(1),
                     hip_deviation=hip_deviation.abs().sum(1),
                     arm_deviation=arm_deviation.abs().sum(1),
                     torso_deviation=torso_deviation.abs().sum(1),
                     joint_acceleration=joint_acceleration.square().sum(1),
                     joint_torque=joint_torque.square().sum(1),
                     # Normalize 354 bounded excitations to a 23-action scale.
                     excitation_rate=23*(excitation-previous).square().mean(1),
                     excitation_effort=excitation.square().mean(1))
        weighted={key:torch.nan_to_num(terms[key],nan=0.,posinf=1e6,neginf=-1e6)*weight*self.dt
                  for key,weight in WEIGHTS.items()}
        weighted['touchdown']=TOUCHDOWN_WEIGHT*terms['touchdown']
        reward=sum(weighted.values())
        return reward, terms, weighted
