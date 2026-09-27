"""Optional kinematic guidance for muscle RL; never writes simulator state.

Fourier targets are smooth across cycle boundaries. Subtracting the standing
pose score prevents this term from rewarding the unchanged reset pose.
"""
import torch
from torch import nn


class PeriodicPoseReward(nn.Module):
    def __init__(self, coefficients, standing_pose, sigma=.2):
        super().__init__()
        coefficients=torch.as_tensor(coefficients,dtype=torch.float32)
        standing_pose=torch.as_tensor(standing_pose,dtype=torch.float32)
        if coefficients.ndim!=2 or coefficients.shape[0]%2!=1:
            raise ValueError('Expected DC followed by cosine/sine coefficient pairs')
        if standing_pose.shape!=(coefficients.shape[1],) or sigma<=0:
            raise ValueError('Invalid standing pose or angular tolerance')
        if not torch.isfinite(coefficients).all() or not torch.isfinite(standing_pose).all():
            raise ValueError('Nonfinite reference pose')
        self.register_buffer('coefficients',coefficients.clone())
        self.register_buffer('standing_pose',standing_pose.clone())
        self.sigma=sigma

    def target(self,phase):
        features=[torch.ones_like(phase)]
        for k in range(1,(self.coefficients.shape[0]+1)//2):
            features.extend([(k*phase).cos(),(k*phase).sin()])
        return torch.stack(features,dim=-1)@self.coefficients

    def forward(self,joint_positions,phase,command):
        target=self.target(phase)
        score=torch.exp(-((joint_positions-target)/self.sigma).square().mean(-1))
        baseline=torch.exp(-((self.standing_pose-target)/self.sigma).square().mean(-1))
        return (score-baseline)*(command.abs()>.1)
