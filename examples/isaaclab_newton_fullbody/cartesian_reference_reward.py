"""Optional phase-conditioned relative-site guidance for muscle walking.

No simulator state/force writes. Removes planar translation and yaw drift, but
preserves world vertical and pelvis tilt so crouching/leaning changes the score.
"""
import torch
from torch import nn
from periodic_pose_reward import PeriodicPoseReward


def aligned_relative_sites(site_positions,pelvis_position,heading_error):
    relative=site_positions-pelvis_position.unsqueeze(-2)
    x,y,z=relative.unbind(-1)
    cosine=heading_error.cos().unsqueeze(-1);sine=heading_error.sin().unsqueeze(-1)
    return torch.stack((cosine*x+sine*y,-sine*x+cosine*y,z),dim=-1)


class CartesianReferenceReward(nn.Module):
    def __init__(self,coefficients,standing_sites,sigma=.1,subtract_standing_baseline=True):
        super().__init__()
        coefficients=torch.as_tensor(coefficients,dtype=torch.float32)
        standing_sites=torch.as_tensor(standing_sites,dtype=torch.float32)
        if coefficients.ndim!=3 or coefficients.shape[-1]!=3:
            raise ValueError('Expected Fourier coefficients x sites x XYZ')
        if standing_sites.shape!=coefficients.shape[1:]:raise ValueError('Site shape mismatch')
        self.site_count=standing_sites.shape[0]
        self.subtract_standing_baseline=subtract_standing_baseline
        self.pose=PeriodicPoseReward(coefficients.flatten(1),standing_sites.flatten(),sigma=sigma)

    def target(self,phase):
        return self.pose.target(phase).reshape(*phase.shape,self.site_count,3)

    def forward(self,site_positions,pelvis_position,heading_error,phase,command):
        sites=aligned_relative_sites(site_positions,pelvis_position,heading_error)
        if not self.subtract_standing_baseline:
            target=self.pose.target(phase)
            score=torch.exp(-((sites.flatten(-2)-target)/self.pose.sigma).square().mean(-1))
            return score*(command.abs()>.1)
        return self.pose(sites.flatten(-2),phase,command)
