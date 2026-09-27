"""Low-dimensional muscle-logit residuals; no joint torques or external forces.

Basis columns must follow the actuator order used by the environment. This is
only an action transform, not a validated walking controller. A baseline muscle
logit vector can retain tonic/co-contraction commands outside the reduced space.
"""
import math

import torch
from torch import nn


class MuscleActionBasis(nn.Module):
    def __init__(self, basis):
        super().__init__()
        value=torch.as_tensor(basis,dtype=torch.float32)
        if value.ndim!=2 or not 0<value.shape[0]<=value.shape[1]:
            raise ValueError('Expected K by muscle-count basis, 0 < K <= muscle-count')
        if not torch.isfinite(value).all():raise ValueError('Nonfinite basis')
        eye=torch.eye(value.shape[0],device=value.device)
        if not torch.allclose(value@value.T,eye,atol=1e-4,rtol=1e-4):
            raise ValueError('Basis rows must be orthonormal')
        self.register_buffer('basis',value.clone())
        # Match total latent variance to independent muscle noise with the same
        # coefficient std. Individual muscles can have different variances.
        self.scale=math.sqrt(value.shape[1]/value.shape[0])

    def forward(self, coefficients, baseline=None):
        residual=(coefficients@self.basis)*self.scale
        return residual if baseline is None else baseline+residual

    def muscle_std(self, independent_coefficient_std):
        """Marginal muscle-logit std; muscle outputs remain correlated."""
        return ((independent_coefficient_std.square()@self.basis.square()).clamp(min=0)).sqrt()*self.scale
