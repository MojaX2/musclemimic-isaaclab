"""Keep the assistance input well-scaled when the curriculum changes."""
import torch
from rsl_rl.modules.normalization import EmpiricalNormalization

class VelocityNormalization(EmpiricalNormalization):
    def forward(self,x):
        normalized=super().forward(x)
        # The last observation is a known, dimensionless bodyweight fraction.
        # Constant assistance during a curriculum stage has zero empirical std.
        # A fixed scale keeps 40% -> 0% in [0, -1], rather than [0, -40].
        if x.shape[-1]==888:
            # Heading is bounded and must not acquire a singular empirical scale.
            center=torch.tensor([1.,0.],dtype=x.dtype,device=x.device)
            return torch.cat([normalized[...,:-3],x[...,-3:-1]-center,(x[...,-1:]-.4)/.4],dim=-1)
        return torch.cat([normalized[...,:-1],(x[...,-1:]-.4)/.4],dim=-1)
