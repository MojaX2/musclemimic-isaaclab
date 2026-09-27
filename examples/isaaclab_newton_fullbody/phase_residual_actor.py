"""Experimental periodic mean-action branch; no simulator forces or state edits.

Zero initialization exactly preserves an existing full-action Gaussian actor.
The Fourier correction participates in the same distribution/log-probability as
PPO's MLP action. This module alone is not a trained walking policy.
"""
import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories
from velocity_normalization import VelocityNormalization


class PhaseResidualActor(MLPModel):
    def __init__(self,*args,harmonics=4,**kwargs):
        super().__init__(*args,**kwargs)
        if harmonics<1 or harmonics>8:raise ValueError('Invalid harmonics')
        if self.obs_groups!=['policy'] or self.obs_dim!=888:
            raise ValueError('Requires the888-dimensional muscle policy observation')
        if not isinstance(self.distribution,GaussianDistribution):
            raise ValueError('Requires Gaussian latent muscle actions')
        if self.obs_normalization:self.obs_normalizer=VelocityNormalization(self.obs_dim)
        self.harmonics=harmonics
        self.phase_head=nn.Linear(2*harmonics,self.distribution.input_dim,bias=False)
        nn.init.zeros_(self.phase_head.weight)

    def load_base_state_dict(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if set(missing)!={'phase_head.weight'} or unexpected:
            raise ValueError(f'Incompatible base checkpoint: {missing}, {unexpected}')
        nn.init.zeros_(self.phase_head.weight)

    def phase_features(self,raw):
        sine,cosine=raw[...,-5],raw[...,-4]
        sk,ck=sine,cosine;features=[]
        for _ in range(self.harmonics):
            features.extend([sk,ck]);sk,ck=sk*cosine+ck*sine,ck*cosine-sk*sine
        return torch.stack(features,dim=-1)*(raw[...,-6].abs()>.1).unsqueeze(-1)

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        if masks is not None:obs=unpad_trajectories(obs,masks)
        mean=super().forward(obs,hidden_state=hidden_state,stochastic_output=False)
        mean=mean+self.phase_head(self.phase_features(obs['policy']))
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def as_jit(self):
        raise NotImplementedError('Export must include the periodic branch')

    def as_onnx(self,verbose=False):
        raise NotImplementedError('Export must include the periodic branch')
