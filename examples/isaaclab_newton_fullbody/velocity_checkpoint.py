"""Extend an existing policy with heading inputs without changing its outputs."""
import torch

def add_heading_inputs(checkpoint):
    old_size=checkpoint['actor_state_dict']['mlp.0.weight'].shape[1]
    if old_size==888:return False
    if old_size!=886:raise ValueError(f'Unsupported observation size: {old_size}')
    def expand(value,fill=0.):
        extra=torch.full((*value.shape[:-1],2),fill,dtype=value.dtype,device=value.device)
        return torch.cat([value[...,:-1],extra,value[...,-1:]],dim=-1)
    for name in ['actor_state_dict','critic_state_dict']:
        state=checkpoint[name]
        state['mlp.0.weight']=expand(state['mlp.0.weight'])
        for key in ['_mean','_var','_std']:
            full='obs_normalizer.'+key
            if full not in state:continue
            state[full]=expand(state[full],0. if key=='_mean' else 1.)
            if key=='_mean':state[full][...,-3]=1.
    # Adam moments for the two first-layer weight matrices must expand as well.
    for state in checkpoint['optimizer_state_dict']['state'].values():
        for key,value in state.items():
            if torch.is_tensor(value) and value.ndim==2 and value.shape[1]==old_size:
                state[key]=expand(value)
    return True
