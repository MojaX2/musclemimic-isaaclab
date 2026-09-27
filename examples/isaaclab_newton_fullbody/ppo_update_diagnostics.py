"""Observe Gaussian PPO updates without altering their likelihoods or RNG draws."""
import torch


def update_with_diagnostics(algorithm):
    actor=algorithm.actor
    original=actor.get_kl_divergence
    had_local='get_kl_divergence' in actor.__dict__
    records=[];rates=[]
    def observe(old,new):
        kl=original(old,new)
        with torch.no_grad():
            old_mean,old_std=old;new_mean,new_std=new
            records.append(torch.stack((kl.mean(),((new_mean-old_mean)/old_std.clamp_min(1e-8)).square().mean().sqrt(),(torch.sigmoid(1.5*new_mean-2.2)-torch.sigmoid(1.5*old_mean-2.2)).abs().mean())).detach())
            rates.append(algorithm.learning_rate)
        return kl
    actor.get_kl_divergence=observe
    try:
        losses=algorithm.update()
    finally:
        if had_local:actor.get_kl_divergence=original
        else:del actor.get_kl_divergence
    if not records:raise RuntimeError('No Gaussian KL observations; requires adaptive PPO')
    values=torch.stack(records).cpu();rates.append(algorithm.learning_rate)
    diagnostics=dict(minibatches=len(records),kl_mean=float(values[:,0].mean()),kl_max=float(values[:,0].max()),mean_shift_rms_std_units=float(values[:,1].mean()),mean_absolute_excitation_mean_change=float(values[:,2].mean()),learning_rate_min=min(rates),learning_rate_max=max(rates),note='Per-minibatch old-rollout versus current distribution. Not final full-rollout KL or gait evidence.')
    return losses,diagnostics
