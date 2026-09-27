"""Checkpointable cap on Gaussian exploration, applied between PPO rollouts."""
import math
import torch


def cap_exploration(distribution, schedule, iteration):
    """Never change actions/log-probabilities within a collected PPO batch."""
    if schedule is None:
        return None
    progress = min(1., max(0., (iteration-schedule['start_iteration'])/schedule['updates']))
    cap = schedule['initial'] * (schedule['final']/schedule['initial'])**progress
    with torch.no_grad():
        distribution.log_std_param.clamp_(max=math.log(cap))
    return cap


def reset_exploration(distribution, optimizer, initial, start_iteration, final=None, updates=600):
    """Explicit between-run covariance reset; preserve the learned mean and other optimizer state."""
    final=initial if final is None else final
    if not all(math.isfinite(x) for x in (initial,final)) or not .05<=final<=initial<=1. or updates<=0:
        raise ValueError('Invalid exploration reset schedule')
    parameter=distribution.log_std_param
    previous=float(parameter.detach().exp().clamp(.05,1.).mean())
    had_state=parameter in optimizer.state
    with torch.no_grad():parameter.fill_(math.log(initial))
    parameter.grad=None
    optimizer.state.pop(parameter,None)
    schedule=dict(start_iteration=start_iteration,initial=initial,final=final,updates=updates)
    metadata=dict(previous_std_mean=previous,initial_std=initial,final_cap=final,std_optimizer_state_cleared=had_state,mean_policy_preserved=True,note='Gaussian mean and other optimizer state preserved. Sigmoid excitation mean can change with covariance. Reset occurs before collecting a new PPO rollout.')
    return schedule,metadata
