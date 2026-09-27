# G1-inspired unassisted muscle walking experiment

`train_velocity_g1.py` is a separate training entrypoint. It leaves
`train_velocity.py`, existing checkpoints, and running experiments untouched.
This experiment starts from scratch with **zero external assistance**. No upward
force callback is registered, positive `--support`/`--eval-support` is rejected,
and checkpoints from the assisted/custom-reward experiment cannot be resumed.
The original implementation was subsequently launched as
`outputs/isaac_velocity_g1/run_001`; current experiment history is recorded in
`outputs/isaac_velocity_g1/EXPERIMENT_STATUS.md`.

## Sources

The design follows the installed Isaac Lab G1 Flat task, including the rewards
inherited from G1 Rough and the generic velocity task:

- `/home/k_miyazawa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py`
- `/home/k_miyazawa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/rough_env_cfg.py`
- `/home/k_miyazawa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/rewards.py`
- `/home/k_miyazawa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py`

This is an independent implementation of that reward structure, adapted to the
musculoskeletal model. It is not a port of the complete G1 environment, terrain,
actuation, command generator, or domain randomization setup.

## Reward rates

Rate terms are multiplied by the 0.02 s control interval, as in Isaac Lab's reward
manager. In particular, a -200 termination rate means -4 on a falling step;
time-limit truncation has no termination penalty. No positive reward clipping
is applied.

| Term | Weight | Interpretation/adaptation |
|---|---:|---|
| Planar velocity tracking | 2 | exp(-squared XY error / 0.25), anatomical heading frame; increased from 1 in v3 |
| World yaw-rate tracking | 1 | exp(-yaw rate squared / 0.25); commanded yaw rate is zero |
| Vertical velocity squared | -0.2 | World root vertical velocity |
| Roll/pitch angular velocity squared | -0.05 | World angular velocity; XY norm is heading-invariant |
| Non-upright orientation | -1 | Squared horizontal projected gravity |
| Falling | -200 | Root height/tilt/nonfinite criteria and severe trunk collapse (v2) |
| Biped air time | 0.75 | Min(stance duration, opposite swing duration), capped at 0.4 s; exactly one foot must contact the floor |
| Foot sliding | -0.1 | Sum of contacted feet's planar speed |
| Ankle soft position limits | -1 | Ankle and subtalar joints; 90% of hard range |
| Hip deviation | -0.1 | Hip adduction/rotation L1 deviation from reset pose |
| Arm deviation | -0.1 | Shoulder, elbow and forearm L1 deviation |
| Torso deviation | -0.1 | Three primary spine DOFs, without counting equality-coupled vertebrae again |
| Joint acceleration squared | -1e-7 | Three hip axes and knee flexion on each leg |
| Joint muscle torque squared | -2e-6 | Same eight leg DOFs; generalized actuator force, not excitation |
| Excitation change | -0.005 | 23 × mean squared change of bounded muscle excitations |
| Excitation effort | -0.05 | Mean squared muscle excitation; additional muscle-specific term |
| Prolonged double support | -0.5 | Moving commands only; zero for first 0.3 s of consecutive double support, ramps to full penalty at 0.6 s (v3) |

The action-change term is normalized to a nominal 23-action scale rather than
summing all 354 muscle channels. Robot PD position commands and muscle
excitations have different units, so this coefficient is an initial tuning
choice, not an assertion of physical equivalence. The model has no articulated
G1 finger counterparts, so that deviation term is omitted. Global leg posture
matching, root-height reward, fixed-heading reward and the old phase-sinusoid
matching reward are removed. Root height still contributes to fall detection.

Version `g1_muscle_v2_torso` additionally terminates when the head-to-pelvis
vertical separation drops below 60% of its reset value (about 0.359 m).
Evaluations of v1 showed a folded trunk while the pelvis still satisfied the
root-only validity checks. This model-specific criterion permits moderate lean
but prevents counting severe trunk collapse as upright survival. Reward weights
are unchanged. Valid unassisted v1 checkpoints can be continued under v2, but
survival metrics before and after this criterion are not directly comparable.

Version `g1_muscle_v3_movement` addresses the stationary solution seen in run_004:
nearly continuous double support and no alternating touchdown reward. It doubles
velocity tracking weight and penalizes prolonged double support while retaining
the 5 cm swing target, flight, scuff and prolonged single-support penalties.
This is an experimental change, not evidence of walking. Falling still costs
-4, and short normal double-support transitions and commanded standing are allowed.

At checkpoint 1176, stochastic evaluation survived 13–16 s on average but barely
moved, whereas both deterministic latent-mean and expected-excitation evaluation
fell near 1.9 s. The normalizer had 138 million observations; missing normalization
updates were ruled out. The muscle activation filter has asymmetric rise/fall
time constants, so averaging excitation need not reproduce noisy actuation.
The precise contribution of that filter remains unisolated.

Optional `--exploration-std-final .15 --exploration-anneal-updates 600` reduces
the maximum latent Gaussian standard deviation geometrically from the loaded
maximum to 0.15, without increasing any smaller learned deviations. Clamping is
between PPO batches, never during an update of a collected batch. The schedule
is checkpointed with its original iteration and survives resumes. This tests
whether adaptation to smaller exploration improves deterministic behavior;
it is not a guarantee. Evaluation uses raw latent-mean actions unless explicitly
requested otherwise. No external force or pretrained assisted policy is added.

`training_episodes_this_run` counts completed training episodes since the current
process began and excludes evaluation episodes. Old cumulative episode counts
cannot be reconstructed exactly from the retained rolling episode summaries.

### v4: dense alternating clearance progress

The v3 experiment completed its 600-update anneal and remained stationary. At
checkpoint 1866, a separate 64-world sampled evaluation survived 18–20 s on
average, while the 4096-world deterministic evaluation survived 1.9–3.3 s.
Neither tracked moving commands. The observation normalization and action
mapping share the same code paths; this does not prove all possible implementation
differences absent. The noise dependence remains unresolved.

`g1_muscle_v4_phase_clearance` adds a rate term of weight 1.0 using the already
observed 1.1 Hz phase. Right and left target clearances are respectively
`0.05 * max(sin(phase), 0)` and `0.05 * max(-sin(phase), 0)` metres. The per-foot
score is `(target² - (height-target)²) / 0.05²`, bounded to [-4,1], with physical
clearance floored at zero. Sum both feet; positive scores require contact of the
intended stance foot, and the term is disabled for commands <=0.1 m/s.

Subtracting the grounded baseline makes stationary feet earn zero. Small lifts
can improve reward before the swing foot's contact force crosses the contact
threshold. Wrong-side lifts and holding one foot up across the cycle lose reward;
flight cannot collect a positive score. Existing 5 cm swing, alternating touchdown,
scuff, flight, long-swing and velocity terms remain. This synthetic foot-height
target is a new shaping term, not a reference motion dataset, imposed joint
trajectory, external force, or imported pretrained policy.

The next experiment retains the learned std cap 0.15 and uses explicitly labelled
sampled evaluation, with deterministic diagnostics kept separately. No deterministic
performance improvement is claimed by switching the evaluation mode. Fifteen CPU
tests cover the new incentives as well as the earlier sensor/posture regressions.

The initial anatomical forward direction is world -X, not necessarily the free
joint's local X axis. Planar tracking accounts for this offset. MuJoCo free-root
angular velocity is converted from local to world coordinates before yaw-rate
and roll/pitch penalties. Evaluation uses the same heading-relative forward
speed as the reward. Commands are forward-only, uniformly sampled from 0–1 m/s,
with an additional 10% exact-standing probability; there is no assistance or
speed curriculum in this variant.

## Extra shaping for lifting the feet

These terms are deliberate additions, **not standard G1 rewards**. They are
active for walking commands above 0.1 m/s unless otherwise stated:

- Swing clearance: weight +0.5, Gaussian target **5 cm** with 2.5 cm width;
  requires single support and a swing duration at most 0.5 s. Both-grounded and
  both-airborne states receive zero clearance reward.
- Scuffing: weight -0.5, normalized clearance deficit below **2.5 cm** during
  0.10–0.35 s of swing. Liftoff and early landing are not directly penalized.
- Alternating touchdown: **+0.15 per event**, not multiplied by dt. Requires a
  preceding 0.12–0.60 s swing with peak clearance at least **4 cm**, and the foot
  must differ from the last rewarded touchdown. Simultaneous landings do not
  earn this bonus.
- Flight: weight -0.5 when neither foot contacts the floor, also during standing.
- Prolonged swing: weight -1 after 0.6 s; the air-time reward is also disabled
  then. This closes the incentive to hold a single foot up indefinitely.

These incentives promote foot clearance and alternating support. They do not
by themselves guarantee human gait, heel-to-toe rolling, or correct knee/hip
coordination. Validate video, speed tracking, swing peaks, contact patterns and
falls before judging the learned policy.

## Sensing and state

`g1_muscle_sensors.py` calls the public MuJoCo Warp contact-force API and sums
positive contact-normal forces for each foot's authored collision shapes
against plane geometries. A foot contacts the floor when the sum exceeds 1 N.
Self-contact does not count. Forces are sampled at the 50 Hz control boundary;
this implementation does not use G1's contact-history maximum. Foot speed is a
control-step finite difference of the heel body's world position, rather than
Isaac Lab's instantaneous link velocity. Terrain support is restricted to the
flat-plane experiment.

Clearance uses the existing collision-shape calculation, including capsule and
ellipsoid extents; it is not ankle-joint height. Air time, stance duration,
swing peak and last rewarded touchdown reset per environment. Selected
anatomical DOFs are mapped by names, avoiding assumptions about Newton's joint
ordering. Individual weighted contributions are logged to TensorBoard/JSON;
`reward_config.json` records the rate weights and touchdown bonus.

## Validation and launch

CPU tests cover the flat-foot loophole, flight, lift versus scuff, prolonged
single support, alternating landings, standing commands, partial resets,
termination/slip scales, anatomical coordinate transforms, real model joint
selection, and contact-force agreement with MuJoCo CPU for both friction cones
and multiple worlds. With the added torso criterion, all 11 CPU tests passed.
A 4-environment GPU smoke run also passed physics, selected resets, PPO update,
and evaluation with zero external wrench. Learning convergence is not established.

For v3, all 13 CPU tests passed, including normal double-support transfer versus
prolonged standing and checkpoint-consistent exploration annealing. A GPU run
with four environments completed two PPO updates, selected reset/clearance
checks and a two-second evaluation with exactly zero external wrench. Its saved
checkpoint retained the exploration schedule. This short smoke run establishes
execution correctness only; it does not establish gait or long-term stability.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/k_miyazawa/IsaacLab/.venv/bin/python -m unittest discover \
  -s examples/isaaclab_newton_fullbody -p test_g1_muscle_rewards.py -v

# Run separately when the training GPU is available (not launched by this change):
/home/k_miyazawa/IsaacLab/.venv/bin/python \
  examples/isaaclab_newton_fullbody/train_velocity_g1.py \
  --run-dir outputs/isaac_velocity_g1/run_001 --num-envs 1024
```

The trainer retains hourly evaluation/video scheduling from the existing
trainer, with its first evaluation after five minutes. The videos use the
existing rollout renderer. Begin with a small `--num-envs 4 --smoke --iterations
1` run before committing GPU time to the new experiment.
