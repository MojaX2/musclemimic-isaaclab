# Isaac Lab / Newton full-body feasibility experiment

An external adapter runs the MuscleMimic full-body model through Isaac Lab's
`SimulationContext -> custom NewtonMJWarpManager -> custom SolverMuJoCo -> MuJoCo Warp`.
No Isaac Lab or installed Newton source files were edited. This is a version-specific
prototype using internal extension hooks, not a completed Articulation or RL environment.

## Verified environment

- Local Isaac Lab: `v3.0.0-EA-26-gbe270ed88` (3.0 early-access checkout, not a verified final release)
- Newton 1.6.0, MuJoCo / MuJoCo Warp 3.12.0, Warp 1.17.0
- NVIDIA RTX 5090, CUDA device 0
- See `results.json` for the observed results. The timings are smoke-test timings,
  not a benchmark comparable to the previous mjlab throughput measurements.

## Run

Uses the existing Isaac Lab environment without dependency synchronization. The adapter
locates the original model in this repository's Python 3.11 virtual environment; adjust
`SOURCE` if the model package is elsewhere. Generated model, cache, and JSON are written
under `/tmp/muscle-feasibility`.

```bash
UV_CACHE_DIR=/tmp/muscle-feasibility/uv-cache \
MPLCONFIGDIR=/tmp/muscle-feasibility/mpl \
uv run --no-sync --project /home/k_miyazawa/IsaacLab python \
  /home/k_miyazawa/musclemimic/examples/isaaclab_newton_fullbody/check_isaaclab_fullbody.py
```

Isaac Sim / AppLauncher is not needed: the test uses Isaac Lab's kitless Newton mode.
GPU access is required.

## What passed

- 4 parallel worlds, 416 muscle actuators, 416 activation states, 424 spatial tendons,
  and all 51 original joint equalities reconstructed in the solver.
- 500 physical steps per world at 2 ms, random independent muscle excitations,
  CUDA Graph enabled, MuJoCo contacts enabled.
- Finite position, velocity, activation and force checks at each 100-step checkpoint;
  overflow flags checked every step; no overflow flags detected.
- Nonzero muscle force and motion; different activation states across environments.
- Solver reset hook clears activation in one selected environment and leaves the other
  three activation buffers unchanged. This is not a test of full RL episode resets.
- Native actuator gain/bias/dynamics parameters, length ranges and tendon target indices
  agree with the source model after conversion.

## Adapter workarounds

1. Resolve mesh paths and normalize the MJCF; use explicit compiled mass/inertia values.
2. Normalize hyphens in names and references: Newton's tendon lookup otherwise misses sites.
3. Expand muscle defaults into explicit general-actuator parameters.
4. Bypass generic equality import and reconstruct the original 51 equalities using the
   scalar-DOF-to-exported-joint mapping; retain the higher-order polynomial coefficients.
5. Restore muscle parameters on the generated MuJoCo spec. Restore GPU
   `actuator_lengthrange` after initialization/model refresh: it was all zeros on the
   uncorrected GPU path and led to explosive forces and NaNs.
6. Select solver options explicitly instead of importing MJCF option overrides.

## Limits

- Initial actuator lengths differ from native MuJoCo by up to **0.271 mm**. The test
  records this difference; it does not assert geometric or trajectory equivalence.
- Newton emits warnings about contact margins being zeroed for its native CCD setup.
  Contact and joint-limit semantics and complete trajectories still need comparison.
- The 51 constraints are injected at the MuJoCo solver layer; Newton's generic constraint
  metadata does not expose them. This adapter targets MuJoCo Warp only.
- Direct `control.mujoco.ctrl` input and solver-state observations are used. No stock
  Isaac Lab spatial-tendon API, Articulation wrapper, ActionTerm, reward computation,
  ManagerBasedRLEnv, rendering, PPO training, or 4096/8192-world scaling was tested here.
- Internal method overrides and module-level model mapping assume this single model and
  these dependency versions. Upstream fixes or a maintained integration are needed for
  general-purpose support.

## Random-motion video

`record_random_motion.py` records six seconds of actual GPU simulation (four worlds,
world 0 saved) with smoothly interpolated random excitation. It checks finite recorded
states and per-step overflow flags. `render_recorded_motion.py` maps the recorded joint
coordinates back to the original anatomical model and renders two views. Rendering uses
`mj_forward` only; it does not run a second dynamics simulation. Playback is 30 fps, 1x.

Output: `/tmp/muscle-feasibility/renders/isaaclab_newton_fullbody_random.mp4`.
The NPZ recording and JSON diagnostics are kept in the same directory. No balance controller
is present, so falling is expected.

Record using the same Isaac Lab `uv run --no-sync --project ... python` command above,
substituting `record_random_motion.py`. For rendering, this workspace's isolated visualization
dependencies can be used without changing Isaac Lab's environment:

```bash
PYTHONPATH=/tmp/muscle-feasibility/mjlab-packages \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
LIBGL_ALWAYS_SOFTWARE=1 \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
UV_CACHE_DIR=/tmp/muscle-feasibility/uv-cache \
uv run --no-sync --project /home/k_miyazawa/IsaacLab python \
  /home/k_miyazawa/musclemimic/examples/isaaclab_newton_fullbody/render_recorded_motion.py
```

## Direct Newton viewer recording

`record_direct_viewer.py` runs one world and records Isaac Lab's actual
`NewtonGLVisualizer.render_rgb_array()` framebuffer during simulation (1280x720,
30 fps, six seconds). It runs the GL viewer headlessly via EGL; it is not a desktop
window/screen recording and does not replay poses through MuJoCo's renderer.

Muscle paths are read live from the GPU solver's `wrap_xpos`, `ten_wrapadr`, and
`ten_wrapnum` buffers and drawn with Newton's `log_lines`. Line color follows current
muscle activation. Non-mesh/non-plane visual shapes are hidden to remove wrapping
helper geometry; physical shapes and contacts remain active. The custom manager's
name starts with `Newton`, as required by Isaac Lab's backend detection for visualizers.

Run with the existing Isaac Lab environment (no PYTHONPATH override):

```bash
UV_CACHE_DIR=/tmp/muscle-feasibility/uv-cache \
MPLCONFIGDIR=/tmp/muscle-feasibility/mpl \
uv run --no-sync --project /home/k_miyazawa/IsaacLab python \
  /home/k_miyazawa/musclemimic/examples/isaaclab_newton_fullbody/record_direct_viewer.py
```

The script currently uses the ffmpeg executable from the isolated
`/tmp/muscle-feasibility/mjlab-packages/imageio_ffmpeg/binaries` installation for encoding.
Outputs are `isaaclab_newton_viewer_direct.mp4`, `isaaclab_direct_viewer.json`, and selected
PNG framebuffer captures under `/tmp/muscle-feasibility/renders`.

## Historical pretrained policy trial (before alignment fixes)

`evaluate_pretrained.py` runs the repository's existing evaluation entry point and
unchanged PPO inference. `MM_BACKEND=isaaclab` starts `pretrained_physics_worker.py`
in Isaac Lab's Python environment and exchanges controls and state through an
authenticated local socket. It exports the actual training environment's MjSpec;
`MM_MODEL_XML` selects that model in the external adapter.

The cached checkpoint is `amathislab/mm-10m-2`, checkpoint 12500, revision
`651078fcdd84bd4fa45a096dc0564125c4bb8a25`. With fingers disabled it uses 354 muscle
actuators, nq=89, nv=88, and 2418 raw observations. Deterministic inference, trajectory
`KIT/314/walking_medium09_poses`, start frame 0, control period 10 ms, physics period
2 ms, and the source model's Euler integrator were used. No retraining was performed.

```bash
MM_BACKEND=isaaclab JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 \
PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/muscle-feasibility/uv-cache \
MPLCONFIGDIR=/tmp/muscle-feasibility/mpl \
.venv/bin/python examples/isaaclab_newton_fullbody/evaluate_pretrained.py \
  --path /home/k_miyazawa/.cache/huggingface/hub/models--amathislab--mm-10m-2/snapshots/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses \
  --use_mujoco --no_render --n_steps 700 --traj_index 0
```

Omit `MM_BACKEND=isaaclab` to run the original native MuJoCo baseline. `--use_mujoco`
selects the original CPU observation/task API; in the Isaac Lab run its `mj_step`
is replaced by the IPC bridge, so **all dynamics integration runs in Isaac Lab's
Newton/MuJoCo Warp backend on the GPU**. JAX policy inference and original task logic
remain in the original Python environment on CPU. Native `mj_forward` reconstructs
observations (including touch sensors); muscle force/length/velocity are copied from
the GPU. This is a hybrid feasibility test, not a complete native Isaac Lab RL task
or a demonstration of identical observations across physics versions.

Results: 700 control steps, three full episodes plus a partial episode. The native
MuJoCo 3.4 baseline fell and terminated at 2.01 s each episode; the Isaac Lab Newton /
MuJoCo Warp 3.12 run terminated at 2.14 s. No GPU overflow flags or non-finite checked
states/muscle outputs occurred. **Loading/inference/control integration succeeded;
stable walking did not.** Failure in the baseline means this trial does not establish
that transfer alone caused the falls. Solver tolerances, native CCD/contact margin
changes, and observation reconstruction remain differences to investigate.

Outputs under `/tmp/muscle-feasibility/pretrained/`: `baseline.json`, `isaaclab.json`,
`worker_results.json`, `model_info.json`, and `pretrained_isaaclab.mp4`. The video is
recorded directly from the live Newton GL viewer at 25 fps, including episode resets.

An additional original MJX/Warp run (MuJoCo 3.4) also lost upright posture near
2 s and reset before step 250 of 300. The initial default contact capacity emitted
an overflow warning; that run was discarded. The completed comparison increased
`nconmax` to 1024 and JIT-compiled the original `env.mjx_step` without changing its
logic. Its log is `/tmp/muscle-feasibility/pretrained_mjx_fixed.log`; the small
comparison wrapper is `/tmp/muscle-feasibility/check_original_mjx.py`. This supports
checking checkpoint/current-code/reference-motion compatibility before attributing
the unsuccessful gait specifically to Isaac Lab.

### Thirty-second video with random starts

The earlier `--traj_index 0` fixed both trajectory and start frame to zero. In the
original `TrajectoryHandler`, defaults are `random_start=True` and
`start_from_random_step=True`: each reset samples a trajectory and a frame, and
`TrajInitialStateHandler` initializes pose and velocity from that frame. This is
random sampling, not a gradual perturbation of the same initial pose.

To record 30 seconds with the original random-start behavior, use the same command
with `--n_steps 3000` and **omit `--traj_index 0`**. Set:

```bash
MM_OUTPUT_DIR=/tmp/muscle-feasibility/pretrained_random_30s
MM_RESET_SEED=42
MM_FOLLOW_CAMERA=1
```

These optional environment variables select a separate output folder, seed the
existing NumPy reset sampler for reproducibility, and keep the live viewer camera
centered on the pelvis horizontally. Output includes `isaaclab_resets.json` with
actual sampled frame indices and `isaaclab.json` with episode outcomes. The policy
remains deterministic; initial states vary between episodes. A single selected
walking motion is used, so only its starting frame varies.

## Corrected MJX policy comparison

The earlier trial above used native `MyoFullBody` physics defaults and reconstructed
observations with a CPU `mj_forward`. These were **not equivalent to the original
MJX evaluation**, even though checkpoint and trajectory were the same. The current
`evaluate_pretrained.py` enables the following fixes by default:

1. When a checkpoint omits `model_option_conf`, preserve `MjxMyoFullBody`'s defaults:
   4 solver iterations, 8 line-search iterations, and `mjDSBL_EULERDAMP`. Pass the
   source iteration counts to the Isaac Lab solver instead of 100/50.
2. Restore source geom margins/gaps and solver tolerance in the exported solver
   spec. Restore margin/gap arrays **again on the GPU after Newton model refresh**;
   fixing only the CPU spec did not fix the actual simulation. The original 1 mm
   margin had been cleared and the implicit gap had become Newton's 0.1 m default.
3. Restore the 110 MuJoCo sensors (including four foot/toe touch sensors) that the
   generic importer had dropped. Copy their GPU outputs and mapped body/site
   kinematics directly into the original observation pipeline. Do not run a new
   CPU `mj_forward` after integration: that changed the sampling time and contact
   forces relative to the original MJX observation.
4. Keep solver generalized coordinates authoritative between physics substeps
   (`update_data_interval=0`). Explicit reset/control-boundary transfers still
   synchronize Newton state for visualization. This setting alone did not explain
   or fix the measured discrepancy.

The switches `MM_MATCH_MJX`, `MM_MATCH_PHYSICS`, `MM_GPU_OBSERVATIONS`, and
`MM_SYNC_INTERVAL` can override these defaults for diagnostic ablations. Other
random-motion examples retain their earlier settings. No installed Isaac Lab or
Newton package files were edited.

`record_original_mjx.py` records the original JIT-compiled `env.mjx_step` and the
unchanged original PPO inference on GPU. It raises contact capacity to 1024 and
samples frames with NumPy seed 42 so reset number N uses the same initial frame as
the Isaac Lab run. Both trials run 3000 control steps (30 s), with resets enabled.
The reset times can differ because each engine uses its own termination decision.

`render_policy_comparison.py` renders the stored states with the same renderer,
camera, muscles and floor for both engines; it does not integrate physics. Its
side-by-side video is an **offline visualization of two actual GPU rollouts**.
The corrected Isaac Lab run also saves its separate live Newton-viewer recording.

```bash
# Original MJX (same checkpoint/motion arguments as above; omit --use_mujoco).
HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python examples/isaaclab_newton_fullbody/record_original_mjx.py \
  --path /home/k_miyazawa/.cache/huggingface/hub/models--amathislab--mm-10m-2/snapshots/checkpoint_12500 \
  --motion_path KIT/314/walking_medium09_poses --no_render --num_envs 1 --n_steps 3000

# Re-run evaluate_pretrained.py with the earlier random-start command and
# MM_OUTPUT_DIR=/tmp/muscle-feasibility/isaac_corrected_30s.

# Meaningful first-transition regression check, requiring both saved trials.
.venv/bin/python examples/isaaclab_newton_fullbody/check_policy_alignment.py \
  /tmp/muscle-feasibility/original_mjx_30s \
  /tmp/muscle-feasibility/isaac_corrected_30s
```

The check fails on the old adapter's saved output and passes after the fixes.
Initial qpos/qvel are identical. At 10 ms, max generalized-coordinate error decreased
from 0.00261 to 8.64e-7, max generalized-velocity error from 0.787 to 0.000147, and raw
observation RMS error from 1.218 to 0.000901. The coordinate/velocity vectors mix
translation and rotation units; these are numerical alignment checks, not a single
physical distance/speed metric. See the JSON results for individual values.
Residual differences remain from float32 model conversion, solver/library versions,
and CPU versus GPU policy arithmetic; long closed-loop trajectories are not bitwise
identical. The original MJX rollout also fails to sustain stable walking with this
cached checkpoint, current code/model and selected reference motion. Aligning the
adapter does not by itself resolve that original-model behavior.

Final 30-second results: original MJX reset 12 times, corrected Isaac Lab reset 11
times; mean rewards were 0.8862 and 0.8834. The first 3 seconds' pelvis-coordinate
RMS difference fell from 0.0327 m before correction to 0.0102 m afterward. This
short-window comparison precedes the first reset in both runs. Original MJX and
corrected Isaac Lab did not sustain stable walking. Contact/constraint capacity
overflows were zero. With the original 4/8 solver budgets, the new Warp version
reported iteration/line-search limits (flags 512/1024); these are distinct from
capacity overflow and were not hidden by increasing solver iterations.

Full metrics: `mjx_alignment_results.json`. Artifacts:

- `/tmp/muscle-feasibility/original_mjx_30s/original_mjx_30s.mp4`
- `/tmp/muscle-feasibility/isaac_corrected_30s/mjx_vs_isaaclab_30s.mp4`
- `/tmp/muscle-feasibility/isaac_corrected_30s/pretrained_isaaclab.mp4` (live viewer)

## Constant upward pelvis assistance

Set `MM_SUPPORT_FRACTION=0.1` on both rollout commands and use `--n_steps 2000`
for 20-second trials. This applies a world +Z force at the `pelvis` body's center
of mass equal to 10% of the complete model's weight (approximately 82.67 N for
84.27 kg). It applies no horizontal force, torque, pose target or velocity target.
The default fraction is zero, preserving the unassisted experiments.

The original MJX route sets `xfrc_applied` in its existing pre-step hook, including
reset initialization. The Isaac Lab route registers a graph-safe Newton state-force
callback before CUDA graph capture so the force is reapplied every physical substep
(after the manager clears accumulated forces). Initial forward evaluation also gets
the same force. Both scripts verify the actual solver external-force values against
the requested force; the Isaac Lab check also excludes accidental accumulation or
forces on other bodies. `assistance.json` records the force and point of application.
The comparison renderer labels the assistance magnitude on both panels.

Set `MM_SUPPORT_BODY=head` to apply the same assistance at the head body's COM
instead. The `head` body contains the `hat_skull` mesh (and jaw); this model has
no separate skull body. The default remains `pelvis`. Force magnitude is still
computed from **total model weight**, not the selected body's weight. The force
metadata and video label identify the selected body.

## Velocity-tracking training (experimental)

`train_velocity.py` trains an independent RSL-RL PPO actor with 354 muscle
excitations, using Isaac Lab's simulation context and Newton/MuJoCo Warp on GPU.
It does not use pretrained imitation weights or reference trajectories in its
reward or observation. An upright pose from the previous reference is used for
reset. The initial command range is 0–0.8 m/s along anatomical forward (-X).
The initial 40% bodyweight head lift is a curriculum aid, not a solved unassisted
walking result. Evaluation success reduces assistance in 10 percentage-point
increments and expands the command range.

```bash
UV_CACHE_DIR=/tmp/muscle-feasibility/uv-cache \
MPLCONFIGDIR=/tmp/muscle-feasibility/mpl PYTHONUNBUFFERED=1 \
uv run --no-sync --project /home/k_miyazawa/IsaacLab python \
  examples/isaaclab_newton_fullbody/train_velocity.py --num-envs 1024 \
  --run-dir outputs/isaac_velocity/run_001
```

The first evaluation is after approximately five minutes, then every hour of
training. Evaluations run for 20 simulated seconds and report survival and speed
error separately at 0, 0.4, 0.8 and 1.2 m/s; assisted and unassisted trials are
separate. Videos show the 0.4 m/s environment, with resets after falls and a
visible assistance label. They render recorded simulation states with MuJoCo;
they are not direct Isaac Lab viewer captures. Checkpoints are saved every five
minutes and before evaluations. `metrics.jsonl`, `evaluations.jsonl`, TensorBoard
and `status.json` expose progress. Creating a `STOP` file inside the run directory
stops after the current PPO iteration and saves `stopped.pt`. `--resume` restores
a checkpoint. Training currently uses identical upright evaluation starts;
robustness to perturbed starts and visible alternating footsteps must be checked
before declaring walking successful. This is a research experiment; reward alone
is not a success criterion.

Training revision 2 fixes assistance-input normalization: empirical variance is
zero during constant-support stages, so evaluating 0% after training at 40%
previously produced a normalized input of -40. `VelocityNormalization` scales
that known dimensionless input with a fixed 0.4 range (0 at 40%, -1 at 0%).
Existing checkpoints retain their other observation statistics and optimizer
state. `--legacy-support-normalization` reproduces the previous behavior.
Run 001 stopped cleanly at iteration 290; run 002 resumes its saved weights with
this correction. The noise-dependent behavior of the assisted policy remains a
separate learning issue, not evidence of walking success. `--eval-only --resume`
can compare checkpoints, and `--stochastic-eval` diagnoses exploration effects.

Run 003 resumes run 002 with `--entropy-coef 0` (now the script default).
The iteration-610 diagnostic still fell after 1.9 seconds deterministically while
training episodes averaged around 8 seconds. Entropy had increased monotonically;
removing the explicit exploration bonus tests whether the inference gap can close.
This does not remove stochastic exploration: the learned Gaussian still samples
during PPO rollouts. No reward, dynamics, action mapping or assistance change is
combined with this experiment. Old runs used `--entropy-coef 0.001`.

`--mean-excitation-eval` diagnoses a second inference gap: applying sigmoid to the
Gaussian latent mean is not the same as the mean muscle excitation under that
Gaussian. This option integrates expected excitation with 9-point Gauss-Hermite
quadrature and runs it deterministically. The integration was checked against
64 points over latent means [-4,4] and std [0.05,1]: maximum absolute excitation
error 5.52e-5. At checkpoint 887, the 0.4 m/s assisted rollout lasted 7.7s instead
of 1.94s using the latent mean. Twenty-second survival was still zero and video
showed simultaneous foot lift; this is not a solved walking result. Training is
unchanged, and ordinary latent-mean evaluation remains available for comparison.

Run 004 resumes run 003 with `--gait-reward 0.5 --mean-excitation-eval`.
The first 6s of the checkpoint-887 mean-excitation rollout spent 28% in flight
at a 0.4 m/s command, with visible simultaneous foot lift. A phase-based foot
clearance reward now encourages alternating 4.5cm swings (60% stance duty cycle),
penalizes simultaneous flight above 5mm, and penalizes vertical root speed.
Commands below 0.1 m/s target both feet down. This adds no motor or prescribed
motion: policy outputs still drive the same 354 muscles. Foot clearance is
computed from actual collision capsule/ellipsoid support extents on GPU. The
formula matches MuJoCo plane geometry distances to floating-point precision on
recorded poses; the smoke check also compares the live GPU values against the
compiled solver model. Recorded videos remain actual integrated rollouts.

Evaluation can additionally use `--eval-phase random` and
`--eval-velocity-perturbation 0.02` to vary phase and initial root velocities.
The perturbation uses a fixed local generator seed (1729) for repeatability.
Initial-velocity perturbation is applied only to the first episode used by the
survival metric; subsequent reset episodes in the video retain the usual reset.
New recordings include exact commanded/actual forward speed, foot clearance,
phase and reset flags. The renderer displays speed and reset count.
Checkpoint 1593 with 40% assistance survived 20s in 8/16 trials at 0.4 m/s,
but only 1/16 at 0.8 m/s; it is not robust enough to reduce support yet.

Run 005 adds heading observations (cos/sin of yaw relative to the reset pose)
and a heading-alignment reward (`--heading-reward 0.5`). The previous observation
contained projected gravity, which does not encode yaw, despite using world-frame
velocity commands. Recorded rollouts drifted about 40 degrees before falling.
This is evidence of a missing observation, not proof that heading explains every
fall. The observation grows from 886 to 888 values; `velocity_checkpoint.py`
inserts zero-weight heading columns and migrates Adam moments and normalization
statistics. Actor/critic outputs before training were checked to agree within
1e-6 on varied inputs. Both heading and support use fixed normalization scales.
The periodic evaluations now use perturbed initial root velocities (std 0.02)
and randomized gait phase. Assistance and muscle dynamics remain unchanged.

At run-005 iteration 2198, all 256 trials each at 0, 0.4 and 0.8 m/s survived
20 seconds with 40% head assistance; mean speed errors were 0.017, 0.026 and
0.132 m/s. The curriculum reduced support to 30%; training lifetime recovered
to ~9.8s in a few minutes. Unassisted trials still fell after ~1.6s.
Run 006 preserves the trained weights and uses `--eval-interval 600` for faster
curriculum checks, with `--video-interval 3600` plus videos at support transitions.
The reward and optimizer configuration are unchanged. `EVALUATE` in the run
folder requests an extra evaluation/video at the next update without restarting.
After each evaluation, the current curriculum state is saved in `latest.pt`.
Support fractions are rounded to avoid a residual floating-point nonzero value.

Final-validation utilities: `--eval-command-sequence 0.4,0.8,0.4,0` divides
`--eval-seconds` into equal command segments (eval-only). Sequence metrics report
survival and speed error over the whole sequence and separately by segment.
The recorded speed overlay follows the changing command. Every evaluation step
checks the actual MuJoCo Warp external wrench: summed vertical force must match
assistance and all other components must be zero. With support zero, the maximum
absolute external wrench component must be exactly zero, not merely a zero
configuration value. These checks run in the evaluator without changing physics.

Important evaluation correction (run 007): the old termination rule used pelvis
height/orientation and missed severe trunk folding. A 0.8 m/s diagnostic at 10%
support folded the upper body while the pelvis remained upright, so old survival
rates alone must not be interpreted as upright walking success. The head-pelvis
vertical gap is now rewarded toward its reset value (~0.599m), and an episode
ends below 60% of that gap (~0.359m). This detects the diagnostic failure at 1.6s.
The previously published 0.4 m/s videos at 40/30/20% support had minimum gaps
0.474/0.443/0.376m respectively; they did not violate this new threshold.
Run 007 resumes run 006 at 10% support, adds `--torso-reward 0.8`, uses 20-second
training episodes (`--episode-seconds 20`), and assigns exact zero commands to
10% of reset trials (`--standing-probability 0.1`). The previous continuous
uniform sampler almost never trained an exact stop command. No hidden joint
controller or extra assistance is introduced. Re-evaluate with the new criterion
before making any final success claim. Trial-level lifetimes and survival flags
are now saved in `trials.npz`; `--video-env-index 2` selects a 0.8 m/s example in
constant-command evaluations.
