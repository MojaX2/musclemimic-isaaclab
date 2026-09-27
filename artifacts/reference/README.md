# Training reference assets

The two small derived arrays used by the supported training presets are included:

- `projected_cycle_walking_states.npz`: 86 reference reset states, projected onto the 51 model joint equalities.
- `periodic_cartesian_fullcycle_targets.npz`: four-harmonic fit of 16 pelvis-relative body sites over a 112-frame walking cycle.

Source: AMASS / KIT `314/walking_medium09_poses`, retargeted for MuscleMimic MyoFullBody; raw frames 443:555 at 100 Hz. Original preparation scripts are archived under `examples/isaaclab_newton_fullbody/prepare_*`. The source motion dataset itself is not included. These derived motion assets retain the applicable source dataset terms; the repository's code license does not replace them. This private experiment archive is not a new public motion dataset.

The arrays are reward/reset guidance, not forces and not policy actions. They are unnecessary for inference from the saved checkpoint. Phase frequency is fixed at 100/112 Hz; it is not command-conditioned. Fresh training and continued training can execute with these arrays; successful convergence across speeds is not guaranteed.
