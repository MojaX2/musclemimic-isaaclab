# Supported entry-point validation

Validated on Linux / RTX 5090 using the pinned existing Isaac Lab environment. A separate working copy was created with **no outputs, virtualenv, or cache copied from the experiment**. `musclemimic_models==1.0.6` and `imageio-ffmpeg==0.6.0` were installed from their distributions into an isolated dependency directory. Model assets were discovered automatically, without pointing to the original workspace.

| Entry point | GPU worlds | PPO updates | Evaluation | Result |
|---|---:|---:|---:|---|
| `scripts/run.py doctor` | — | — | — | Versions, GPU, model and encoder found |
| `scripts/run.py train --iterations 2` | 4 | 2 fresh | 1 s | Checkpoint, evaluation JSON, MP4 generated |
| `scripts/run.py evaluate` | 4 | 0 | 20 s | Checkpoint loaded, JSON and MP4 generated; all four survived |
| `scripts/run.py train --preset velocity --resume … --iterations 2` | 4 | 2 resumed | 1 s | Iteration 5716→5718; actor tensors changed; checkpoint, JSON, MP4 generated |

All tested evaluation external-wrench maxima were exactly zero. Fresh training is only a software smoke test; a policy trained for two updates is not expected to walk. The four-world inference is one sample per command and does not replace the original 4096-world performance evaluation. The resumed two-update run is likewise not a new gait-quality experiment.

See `summary.json` and the saved metrics. Video files from these tests remain local to avoid duplicating media in Git; the selected checkpoint GIF is in `artifacts/best-policy/`.

Syntax and machine-specific path checks passed for the supported runner, adapter, trainer, renderer and compatibility evaluation entry point. Historical auxiliary experiment scripts are retained as archived research utilities; the supported end-to-end instructions are the root README.

Limit: a complete fresh install of Isaac Lab, OS drivers and system EGL libraries on a second machine was not performed. The README pins the same upstream commit and package versions as the validated runtime. Installation success and numerical reproducibility on other hardware are not guaranteed.
