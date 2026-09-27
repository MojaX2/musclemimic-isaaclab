"""Render recorded GPU states only; this script never integrates dynamics."""
from pathlib import Path
import json

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path('/tmp/muscle-feasibility/renders')
SOURCE = Path(__file__).resolve().parents[2] / '.venv/lib/python3.11/site-packages/musclemimic_models/model/body/myofullbody.xml'
recording = np.load(OUT / 'isaaclab_random_motion.npz')
model = mujoco.MjModel.from_xml_path(str(SOURCE))
data = mujoco.MjData(model)
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 720
model.tendon_width[:] = np.minimum(model.tendon_width, .0025)
options = mujoco.MjvOption()
options.geomgroup[:] = [1, 1, 0, 0, 0, 0]
options.sitegroup[:] = 0
options.tendongroup[:] = 1
options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
camera = mujoco.MjvCamera()
camera.distance = 3.3
camera.elevation = -12
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
path = OUT / 'isaaclab_newton_fullbody_random.mp4'
writer = imageio_ffmpeg.write_frames(str(path), (1280, 832), fps=int(recording['fps']),
    codec='libx264', pix_fmt_out='yuv420p', output_params=['-crf', '19', '-movflags', '+faststart'])
writer.send(None)
previews = []
try:
    with mujoco.Renderer(model, height=720, width=640) as renderer:
        for frame, (pose, activation, time) in enumerate(zip(recording['qpos'], recording['act'], recording['times'])):
            data.qpos[:] = pose
            data.act[:] = activation
            # Compute transforms and tendon paths for visualization; do NOT call mj_step.
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [pose[0], pose[1], max(.42, pose[2]*.65)]
            bright = np.clip(activation/.35, 0, 1)
            model.tendon_rgba[:] = [.22, .24, .28, 1]
            model.tendon_rgba[model.actuator_trnid[:, 0], :3] = np.array([.30, .08, .10]) + bright[:, None]*np.array([.65, .47, .02])
            canvas = Image.new('RGB', (1280, 832), '#111720')
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 12), 'MyoFullBody | 416 muscles | Random excitation', font=font, fill='#eef2f8')
            draw.text((22, 43), 'Physics: Isaac Lab -> Newton -> MuJoCo Warp (GPU) | No balance controller | 1x speed', font=small, fill='#bac8d8')
            for view, angle in enumerate([90, 140]):
                camera.azimuth = angle
                renderer.update_scene(data, camera=camera, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
                canvas.paste(Image.fromarray(renderer.render()), (view*640, 76))
            draw.text((22, 808), f't = {time:.2f} s | Brighter muscle paths = greater activation | Offline rendering of recorded GPU poses', font=small, fill='#eef2f8')
            writer.send(np.asarray(canvas))
            if frame in [0, 30, 90, 150]:
                previews.append(canvas.copy().resize((640, 416)))
            if frame % 30 == 0:
                print('RENDER', frame, '/', len(recording['qpos']), flush=True)
finally:
    writer.close()
preview = Image.new('RGB', (1280, 832))
for i, frame in enumerate(previews):
    preview.paste(frame, ((i%2)*640, (i//2)*416))
preview.save(OUT/'isaaclab_random_preview.jpg')
print('VIDEO', path, 'bytes', path.stat().st_size, flush=True)
