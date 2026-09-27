"""Render native MuJoCo models compiled through mjlab Entity (no dynamics)."""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg

ROOT = Path('/home/k_miyazawa/musclemimic/.venv/lib/python3.11/site-packages/musclemimic_models/model')
OUT = Path('/tmp/muscle-feasibility/renders')
OUT.mkdir(exist_ok=True)
for name, relative in [('bimanual', 'arm/myoarm_bimanual.xml'), ('fullbody', 'body/myofullbody.xml')]:
    path = ROOT / relative
    source = mujoco.MjSpec.from_file(str(path))
    entity = Entity(EntityCfg(
        spec_fn=lambda path=path: mujoco.MjSpec.from_file(str(path)),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(
            target_names_expr=tuple(a.target for a in source.actuators),
            transmission_type=TransmissionType.TENDON,
        ),)),
    ))
    model = entity.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Display only: hide collision/wrapping guides, show anatomical meshes and paths.
    options = mujoco.MjvOption()
    options.geomgroup[:] = [1, 1, 0, 0, 0, 0]
    options.sitegroup[:] = 0
    options.tendongroup[:] = 1
    options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    model.tendon_rgba[:] = [0.9, 0.13, 0.17, 1]
    model.tendon_width[:] = np.minimum(model.tendon_width, 0.003)
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 1000
    low, high = data.xpos[1:].min(axis=0), data.xpos[1:].max(axis=0)
    # Ignore world-anchored wrapper when framing the fixed-base arms.
    if name == 'bimanual':
        positions = data.xpos[2:]
        low, high = positions.min(axis=0), positions.max(axis=0)
    center = (low + high) / 2
    extent = float(np.max(high - low))
    print(name, 'bounds', low, high, 'center', center, flush=True)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(1.0, extent * 2.0)
    cam.elevation = -8
    frames = []
    with mujoco.Renderer(model, height=1000, width=900) as renderer:
        for label, azimuth in [('Front', 90), ('Three-quarter', 135)]:
            cam.azimuth = azimuth
            renderer.update_scene(data, camera=cam, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            frame = Image.fromarray(renderer.render())
            frame.save(OUT / f'{name}_{label.lower()}.png')
            frames.append(frame)
    canvas = Image.new('RGB', (1800, 1080), '#10141c')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    title = f'{"MyoFullBody" if name == "fullbody" else "MyoBimanualArm"}  |  {model.nu} muscle actuators  |  Loaded through mjlab'
    draw.text((30, 20), title, font=font, fill='#edf1f7')
    for i, frame in enumerate(frames):
        canvas.paste(frame, (i * 900, 80))
    canvas.save(OUT / f'{name}.png')
    print('SAVED', OUT / f'{name}.png', flush=True)
