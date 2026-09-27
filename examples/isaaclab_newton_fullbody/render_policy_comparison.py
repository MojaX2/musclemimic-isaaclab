"""Render saved physics rollouts with a common camera; never integrate dynamics."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('--right');p.add_argument('--output',required=True);p.add_argument('--left-label',default='Original MJX + Warp');a=p.parse_args()
folders=[Path(a.left)]+([Path(a.right)] if a.right else [])
labels=[a.left_label]+(['Isaac Lab + Newton + MuJoCo Warp (corrected)'] if a.right else [])
models=[];data=[];recordings=[];renderers=[];supports=[]
for folder in folders:
    support_path=folder/'assistance.json'
    supports.append(json.loads(support_path.read_text()) if support_path.exists() else None)
    path=folder/'model.xml'
    if not path.exists():path=folder/'trained_model.xml'
    model=mujoco.MjModel.from_xml_path(str(path));model.vis.global_.offwidth=640;model.vis.global_.offheight=640
    floor=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    if floor>=0:model.geom_group[floor]=0
    model.tendon_width[:]=np.minimum(model.tendon_width,.0025)
    models.append(model);data.append(mujoco.MjData(model));recordings.append(np.load(folder/'rollout.npz'));renderers.append(mujoco.Renderer(model,height=640,width=640))
width=640*len(folders);height=720;fps=25
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
ffmpeg=next(Path('/tmp/muscle-feasibility/mjlab-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux*'))
writer=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',a.output],stdin=subprocess.PIPE)
opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0];opt.sitegroup[:]=0;opt.tendongroup[:]=1;opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]=True
cam=mujoco.MjvCamera();cam.distance=3.;cam.azimuth=135;cam.elevation=-12
strides=[max(1,int(round(1/(fps*float(r['dt']))))) for r in recordings]
frames=min(len(r['qpos'])//stride for r,stride in zip(recordings,strides))
try:
    for frame in range(frames):
        canvas=Image.new('RGB',(width,height),'#111720');draw=ImageDraw.Draw(canvas)
        for col,(m,d,r,renderer) in enumerate(zip(models,data,recordings,renderers)):
            i=frame*strides[col]
            d.qpos[:]=r['qpos'][i];d.act[:]=r['act'][i];mujoco.mj_forward(m,d)
            cam.lookat[:]=[d.qpos[0],d.qpos[1],.9]
            m.tendon_rgba[:]=[.25,.1,.12,1]
            m.tendon_rgba[m.actuator_trnid[:,0],:3]=[.3,.08,.1]+np.clip(d.act[:,None]/.35,0,1)*[.65,.47,.02]
            renderer.update_scene(d,camera=cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]=False
            canvas.paste(Image.fromarray(renderer.render()),(col*640,45));draw.text((col*640+12,12),labels[col],font=font,fill='white')
        for col,r in enumerate(recordings):
            i=frame*strides[col]
            if 'command' in r and 'velocity' in r:
                resets=int(np.sum(r['reset'][:i+1])) if 'reset' in r else 0
                footer=f"{frame/fps:05.2f}s | Target {r['command'][i]:.2f} | Speed {r['velocity'][i]:.2f} m/s | Resets {resets}"
            else:footer=f'{frame/fps:05.2f} s | 1x | Resets on falls | Recorded states'
            draw.text((col*640+12,692),footer,font=font,fill='white')
        for col,support in enumerate(supports):
            if support and support.get('controller')=='pelvis_pd':
                text=(f"Pelvis PD {support['strength']:.0%} | gravity {support.get('weight_support',.3):.0%} BW | cap {support.get('vertical_limit',.6):.0%} BW" if support['strength'] else "Pelvis PD OFF | Zero external force")
                draw.rectangle((col*640+6,49,col*640+634,77),fill='#111720')
                draw.text((col*640+12,52),text,font=font,fill='#79e8ff')
            elif support and support['fraction_body_weight']>0:
                text=f"{support['body'].capitalize()} lift: {support['force_world_N'][2]:.1f} N upward ({support['fraction_body_weight']:.0%} body weight)"
                draw.rectangle((col*640+6,49,col*640+634,77),fill='#111720')
                draw.text((col*640+12,52),text,font=font,fill='#79e8ff')
        writer.stdin.write(np.asarray(canvas).tobytes())
        if frame in [0,25,75,150,300,600]:canvas.save(Path(a.output).with_suffix(f'.{frame:03d}.png'))
finally:
    writer.stdin.close();assert writer.wait()==0
    for r in renderers:r.close()
print('VIDEO',a.output,'frames',frames,'seconds',frames/fps)
