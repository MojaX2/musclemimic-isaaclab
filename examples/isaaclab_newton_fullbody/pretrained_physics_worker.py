"""Isaac Lab/Newton physics worker for the original pretrained policy."""
import os,sys,json,subprocess,re
os.environ.pop('DISPLAY',None)
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
import mujoco,newton,warp as wp
from PIL import Image
from isaaclab.sim import SimulationCfg,build_simulation_context
from isaaclab_newton.physics import NewtonCfg,MJWarpSolverCfg,NewtonManager,NewtonMJWarpManager
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
from newton.solvers import SolverMuJoCo
from newton_fullbody_adapter import FullBodySolver,make_builder,REFERENCE,JOINT_DOF
wp.config.kernel_cache_dir='/tmp/muscle-feasibility/warp-cache'
OUT=Path(os.environ.get('MM_OUTPUT_DIR','/tmp/muscle-feasibility/pretrained'))
SUPPORT_BODY=os.environ.get('MM_SUPPORT_BODY','pelvis')
SUPPORT_FRACTION=float(os.environ.get('MM_SUPPORT_FRACTION','0'))
assert 0 <= SUPPORT_FRACTION < 1
SUPPORT_FORCE=float(np.sum(REFERENCE.body_mass)*abs(REFERENCE.opt.gravity[2])*SUPPORT_FRACTION)
@wp.kernel
def add_body_support(body_f:wp.array(dtype=wp.spatial_vector),body_id:int,force:float):
    body_f[body_id]=body_f[body_id]+wp.spatial_vector(0.,0.,force,0.,0.,0.)
class NewtonMuscleManager(NewtonMJWarpManager):
    @classmethod
    def _create_solver(cls,model,solver_cfg):
        return FullBodySolver(model,**cls._filter_solver_kwargs(SolverMuJoCo,solver_cfg))
cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True,update_data_interval=int(os.environ.get('MM_SYNC_INTERVAL','1')),integrator={0:'euler',1:'rk4',2:'implicit',3:'implicitfast'}[int(REFERENCE.opt.integrator)],iterations=int(REFERENCE.opt.iterations),ls_iterations=int(REFERENCE.opt.ls_iterations),nconmax=1024,njmax=2048),use_cuda_graph=True,num_substeps=1)
cfg.class_type=NewtonMuscleManager
viewer_cfg=NewtonGLVisualizerCfg(headless=True,window_width=1280,window_height=720,eye=(4,-4,2),lookat=(0,0,.9),focal_length=24,max_visible_envs=1,show_collision=False,enable_picking=False,streaming_view=False)
with build_simulation_context(sim_cfg=SimulationCfg(dt=float(REFERENCE.opt.timestep),device='cuda:0',physics=cfg,visualizer_cfgs=[viewer_cfg])) as sim:
    template=make_builder()
    for i,t in enumerate(template.shape_type):
        if t not in [newton.GeoType.MESH,newton.GeoType.PLANE]: template.shape_flags[i]&=~int(newton.ShapeFlags.VISIBLE)
    builder=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(builder);builder.add_world(template)
    NewtonManager.set_builder(builder)
    if SUPPORT_FORCE:
        support_indices=[i for i,name in enumerate(builder.body_label) if name.endswith('/'+SUPPORT_BODY)]
        assert len(support_indices)==1,support_indices
        def support_callback(state):
            wp.launch(add_body_support,dim=1,inputs=[state.body_f,support_indices[0],SUPPORT_FORCE],device='cuda:0')
        NewtonManager.register_state_force_callback(support_callback)
    sim.reset()
    s=NewtonManager._solver; d=s.mjw_data; m=s.mj_model
    support_ids=[i for i in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('_'+SUPPORT_BODY)]
    assert len(support_ids)==1
    support_id=support_ids[0]
    external_wrench=np.zeros((1,m.nbody,6),np.float32);external_wrench[0,support_id,2]=SUPPORT_FORCE
    (OUT/'assistance.json').write_text(json.dumps(dict(body=SUPPORT_BODY,fraction_body_weight=SUPPORT_FRACTION,force_world_N=[0.,0.,SUPPORT_FORCE],torque_Nm=[0.,0.,0.],point='body center of mass'),indent=2))
    mujoco.mj_saveModel(m,str(OUT/'solver.mjb'))
    fields=['actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_acc0','actuator_lengthrange','dof_damping','dof_armature','eq_data','body_mass','body_inertia','geom_margin','geom_gap','eq_solref','eq_solimp','jnt_range','jnt_solref','jnt_solimp','jnt_stiffness','qpos_spring','dof_frictionloss','tendon_lengthspring','tendon_solref_lim','tendon_solimp_lim','tendon_solref_fri','tendon_solimp_fri','geom_solref','geom_solimp']
    np.savez(OUT/'gpu_model.npz',**{k:getattr(s.mjw_model,k).numpy() for k in fields})
    dofs=s.mjc_jnt_to_newton_dof.numpy().reshape(-1)
    srcq=list(range(7));dstq=list(range(7));srcv=list(range(6));dstv=list(range(6))
    for j in range(REFERENCE.njnt):
        if REFERENCE.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
        name=mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_JOINT,j).replace('-','_')
        k=np.flatnonzero(dofs==JOINT_DOF[name]);assert len(k)==1
        k=int(k[0]);srcq.append(int(REFERENCE.jnt_qposadr[j]));dstq.append(int(m.jnt_qposadr[k]));srcv.append(int(REFERENCE.jnt_dofadr[j]));dstv.append(int(m.jnt_dofadr[k]))
    assert len(srcq)==REFERENCE.nq and len(srcv)==REFERENCE.nv
    np.savez(OUT/'joint_mapping.npz',srcq=srcq,dstq=dstq,srcv=srcv,dstv=dstv)
    ffmpeg=next(Path('/tmp/muscle-feasibility/mjlab-packages/imageio_ffmpeg/binaries').glob('ffmpeg-linux*'))
    writer=subprocess.Popen([str(ffmpeg),'-y','-f','rawvideo','-vcodec','rawvideo','-s','1280x720','-pix_fmt','rgb24','-r','25','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','19',str(OUT/'pretrained_isaaclab.mp4')],stdin=subprocess.PIPE)
    conn=Client(('127.0.0.1',int(sys.argv[1])),authkey=b'muscle-local-test');conn.send(dict(nq=m.nq,nv=m.nv,nu=m.nu))
    body_map=[];site_map=[]
    if os.environ.get('MM_GPU_OBSERVATIONS')=='1':
        for kind,enum,output in [('body',mujoco.mjtObj.mjOBJ_BODY,body_map),('site',mujoco.mjtObj.mjOBJ_SITE,site_map)]:
            for i in range(getattr(REFERENCE,'n'+kind)):
                name=mujoco.mj_id2name(REFERENCE,enum,i).replace('-','_')
                candidates=[]
                for j in range(getattr(m,'n'+kind)):
                    new=mujoco.mj_id2name(m,enum,j)
                    match=(new.endswith('_'+name) or new==name) if kind=='body' else re.sub(r'_\d+$','',new.rsplit('/',1)[-1])==name
                    if match:candidates.append(j)
                assert len(candidates)==1,(kind,name,candidates)
                output.append(candidates[0])
        assert m.nsensordata==REFERENCE.nsensordata
    counter=0; overflow=[]
    try:
        while True:
            request=conn.recv()
            if request is None:break
            q=np.zeros((1,m.nq),np.float32);v=np.zeros((1,m.nv),np.float32)
            q[0,dstq]=request['qpos'][srcq];v[0,dstv]=request['qvel'][srcv]
            d.qpos.assign(q);d.qvel.assign(v);d.act.assign(request['act'][None].astype(np.float32))
            if request['time']==0:
                d.qacc_warmstart.zero_(); d.qfrc_applied.zero_(); d.xfrc_applied.zero_()
            if SUPPORT_FORCE:d.xfrc_applied.assign(external_wrench)
            import mujoco_warp
            mujoco_warp.forward(s.mjw_model,d)
            state=NewtonManager.get_state_0();s._update_newton_state(s.model,state,d,state_prev=state)
            NewtonManager.get_control().mujoco.ctrl.assign(request['ctrl'].astype(np.float32))
            for _ in range(request['nstep']):sim.step(render=False)
            if SUPPORT_FORCE:assert np.allclose(d.xfrc_applied.numpy(),external_wrench,rtol=1e-6,atol=1e-6)
            rawq=d.qpos.numpy()[0];rawv=d.qvel.numpy()[0]
            outq=np.empty(REFERENCE.nq);outv=np.empty(REFERENCE.nv);outq[srcq]=rawq[dstq];outv[srcv]=rawv[dstv]
            flags=d.overflow.numpy().tolist()
            if np.any(flags):overflow.append([counter,flags])
            assert np.isfinite(outq).all() and np.isfinite(outv).all()
            result=dict(qpos=outq,qvel=outv,act=d.act.numpy()[0],time=request['time']+request['nstep']*REFERENCE.opt.timestep)
            for name in ['actuator_force','actuator_length','actuator_velocity']:result[name]=getattr(d,name).numpy()[0]
            assert all(np.isfinite(result[k]).all() for k in ['act','actuator_force','actuator_length','actuator_velocity'])
            if os.environ.get('MM_GPU_OBSERVATIONS')=='1':
                for name in ['xpos','xquat','xmat','xipos','ximat','cvel','subtree_com']:
                    result[name]=getattr(d,name).numpy()[0][body_map]
                for name in ['site_xpos','site_xmat']:
                    result[name]=getattr(d,name).numpy()[0][site_map]
                result['sensordata']=d.sensordata.numpy()[0]
            if request['nstep'] and counter%4==0:
                viewer=sim._visualizers[0]
                if os.environ.get('MM_FOLLOW_CAMERA')=='1':
                    x,y=outq[:2]
                    viewer.set_camera_view(eye=(x+3.,y-3.,1.8),target=(x,y,.9))
                sim.render()
                points=d.wrap_xpos.numpy()[0].reshape(-1,3);adr=d.ten_wrapadr.numpy()[0];nums=d.ten_wrapnum.numpy()[0]
                starts=[];ends=[];colors=[]
                for muscle,tendon in enumerate(m.actuator_trnid[:,0]):
                    path=points[int(adr[tendon]):int(adr[tendon]+nums[tendon])]
                    color=np.array([.35,.08,.10])+np.clip(result['act'][muscle]/.35,0,1)*np.array([.60,.47,.02])
                    for k in range(len(path)-1):starts.append(path[k]);ends.append(path[k+1]);colors.append(color)
                if starts:viewer._viewer.log_lines('muscles',wp.array(starts,dtype=wp.vec3,device='cuda:0'),wp.array(ends,dtype=wp.vec3,device='cuda:0'),wp.array(colors,dtype=wp.vec3,device='cuda:0'))
                pixels=np.ascontiguousarray(viewer.render_rgb_array());writer.stdin.write(pixels.tobytes())
                if counter in [0,100,200,400,600]:Image.fromarray(pixels).save(OUT/f'policy_{counter}.png')
            counter+=int(request['nstep']>0);conn.send(result)
    finally:
        writer.stdin.close();writer.wait();(OUT/'worker_results.json').write_text(json.dumps(dict(steps=counter,overflow=overflow)))
