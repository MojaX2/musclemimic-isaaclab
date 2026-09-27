"""External model adapter; preserves native equalities through multi-DOF import."""
from pathlib import Path
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import mujoco,newton,warp as wp
from newton.solvers import SolverMuJoCo
ROOT=Path('/tmp/muscle-feasibility')
SOURCE=Path(os.environ.get('MM_MODEL_XML', str(Path(__file__).resolve().parents[2]/'.venv/lib/python3.11/site-packages/musclemimic_models/model/body/myofullbody.xml')))
ROOT.mkdir(parents=True,exist_ok=True)
REFERENCE=mujoco.MjModel.from_xml_path(str(SOURCE))
JOINT_DOF={}
def make_builder():
    spec=mujoco.MjSpec.from_file(str(SOURCE))
    spec.meshdir=str((SOURCE.parent/spec.meshdir).resolve())
    spec.texturedir=str((SOURCE.parent/spec.texturedir).resolve())
    p=ROOT/'fullbody_adapter.xml'
    compiled=spec.compile();mujoco.mj_saveLastXML(str(p),compiled)
    tree=ET.parse(p);compiler=tree.getroot().find("compiler")
    compiler.set("meshdir",spec.meshdir);compiler.set("texturedir",spec.texturedir)
    # Expand defaults here too: Newton refreshes actuator parameters after compilation.
    for i,a in enumerate(tree.getroot().find('actuator')):
        a.tag='general'
        for field in ['gainprm','biasprm','dynprm','lengthrange','ctrlrange','forcerange','actrange','gear']:
            a.set(field,' '.join(map(str,getattr(REFERENCE,'actuator_'+field)[i])))
        for field in ['gaintype','biastype','dyntype']: a.set(field,'muscle')
        for field in ['ctrllimited','forcelimited','actlimited','actearly']:
            a.set(field,'true' if getattr(REFERENCE,'actuator_'+field)[i] else 'false')
    # Materialize MuJoCo compiler bounds and inferred inertias before Newton import.
    for body in tree.getroot().find('worldbody').iter('body'):
        bid=mujoco.mj_name2id(REFERENCE,mujoco.mjtObj.mjOBJ_BODY,body.get('name'))
        assert bid>0
        inertial=body.find('inertial')
        if inertial is not None: body.remove(inertial)
        ET.SubElement(body,'inertial',mass=str(REFERENCE.body_mass[bid]),
            pos=' '.join(map(str,REFERENCE.body_ipos[bid])),
            quat=' '.join(map(str,REFERENCE.body_iquat[bid])),
            diaginertia=' '.join(map(str,REFERENCE.body_inertia[bid])))
    # Newton sanitizes tendon references but not the corresponding site labels.
    for el in tree.iter():
        for key in ['name','site','sidesite','geom','joint','joint1','joint2','tendon','body','body1','body2','objname']:
            if key in el.attrib: el.set(key,el.get(key).replace('-', '_'))
    tree.write(p)
    b=newton.ModelBuilder();SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(p),skip_equality_constraints=True,parse_mujoco_options=False)
    xml=ET.parse(p).getroot()
    # Match each original scalar joint to its Newton DOF, before solver naming.
    for body in xml.find('worldbody').iter('body'):
        js=body.findall('joint')
        if not js: continue
        bi=[i for i,l in enumerate(b.body_label) if l.endswith('/'+body.get('name',''))]
        assert len(bi)==1,(body.get('name'),bi)
        ji=[i for i,c in enumerate(b.joint_child) if c==bi[0]]
        assert len(ji)==1
        ordered=[j for j in js if j.get('type','hinge')=='slide']+[j for j in js if j.get('type','hinge')!='slide']
        for off,j in enumerate(ordered): JOINT_DOF[j.get('name')]=b.joint_qd_start[ji[0]]+off
    return b
class FullBodySolver(SolverMuJoCo):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._restore_muscle_ranges()
    def _restore_muscle_ranges(self):
        # Newton 1.6 leaves GPU actuator_lengthrange zero for this model.
        # Repair after initialization/model refresh, outside CUDA graph capture.
        if not self.use_mujoco_cpu:
            arr=self.mjw_model.actuator_lengthrange
            arr.assign(np.broadcast_to(self.mj_model.actuator_lengthrange,arr.numpy().shape).copy())
            if os.environ.get('MM_MATCH_PHYSICS') == '1':
                for key in ['geom_margin','geom_gap']:
                    target=getattr(self.mjw_model,key)
                    target.assign(np.broadcast_to(getattr(self.mj_model,key),target.shape).copy())
    def notify_model_changed(self,*args,**kwargs):
        super().notify_model_changed(*args,**kwargs)
        self._restore_muscle_ranges()

    def _init_actuators(self,*args,**kwargs):
        import inspect
        bound=inspect.signature(SolverMuJoCo._init_actuators).bind(self,*args,**kwargs).arguments
        result=super()._init_actuators(*args,**kwargs)
        spec=bound['spec'];dofmap=bound['dof_to_mjc_joint'];names=bound['mjc_joint_names']
        if os.environ.get('MM_MATCH_PHYSICS') == '1':
            # Preserve authored contact distances; Newton defaults gap to 0.1 m
            # and drops margins even on the analytic primitives in this model.
            source_geoms = {mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_GEOM,i).replace('-', '_'): i
                            for i in range(REFERENCE.ngeom)
                            if mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_GEOM,i)}
            for geom in spec.geoms:
                name = re.sub(r'_\d+$', '', geom.name.rsplit('/', 1)[-1])
                if name in source_geoms:
                    i = source_geoms[name]
                    geom.margin = float(REFERENCE.geom_margin[i])
                    geom.gap = float(REFERENCE.geom_gap[i])
            spec.option.tolerance = float(REFERENCE.opt.tolerance)
            spec.option.disableflags |= int(REFERENCE.opt.disableflags)

        if os.environ.get('MM_GPU_OBSERVATIONS') == '1':
            # The generic Newton importer does not retain MuJoCo sensors.
            def sensor_target(objtype, oid):
                if oid < 0: return ''
                name = mujoco.mj_id2name(REFERENCE,objtype,int(oid)).replace('-', '_')
                if objtype == mujoco.mjtObj.mjOBJ_BODY:
                    candidates=[body.name for body in spec.bodies if body.name.endswith('_'+name) or body.name==name]
                elif objtype == mujoco.mjtObj.mjOBJ_SITE:
                    candidates=[site.name for site in spec.sites if re.sub(r'_\d+$','',site.name.rsplit('/',1)[-1])==name]
                else: raise ValueError(f'Unsupported sensor target {objtype}')
                assert len(candidates)==1,(name,candidates)
                return candidates[0]
            assert not list(spec.sensors)
            for i in range(REFERENCE.nsensor):
                objtype=int(REFERENCE.sensor_objtype[i]); reftype=int(REFERENCE.sensor_reftype[i])
                spec.add_sensor(name=mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_SENSOR,i),
                    type=int(REFERENCE.sensor_type[i]),objtype=objtype,
                    objname=sensor_target(objtype,int(REFERENCE.sensor_objid[i])),
                    reftype=reftype,refname=sensor_target(reftype,int(REFERENCE.sensor_refid[i])),
                    cutoff=float(REFERENCE.sensor_cutoff[i]))
        assert len(list(spec.actuators))==REFERENCE.nu
        for i,a in enumerate(spec.actuators):
            for field in ['gainprm','biasprm','dynprm','lengthrange','ctrlrange','forcerange','actrange','gear']:
                setattr(a,field,getattr(REFERENCE,'actuator_'+field)[i])
            for field in ['gaintype','biastype','dyntype','ctrllimited','forcelimited','actlimited','actearly']:
                setattr(a,field,int(getattr(REFERENCE,'actuator_'+field)[i]))
            a.name=mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_ACTUATOR,i)

        for i in range(REFERENCE.neq):
            assert REFERENCE.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT
            def mapped(oid):
                if oid<0:return ''
                original=mujoco.mj_id2name(REFERENCE,mujoco.mjtObj.mjOBJ_JOINT,int(oid))
                return names[int(dofmap[JOINT_DOF[original]])]
            spec.add_equality(name=f'original_eq_{i}',type=mujoco.mjtEq.mjEQ_JOINT,
                name1=mapped(REFERENCE.eq_obj1id[i]),name2=mapped(REFERENCE.eq_obj2id[i]),
                data=REFERENCE.eq_data[i],solref=REFERENCE.eq_solref[i],solimp=REFERENCE.eq_solimp[i],active=bool(REFERENCE.eq_active0[i]))
        return result
