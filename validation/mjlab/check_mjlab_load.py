"""CPU-only check of MuscleMimic models through mjlab's real Entity API."""
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import traceback

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg

ROOT = Path('/home/k_miyazawa/musclemimic/.venv/lib/python3.11/site-packages/musclemimic_models/model')
results = {'versions': {p: version(p) for p in ['mjlab', 'mujoco', 'mujoco-warp', 'torch']}, 'models': {}}
print('VERSIONS', results['versions'], flush=True)
for name, relative in [('bimanual', 'arm/myoarm_bimanual.xml'), ('fullbody', 'body/myofullbody.xml')]:
    path = ROOT / relative
    try:
        ref = mujoco.MjModel.from_xml_path(str(path))
        source = mujoco.MjSpec.from_file(str(path))
        targets = tuple(a.target for a in source.actuators)
        entity = Entity(EntityCfg(
            spec_fn=lambda path=path: mujoco.MjSpec.from_file(str(path)),
            articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(
                target_names_expr=targets,
                transmission_type=TransmissionType.TENDON,
            ),)),
        ))
        model = entity.compile()
        counts = {key: int(getattr(model, key)) for key in ['nbody', 'njnt', 'nq', 'nv', 'nu', 'na', 'ntendon', 'neq']}
        for key in ['njnt', 'nq', 'nv', 'nu', 'na', 'ntendon', 'neq']:
            assert getattr(model, key) == getattr(ref, key), key
        compared = []
        for field in [
            'actuator_dyntype', 'actuator_gaintype', 'actuator_biastype',
            'actuator_dynprm', 'actuator_gainprm', 'actuator_biasprm',
            'actuator_lengthrange', 'actuator_ctrlrange', 'actuator_trnid',
            'tendon_adr', 'tendon_num', 'wrap_type', 'wrap_objid', 'wrap_prm',
            'eq_type', 'eq_obj1id', 'eq_obj2id', 'eq_data',
        ]:
            np.testing.assert_allclose(getattr(model, field), getattr(ref, field), rtol=0, atol=0, err_msg=field)
            compared.append(field)
        original_names = [mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(ref.nu)]
        wrapped_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        assert original_names == wrapped_names
        data = mujoco.MjData(model)
        data.ctrl[:] = 0.1
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.actuator_force).all()
        results['models'][name] = dict(status='PASS', counts=counts,
            muscle_dyntypes=dict(Counter(model.actuator_dyntype.tolist())),
            exact_match_fields=compared, actuator_order_preserved=True,
            command_field=entity.actuators[0].command_field,
            forward_force_finite=True)
        print(name, json.dumps(results['models'][name]), flush=True)
    except Exception as exc:
        results['models'][name] = dict(status='FAIL', error=str(exc))
        traceback.print_exc()
Path('/tmp/muscle-feasibility/mjlab_load_results.json').write_text(json.dumps(results, indent=2))
if any(r['status'] != 'PASS' for r in results['models'].values()):
    raise SystemExit(1)
