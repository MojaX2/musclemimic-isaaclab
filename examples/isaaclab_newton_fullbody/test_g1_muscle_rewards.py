"""CPU regressions: reward incentives, resets, model mapping, and physical contacts."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from g1_muscle_rewards import G1MuscleRewards, forward_velocity


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.r=G1MuscleRewards(2,'cpu')

    def gait(self, contact, height, command=1.):
        return self.r.gait_terms(torch.tensor([contact]*2),torch.tensor([height]*2),
                                 torch.zeros(2,2,2),torch.full((2,),command))

    def test_grounded_feet_cannot_collect_gait_reward(self):
        for _ in range(50):
            t=self.gait([True,True],[0.,0.])
            self.assertEqual(float(t['air_time'].sum()+t['swing_clearance'].sum()+t['touchdown'].sum()),0.)

    def test_prolonged_double_support_allows_transfer_and_standing_commands(self):
        for _ in range(10):t=self.gait([True,True],[0.,0.])
        self.assertTrue((t['prolonged_double_support']==0).all())
        for _ in range(30):t=self.gait([True,True],[0.,0.])
        self.assertTrue((t['prolonged_double_support']==1).all())
        self.r.reset(torch.tensor([True,False]))
        self.assertEqual(float(self.r.double_support[0]),0.)
        self.assertGreater(float(self.r.double_support[1]),.6)
        t=self.gait([True,False],[0.,.05])
        self.assertTrue((t['prolonged_double_support']==0).all())
        for _ in range(50):t=self.gait([True,True],[0.,0.],command=0.)
        self.assertTrue((t['prolonged_double_support']==0).all())

    def test_noise_cap_preserves_small_std_and_resumes_by_iteration(self):
        from velocity_exploration import cap_exploration
        from rsl_rl.modules import GaussianDistribution
        d=GaussianDistribution(2,init_std=.5,std_type='log',std_range=[.05,1.])
        with torch.no_grad():d.log_std_param[0]=math.log(.08)
        schedule=dict(start_iteration=100,initial=.5,final=.15,updates=600)
        cap_exploration(d,schedule,400)
        middle=d.log_std_param.detach().exp().clone()
        self.assertAlmostEqual(float(middle[0]),.08,places=6)
        self.assertLess(float(middle[1]),.5)
        # Reloading policy and schedule at the same iteration must not restart annealing.
        other=GaussianDistribution(2,init_std=.5,std_type='log',std_range=[.05,1.])
        other.load_state_dict(d.state_dict())
        cap_exploration(other,schedule,400)
        torch.testing.assert_close(other.log_std_param,d.log_std_param)
        cap_exploration(other,schedule,900)
        self.assertAlmostEqual(float(other.log_std_param[1].detach().exp()),.15,places=6)
        other.update(torch.zeros(3,2))
        self.assertTrue(torch.isfinite(other.log_prob(other.sample())).all())

    def test_flight_is_penalized_not_rewarded(self):
        for _ in range(20):
            t=self.gait([False,False],[.05,.05])
        self.assertTrue((t['air_time']==0).all() and (t['swing_clearance']==0).all())
        self.assertTrue((t['flight']==1).all())

    def test_phase_progress_rewards_small_lift_before_contact_breaks(self):
        phase=torch.full((2,),math.pi/2)
        def score(height,contact=(True,True),command=1.):
            return self.r.gait_terms(torch.tensor([contact]*2),torch.tensor([height]*2),
                torch.zeros(2,2,2),torch.full((2,),command),phase)['phase_clearance']
        grounded=score([0.,0.])
        small=score([.005,0.])
        target=score([.05,0.],(False,True))
        self.assertTrue((grounded==0).all())
        self.assertTrue((small>grounded).all() and (target>small).all())
        self.assertTrue((score([0.,.05])<0).all())
        self.assertTrue((score([.05,.05],(False,False))<=0).all())
        self.assertTrue((score([.05,0.],command=0.)==0).all())
        self.assertTrue((score([.12,0.],(False,True))<target).all())

    def test_phase_progress_prefers_alternation_to_held_foot(self):
        phase=torch.arange(100,dtype=torch.float32)*2*math.pi/100
        r=G1MuscleRewards(100,'cpu')
        wave=phase.sin()
        target=.05*torch.stack([wave.clamp(min=0),(-wave).clamp(min=0)],1)
        command=torch.ones(100);vel=torch.zeros(100,2,2)
        contact=target<=0
        walking=r.gait_terms(contact,target,vel,command,phase)['phase_clearance']
        held=torch.tensor([[.05,0.]]).repeat(100,1)
        hovering=r.gait_terms(torch.tensor([[False,True]]).repeat(100,1),held,vel,command,phase)['phase_clearance']
        self.assertGreater(float(walking.mean()),float(hovering.mean()))
        self.assertLess(float(hovering.mean()),0.)
        opposite=r.gait_terms(contact.flip(1),target.flip(1),vel,command,phase)['phase_clearance']
        self.assertGreater(float(walking.mean()),float(opposite.mean()))

    def test_lifted_swing_beats_scuff(self):
        for _ in range(10):
            low=self.gait([True,False],[0.,.003])
        low_score=low['swing_clearance']-low['scuff']
        self.r.reset(torch.ones(2,dtype=torch.bool))
        for _ in range(10):
            lifted=self.gait([True,False],[0.,.05])
        self.assertTrue((lifted['air_time']>0).all())
        self.assertTrue(((lifted['swing_clearance']-lifted['scuff'])>low_score).all())

    def test_hover_has_no_positive_gait_reward(self):
        for _ in range(50):
            t=self.gait([True,False],[0.,.05])
        self.assertTrue((t['air_time']==0).all() and (t['swing_clearance']==0).all())
        self.assertTrue((t['long_swing']==1).all())

    def test_touchdown_requires_clearance_and_alternation(self):
        for _ in range(10):self.gait([True,False],[0.,.05])
        self.assertTrue((self.gait([True,True],[0.,0.])['touchdown']==1).all())
        for _ in range(10):self.gait([True,False],[0.,.05])
        self.assertTrue((self.gait([True,True],[0.,0.])['touchdown']==0).all())
        for _ in range(10):self.gait([False,True],[.05,0.])
        self.assertTrue((self.gait([True,True],[0.,0.])['touchdown']==1).all())
        for _ in range(10):self.gait([True,False],[0.,.01])
        self.assertTrue((self.gait([True,True],[0.,0.])['touchdown']==0).all())

    def test_stationary_command_and_partial_reset(self):
        for _ in range(10):t=self.gait([True,False],[0.,.05],command=0.)
        self.assertTrue((t['air_time']==0).all() and (t['swing_clearance']==0).all())
        self.r.reset(torch.tensor([True,False]))
        self.assertEqual(float(self.r.air[0].sum()),0.)
        self.assertGreater(float(self.r.air[1].sum()),.1)
        self.assertEqual(float(self.r.peak[0].sum()),0.)

    def test_slip_and_termination_weights(self):
        z=torch.zeros(2,1)
        args=dict(velocity_xy=torch.zeros(2,2),command=torch.zeros(2),root_velocity=torch.zeros(2,6),
                  gravity_xy=torch.zeros(2,2),fallen=torch.tensor([False,True]),
                  contact=torch.ones(2,2,dtype=torch.bool),clearance=torch.zeros(2,2),
                  foot_velocity_xy=torch.tensor([[[0.,0.],[0.,0.]],[[1.,0.],[0.,0.]]]),
                  ankle_violation=z,hip_deviation=z,arm_deviation=z,torso_deviation=z,
                  joint_acceleration=z,joint_torque=z,excitation=torch.zeros(2,354),previous=torch.zeros(2,354))
        reward,terms,weighted=self.r(**args)
        self.assertAlmostEqual(float(weighted['termination'][1]),-4.)
        self.assertAlmostEqual(float(weighted['feet_slide'][1]),-.002,places=6)
        self.assertTrue(torch.isfinite(reward).all())
        self.assertGreater(float(reward[0]),float(reward[1]))

    def test_anatomical_velocity_frame(self):
        vel=torch.tensor([[-1.,0.,0.],[0.,-1.,0.]])
        out=forward_velocity(vel,torch.tensor([0.,math.pi/2]))
        torch.testing.assert_close(out,torch.tensor([[1.,0.],[1.,0.]]),atol=1e-6,rtol=1e-6)


class SensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        wp.config.kernel_cache_dir='/tmp/muscle-g1-reward-test-warp-cache'

    def test_anatomical_joint_mapping(self):
        import mujoco
        from g1_muscle_sensors import joint_groups, root_world_velocity
        model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2]/'outputs/isaac_velocity/assets/model.xml'))
        groups=joint_groups(model,'cpu')
        self.assertEqual(len(groups['leg']),8)
        self.assertEqual(len(groups['ankle']),4)
        self.assertEqual(len(groups['arms']),10)
        self.assertFalse(set(groups['leg'].tolist()) & set(groups['arms'].tolist()))
        # Newton renames multi-axis joints. Select by the importer's DOF mapping,
        # even when none of the resulting names retain their anatomical labels.
        mapping={model.joint(i).name:i for i in range(model.njnt)}
        from unittest.mock import patch
        with patch('mujoco.mj_id2name', return_value='converted_joint'):
            mapped=joint_groups(model,'cpu',mapping)
            for key in groups:torch.testing.assert_close(mapped[key],groups[key])
        q=torch.zeros(1,7);q[0,3]=math.sqrt(.5);q[0,6]=math.sqrt(.5)
        v=torch.tensor([[0.,0.,0.,1.,0.,0.]])
        torch.testing.assert_close(root_world_velocity(q,v)[0,3:],torch.tensor([0.,1.,0.]),atol=1e-6,rtol=1e-6)

    def test_warp_contact_forces_match_mujoco_cpu(self):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from g1_muscle_sensors import FootContactForces
        for cone in ['pyramidal','elliptic']:
            model=mujoco.MjModel.from_xml_string(f'''<mujoco><option cone="{cone}"/><worldbody>
                <geom type="plane" size="3 3 .1"/>
                <body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
                <body pos="1 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
                </worldbody></mujoco>''')
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            self.assertEqual(data.ncon,1)
            force=np.zeros(6);mujoco.mj_contactForce(model,data,0,force)
            with wp.ScopedDevice('cpu'):
                wm=mjw.put_model(model);wd=mjw.put_data(model,data,nworld=2,nconmax=8,njmax=32)
                sensor=FootContactForces(model,wm,wd,SimpleNamespace(feet=[([1],None,None),([2],None,None)]),'cpu')
                contact=sensor()
                np.testing.assert_allclose(sensor.normal_t.numpy()[:,0],force[0],rtol=1e-5)
                self.assertTrue(contact[:,0].all() and not contact[:,1].any())
                # Changing active contact count must not leave stale force behind.
                wd.nacon.zero_()
                self.assertFalse(sensor().any())


if __name__=='__main__':unittest.main()
