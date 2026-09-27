import unittest
import mujoco
import numpy as np
from project_joint_equalities import project_joint_equalities


class EqualityProjectionTest(unittest.TestCase):
    def test_chained_polynomial_position_and_velocity(self):
        m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><joint name="a" ref="10"/><geom size=".1"/><body><joint name="b"/><geom size=".1"/><body><joint name="c"/><geom size=".1"/></body></body></body></worldbody><equality><joint joint1="c" joint2="b" polycoef="0 2 0 0 0"/><joint joint1="b" joint2="a" polycoef="0 0 1 0 0"/></equality></mujoco>''')
        q=m.qpos0.copy();q[0]+=.3;v=np.array([.4,7.,8.]);qp,vp=project_joint_equalities(m,q,v)
        np.testing.assert_allclose(qp,[q[0],.09,.18]);np.testing.assert_allclose(vp,[.4,.24,.48])
        d=mujoco.MjData(m);d.qpos[:]=qp;d.qvel[:]=vp;mujoco.mj_forward(m,d)
        rows=d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        np.testing.assert_allclose(d.efc_pos[rows],0,atol=1e-12);np.testing.assert_allclose(d.efc_vel[rows],0,atol=1e-12)
        np.testing.assert_array_equal(q,[q[0],0,0]);np.testing.assert_array_equal(v,[.4,7,8])

    def test_batch_idempotent(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="a"/><geom size=".1"/></body></worldbody><equality><joint joint1="a" polycoef=".1 0 0 0 0"/></equality></mujoco>')
        q,v=project_joint_equalities(m,np.zeros((3,1)),np.ones((3,1)))
        np.testing.assert_allclose(q,.1);np.testing.assert_allclose(v,0)
        q2,v2=project_joint_equalities(m,q,v);np.testing.assert_array_equal(q,q2);np.testing.assert_array_equal(v,v2)

if __name__=='__main__':unittest.main()
