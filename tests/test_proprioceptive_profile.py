from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'real-wbc'))
from modules.proprioceptive_math import contact_velocity,foot_kinematics
from modules.unified_policy_contract import UnifiedPolicyInput
from test_unified_policy_contract import extract_node_function,HOME
from types import SimpleNamespace as NS
from modules.runtime_safety import RuntimeSafetyFault


def test_without_external_pose_and_yaw_from_imu():
    a=UnifiedPolicyInput(HOME,use_xy_anchor=False);p=np.zeros(260)
    args=(p,[0,0,0],HOME,np.zeros(6),HOME,None)
    x=a.build(*args,[1,0,0,0],0)
    y=a.build(*args,[np.cos(.05),0,0,np.sin(.05)],1)
    np.testing.assert_array_equal(x[263:265],[0,0]);np.testing.assert_array_equal(y[263:265],[0,0])
    np.testing.assert_allclose(y[265:267],[np.sin(.1),np.cos(.1)-1],atol=1e-7)
    with pytest.raises(ValueError):UnifiedPolicyInput(HOME).build(*args,[1,0,0,0],0)


def test_node_proprioception_needs_no_pose_but_requires_support():
    f=extract_node_function('build_unified_observation',{'contact_velocity':contact_velocity})
    arm=NS(state_valid=True,state_fresh=True,target_valid=True,target_fresh=True,joint_pos=HOME,joint_vel=np.zeros(6),joint_target=HOME)
    q=np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
    node=NS(proprioceptive_policy=True,unified_pose=None,unified_pose_time=None,
            arm_observation_cache=NS(get=lambda now:arm),obs=np.zeros(260),unified_height_mode='zero',
            default_dof_pos=np.r_[q,HOME],prev_action=np.zeros(18),latest_imu_quaternion=np.array([1,0,0,0]),
            interface_to_policy_leg_order=lambda x:x,quadruped_q=q,quadruped_dq=np.zeros(12),
            latest_foot_force=np.ones(4)*50,foot_contact_thres=20.,proprio_velocity=np.zeros(3),
            proprio_no_contact_seconds=0.,unified_input=UnifiedPolicyInput(HOME,use_xy_anchor=False),
            fixed_commands=np.zeros(3),policy_ctrl_iter=0)
    out=f(node);assert out.shape==(483,);np.testing.assert_array_equal(out[66:253],0);np.testing.assert_array_equal(out[263:265],0)
    node.latest_foot_force[:]=0
    for i in range(12):node.policy_ctrl_iter+=1;f(node)
    node.policy_ctrl_iter+=1
    with pytest.raises(RuntimeSafetyFault,match='support contacts'):f(node)


def test_fk_derivative_and_single_contact_rotation():
    rng=np.random.default_rng(44);q=rng.normal(size=(4,3));dq=rng.normal(size=(4,3))
    r,dr=foot_kinematics(q,dq,np);plus,_=foot_kinematics(q+1e-6*dq,dq,np);minus,_=foot_kinematics(q-1e-6*dq,dq,np)
    np.testing.assert_allclose(dr,(plus-minus)/2e-6,atol=1e-9)
    w=np.array([.1,.2,.3]);v,n=contact_velocity(q,np.zeros_like(q),w,np.array([True,False,False,False]),np.zeros(3),.02,np)
    np.testing.assert_allclose(v,-np.cross(w,r[0])*(1-np.exp(-.02/.06)),atol=1e-10);assert n==1


def test_launcher_rejects_old_profile_before_hardware():
    import os,subprocess
    env=os.environ.copy();env['GX_REAL_PROPRIO_BUNDLE']=str(ROOT/'policies/unified483_model13493');env['GX_REAL_PYTHON_BIN']=sys.executable
    result=subprocess.run(['bash',str(ROOT/'scripts/run_proprioceptive_flat_real.sh')],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'not exported for the requested observation profile' in result.stderr
