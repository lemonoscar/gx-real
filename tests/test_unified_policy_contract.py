"""Offline contract and runtime fault tests; never imports the hardware node."""
import ast
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'real-wbc'))
from modules.unified_policy_contract import UnifiedPolicyInput, filter_command, FEATURES, MEASUREMENT, MEASUREMENTS
from modules.runtime_safety import RuntimeSafetyFault
from modules.height_scan_policy_validation import ZERO_HEIGHT_SCAN_FUNC, validate_height_scan_runtime_mode

HOME=np.array([0,.3,.5,0,0,0])

def build(adapter, step, prefix=None, command=(0,0,0), q=HOME, dq=None, target=HOME, position=(0,0,0), quat=(1,0,0,0)):
    if prefix is None: prefix=np.zeros(260)
    if dq is None: dq=np.zeros(6)
    return adapter.build(prefix,command,q,dq,target,position,quat,step)


def test_dead_zones_signed_axes_and_ellipsoid():
    np.testing.assert_array_equal(filter_command([.1,-.05,.05]),0)
    for axis,limit in enumerate([.75,.2,.3]):
        for sign in [-1,1]:
            c=np.zeros(3); c[axis]=sign*10
            expected=(-.5 if axis==0 and sign<0 else sign*limit)
            assert filter_command(c)[axis]==pytest.approx(expected)
    c=filter_command([.75,.2,.3]); assert np.linalg.norm(c/[.75,.2,.3])==pytest.approx(1)
    with pytest.raises(ValueError): filter_command([np.nan,0,0])


def test_stow_gate_hysteresis_and_phase():
    a=UnifiedPolicyInput(HOME)
    o=build(a,0,command=(.5,0,0),q=HOME+.12)
    np.testing.assert_array_equal(o[9:12],0); assert o[262]==1
    o=build(a,1,command=(.5,0,0)); assert o[9]==.5 and o[262]==0
    assert o[260]==pytest.approx(np.sin(2*np.pi*.02*(.7+1.3*.5/.75)))
    o=build(a,2,command=(.5,0,0),q=HOME+.12,dq=np.ones(6)); assert o[9]==.5
    o=build(a,3,command=(.5,0,0),target=HOME+.04); assert o[9]==0
    o=build(a,4,command=(.5,0,0),dq=np.ones(6)); assert o[9]==0
    a.reset(); assert build(a,0)[262]==1


def test_history_previous_unscaled_oldest_first_and_duplicate_reads():
    a=UnifiedPolicyInput(HOME)
    for step in range(7):
        p=np.full(260,float(step)); p[3:6]*=.25; p[30:48]*=.05
        o=build(a,step,prefix=p)
        expected=np.maximum(0,np.arange(step-4,step))
        np.testing.assert_allclose(o[267:].reshape(4,54),np.repeat(expected[:,None],54,axis=1))
        np.testing.assert_array_equal(build(a,step,prefix=p),o)
        assert o[259]==0
    with pytest.raises(ValueError): build(a,9)


def test_anchor_is_full_body_inverse_and_yaw_error():
    a=UnifiedPolicyInput(HOME); build(a,0,position=(1,2,.4))
    o=build(a,1,position=(1.02,2.01,.4),quat=(np.sqrt(.5),0,0,np.sqrt(.5)))
    np.testing.assert_allclose(o[263:267],[-.01,.02,1,-1],atol=1e-6)
    with pytest.raises(ValueError): build(a,2,quat=(0,0,0,0))


def extract_node_function(name, extras=None):
    tree=ast.parse((ROOT/'real-wbc/modules/wbc_node_leg12_arm_passthrough.py').read_text())
    item=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'np':np,'time':time,'RuntimeSafetyFault':RuntimeSafetyFault}
    namespace.update(extras or {})
    exec(compile(ast.Module(body=[item],type_ignores=[]),'<offline-node-method>','exec'),namespace)
    return namespace[name]


def test_runtime_requires_live_pose_arm_and_scan():
    f=extract_node_function('build_unified_observation')
    arm=NS(state_valid=True,state_fresh=True,target_valid=True,target_fresh=True,
           joint_pos=HOME,joint_vel=np.zeros(6),joint_target=HOME)
    node=NS(unified_height_mode="pointcloud",unified_pose=(np.zeros(3),np.array([1,0,0,0])),unified_pose_time=time.monotonic(),
            height_scan_timeout=.25,arm_observation_cache=NS(get=lambda now:arm),obs=np.zeros(260),
            get_height_scan_observation=lambda:np.zeros(187),latest_height_scan_diag={'height_scan_ok':True,'used_fallback':False},
            default_dof_pos=np.r_[np.zeros(12),HOME],prev_action=np.zeros(18),unified_input=UnifiedPolicyInput(HOME),fixed_commands=np.zeros(3),policy_ctrl_iter=0)
    assert f(node).shape==(483,)
    node.unified_pose_time-=1
    with pytest.raises(RuntimeSafetyFault,match='odometry'): f(node)
    node.unified_pose_time=time.monotonic();arm.state_fresh=False
    with pytest.raises(RuntimeSafetyFault,match='arm'): f(node)
    arm.state_fresh=True;node.latest_height_scan_diag['used_fallback']=True
    with pytest.raises(RuntimeSafetyFault,match='scan'): f(node)
    node.unified_height_mode="zero"
    node.obs[66:253]=123.
    node.unified_input.reset()
    node.get_height_scan_observation=lambda: (_ for _ in ()).throw(AssertionError("zero mode must not read cloud"))
    output=f(node)
    np.testing.assert_array_equal(output[66:253],np.zeros(187))
    arm.target_fresh=False
    with pytest.raises(RuntimeSafetyFault,match='arm'): f(node)


def test_real_bundle_configuration_is_accepted_and_wrong_kind_rejected():
    source=ast.parse((ROOT/'real-wbc/modules/wbc_node_leg12_arm_passthrough.py').read_text())
    constant=next(n for n in source.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='EXPECTED_POLICY_OBS_FUNCS')
    ns={'ZERO_HEIGHT_SCAN_FUNC':ZERO_HEIGHT_SCAN_FUNC}
    exec(compile(ast.Module(body=[constant],type_ignores=[]),'<constant>','exec'),ns)
    from modules.leg_joint_limits import INTERFACE_LEG_JOINT_NAMES
    extras=dict(Dict=dict,List=list,Optional=__import__('typing').Optional,LEG_DOF=12,
                INTERFACE_LEG_JOINT_NAMES=INTERFACE_LEG_JOINT_NAMES,FEATURES=FEATURES,MEASUREMENT=MEASUREMENT,
                MEASUREMENTS=MEASUREMENTS,EXPECTED_POLICY_OBS_FUNCS=ns['EXPECTED_POLICY_OBS_FUNCS'],
                validate_height_scan_runtime_mode=validate_height_scan_runtime_mode)
    f=extract_node_function('_validate_policy_config',extras)
    cfg=yaml.safe_load((ROOT/'policies/unified483_model13493/env.yaml').read_text())
    f(cfg,cfg['dog_joint_names'],cfg['joint_names'],enable_height_scan=True)
    f(cfg,cfg['dog_joint_names'],cfg['joint_names'],enable_height_scan=False,unified_height_mode="zero")
    with pytest.raises(RuntimeError):
        f(cfg,cfg['dog_joint_names'],cfg['joint_names'],enable_height_scan=False)
    cfg['observations']['policy']['joint_vel']['params']['kind']='q'
    with pytest.raises(RuntimeError): f(cfg,cfg['dog_joint_names'],cfg['joint_names'],enable_height_scan=True)


@pytest.mark.parametrize('entry,mode', [('run_unified483_real.sh','zero'),('run_unified483_pointcloud_real.sh','pointcloud')])
def test_launchers_select_mode_without_hardware(tmp_path,entry,mode):
    import os,shutil,subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ('run_unified483_real.sh','run_unified483_pointcloud_real.sh'):
        shutil.copy(ROOT/'scripts'/name,scripts/name)
    # Replace downstream hardware launcher with argument-only output.
    (scripts/'run_leg12_real.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    env=os.environ.copy();env.pop('GX_REAL_UNIFIED_HEIGHT_MODE',None)
    args=subprocess.check_output(['bash',str(scripts/entry)],env=env,text=True).splitlines()
    assert args[args.index('--unified-height-mode')+1]==mode
    assert ('--enable-height-scan' in args)==(mode=='pointcloud')


@pytest.mark.parametrize('mode',['zero','pointcloud'])
def test_joystick_launcher_routes_both_height_modes(tmp_path,mode):
    import os,shutil,subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    for name in ('run_unified483_real.sh','run_unified483_joystick_real.sh'):
        shutil.copy(ROOT/'scripts'/name,scripts/name)
    (scripts/'run_leg12_real.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    env=os.environ.copy();env['GX_REAL_UNIFIED_HEIGHT_MODE']=mode
    args=subprocess.check_output(['bash',str(scripts/'run_unified483_joystick_real.sh')],env=env,text=True).splitlines()
    assert args[args.index('--base-command-source')+1]=='wireless_joystick'
    assert args[args.index('--unified-height-mode')+1]==mode
    assert args[args.index('--joy-max-vx')+1]=='0.75'
    assert args[args.index('--joy-max-vy')+1]=='0.2'
    assert args[args.index('--joy-max-yaw')+1]=='0.3'
    assert ('--enable-height-scan' in args)==(mode=='pointcloud')


def test_production_parser_default_model_and_arm_feedback_requirement():
    # Execute only argparse setup; no imports/initialization of ROS or hardware node.
    import argparse,os
    tree=ast.parse((ROOT/'real-wbc/scripts/run_wbc_leg12.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
    setup=[]
    for stmt in main.body:
        if isinstance(stmt,ast.Assign) and any(getattr(t,'id',None)=='args' for t in stmt.targets): break
        setup.append(stmt)
    scope={'argparse':argparse,'np':np,'os':os,'DEFAULT_POLICY_PATH':str(ROOT/'policies/policy.onnx'),
           'DEFAULT_POLICY_BUNDLE':str(ROOT/'policies/policy_bundle.json'),
           'DEFAULT_HEIGHT_SCAN_CONTRACT':str(ROOT/'policies/height_scan_contract.yaml'),
           'DEFAULT_LOG_DIR':str(ROOT/'logs')}
    exec(compile(ast.Module(body=setup,type_ignores=[]),'<parser-only>','exec'),scope)
    args=scope['parser'].parse_args([])
    assert args.require_arm_state_for_rl is True
    assert args.enable_height_scan is False
    assert args.unified_height_mode is None  # Unified init selects zero when scan disabled.
    assert Path(args.policy_path).resolve().parent.name=='unified483_model13493'
    assert args.base_command_source=='fixed'
