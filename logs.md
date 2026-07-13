2026-06-04 21:51:12,564 INFO Run logs: /home/unitree/gx-real/logs/20260604_215112
2026-06-04 21:51:12,936 INFO Pose estimator disabled for leg-only deployment
2026-06-04 21:51:12,945 INFO Preparing policy
2026-06-04 21:51:13,232 INFO Policy inference time: 6.524085998535156e-05 (9.108558323667904e-06)
2026-06-04 21:51:13,236 INFO starting to play policy
2026-06-04 21:51:13,241 INFO kp: [ 32.  32.  36.  32.  32.  36.  32.  32.  36.  32.  32.  36. 100. 100.
 100.  20.  20.   5.], kd: [0.8 0.8 0.9 0.8 0.8 0.9 0.8 0.8 0.9 0.8 0.8 0.9 3.  3.  3.  2.  1.  0.5], torque_limits: {<MotorId.FR_HIP: 0>: 40, <MotorId.FR_THIGH: 1>: 60, <MotorId.FR_CALF: 2>: 75, <MotorId.FL_HIP: 3>: 40, <MotorId.FL_THIGH: 4>: 60, <MotorId.FL_CALF: 5>: 75, <MotorId.RR_HIP: 6>: 40, <MotorId.RR_THIGH: 7>: 60, <MotorId.RR_CALF: 8>: 75, <MotorId.RL_HIP: 9>: 40, <MotorId.RL_THIGH: 10>: 60, <MotorId.RL_CALF: 11>: 75}, commanded_leg_kp: [200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200.], commanded_leg_kd: [10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10.], deploy_policy_kp: [200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200.], deploy_policy_kd: [10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10.], manual_takeover_kp: [200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200. 200.], manual_takeover_kd: [10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10. 10.], obs_dof_pos_scale: 1.0, train_leg_default_offset: [ 0.1  0.8 -1.5 -0.1  0.8 -1.5  0.1  1.  -1.5 -0.1  1.  -1.5], real_deploy_leg_offset: [-0.035  0.852 -1.57   0.011  0.846 -1.597  0.006  0.936 -1.578  0.021
  0.919 -1.564],obs_dof_pos_offset: [-0.035  0.852 -1.57   0.011  0.846 -1.597  0.006  0.936 -1.578  0.021
  0.919 -1.564  0.     0.3    0.5    0.     0.     0.   ], obs_dof_vel_scale: 0.05, train_leg_action_offset: [ 0.1  0.8 -1.5 -0.1  0.8 -1.5  0.1  1.  -1.5 -0.1  1.  -1.5],leg_action_offset: [-0.035  0.852 -1.57   0.011  0.846 -1.597  0.006  0.936 -1.578  0.021
  0.919 -1.564], train_leg_action_scale: [0.18 0.32 0.32 0.18 0.32 0.32 0.18 0.32 0.32 0.18 0.32 0.32], leg_action_scale: [0.18 0.32 0.32 0.18 0.32 0.32 0.18 0.32 0.32 0.18 0.32 0.32], train_sim2sim_action_delay_range: (1, 1), deploy_sim2sim_action_delay_range: (1, 1), sim2sim_action_hold_prob: 0.05, sim2sim_action_noise_std: 0.0, sim2sim_obs_delay_steps: 0, policy_leg_joint_names: ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'], policy_leg_indices_from_interface: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], policy_freq: 50.0, config_path: /home/unitree/gx-real/policies/env.yaml, fixed_commands: [0. 0. 0.], policy_takeover_commands: [0. 0. 0.], policy_move_commands: [0. 0. 0.], base_command_source: wireless_joystick, policy_command_ramp_duration: 1.5, startup_action_limit_sec: 3.0, startup_action_abs_limit: 1.0, startup_action_delta_limit: 0.35, arm_control_owner: external_spacemouse, arm_state_topic: /arm/state, arm_target_topic: /arm/target_state, require_arm_state_for_rl: True, allow_unknown_sport_mode: False, lowstate_watchdog_sec: 0.25, sport_state_watchdog_sec: 0.5, live_ready_pose_calibration: True, fixed_gripper_cmd: 0.0
2026-06-04 21:51:13,267 INFO Height scan provider disabled; using zero height_scan observation
2026-06-04 21:51:13,270 INFO Runtime targets | standup_mode=internal base_command_source=wireless_joystick arm_control_owner=external_spacemouse arm_pose_source=user arm_hold_pose=[0.000 0.500 0.300 0.000 0.000 0.000] button_arm_pose=None arm_reset_pose=[0.000 0.500 0.300 0.000 0.000 0.000] commanded_leg_kp=[200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0 200.0] commanded_leg_kd=[10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0 10.0] move_commands=[0.000 0.000 0.000]
2026-06-04 21:51:13,276 INFO Arm control owner: external_spacemouse
2026-06-04 21:51:13,278 INFO WBC will only consume arm state from /arm/state and target from /arm/target_state
2026-06-04 21:51:13,281 INFO Press R1 to start unitree_mujoco get-up
2026-06-04 21:51:13,283 INFO Press L2 to start policy after stand-up completes
2026-06-04 21:51:13,285 INFO Press Y to zero base command; in joystick mode it inhibits until sticks return to center
2026-06-04 21:51:13,287 INFO A/X/B/D-pad arm controls are no-op; X5 control moved to standalone SpaceMouse Arm Node
2026-06-04 21:51:13,289 INFO Press L1 for emergency stop
2026-06-04 21:51:13,290 WARNING WBC ARX5 write controller disabled; arm_control_owner=external_spacemouse. Leg control remains active and arm observation uses external topics or fallback.
2026-06-04 21:51:13,293 INFO Deploy node ready
2026-06-04 21:51:17,686 INFO Joystick input received before policy | axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} mapped_cmd=[-0.000 -0.000 -0.000] formula=[vx=-1*ly*0.100, vy=-1*lx*0.050, yaw=-1*rx*0.200] deadzone=0.120 valid=True reason=
2026-06-04 21:51:17,691 INFO standing up
2026-06-04 21:51:17,693 INFO Runtime leg offset update | source=r1_current_standing_pose leg_action_offset=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354]
2026-06-04 21:51:17,696 INFO Internal ready-pose alignment: current posture is standing-like (ready_error=0.000, crouch_error=1.090); skipping crouch phase
2026-06-04 21:51:18,189 INFO Joystick input received before policy | axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} mapped_cmd=[-0.000 -0.000 -0.000] formula=[vx=-1*ly*0.100, vy=-1*lx*0.050, yaw=-1*rx*0.200] deadzone=0.120 valid=True reason=
2026-06-04 21:51:20,536 INFO Joystick input received before policy | axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} mapped_cmd=[-0.000 -0.000 -0.000] formula=[vx=-1*ly*0.100, vy=-1*lx*0.050, yaw=-1*rx*0.200] deadzone=0.120 valid=True reason=
2026-06-04 21:51:20,541 INFO Starting dog-only startup before rollout; command ramp [0.0, 0.0, 0.0] -> [0.0, 0.0, 0.0] over 1.50s
2026-06-04 21:51:20,546 INFO Startup diag | elapsed=0.00 ratio=0.000 current_leg_q=[-0.018  0.658 -1.386  0.017  0.660 -1.391 -0.090  0.648 -1.375  0.087
  0.658 -1.370] target_leg_q=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] leg_q_error=[ 1.393e-03  4.844e-03  9.873e-03 -9.084e-05  4.239e-03  9.747e-03
  4.935e-03  3.028e-03  1.603e-02 -2.180e-03  1.756e-03  1.613e-02] max_leg_error=0.016 rear_thigh_error=0.003 current_leg_dq=[-0.120 -0.058  0.107  0.047 -0.081  0.105 -0.399 -0.039 -0.044 -0.085
  0.062 -0.028] current_tau_est=[ 1.088  1.286  8.250 -0.866  1.311  8.961  0.322  2.202  3.983 -8.955
  0.965  3.888] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] foot_force=[48.000 49.000 46.000 46.000]
2026-06-04 21:51:21,036 INFO Joystick input received before policy | axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} mapped_cmd=[-0.000 -0.000 -0.000] formula=[vx=-1*ly*0.100, vy=-1*lx*0.050, yaw=-1*rx*0.200] deadzone=0.120 valid=True reason=
2026-06-04 21:51:21,057 INFO Startup diag | elapsed=0.51 ratio=0.272 current_leg_q=[-0.018  0.656 -1.389  0.017  0.658 -1.394 -0.090  0.647 -1.380  0.086
  0.657 -1.374] target_leg_q=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] leg_q_error=[ 0.001  0.006  0.013 -0.000  0.006  0.013  0.005  0.004  0.021 -0.001
  0.003  0.020] max_leg_error=0.021 rear_thigh_error=0.004 current_leg_dq=[-0.120 -0.047 -0.105  0.019 -0.078 -0.125 -0.279 -0.101 -0.079 -0.298
  0.023 -0.075] current_tau_est=[ 0.767  1.113  3.082 -0.445  1.187  2.892  0.074  0.544  3.224 -7.941
  0.148  3.035] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] foot_force=[47.000 49.000 44.000 45.000]
2026-06-04 21:51:21,569 INFO Startup diag | elapsed=1.03 ratio=0.764 current_leg_q=[-0.018  0.657 -1.388  0.017  0.658 -1.393 -0.090  0.648 -1.381  0.087
  0.657 -1.374] target_leg_q=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] leg_q_error=[ 0.002  0.006  0.012 -0.000  0.006  0.012  0.005  0.003  0.021 -0.002
  0.002  0.020] max_leg_error=0.021 rear_thigh_error=0.003 current_leg_dq=[ 0.019 -0.054  0.121  0.132 -0.093  0.253 -0.667  0.109  0.008  0.690
 -0.016  0.008] current_tau_est=[ 2.078  0.940  8.061 -0.668  1.385  8.440  5.195  1.831  3.888 -5.888
  1.583  3.841] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] foot_force=[46.000 50.000 46.000 46.000]
2026-06-04 21:51:22,079 INFO Startup diag | elapsed=1.54 ratio=1.000 current_leg_q=[-0.017  0.657 -1.388  0.018  0.659 -1.392 -0.092  0.648 -1.380  0.089
  0.658 -1.374] target_leg_q=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] leg_q_error=[ 0.001  0.005  0.012 -0.001  0.006  0.011  0.007  0.003  0.021 -0.005
  0.002  0.020] max_leg_error=0.021 rear_thigh_error=0.003 current_leg_dq=[ 0.105  0.081 -0.006 -0.054 -0.043  0.020 -0.136  0.112 -0.002  0.314
 -0.008  0.061] current_tau_est=[  1.311   0.940   9.056  -1.732   1.361   9.246  11.701   2.152   8.013
 -11.033   0.940   7.207] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] foot_force=[47.000 50.000 44.000 45.000]
2026-06-04 21:51:22,444 INFO Dog-only startup completed; starting rollout with residual errors max=0.021 rear_thigh=0.003
2026-06-04 21:51:22,447 INFO Policy command target update | source=policy_start start=[0.000 0.000 0.000] target=[0.000 0.000 0.000] ramp=1.50s
2026-06-04 21:51:22,454 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:22,460 WARNING Startup action limiter | elapsed=0.012/3.000s abs_limit=1.000 delta_limit=0.350 abs_clipped=False delta_clipped=True requested=[-0.025 -0.172  0.980  0.162 -0.178  0.855 -0.099 -0.060  0.514 -0.056
 -0.138  0.655] limited=[-0.025 -0.172  0.350  0.162 -0.178  0.350 -0.099 -0.060  0.350 -0.056
 -0.138  0.350]
2026-06-04 21:51:22,469 INFO Policy diag | handover=0.010 est_lin_vel=[-6.873e-05 -4.906e-06 -8.549e-04] commands=[0.000 0.000 0.000] raw_action=[-0.025 -0.172  0.980  0.162 -0.178  0.855 -0.099 -0.060  0.514 -0.056
 -0.138  0.655] clipped_action=[-0.025 -0.172  0.980  0.162 -0.178  0.855 -0.099 -0.060  0.514 -0.056
 -0.138  0.655] startup_limited_action=[-0.025 -0.172  0.350  0.162 -0.178  0.350 -0.099 -0.060  0.350 -0.056
 -0.138  0.350] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=True timed_action=[-0.025 -0.172  0.350  0.162 -0.178  0.350 -0.099 -0.060  0.350 -0.056
 -0.138  0.350] applied_action=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] commanded_leg_q=[-0.018  0.656 -1.388  0.017  0.659 -1.392 -0.091  0.648 -1.380  0.087
  0.658 -1.374] current_leg_q=[-0.018  0.656 -1.388  0.017  0.659 -1.392 -0.089  0.648 -1.380  0.085
  0.658 -1.374] leg_q_error=[ 0.000  0.000  0.000 -0.000  0.000  0.000 -0.002 -0.000  0.000  0.002
  0.000  0.000] current_leg_dq=[ 0.124 -0.043 -0.087 -0.171 -0.089 -0.022  0.229  0.074 -0.047 -0.128
  0.050 -0.026] current_tau_est=[ 1.410  1.905  9.056 -1.113  1.509  9.436  7.842  2.251  8.250 -7.916
  1.113  7.681] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] lowcmd_leg_q_hw=[-0.016  0.662 -1.376  0.017  0.664 -1.381 -0.085  0.651 -1.359  0.085
  0.660 -1.354] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[45.000 49.000 44.000 46.000]
2026-06-04 21:51:22,966 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:23,009 INFO Policy diag | handover=0.465 est_lin_vel=[0.003 0.003 0.005] commands=[0.000 0.000 0.000] raw_action=[-0.022  0.014  0.837  0.164 -0.071  0.550 -0.134 -0.134  0.347  0.081
 -0.110  0.512] clipped_action=[-0.022  0.014  0.837  0.164 -0.071  0.550 -0.134 -0.134  0.347  0.081
 -0.110  0.512] startup_limited_action=[-0.022  0.014  0.837  0.164 -0.071  0.550 -0.134 -0.134  0.347  0.081
 -0.110  0.512] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.022  0.014  0.837  0.164 -0.071  0.550 -0.134 -0.134  0.347  0.081
 -0.110  0.512] applied_action=[-0.033  0.002  0.849  0.151 -0.060  0.514 -0.133 -0.132  0.348  0.100
 -0.116  0.514] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.022  0.663 -1.104  0.044  0.645 -1.217 -0.109  0.609 -1.248  0.103
  0.623 -1.190] commanded_leg_q=[-0.020  0.659 -1.256  0.030  0.652 -1.311 -0.099  0.630 -1.319  0.094
  0.642 -1.288] current_leg_q=[-0.019  0.650 -1.362  0.016  0.652 -1.376 -0.088  0.648 -1.374  0.088
  0.654 -1.364] leg_q_error=[-0.001  0.010  0.106  0.014 -0.000  0.065 -0.011 -0.018  0.055  0.006
 -0.013  0.076] current_leg_dq=[-0.031 -0.027 -0.182 -0.128 -0.101 -0.042 -0.422 -0.341  0.018  0.915
 -0.322 -0.127] current_tau_est=[ 2.078  2.647  7.681 -0.569  1.583  8.061 -1.583 -1.484  7.444 -5.616
 -0.074  7.634] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.019  0.661 -1.253  0.027  0.655 -1.312 -0.101  0.627 -1.318  0.095
  0.639 -1.285] lowcmd_leg_q_hw=[-0.019  0.661 -1.253  0.027  0.655 -1.312 -0.101  0.627 -1.318  0.095
  0.639 -1.285] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[50.000 46.000 43.000 47.000]
2026-06-04 21:51:23,467 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:23,549 INFO Policy diag | handover=0.915 est_lin_vel=[0.009 0.005 0.007] commands=[0.000 0.000 0.000] raw_action=[ 0.016 -0.072  0.682  0.086 -0.104  0.267 -0.046 -0.078  0.302  0.010
 -0.125  0.474] clipped_action=[ 0.016 -0.072  0.682  0.086 -0.104  0.267 -0.046 -0.078  0.302  0.010
 -0.125  0.474] startup_limited_action=[ 0.016 -0.072  0.682  0.086 -0.104  0.267 -0.046 -0.078  0.302  0.010
 -0.125  0.474] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.016 -0.072  0.682  0.086 -0.104  0.267 -0.046 -0.078  0.302  0.010
 -0.125  0.474] applied_action=[ 0.022 -0.104  0.720  0.104 -0.126  0.284 -0.040 -0.083  0.263 -0.019
 -0.106  0.466] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.013  0.629 -1.146  0.036  0.624 -1.291 -0.092  0.624 -1.275  0.081
  0.626 -1.205] commanded_leg_q=[-0.013  0.632 -1.166  0.034  0.627 -1.299 -0.092  0.626 -1.284  0.082
  0.629 -1.219] current_leg_q=[-0.019  0.645 -1.334  0.019  0.646 -1.351 -0.089  0.643 -1.349  0.089
  0.645 -1.329] leg_q_error=[ 0.006 -0.013  0.168  0.015 -0.019  0.052 -0.004 -0.017  0.065 -0.007
 -0.016  0.110] current_leg_dq=[-0.039 -0.798  0.346 -0.105 -0.802  0.089  0.713 -0.624 -0.245  0.887
 -0.639  0.202] current_tau_est=[ 2.350  2.944 27.975 -2.078  0.866 12.186  7.966  0.965  6.496 -7.421
 -0.173 19.393] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.012  0.633 -1.170  0.034  0.627 -1.289 -0.094  0.624 -1.284  0.084
  0.627 -1.221] lowcmd_leg_q_hw=[-0.012  0.633 -1.170  0.034  0.627 -1.289 -0.094  0.624 -1.284  0.084
  0.627 -1.221] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[53.000 48.000 39.000 52.000]
2026-06-04 21:51:23,967 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:24,090 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.017 -0.003  0.001] commands=[0.000 0.000 0.000] raw_action=[ 0.059 -0.118  0.559  0.085 -0.159  0.298 -0.064 -0.086  0.232  0.001
 -0.103  0.456] clipped_action=[ 0.059 -0.118  0.559  0.085 -0.159  0.298 -0.064 -0.086  0.232  0.001
 -0.103  0.456] startup_limited_action=[ 0.059 -0.118  0.559  0.085 -0.159  0.298 -0.064 -0.086  0.232  0.001
 -0.103  0.456] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.059 -0.118  0.559  0.085 -0.159  0.298 -0.064 -0.086  0.232  0.001
 -0.103  0.456] applied_action=[ 0.052 -0.107  0.649  0.077 -0.130  0.234 -0.065 -0.076  0.250 -0.017
 -0.120  0.534] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.007  0.628 -1.168  0.031  0.623 -1.307 -0.097  0.627 -1.279  0.082
  0.621 -1.183] commanded_leg_q=[-0.007  0.628 -1.168  0.031  0.623 -1.307 -0.097  0.627 -1.279  0.082
  0.621 -1.183] current_leg_q=[-0.018  0.641 -1.327  0.019  0.645 -1.346 -0.088  0.643 -1.340  0.088
  0.645 -1.326] leg_q_error=[ 0.011 -0.013  0.159  0.012 -0.022  0.040 -0.009 -0.016  0.061 -0.006
 -0.024  0.142] current_leg_dq=[-0.074  0.605  1.122 -0.636  0.787  0.190  1.353  0.558  0.344 -0.577
 -0.670  0.843] current_tau_est=[ 1.930  0.371  0.000  0.025 -0.124  8.487  4.181  0.346  9.388 -6.358
  1.732 16.738] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.011  0.624 -1.164  0.033  0.613 -1.296 -0.093  0.626 -1.275  0.082
  0.621 -1.200] lowcmd_leg_q_hw=[-0.011  0.624 -1.164  0.033  0.613 -1.296 -0.093  0.626 -1.275  0.082
  0.621 -1.200] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[49.000 49.000 37.000 45.000]
2026-06-04 21:51:24,487 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:24,629 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.009 -0.001 -0.002] commands=[0.000 0.000 0.000] raw_action=[ 0.181 -0.155  0.514  0.144 -0.121  0.474 -0.099 -0.131  0.185 -0.109
 -0.086  0.437] clipped_action=[ 0.181 -0.155  0.514  0.144 -0.121  0.474 -0.099 -0.131  0.185 -0.109
 -0.086  0.437] startup_limited_action=[ 0.181 -0.155  0.514  0.144 -0.121  0.474 -0.099 -0.131  0.185 -0.109
 -0.086  0.437] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.181 -0.155  0.514  0.144 -0.121  0.474 -0.099 -0.131  0.185 -0.109
 -0.086  0.437] applied_action=[ 0.148 -0.156  0.579  0.115 -0.127  0.363 -0.120 -0.127  0.217 -0.055
 -0.093  0.438] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[ 0.010  0.612 -1.191  0.037  0.624 -1.265 -0.107  0.610 -1.290  0.075
  0.630 -1.214] commanded_leg_q=[ 0.010  0.612 -1.191  0.037  0.624 -1.265 -0.107  0.610 -1.290  0.075
  0.630 -1.214] current_leg_q=[-0.018  0.642 -1.317  0.016  0.646 -1.344 -0.082  0.645 -1.339  0.091
  0.646 -1.318] leg_q_error=[ 0.029 -0.029  0.126  0.021 -0.022  0.079 -0.024 -0.035  0.049 -0.016
 -0.016  0.104] current_leg_dq=[ 0.047 -0.504 -0.042  0.570 -0.616  0.055 -0.926  0.701  0.127  0.636
  0.264  0.493] current_tau_est=[ 1.880  0.148 24.893  0.965  0.223  4.599  1.286 -0.693  3.888 -7.595
  1.113  1.043] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[ 0.002  0.625 -1.194  0.037  0.625 -1.303 -0.094  0.622 -1.278  0.074
  0.634 -1.200] lowcmd_leg_q_hw=[ 0.002  0.625 -1.194  0.037  0.625 -1.303 -0.094  0.622 -1.278  0.074
  0.634 -1.200] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[56.000 38.000 41.000 49.000]
2026-06-04 21:51:25,006 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.908, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.091 -0.000 -0.000] safe_cmd=[-0.091  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:25,171 INFO Policy diag | handover=1.000 est_lin_vel=[-0.003  0.006 -0.000] commands=[-0.091  0.000  0.000] raw_action=[ 0.080 -0.207  0.484  0.204 -0.228  0.520 -0.056 -0.160  0.248 -0.067
 -0.065  0.397] clipped_action=[ 0.080 -0.207  0.484  0.204 -0.228  0.520 -0.056 -0.160  0.248 -0.067
 -0.065  0.397] startup_limited_action=[ 0.080 -0.207  0.484  0.204 -0.228  0.520 -0.056 -0.160  0.248 -0.067
 -0.065  0.397] startup_limiter_active=True startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.080 -0.207  0.484  0.204 -0.228  0.520 -0.056 -0.160  0.248 -0.067
 -0.065  0.397] applied_action=[ 0.031 -0.130  0.491  0.147 -0.258  0.467 -0.007 -0.129  0.303 -0.068
 -0.104  0.423] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.011  0.621 -1.219  0.043  0.582 -1.232 -0.086  0.610 -1.262  0.072
  0.626 -1.219] commanded_leg_q=[-0.011  0.621 -1.219  0.043  0.582 -1.232 -0.086  0.610 -1.262  0.072
  0.626 -1.219] current_leg_q=[-0.019  0.637 -1.314  0.017  0.638 -1.337 -0.083  0.640 -1.334  0.088
  0.640 -1.320] leg_q_error=[ 0.008 -0.016  0.095  0.026 -0.056  0.106 -0.003 -0.030  0.071 -0.015
 -0.014  0.102] current_leg_dq=[-0.221 -0.426  0.277 -0.395 -0.585  0.202  0.128 -0.748  0.176 -0.124
 -0.388  0.825] current_tau_est=[ 1.311 -0.544 19.251  0.470 -5.442  9.104  7.100  1.212 14.035 -8.065
 -3.265  0.095] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.019  0.624 -1.198  0.033  0.604 -1.283 -0.079  0.624 -1.254  0.077
  0.612 -1.183] lowcmd_leg_q_hw=[-0.019  0.624 -1.198  0.033  0.604 -1.283 -0.079  0.624 -1.254  0.077
  0.612 -1.183] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 40.000 42.000 50.000]
2026-06-04 21:51:25,528 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.913, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.091 -0.000 -0.000] safe_cmd=[-0.091  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:25,710 INFO Policy diag | handover=1.000 est_lin_vel=[0.010 0.002 0.000] commands=[-0.091  0.000  0.000] raw_action=[-0.047 -0.338  0.551  0.143 -0.327  0.418  0.017 -0.107  0.284 -0.028
 -0.111  0.348] clipped_action=[-0.047 -0.338  0.551  0.143 -0.327  0.418  0.017 -0.107  0.284 -0.028
 -0.111  0.348] startup_limited_action=[-0.047 -0.338  0.551  0.143 -0.327  0.418  0.017 -0.107  0.284 -0.028
 -0.111  0.348] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.047 -0.338  0.551  0.143 -0.327  0.418  0.017 -0.107  0.284 -0.028
 -0.111  0.348] applied_action=[-0.011 -0.286  0.509  0.131 -0.298  0.389 -0.008 -0.124  0.298 -0.046
 -0.102  0.391] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.018  0.571 -1.213  0.040  0.569 -1.257 -0.087  0.611 -1.264  0.076
  0.627 -1.229] commanded_leg_q=[-0.018  0.571 -1.213  0.040  0.569 -1.257 -0.087  0.611 -1.264  0.076
  0.627 -1.229] current_leg_q=[-0.020  0.635 -1.323  0.018  0.636 -1.334 -0.086  0.637 -1.332  0.087
  0.639 -1.316] leg_q_error=[ 0.001 -0.064  0.110  0.022 -0.067  0.077 -0.000 -0.025  0.069 -0.011
 -0.012  0.087] current_leg_dq=[-0.353  0.233  1.005  0.756 -0.074  0.253 -0.655  0.151  0.204  0.701
  0.124  0.423] current_tau_est=[ 1.311  1.509 -0.142  1.509  3.068  9.625  5.640  3.364  2.987 -8.015
  2.820  0.996] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.021  0.578 -1.189  0.042  0.584 -1.273 -0.086  0.614 -1.257  0.076
  0.629 -1.236] lowcmd_leg_q_hw=[-0.021  0.578 -1.189  0.042  0.584 -1.273 -0.086  0.614 -1.257  0.076
  0.629 -1.236] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 40.000 40.000 44.000]
2026-06-04 21:51:26,045 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.956, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.096 -0.000 -0.000] safe_cmd=[-0.096  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:26,250 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.014 -0.004 -0.006] commands=[-0.099  0.000  0.000] raw_action=[-0.023 -0.249  0.464  0.176 -0.340  0.512 -0.029 -0.140  0.259 -0.038
 -0.092  0.397] clipped_action=[-0.023 -0.249  0.464  0.176 -0.340  0.512 -0.029 -0.140  0.259 -0.038
 -0.092  0.397] startup_limited_action=[-0.023 -0.249  0.464  0.176 -0.340  0.512 -0.029 -0.140  0.259 -0.038
 -0.092  0.397] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.023 -0.249  0.464  0.176 -0.340  0.512 -0.029 -0.140  0.259 -0.038
 -0.092  0.397] applied_action=[ 0.015 -0.218  0.493  0.175 -0.268  0.470 -0.037 -0.145  0.284 -0.087
 -0.091  0.421] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.014  0.593 -1.218  0.048  0.578 -1.231 -0.092  0.605 -1.268  0.069
  0.631 -1.219] commanded_leg_q=[-0.014  0.593 -1.218  0.048  0.578 -1.231 -0.092  0.605 -1.268  0.069
  0.631 -1.219] current_leg_q=[-0.020  0.636 -1.319  0.018  0.631 -1.332 -0.084  0.636 -1.337  0.087
  0.641 -1.322] leg_q_error=[ 0.007 -0.043  0.101  0.030 -0.053  0.101 -0.007 -0.031  0.069 -0.018
 -0.011  0.103] current_leg_dq=[-0.016  1.147 -0.481 -0.736 -0.415  0.461  0.981  0.574  0.182 -0.512
  0.074  0.560] current_tau_est=[ 2.226 -0.990  5.121  0.693  7.347  2.039  3.364  2.944  3.414 -6.308
  2.820  0.996] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.018  0.572 -1.200  0.045  0.567 -1.229 -0.091  0.607 -1.276  0.076
  0.628 -1.241] lowcmd_leg_q_hw=[-0.018  0.572 -1.200  0.045  0.567 -1.229 -0.091  0.607 -1.276  0.076
  0.628 -1.241] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[47.000 49.000 40.000 42.000]
2026-06-04 21:51:26,546 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:26,790 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.004 -0.008 -0.011] commands=[-0.100  0.000  0.000] raw_action=[-0.062 -0.220  0.524  0.153 -0.295  0.383  0.008 -0.107  0.309 -0.036
 -0.123  0.457] clipped_action=[-0.062 -0.220  0.524  0.153 -0.295  0.383  0.008 -0.107  0.309 -0.036
 -0.123  0.457] startup_limited_action=[-0.062 -0.220  0.524  0.153 -0.295  0.383  0.008 -0.107  0.309 -0.036
 -0.123  0.457] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.062 -0.220  0.524  0.153 -0.295  0.383  0.008 -0.107  0.309 -0.036
 -0.123  0.457] applied_action=[-0.083 -0.263  0.554  0.133 -0.266  0.328  0.021 -0.093  0.342 -0.040
 -0.133  0.455] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.031  0.578 -1.199  0.041  0.579 -1.276 -0.081  0.621 -1.250  0.077
  0.617 -1.209] commanded_leg_q=[-0.031  0.578 -1.199  0.041  0.579 -1.276 -0.081  0.621 -1.250  0.077
  0.617 -1.209] current_leg_q=[-0.020  0.625 -1.319  0.018  0.631 -1.337 -0.086  0.635 -1.336  0.087
  0.638 -1.326] leg_q_error=[-0.011 -0.047  0.120  0.022 -0.052  0.060  0.005 -0.014  0.086 -0.009
 -0.021  0.118] current_leg_dq=[ 0.190  0.403  0.510 -0.546 -0.174  0.117  0.771  0.488  0.040 -0.450
  0.074  0.499] current_tau_est=[ 1.212  4.577 18.160  0.000 -3.488 10.052  5.715 -8.040  4.125 -6.803
 -2.523  1.091] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.028  0.583 -1.195  0.040  0.583 -1.278 -0.083  0.620 -1.254  0.078
  0.617 -1.204] lowcmd_leg_q_hw=[-0.028  0.583 -1.195  0.040  0.583 -1.278 -0.083  0.620 -1.254  0.078
  0.617 -1.204] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 40.000 44.000 51.000]
2026-06-04 21:51:27,065 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.171, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.017 -0.000 -0.000] safe_cmd=[-0.089  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:27,329 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.033  0.000 -0.002] commands=[-0.010  0.000  0.000] raw_action=[-0.024 -0.107  0.613  0.088 -0.178  0.304 -0.013 -0.078  0.276 -0.012
 -0.145  0.477] clipped_action=[-0.024 -0.107  0.613  0.088 -0.178  0.304 -0.013 -0.078  0.276 -0.012
 -0.145  0.477] startup_limited_action=[-0.024 -0.107  0.613  0.088 -0.178  0.304 -0.013 -0.078  0.276 -0.012
 -0.145  0.477] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.024 -0.107  0.613  0.088 -0.178  0.304 -0.013 -0.078  0.276 -0.012
 -0.145  0.477] applied_action=[-0.024 -0.145  0.594  0.118 -0.187  0.338 -0.018 -0.099  0.272 -0.026
 -0.128  0.450] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.021  0.616 -1.186  0.038  0.604 -1.273 -0.088  0.620 -1.272  0.080
  0.619 -1.210] commanded_leg_q=[-0.021  0.616 -1.186  0.038  0.604 -1.273 -0.088  0.620 -1.272  0.080
  0.619 -1.210] current_leg_q=[-0.020  0.635 -1.328  0.018  0.639 -1.340 -0.088  0.638 -1.343  0.086
  0.643 -1.330] leg_q_error=[-2.531e-04 -1.957e-02  1.419e-01  2.012e-02 -3.432e-02  6.710e-02
 -1.307e-04 -1.887e-02  7.093e-02 -6.061e-03 -2.459e-02  1.197e-01] current_leg_dq=[-0.105 -0.190 -0.542  0.713  0.271 -0.305 -1.271 -0.019 -0.216 -0.167
 -0.151  0.253] current_tau_est=[ 0.866 -3.538  6.069  1.113  1.682  6.022  6.976  3.043  5.595 -8.065
  3.068 21.147] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.021  0.616 -1.186  0.038  0.604 -1.273 -0.088  0.620 -1.272  0.080
  0.619 -1.210] lowcmd_leg_q_hw=[-0.021  0.616 -1.186  0.038  0.604 -1.273 -0.088  0.620 -1.272  0.080
  0.619 -1.210] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[51.000 52.000 42.000 48.000]
2026-06-04 21:51:27,566 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.066  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:27,871 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.002 -0.001 -0.002] commands=[-0.100  0.000  0.000] raw_action=[-0.009 -0.262  0.541  0.150 -0.325  0.481 -0.002 -0.134  0.292 -0.045
 -0.081  0.378] clipped_action=[-0.009 -0.262  0.541  0.150 -0.325  0.481 -0.002 -0.134  0.292 -0.045
 -0.081  0.378] startup_limited_action=[-0.009 -0.262  0.541  0.150 -0.325  0.481 -0.002 -0.134  0.292 -0.045
 -0.081  0.378] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.009 -0.262  0.541  0.150 -0.325  0.481 -0.002 -0.134  0.292 -0.045
 -0.081  0.378] applied_action=[ 0.018 -0.208  0.563  0.151 -0.241  0.420 -0.014 -0.132  0.319 -0.102
 -0.108  0.403] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.013  0.596 -1.196  0.044  0.587 -1.247 -0.088  0.609 -1.257  0.066
  0.625 -1.225] commanded_leg_q=[-0.013  0.596 -1.196  0.044  0.587 -1.247 -0.088  0.609 -1.257  0.066
  0.625 -1.225] current_leg_q=[-0.019  0.629 -1.322  0.017  0.631 -1.337 -0.085  0.635 -1.343  0.090
  0.639 -1.328] leg_q_error=[ 0.006 -0.033  0.126  0.027 -0.044  0.090 -0.002 -0.026  0.086 -0.023
 -0.014  0.103] current_leg_dq=[ 0.116  1.314 -0.117 -0.186  0.911  0.176  0.287  1.097  0.352 -0.543
  0.639 -0.020] current_tau_est=[ 1.460 -0.569  3.841 -1.113  0.124  3.509  7.298 -0.767  3.319 -7.619
  0.371 22.949] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.019  0.601 -1.185  0.041  0.602 -1.284 -0.085  0.615 -1.260  0.071
  0.621 -1.207] lowcmd_leg_q_hw=[-0.019  0.601 -1.185  0.041  0.602 -1.284 -0.085  0.615 -1.260  0.071
  0.621 -1.207] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 41.000 47.000 48.000]
2026-06-04 21:51:28,066 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:28,410 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.006  0.001 -0.003] commands=[-0.100  0.000  0.000] raw_action=[-0.071 -0.313  0.564  0.176 -0.270  0.348  0.003 -0.112  0.309 -0.048
 -0.108  0.406] clipped_action=[-0.071 -0.313  0.564  0.176 -0.270  0.348  0.003 -0.112  0.309 -0.048
 -0.108  0.406] startup_limited_action=[-0.071 -0.313  0.564  0.176 -0.270  0.348  0.003 -0.112  0.309 -0.048
 -0.108  0.406] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.071 -0.313  0.564  0.176 -0.270  0.348  0.003 -0.112  0.309 -0.048
 -0.108  0.406] applied_action=[-0.056 -0.263  0.563  0.133 -0.268  0.321  0.007 -0.107  0.318 -0.035
 -0.110  0.384] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.027  0.578 -1.196  0.041  0.578 -1.279 -0.084  0.617 -1.257  0.078
  0.624 -1.231] commanded_leg_q=[-0.027  0.578 -1.196  0.041  0.578 -1.279 -0.084  0.617 -1.257  0.078
  0.624 -1.231] current_leg_q=[-0.020  0.629 -1.319  0.019  0.629 -1.335 -0.087  0.634 -1.342  0.088
  0.637 -1.324] leg_q_error=[-0.007 -0.051  0.123  0.022 -0.051  0.056  0.003 -0.017  0.085 -0.010
 -0.012  0.093] current_leg_dq=[-0.043 -0.236  1.296  0.225 -0.217  0.073 -0.422 -0.926  0.237  0.194
 -0.891  0.477] current_tau_est=[-0.049  4.676 12.992 -3.117  4.601 12.660 11.058  7.842 13.608 -9.821
  3.439  1.422] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.028  0.585 -1.190  0.045  0.577 -1.265 -0.083  0.616 -1.259  0.077
  0.620 -1.217] lowcmd_leg_q_hw=[-0.028  0.585 -1.190  0.045  0.577 -1.265 -0.083  0.616 -1.259  0.077
  0.620 -1.217] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 43.000 48.000 49.000]
2026-06-04 21:51:28,585 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:28,949 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.008 -0.002 -0.005] commands=[-0.100  0.000  0.000] raw_action=[-0.089 -0.255  0.569  0.146 -0.329  0.400  0.016 -0.098  0.328 -0.012
 -0.112  0.402] clipped_action=[-0.089 -0.255  0.569  0.146 -0.329  0.400  0.016 -0.098  0.328 -0.012
 -0.112  0.402] startup_limited_action=[-0.089 -0.255  0.569  0.146 -0.329  0.400  0.016 -0.098  0.328 -0.012
 -0.112  0.402] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.089 -0.255  0.569  0.146 -0.329  0.400  0.016 -0.098  0.328 -0.012
 -0.112  0.402] applied_action=[-0.069 -0.247  0.564  0.137 -0.286  0.354  0.018 -0.100  0.320 -0.037
 -0.126  0.405] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.029  0.583 -1.195  0.041  0.573 -1.268 -0.082  0.619 -1.257  0.078
  0.619 -1.224] commanded_leg_q=[-0.029  0.583 -1.195  0.041  0.573 -1.268 -0.082  0.619 -1.257  0.078
  0.619 -1.224] current_leg_q=[-0.018  0.638 -1.329  0.019  0.641 -1.337 -0.083  0.641 -1.339  0.087
  0.646 -1.328] leg_q_error=[-0.011 -0.055  0.134  0.022 -0.068  0.069  0.002 -0.021  0.083 -0.009
 -0.027  0.104] current_leg_dq=[ 0.105  0.469 -0.374 -0.089  0.256 -0.113  0.283  0.205 -0.313 -0.194
  0.233 -0.358] current_tau_est=[ 1.311  0.792  4.836 -0.841  1.930  4.647  7.496  3.241  5.405 -8.337
  1.361  4.836] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.025  0.574 -1.213  0.047  0.572 -1.267 -0.082  0.611 -1.268  0.078
  0.619 -1.203] lowcmd_leg_q_hw=[-0.025  0.574 -1.213  0.047  0.572 -1.267 -0.082  0.611 -1.268  0.078
  0.619 -1.203] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 48.000 43.000 41.000]
2026-06-04 21:51:29,088 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:29,490 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.022 -0.002 -0.003] commands=[-0.082  0.000  0.000] raw_action=[-0.036 -0.255  0.504  0.108 -0.299  0.324  0.024 -0.093  0.303 -0.032
 -0.144  0.428] clipped_action=[-0.036 -0.255  0.504  0.108 -0.299  0.324  0.024 -0.093  0.303 -0.032
 -0.144  0.428] startup_limited_action=[-0.036 -0.255  0.504  0.108 -0.299  0.324  0.024 -0.093  0.303 -0.032
 -0.144  0.428] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.075 -0.332  0.564  0.129 -0.362  0.400  0.018 -0.104  0.313 -0.011
 -0.112  0.363] applied_action=[-0.075 -0.332  0.564  0.129 -0.362  0.400  0.018 -0.104  0.313 -0.011
 -0.112  0.363] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.030  0.556 -1.196  0.040  0.548 -1.253 -0.082  0.618 -1.259  0.083
  0.624 -1.238] commanded_leg_q=[-0.030  0.556 -1.196  0.040  0.548 -1.253 -0.082  0.618 -1.259  0.083
  0.624 -1.238] current_leg_q=[-0.020  0.624 -1.316  0.021  0.627 -1.333 -0.086  0.634 -1.336  0.088
  0.636 -1.322] leg_q_error=[-0.010 -0.068  0.120  0.019 -0.079  0.080  0.004 -0.016  0.077 -0.005
 -0.012  0.084] current_leg_dq=[-0.167  1.752 -0.384  0.395  1.384  0.319 -0.236  0.434  0.129 -0.124
  0.543  0.008] current_tau_est=[ 1.781 -1.880  4.220 -2.029 -0.025  2.892 10.637  0.569  3.509 -8.856
  0.693  3.509] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.021  0.582 -1.197  0.041  0.560 -1.238 -0.085  0.619 -1.263  0.070
  0.614 -1.226] lowcmd_leg_q_hw=[-0.021  0.582 -1.197  0.041  0.560 -1.238 -0.085  0.619 -1.263  0.070
  0.614 -1.226] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[51.000 45.000 42.000 49.000]
2026-06-04 21:51:29,606 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.000 -0.000 -0.000] safe_cmd=[-0.046  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:30,030 INFO Policy diag | handover=1.000 est_lin_vel=[0.013 0.000 0.002] commands=[-0.060  0.000  0.000] raw_action=[ 0.036 -0.170  0.558  0.145 -0.158  0.328 -0.052 -0.119  0.272 -0.066
 -0.098  0.413] clipped_action=[ 0.036 -0.170  0.558  0.145 -0.158  0.328 -0.052 -0.119  0.272 -0.066
 -0.098  0.413] startup_limited_action=[ 0.036 -0.170  0.558  0.145 -0.158  0.328 -0.052 -0.119  0.272 -0.066
 -0.098  0.413] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.036 -0.170  0.558  0.145 -0.158  0.328 -0.052 -0.119  0.272 -0.066
 -0.098  0.413] applied_action=[ 0.036 -0.235  0.574  0.109 -0.218  0.350 -0.037 -0.127  0.264 -0.035
 -0.103  0.357] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.010  0.587 -1.192  0.036  0.595 -1.269 -0.092  0.610 -1.275  0.078
  0.627 -1.240] commanded_leg_q=[-0.010  0.587 -1.192  0.036  0.595 -1.269 -0.092  0.610 -1.275  0.078
  0.627 -1.240] current_leg_q=[-0.020  0.639 -1.319  0.016  0.641 -1.338 -0.084  0.640 -1.332  0.087
  0.640 -1.314] leg_q_error=[ 0.010 -0.052  0.126  0.021 -0.047  0.069 -0.007 -0.030  0.057 -0.008
 -0.014  0.074] current_leg_dq=[-0.109 -1.031  0.160 -0.174 -1.105 -0.028 -0.202 -1.155 -0.093  0.260
 -0.798 -0.158] current_tau_est=[ 2.350 -1.212 24.703 -0.792 -3.340  4.410  9.401  2.325  3.841 -9.722
  1.064  2.797] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.018  0.609 -1.181  0.033  0.600 -1.279 -0.091  0.616 -1.270  0.087
  0.620 -1.223] lowcmd_leg_q_hw=[-0.018  0.609 -1.181  0.033  0.600 -1.279 -0.091  0.616 -1.270  0.087
  0.620 -1.223] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[56.000 41.000 39.000 49.000]
2026-06-04 21:51:30,108 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.085  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:30,569 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.007  0.001 -0.000] commands=[-0.100  0.000  0.000] raw_action=[-0.103 -0.283  0.588  0.168 -0.211  0.270  0.020 -0.103  0.334 -0.033
 -0.116  0.472] clipped_action=[-0.103 -0.283  0.588  0.168 -0.211  0.270  0.020 -0.103  0.334 -0.033
 -0.116  0.472] startup_limited_action=[-0.103 -0.283  0.588  0.168 -0.211  0.270  0.020 -0.103  0.334 -0.033
 -0.116  0.472] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.103 -0.283  0.588  0.168 -0.211  0.270  0.020 -0.103  0.334 -0.033
 -0.116  0.472] applied_action=[-0.070 -0.224  0.597  0.118 -0.270  0.324  0.022 -0.094  0.345 -0.016
 -0.125  0.442] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.029  0.591 -1.185  0.038  0.578 -1.278 -0.081  0.621 -1.249  0.082
  0.620 -1.213] commanded_leg_q=[-0.029  0.591 -1.185  0.038  0.578 -1.278 -0.081  0.621 -1.249  0.082
  0.620 -1.213] current_leg_q=[-0.019  0.630 -1.325  0.017  0.633 -1.341 -0.089  0.633 -1.344  0.087
  0.640 -1.336] leg_q_error=[-0.010 -0.039  0.140  0.021 -0.055  0.063  0.007 -0.012  0.095 -0.005
 -0.020  0.123] current_leg_dq=[ 0.097  0.667  0.479 -0.291 -1.314  0.261  0.054 -1.124  0.364 -0.198
 -0.512 -0.360] current_tau_est=[ 1.163  0.272 20.815 -0.074 -4.379  3.698  7.694  4.750  3.698 -7.298
 -2.152  6.022] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.037  0.589 -1.190  0.044  0.575 -1.273 -0.079  0.623 -1.251  0.086
  0.615 -1.204] lowcmd_leg_q_hw=[-0.037  0.589 -1.190  0.044  0.575 -1.273 -0.079  0.623 -1.251  0.086
  0.615 -1.204] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 49.000 46.000 53.000]
2026-06-04 21:51:30,628 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:31,109 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.017  0.002 -0.003] commands=[-0.100  0.000  0.000] raw_action=[-0.016 -0.167  0.453  0.117 -0.314  0.388  0.008 -0.087  0.309 -0.053
 -0.157  0.504] clipped_action=[-0.016 -0.167  0.453  0.117 -0.314  0.388  0.008 -0.087  0.309 -0.053
 -0.157  0.504] startup_limited_action=[-0.016 -0.167  0.453  0.117 -0.314  0.388  0.008 -0.087  0.309 -0.053
 -0.157  0.504] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.016 -0.167  0.453  0.117 -0.314  0.388  0.008 -0.087  0.309 -0.053
 -0.157  0.504] applied_action=[-0.066 -0.267  0.518  0.152 -0.276  0.324  0.017 -0.094  0.306 -0.068
 -0.155  0.470] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.028  0.577 -1.210  0.044  0.576 -1.278 -0.082  0.621 -1.261  0.072
  0.610 -1.204] commanded_leg_q=[-0.028  0.577 -1.210  0.044  0.576 -1.278 -0.082  0.621 -1.261  0.072
  0.610 -1.204] current_leg_q=[-0.019  0.629 -1.320  0.017  0.631 -1.334 -0.085  0.635 -1.339  0.088
  0.637 -1.329] leg_q_error=[-0.009 -0.052  0.110  0.027 -0.055  0.056  0.003 -0.014  0.078 -0.015
 -0.027  0.125] current_leg_dq=[-0.012  1.310 -0.510 -0.244  1.310 -0.340  0.705  0.550 -0.269 -0.384
  0.128 -0.445] current_tau_est=[ 1.682 -0.792  4.457 -0.792 -0.470  5.121  5.517  0.643  5.595 -7.718
  2.944  5.405] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.020  0.569 -1.215  0.039  0.560 -1.263 -0.088  0.614 -1.268  0.075
  0.618 -1.227] lowcmd_leg_q_hw=[-0.020  0.569 -1.215  0.039  0.560 -1.263 -0.088  0.614 -1.268  0.075
  0.618 -1.227] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 47.000 47.000 51.000]
2026-06-04 21:51:31,134 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:31,644 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.000 -0.000 -0.000] safe_cmd=[-0.082  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:31,654 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.011  0.002 -0.002] commands=[-0.082  0.000  0.000] raw_action=[-0.081 -0.179  0.562  0.170 -0.264  0.364 -0.006 -0.109  0.310 -0.035
 -0.137  0.476] clipped_action=[-0.081 -0.179  0.562  0.170 -0.264  0.364 -0.006 -0.109  0.310 -0.035
 -0.137  0.476] startup_limited_action=[-0.081 -0.179  0.562  0.170 -0.264  0.364 -0.006 -0.109  0.310 -0.035
 -0.137  0.476] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.081 -0.179  0.562  0.170 -0.264  0.364 -0.006 -0.109  0.310 -0.035
 -0.137  0.476] applied_action=[-0.046 -0.192  0.536  0.157 -0.329  0.445 -0.002 -0.112  0.310 -0.039
 -0.125  0.425] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.025  0.601 -1.204  0.045  0.559 -1.239 -0.086  0.615 -1.260  0.078
  0.620 -1.218] commanded_leg_q=[-0.025  0.601 -1.204  0.045  0.559 -1.239 -0.086  0.615 -1.260  0.078
  0.620 -1.218] current_leg_q=[-0.020  0.620 -1.317  0.020  0.627 -1.333 -0.088  0.631 -1.339  0.089
  0.634 -1.328] leg_q_error=[-0.005 -0.020  0.113  0.026 -0.068  0.094  0.002 -0.016  0.079 -0.011
 -0.014  0.109] current_leg_dq=[ 0.171 -0.636  0.119 -0.705 -1.039  0.089  0.907 -0.876  0.053 -0.651
  0.000  0.067] current_tau_est=[ 1.039  8.733  1.991  0.223 -5.962  3.841  5.616 -0.891  4.267 -7.174
 -2.152  3.272] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.026  0.578 -1.214  0.048  0.557 -1.250 -0.084  0.614 -1.258  0.077
  0.624 -1.225] lowcmd_leg_q_hw=[-0.026  0.578 -1.214  0.048  0.557 -1.250 -0.084  0.614 -1.258  0.077
  0.624 -1.225] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[43.000 43.000 44.000 52.000]
2026-06-04 21:51:32,145 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.017  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:32,230 INFO Policy diag | handover=1.000 est_lin_vel=[-0.003 -0.003 -0.003] commands=[-0.042  0.000  0.000] raw_action=[ 0.104 -0.145  0.527  0.146 -0.161  0.405 -0.070 -0.127  0.241 -0.072
 -0.062  0.433] clipped_action=[ 0.104 -0.145  0.527  0.146 -0.161  0.405 -0.070 -0.127  0.241 -0.072
 -0.062  0.433] startup_limited_action=[ 0.104 -0.145  0.527  0.146 -0.161  0.405 -0.070 -0.127  0.241 -0.072
 -0.062  0.433] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.104 -0.145  0.527  0.146 -0.161  0.405 -0.070 -0.127  0.241 -0.072
 -0.062  0.433] applied_action=[ 0.060 -0.120  0.596  0.121 -0.160  0.362 -0.066 -0.117  0.271 -0.047
 -0.100  0.430] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.006  0.624 -1.185  0.039  0.613 -1.265 -0.097  0.614 -1.272  0.076
  0.628 -1.217] commanded_leg_q=[-0.006  0.624 -1.185  0.039  0.613 -1.265 -0.097  0.614 -1.272  0.076
  0.628 -1.217] current_leg_q=[-0.017  0.640 -1.323  0.018  0.644 -1.343 -0.085  0.642 -1.339  0.089
  0.642 -1.321] leg_q_error=[ 0.012 -0.016  0.138  0.021 -0.031  0.078 -0.012 -0.028  0.066 -0.013
 -0.014  0.105] current_leg_dq=[-0.190  0.147  0.744 -0.221  0.066 -0.004  0.124  0.109  0.081  0.027
 -0.419  0.071] current_tau_est=[ 2.251  2.127  0.616 -0.272  2.152  3.651  8.535  2.400 12.944 -9.302
  0.272 22.380] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.013  0.614 -1.179  0.035  0.633 -1.328 -0.091  0.626 -1.269  0.079
  0.623 -1.204] lowcmd_leg_q_hw=[-0.013  0.614 -1.179  0.035  0.633 -1.328 -0.091  0.626 -1.269  0.079
  0.623 -1.204] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[53.000 40.000 40.000 52.000]
2026-06-04 21:51:32,667 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:32,770 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.019 -0.005 -0.004] commands=[-0.100  0.000  0.000] raw_action=[-0.151 -0.293  0.601  0.155 -0.345  0.387  0.029 -0.083  0.344 -0.010
 -0.143  0.424] clipped_action=[-0.151 -0.293  0.601  0.155 -0.345  0.387  0.029 -0.083  0.344 -0.010
 -0.143  0.424] startup_limited_action=[-0.151 -0.293  0.601  0.155 -0.345  0.387  0.029 -0.083  0.344 -0.010
 -0.143  0.424] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.151 -0.293  0.601  0.155 -0.345  0.387  0.029 -0.083  0.344 -0.010
 -0.143  0.424] applied_action=[-0.149 -0.349  0.631  0.174 -0.263  0.314  0.027 -0.095  0.331 -0.037
 -0.139  0.445] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.043  0.551 -1.174  0.048  0.580 -1.281 -0.080  0.621 -1.253  0.078
  0.615 -1.212] commanded_leg_q=[-0.043  0.551 -1.174  0.048  0.580 -1.281 -0.080  0.621 -1.253  0.078
  0.615 -1.212] current_leg_q=[-0.020  0.632 -1.328  0.020  0.632 -1.338 -0.088  0.637 -1.342  0.089
  0.641 -1.331] leg_q_error=[-0.023 -0.081  0.154  0.028 -0.052  0.058  0.008 -0.016  0.089 -0.011
 -0.026  0.119] current_leg_dq=[-0.050 -0.302 -0.404  0.283 -0.213 -0.295 -0.167 -1.023 -0.210  0.391
 -0.403 -0.358] current_tau_est=[  2.400 -10.341   5.310  -3.117 -13.977   5.974  10.019   1.064   5.832
 -10.514  -0.470   5.595] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.025  0.577 -1.216  0.041  0.556 -1.251 -0.084  0.616 -1.262  0.077
  0.624 -1.227] lowcmd_leg_q_hw=[-0.025  0.577 -1.216  0.041  0.556 -1.251 -0.084  0.616 -1.262  0.077
  0.624 -1.227] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[49.000 50.000 44.000 49.000]
2026-06-04 21:51:33,185 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:33,311 INFO Policy diag | handover=1.000 est_lin_vel=[-0.010 -0.004 -0.006] commands=[-0.100  0.000  0.000] raw_action=[-0.062 -0.214  0.547  0.186 -0.343  0.522  0.010 -0.125  0.291 -0.018
 -0.091  0.418] clipped_action=[-0.062 -0.214  0.547  0.186 -0.343  0.522  0.010 -0.125  0.291 -0.018
 -0.091  0.418] startup_limited_action=[-0.062 -0.214  0.547  0.186 -0.343  0.522  0.010 -0.125  0.291 -0.018
 -0.091  0.418] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.062 -0.214  0.547  0.186 -0.343  0.522  0.010 -0.125  0.291 -0.018
 -0.091  0.418] applied_action=[-0.029 -0.230  0.517  0.154 -0.315  0.439  0.005 -0.114  0.299 -0.021
 -0.088  0.391] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.022  0.589 -1.211  0.044  0.563 -1.241 -0.084  0.615 -1.263  0.081
  0.632 -1.229] commanded_leg_q=[-0.022  0.589 -1.211  0.044  0.563 -1.241 -0.084  0.615 -1.263  0.081
  0.632 -1.229] current_leg_q=[-0.018  0.633 -1.330  0.017  0.634 -1.342 -0.083  0.638 -1.343  0.086
  0.642 -1.334] leg_q_error=[-0.004 -0.044  0.119  0.028 -0.071  0.101 -0.001 -0.023  0.079 -0.005
 -0.011  0.105] current_leg_dq=[ 0.244  0.515  0.473 -0.225 -1.124 -0.049  0.155  0.477 -0.020 -0.171
  0.198 -0.144] current_tau_est=[ 0.693  1.039 21.242 -0.742 -3.686  4.789  8.139  0.643  4.410 -8.263
  1.781  4.267] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.025  0.589 -1.193  0.044  0.573 -1.254 -0.085  0.614 -1.254  0.079
  0.632 -1.219] lowcmd_leg_q_hw=[-0.025  0.589 -1.193  0.044  0.573 -1.254 -0.085  0.614 -1.254  0.079
  0.632 -1.219] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[50.000 54.000 42.000 49.000]
2026-06-04 21:51:33,687 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:33,851 INFO Policy diag | handover=1.000 est_lin_vel=[-0.004 -0.003 -0.009] commands=[-0.100  0.000  0.000] raw_action=[-0.069 -0.180  0.606  0.179 -0.246  0.385 -0.017 -0.110  0.321 -0.072
 -0.128  0.506] clipped_action=[-0.069 -0.180  0.606  0.179 -0.246  0.385 -0.017 -0.110  0.321 -0.072
 -0.128  0.506] startup_limited_action=[-0.069 -0.180  0.606  0.179 -0.246  0.385 -0.017 -0.110  0.321 -0.072
 -0.128  0.506] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.069 -0.180  0.606  0.179 -0.246  0.385 -0.017 -0.110  0.321 -0.072
 -0.128  0.506] applied_action=[-0.002 -0.209  0.479  0.154 -0.262  0.349 -0.003 -0.110  0.303 -0.078
 -0.119  0.497] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.017  0.596 -1.223  0.044  0.580 -1.270 -0.086  0.616 -1.262  0.071
  0.622 -1.195] commanded_leg_q=[-0.017  0.596 -1.223  0.044  0.580 -1.270 -0.086  0.616 -1.262  0.071
  0.622 -1.195] current_leg_q=[-0.020  0.619 -1.321  0.019  0.625 -1.339 -0.086  0.630 -1.339  0.087
  0.634 -1.330] leg_q_error=[ 0.003 -0.023  0.098  0.025 -0.045  0.069  0.001 -0.015  0.077 -0.016
 -0.012  0.135] current_leg_dq=[ 0.062 -1.252 -0.574 -0.767 -0.775 -0.032  0.783 -0.915 -0.024 -0.403
 -0.225 -0.105] current_tau_est=[ 1.410  5.096  5.121  0.371  2.523  4.504  5.195  0.841  4.647 -7.545
  0.643  4.315] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.021  0.563 -1.190  0.044  0.596 -1.294 -0.088  0.613 -1.261  0.068
  0.627 -1.217] lowcmd_leg_q_hw=[-0.021  0.563 -1.190  0.044  0.596 -1.294 -0.088  0.613 -1.261  0.068
  0.627 -1.217] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[45.000 45.000 45.000 52.000]
2026-06-04 21:51:34,206 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:34,389 INFO Policy diag | handover=1.000 est_lin_vel=[0.013 0.002 0.000] commands=[-0.100  0.000  0.000] raw_action=[-0.041 -0.244  0.499  0.143 -0.281  0.362  0.002 -0.107  0.303 -0.071
 -0.123  0.459] clipped_action=[-0.041 -0.244  0.499  0.143 -0.281  0.362  0.002 -0.107  0.303 -0.071
 -0.123  0.459] startup_limited_action=[-0.041 -0.244  0.499  0.143 -0.281  0.362  0.002 -0.107  0.303 -0.071
 -0.123  0.459] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.037 -0.227  0.507  0.138 -0.356  0.472  0.003 -0.111  0.294 -0.060
 -0.130  0.420] applied_action=[-0.037 -0.227  0.507  0.138 -0.356  0.472  0.003 -0.111  0.294 -0.060
 -0.130  0.420] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.023  0.590 -1.214  0.042  0.550 -1.230 -0.085  0.615 -1.265  0.074
  0.618 -1.220] commanded_leg_q=[-0.023  0.590 -1.214  0.042  0.550 -1.230 -0.085  0.615 -1.265  0.074
  0.618 -1.220] current_leg_q=[-0.019  0.631 -1.324  0.020  0.635 -1.334 -0.087  0.636 -1.336  0.085
  0.642 -1.326] leg_q_error=[-0.005 -0.041  0.110  0.021 -0.085  0.104  0.002 -0.020  0.071 -0.011
 -0.024  0.106] current_leg_dq=[ 0.372  0.659 -0.344  0.109  0.070  0.091  0.116  0.213  0.326 -0.085
 -0.027  0.463] current_tau_est=[ 0.421  0.792  4.315 -3.241  3.265  3.414  9.302  2.350  2.655 -8.733
  2.548  1.802] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.033  0.563 -1.190  0.041  0.584 -1.295 -0.082  0.623 -1.262  0.076
  0.612 -1.220] lowcmd_leg_q_hw=[-0.033  0.563 -1.190  0.041  0.584 -1.295 -0.082  0.623 -1.262  0.076
  0.612 -1.220] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[41.000 45.000 40.000 45.000]
2026-06-04 21:51:34,707 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.000 -0.000 -0.000] safe_cmd=[-0.034  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:34,929 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.034 -0.001  0.002] commands=[-0.051  0.000  0.000] raw_action=[-0.034 -0.232  0.581  0.120 -0.258  0.358 -0.031 -0.112  0.267 -0.008
 -0.108  0.359] clipped_action=[-0.034 -0.232  0.581  0.120 -0.258  0.358 -0.031 -0.112  0.267 -0.008
 -0.108  0.359] startup_limited_action=[-0.034 -0.232  0.581  0.120 -0.258  0.358 -0.031 -0.112  0.267 -0.008
 -0.108  0.359] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.034 -0.232  0.581  0.120 -0.258  0.358 -0.031 -0.112  0.267 -0.008
 -0.108  0.359] applied_action=[ 0.007 -0.187  0.598  0.099 -0.209  0.359 -0.052 -0.121  0.247 -0.022
 -0.119  0.366] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.015  0.603 -1.185  0.035  0.597 -1.266 -0.095  0.612 -1.280  0.081
  0.622 -1.237] commanded_leg_q=[-0.015  0.603 -1.185  0.035  0.597 -1.266 -0.095  0.612 -1.280  0.081
  0.622 -1.237] current_leg_q=[-0.018  0.641 -1.329  0.020  0.642 -1.341 -0.086  0.642 -1.340  0.088
  0.644 -1.324] leg_q_error=[ 0.002 -0.038  0.145  0.015 -0.045  0.075 -0.009 -0.030  0.059 -0.007
 -0.023  0.087] current_leg_dq=[ 0.000  0.868  0.012 -0.492  0.787  0.156  1.132  0.481  0.621 -0.430
  0.333  1.464] current_tau_est=[ 1.781  0.000  3.604 -0.421 -0.074  3.888  5.220  0.173  6.496 -7.001
  0.965  7.966] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.011  0.610 -1.194  0.041  0.616 -1.293 -0.096  0.618 -1.277  0.076
  0.627 -1.214] lowcmd_leg_q_hw=[-0.011  0.610 -1.194  0.041  0.616 -1.293 -0.096  0.618 -1.277  0.076
  0.627 -1.214] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[52.000 49.000 40.000 46.000]
2026-06-04 21:51:35,207 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:35,469 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.011  0.003 -0.004] commands=[-0.100  0.000  0.000] raw_action=[-0.095 -0.324  0.649  0.149 -0.335  0.396 -0.014 -0.102  0.310 -0.027
 -0.125  0.370] clipped_action=[-0.095 -0.324  0.649  0.149 -0.335  0.396 -0.014 -0.102  0.310 -0.027
 -0.125  0.370] startup_limited_action=[-0.095 -0.324  0.649  0.149 -0.335  0.396 -0.014 -0.102  0.310 -0.027
 -0.125  0.370] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.095 -0.324  0.649  0.149 -0.335  0.396 -0.014 -0.102  0.310 -0.027
 -0.125  0.370] applied_action=[-0.079 -0.274  0.606  0.146 -0.275  0.349  0.005 -0.110  0.336 -0.050
 -0.113  0.409] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.031  0.575 -1.182  0.043  0.576 -1.270 -0.084  0.616 -1.252  0.076
  0.623 -1.223] commanded_leg_q=[-0.031  0.575 -1.182  0.043  0.576 -1.270 -0.084  0.616 -1.252  0.076
  0.623 -1.223] current_leg_q=[-0.021  0.625 -1.327  0.021  0.628 -1.337 -0.088  0.633 -1.339  0.088
  0.640 -1.323] leg_q_error=[-0.010 -0.050  0.145  0.022 -0.051  0.067  0.004 -0.017  0.088 -0.012
 -0.017  0.100] current_leg_dq=[-0.031 -0.221 -0.645  0.674 -0.085 -0.415  0.484 -0.605  0.206  0.054
 -0.097  0.014] current_tau_est=[  0.148 -11.231   6.591   4.008   4.478   6.211   5.715   7.125  12.612
  -8.634   2.499  18.729] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.025  0.569 -1.187  0.046  0.573 -1.250 -0.086  0.612 -1.264  0.072
  0.626 -1.226] lowcmd_leg_q_hw=[-0.025  0.569 -1.187  0.046  0.573 -1.250 -0.086  0.612 -1.264  0.072
  0.626 -1.226] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[50.000 50.000 44.000 47.000]
2026-06-04 21:51:35,725 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:36,010 INFO Policy diag | handover=1.000 est_lin_vel=[-0.006 -0.001 -0.003] commands=[-0.100  0.000  0.000] raw_action=[ 0.021 -0.257  0.486  0.137 -0.240  0.317  0.021 -0.097  0.305 -0.085
 -0.111  0.455] clipped_action=[ 0.021 -0.257  0.486  0.137 -0.240  0.317  0.021 -0.097  0.305 -0.085
 -0.111  0.455] startup_limited_action=[ 0.021 -0.257  0.486  0.137 -0.240  0.317  0.021 -0.097  0.305 -0.085
 -0.111  0.455] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.021 -0.257  0.486  0.137 -0.240  0.317  0.021 -0.097  0.305 -0.085
 -0.111  0.455] applied_action=[-0.076 -0.289  0.554  0.150 -0.285  0.342  0.017 -0.102  0.322  0.001
 -0.122  0.444] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.030  0.570 -1.199  0.044  0.573 -1.272 -0.082  0.618 -1.256  0.085
  0.621 -1.212] commanded_leg_q=[-0.030  0.570 -1.199  0.044  0.573 -1.272 -0.082  0.618 -1.256  0.085
  0.621 -1.212] current_leg_q=[-0.019  0.629 -1.315  0.018  0.633 -1.336 -0.084  0.637 -1.338  0.088
  0.639 -1.328] leg_q_error=[-0.011 -0.059  0.116  0.025 -0.060  0.064  0.002 -0.019  0.082 -0.003
 -0.018  0.116] current_leg_dq=[-0.372  1.008  1.207  1.039  0.953  0.113 -1.302 -0.717  0.190  0.825
 -0.632  0.404] current_tau_est=[ 1.163 -0.940 -1.470 -2.078 -1.163  8.250 10.316 -0.643 15.457 -9.425
 -0.049 16.832] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.023  0.587 -1.171  0.036  0.593 -1.289 -0.081  0.622 -1.251  0.073
  0.615 -1.220] lowcmd_leg_q_hw=[-0.023  0.587 -1.171  0.036  0.593 -1.289 -0.081  0.622 -1.251  0.073
  0.615 -1.220] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 43.000 38.000 42.000]
2026-06-04 21:51:36,226 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:36,551 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.006  0.003 -0.002] commands=[-0.052  0.000  0.000] raw_action=[ 0.028 -0.100  0.554  0.126 -0.191  0.412 -0.024 -0.120  0.284 -0.069
 -0.118  0.437] clipped_action=[ 0.028 -0.100  0.554  0.126 -0.191  0.412 -0.024 -0.120  0.284 -0.069
 -0.118  0.437] startup_limited_action=[ 0.028 -0.100  0.554  0.126 -0.191  0.412 -0.024 -0.120  0.284 -0.069
 -0.118  0.437] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.028 -0.100  0.554  0.126 -0.191  0.412 -0.024 -0.120  0.284 -0.069
 -0.118  0.437] applied_action=[-0.009 -0.207  0.582  0.137 -0.206  0.345 -0.024 -0.108  0.274 -0.044
 -0.101  0.448] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.018  0.596 -1.190  0.042  0.598 -1.271 -0.089  0.617 -1.272  0.077
  0.627 -1.211] commanded_leg_q=[-0.018  0.596 -1.190  0.042  0.598 -1.271 -0.089  0.617 -1.272  0.077
  0.627 -1.211] current_leg_q=[-0.022  0.630 -1.321  0.024  0.633 -1.336 -0.087  0.635 -1.337  0.086
  0.640 -1.325] leg_q_error=[ 0.004 -0.034  0.132  0.018 -0.034  0.065 -0.002 -0.018  0.065 -0.010
 -0.012  0.115] current_leg_dq=[-0.155 -0.213 -0.590  0.256  0.298 -0.368 -0.515  0.515 -0.447  0.306
  0.372  0.142] current_tau_est=[ 2.721 -1.806  5.405 -5.096  2.721  5.879  0.891  0.767  6.211 -3.265
  1.509 21.242] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.026  0.612 -1.188  0.046  0.585 -1.248 -0.089  0.612 -1.264  0.081
  0.622 -1.216] lowcmd_leg_q_hw=[-0.026  0.612 -1.188  0.046  0.585 -1.248 -0.089  0.612 -1.264  0.081
  0.622 -1.216] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[56.000 45.000 40.000 46.000]
2026-06-04 21:51:36,727 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.021  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:37,090 INFO Policy diag | handover=1.000 est_lin_vel=[-0.001  0.000 -0.004] commands=[-0.100  0.000  0.000] raw_action=[-0.034 -0.207  0.604  0.116 -0.210  0.292  0.010 -0.105  0.348 -0.072
 -0.116  0.425] clipped_action=[-0.034 -0.207  0.604  0.116 -0.210  0.292  0.010 -0.105  0.348 -0.072
 -0.116  0.425] startup_limited_action=[-0.034 -0.207  0.604  0.116 -0.210  0.292  0.010 -0.105  0.348 -0.072
 -0.116  0.425] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.034 -0.207  0.604  0.116 -0.210  0.292  0.010 -0.105  0.348 -0.072
 -0.116  0.425] applied_action=[-0.080 -0.250  0.625  0.116 -0.200  0.235  0.020 -0.087  0.354 -0.023
 -0.128  0.394] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.031  0.582 -1.176  0.038  0.600 -1.306 -0.082  0.623 -1.246  0.081
  0.619 -1.228] commanded_leg_q=[-0.031  0.582 -1.176  0.038  0.600 -1.306 -0.082  0.623 -1.246  0.081
  0.619 -1.228] current_leg_q=[-0.019  0.636 -1.326  0.020  0.638 -1.340 -0.088  0.639 -1.340  0.088
  0.642 -1.327] leg_q_error=[-0.012 -0.054  0.150  0.017 -0.038  0.034  0.006 -0.016  0.094 -0.008
 -0.023  0.099] current_leg_dq=[ 0.221  0.047 -0.560  0.279  0.694 -0.226 -0.407 -0.209 -0.216  0.271
  0.430 -0.281] current_tau_est=[  1.064   2.103   6.259  -3.142  -0.470   5.832  11.231   4.008   5.690
 -10.143   0.173   4.647] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.035  0.595 -1.171  0.041  0.582 -1.271 -0.078  0.622 -1.250  0.080
  0.612 -1.207] lowcmd_leg_q_hw=[-0.035  0.595 -1.171  0.041  0.582 -1.271 -0.078  0.622 -1.250  0.080
  0.612 -1.207] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[50.000 46.000 43.000 50.000]
2026-06-04 21:51:37,247 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:37,630 INFO Policy diag | handover=1.000 est_lin_vel=[-0.002  0.001 -0.003] commands=[-0.100  0.000  0.000] raw_action=[ 0.010 -0.139  0.518  0.156 -0.296  0.505 -0.025 -0.116  0.304 -0.072
 -0.116  0.385] clipped_action=[ 0.010 -0.139  0.518  0.156 -0.296  0.505 -0.025 -0.116  0.304 -0.072
 -0.116  0.385] startup_limited_action=[ 0.010 -0.139  0.518  0.156 -0.296  0.505 -0.025 -0.116  0.304 -0.072
 -0.116  0.385] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.010 -0.139  0.518  0.156 -0.296  0.505 -0.025 -0.116  0.304 -0.072
 -0.116  0.385] applied_action=[-0.061 -0.222  0.534  0.193 -0.296  0.438  0.016 -0.109  0.306 -0.055
 -0.112  0.458] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.027  0.592 -1.205  0.052  0.570 -1.241 -0.082  0.616 -1.261  0.075
  0.624 -1.208] commanded_leg_q=[-0.027  0.592 -1.205  0.052  0.570 -1.241 -0.082  0.616 -1.261  0.075
  0.624 -1.208] current_leg_q=[-0.019  0.634 -1.319  0.019  0.633 -1.335 -0.085  0.638 -1.340  0.087
  0.643 -1.333] leg_q_error=[-0.009 -0.043  0.114  0.033 -0.064  0.094  0.003 -0.022  0.079 -0.012
 -0.019  0.125] current_leg_dq=[-0.050 -1.341  0.431 -0.287  1.744 -0.097  0.236  0.357 -0.156 -0.031
  0.240 -0.269] current_tau_est=[ 1.905 -1.361 18.444 -0.544 -2.622  4.836  7.446  0.544  4.979 -8.386
  1.633  4.647] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.027  0.592 -1.205  0.052  0.570 -1.241 -0.082  0.616 -1.261  0.075
  0.624 -1.208] lowcmd_leg_q_hw=[-0.027  0.592 -1.205  0.052  0.570 -1.241 -0.082  0.616 -1.261  0.075
  0.624 -1.208] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 50.000 44.000 50.000]
2026-06-04 21:51:37,766 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:38,172 INFO Policy diag | handover=1.000 est_lin_vel=[ 1.588e-02 -6.505e-06 -4.268e-03] commands=[-0.087  0.000  0.000] raw_action=[-0.028 -0.298  0.552  0.148 -0.296  0.399 -0.005 -0.117  0.289 -0.055
 -0.100  0.374] clipped_action=[-0.028 -0.298  0.552  0.148 -0.296  0.399 -0.005 -0.117  0.289 -0.055
 -0.100  0.374] startup_limited_action=[-0.028 -0.298  0.552  0.148 -0.296  0.399 -0.005 -0.117  0.289 -0.055
 -0.100  0.374] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.028 -0.298  0.552  0.148 -0.296  0.399 -0.005 -0.117  0.289 -0.055
 -0.100  0.374] applied_action=[-0.011 -0.227  0.495  0.164 -0.308  0.430 -0.003 -0.117  0.294 -0.078
 -0.117  0.431] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.018  0.590 -1.218  0.046  0.566 -1.244 -0.086  0.614 -1.265  0.071
  0.622 -1.216] commanded_leg_q=[-0.018  0.590 -1.218  0.046  0.566 -1.244 -0.086  0.614 -1.265  0.071
  0.622 -1.216] current_leg_q=[-0.019  0.635 -1.321  0.017  0.633 -1.334 -0.084  0.639 -1.339  0.088
  0.641 -1.323] leg_q_error=[ 0.000 -0.045  0.103  0.029 -0.067  0.090 -0.002 -0.026  0.074 -0.017
 -0.018  0.107] current_leg_dq=[ 0.279  0.039  0.214  0.399  0.496  0.075 -0.523  0.612  0.156  0.023
  0.360  0.000] current_tau_est=[ 0.940  2.622  1.707  2.944  1.707  3.509  4.156  0.693  3.461 -9.178
  1.979  3.509] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.022  0.589 -1.210  0.043  0.587 -1.273 -0.083  0.618 -1.265  0.067
  0.613 -1.201] lowcmd_leg_q_hw=[-0.022  0.589 -1.210  0.043  0.587 -1.273 -0.083  0.618 -1.265  0.067
  0.613 -1.201] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[52.000 47.000 45.000 50.000]
2026-06-04 21:51:38,287 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.000 -0.000 -0.000] safe_cmd=[-0.052  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:38,709 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.007 -0.006 -0.004] commands=[-0.006  0.000  0.000] raw_action=[ 0.063 -0.130  0.609  0.069 -0.154  0.210 -0.033 -0.069  0.223 -0.034
 -0.121  0.513] clipped_action=[ 0.063 -0.130  0.609  0.069 -0.154  0.210 -0.033 -0.069  0.223 -0.034
 -0.121  0.513] startup_limited_action=[ 0.063 -0.130  0.609  0.069 -0.154  0.210 -0.033 -0.069  0.223 -0.034
 -0.121  0.513] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.063 -0.130  0.609  0.069 -0.154  0.210 -0.033 -0.069  0.223 -0.034
 -0.121  0.513] applied_action=[ 0.075 -0.108  0.570  0.079 -0.164  0.256 -0.033 -0.068  0.222 -0.070
 -0.109  0.506] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.003  0.628 -1.193  0.031  0.612 -1.299 -0.091  0.629 -1.288  0.072
  0.625 -1.192] commanded_leg_q=[-0.003  0.628 -1.193  0.031  0.612 -1.299 -0.091  0.629 -1.288  0.072
  0.625 -1.192] current_leg_q=[-0.017  0.642 -1.321  0.019  0.646 -1.339 -0.088  0.645 -1.340  0.089
  0.648 -1.321] leg_q_error=[ 0.014 -0.014  0.127  0.012 -0.034  0.040 -0.003 -0.016  0.052 -0.017
 -0.023  0.129] current_leg_dq=[-0.093 -0.078 -0.087  0.085 -0.465  0.044  0.182 -0.004  0.000  0.391
 -0.236  0.550] current_tau_est=[ 1.682 -2.523  3.698 -2.350  5.467  4.457  9.302  2.820  4.599 -6.877
  3.463  1.802] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[ 0.003  0.615 -1.187  0.031  0.610 -1.291 -0.099  0.624 -1.291  0.077
  0.624 -1.179] lowcmd_leg_q_hw=[ 0.003  0.615 -1.187  0.031  0.610 -1.291 -0.099  0.624 -1.291  0.077
  0.624 -1.179] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[54.000 49.000 38.000 53.000]
2026-06-04 21:51:38,807 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.037  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:39,248 INFO Policy diag | handover=1.000 est_lin_vel=[0.021 0.001 0.000] commands=[-0.073  0.000  0.000] raw_action=[-0.116 -0.214  0.524  0.137 -0.287  0.360  0.024 -0.085  0.303  0.064
 -0.148  0.474] clipped_action=[-0.116 -0.214  0.524  0.137 -0.287  0.360  0.024 -0.085  0.303  0.064
 -0.148  0.474] startup_limited_action=[-0.116 -0.214  0.524  0.137 -0.287  0.360  0.024 -0.085  0.303  0.064
 -0.148  0.474] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.116 -0.214  0.524  0.137 -0.287  0.360  0.024 -0.085  0.303  0.064
 -0.148  0.474] applied_action=[-0.094 -0.225  0.541  0.152 -0.271  0.334  0.017 -0.082  0.315  0.005
 -0.123  0.420] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.033  0.591 -1.203  0.044  0.577 -1.275 -0.082  0.625 -1.258  0.086
  0.620 -1.220] commanded_leg_q=[-0.033  0.591 -1.203  0.044  0.577 -1.275 -0.082  0.625 -1.258  0.086
  0.620 -1.220] current_leg_q=[-0.018  0.636 -1.324  0.020  0.634 -1.337 -0.087  0.637 -1.339  0.089
  0.641 -1.323] leg_q_error=[-0.015 -0.045  0.121  0.024 -0.056  0.063  0.005 -0.012  0.081 -0.003
 -0.021  0.103] current_leg_dq=[ 0.151  0.992  1.037 -0.763 -1.325  0.249  1.015 -0.279  0.055 -0.903
 -0.217 -0.210] current_tau_est=[ 1.385 -0.025 11.380  1.187  8.733 15.173  3.760  5.269 11.380 -5.245
  3.785  4.267] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.015  0.614 -1.218  0.040  0.582 -1.247 -0.088  0.614 -1.280  0.081
  0.624 -1.225] lowcmd_leg_q_hw=[-0.015  0.614 -1.218  0.040  0.582 -1.247 -0.088  0.614 -1.280  0.081
  0.624 -1.225] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[48.000 46.000 42.000 52.000]
2026-06-04 21:51:39,308 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.451, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.045 -0.000 -0.000] safe_cmd=[-0.079  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:39,790 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.018  0.002 -0.002] commands=[-0.100  0.000  0.000] raw_action=[-0.051 -0.203  0.522  0.163 -0.340  0.438 -0.013 -0.117  0.314 -0.034
 -0.112  0.377] clipped_action=[-0.051 -0.203  0.522  0.163 -0.340  0.438 -0.013 -0.117  0.314 -0.034
 -0.112  0.377] startup_limited_action=[-0.051 -0.203  0.522  0.163 -0.340  0.438 -0.013 -0.117  0.314 -0.034
 -0.112  0.377] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.051 -0.203  0.522  0.163 -0.340  0.438 -0.013 -0.117  0.314 -0.034
 -0.112  0.377] applied_action=[-0.028 -0.228  0.527  0.154 -0.371  0.499 -0.010 -0.121  0.295 -0.049
 -0.116  0.362] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.022  0.589 -1.207  0.044  0.546 -1.222 -0.087  0.612 -1.265  0.076
  0.623 -1.238] commanded_leg_q=[-0.022  0.589 -1.207  0.044  0.546 -1.222 -0.087  0.612 -1.265  0.076
  0.623 -1.238] current_leg_q=[-0.020  0.634 -1.329  0.020  0.632 -1.339 -0.086  0.636 -1.340  0.087
  0.641 -1.329] leg_q_error=[-0.001 -0.045  0.122  0.025 -0.086  0.117 -0.001 -0.024  0.075 -0.011
 -0.018  0.091] current_leg_dq=[-0.186  0.608 -0.394 -0.736  1.515  0.309  0.543 -0.481  0.044 -0.167
 -0.178  0.067] current_tau_est=[ 0.322  0.124  5.216  0.940 -3.241 21.337  5.937  5.566 15.410 -7.966
  3.859 18.966] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.027  0.594 -1.214  0.047  0.539 -1.214 -0.086  0.614 -1.260  0.078
  0.626 -1.230] lowcmd_leg_q_hw=[-0.027  0.594 -1.214  0.047  0.539 -1.214 -0.086  0.614 -1.260  0.078
  0.626 -1.230] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[52.000 56.000 46.000 53.000]
2026-06-04 21:51:39,816 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:40,328 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:40,339 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.016  0.002 -0.003] commands=[-0.100  0.000  0.000] raw_action=[-0.067 -0.226  0.542  0.159 -0.314  0.400  0.016 -0.100  0.298 -0.049
 -0.125  0.448] clipped_action=[-0.067 -0.226  0.542  0.159 -0.314  0.400  0.016 -0.100  0.298 -0.049
 -0.125  0.448] startup_limited_action=[-0.067 -0.226  0.542  0.159 -0.314  0.400  0.016 -0.100  0.298 -0.049
 -0.125  0.448] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.067 -0.226  0.542  0.159 -0.314  0.400  0.016 -0.100  0.298 -0.049
 -0.125  0.448] applied_action=[-0.070 -0.171  0.523  0.169 -0.340  0.477  0.004 -0.115  0.319 -0.040
 -0.113  0.419] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.029  0.608 -1.209  0.047  0.555 -1.229 -0.084  0.614 -1.257  0.077
  0.624 -1.220] commanded_leg_q=[-0.029  0.608 -1.209  0.047  0.555 -1.229 -0.084  0.614 -1.257  0.077
  0.624 -1.220] current_leg_q=[-0.018  0.636 -1.330  0.019  0.638 -1.341 -0.087  0.638 -1.346  0.087
  0.646 -1.338] leg_q_error=[-0.011 -0.028  0.122  0.028 -0.082  0.112  0.003 -0.024  0.089 -0.009
 -0.022  0.118] current_leg_dq=[-0.360 -0.345  0.704  0.977 -0.891  0.495 -0.639 -0.322  0.218  0.182
 -0.244  0.263] current_tau_est=[  1.187   5.517  17.923  -3.142   8.312  16.975   9.598   5.863  15.220
 -10.118   3.538  18.539] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.026  0.601 -1.195  0.046  0.560 -1.231 -0.086  0.614 -1.258  0.076
  0.625 -1.228] lowcmd_leg_q_hw=[-0.026  0.601 -1.195  0.046  0.560 -1.231 -0.086  0.614 -1.258  0.076
  0.625 -1.228] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 53.000 43.000 46.000]
2026-06-04 21:51:40,845 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:40,892 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.014  0.007 -0.004] commands=[-0.100  0.000  0.000] raw_action=[ 0.041 -0.173  0.519  0.176 -0.260  0.447 -0.037 -0.142  0.304 -0.103
 -0.099  0.432] clipped_action=[ 0.041 -0.173  0.519  0.176 -0.260  0.447 -0.037 -0.142  0.304 -0.103
 -0.099  0.432] startup_limited_action=[ 0.041 -0.173  0.519  0.176 -0.260  0.447 -0.037 -0.142  0.304 -0.103
 -0.099  0.432] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.041 -0.173  0.519  0.176 -0.260  0.447 -0.037 -0.142  0.304 -0.103
 -0.099  0.432] applied_action=[-0.002 -0.229  0.530  0.154 -0.257  0.355 -0.020 -0.125  0.306 -0.084
 -0.114  0.423] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.017  0.589 -1.206  0.044  0.582 -1.268 -0.089  0.611 -1.261  0.070
  0.623 -1.219] commanded_leg_q=[-0.017  0.589 -1.206  0.044  0.582 -1.268 -0.089  0.611 -1.261  0.070
  0.623 -1.219] current_leg_q=[-0.018  0.629 -1.323  0.017  0.633 -1.336 -0.084  0.637 -1.342  0.087
  0.640 -1.332] leg_q_error=[ 0.001 -0.040  0.117  0.028 -0.051  0.068 -0.005 -0.026  0.081 -0.017
 -0.017  0.113] current_leg_dq=[ 0.019  0.438 -0.433 -0.178  0.442 -0.146  0.047  0.047 -0.196  0.054
  0.198 -0.366] current_tau_est=[ 1.559  1.707  4.457 -0.792  1.559  4.504  8.386  2.870  5.121 -9.203
  2.325  5.310] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.020  0.590 -1.233  0.045  0.560 -1.251 -0.083  0.616 -1.262  0.077
  0.621 -1.215] lowcmd_leg_q_hw=[-0.020  0.590 -1.233  0.045  0.560 -1.251 -0.083  0.616 -1.262  0.077
  0.621 -1.215] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[46.000 42.000 45.000 51.000]
2026-06-04 21:51:41,346 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:41,429 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.005 -0.005 -0.006] commands=[-0.100  0.000  0.000] raw_action=[-0.036 -0.204  0.555  0.166 -0.270  0.395 -0.009 -0.107  0.305 -0.072
 -0.115  0.464] clipped_action=[-0.036 -0.204  0.555  0.166 -0.270  0.395 -0.009 -0.107  0.305 -0.072
 -0.115  0.464] startup_limited_action=[-0.036 -0.204  0.555  0.166 -0.270  0.395 -0.009 -0.107  0.305 -0.072
 -0.115  0.464] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.036 -0.204  0.555  0.166 -0.270  0.395 -0.009 -0.107  0.305 -0.072
 -0.115  0.464] applied_action=[-0.024 -0.205  0.559  0.172 -0.271  0.447 -0.026 -0.129  0.304 -0.081
 -0.101  0.420] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.021  0.597 -1.197  0.048  0.578 -1.238 -0.090  0.610 -1.262  0.070
  0.627 -1.220] commanded_leg_q=[-0.021  0.597 -1.197  0.048  0.578 -1.238 -0.090  0.610 -1.262  0.070
  0.627 -1.220] current_leg_q=[-0.019  0.632 -1.328  0.017  0.633 -1.338 -0.084  0.634 -1.340  0.086
  0.640 -1.329] leg_q_error=[-0.002 -0.035  0.131  0.031 -0.056  0.100 -0.006 -0.024  0.078 -0.016
 -0.012  0.109] current_leg_dq=[-0.023 -0.434  0.235  0.035 -0.326 -0.051  0.186 -0.101 -0.081  0.120
 -0.372 -0.016] current_tau_est=[  1.707   5.962  25.414  -1.039   5.245  13.276   8.164   5.269  15.031
 -10.019  -0.668  22.380] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.014  0.597 -1.181  0.044  0.603 -1.271 -0.095  0.608 -1.265  0.063
  0.619 -1.212] lowcmd_leg_q_hw=[-0.014  0.597 -1.181  0.044  0.603 -1.271 -0.095  0.608 -1.265  0.063
  0.619 -1.212] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[41.000 48.000 44.000 45.000]
2026-06-04 21:51:41,866 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:41,969 INFO Policy diag | handover=1.000 est_lin_vel=[-0.001  0.004 -0.002] commands=[-0.100  0.000  0.000] raw_action=[-0.041 -0.266  0.539  0.147 -0.315  0.412  0.013 -0.128  0.292 -0.030
 -0.099  0.391] clipped_action=[-0.041 -0.266  0.539  0.147 -0.315  0.412  0.013 -0.128  0.292 -0.030
 -0.099  0.391] startup_limited_action=[-0.041 -0.266  0.539  0.147 -0.315  0.412  0.013 -0.128  0.292 -0.030
 -0.099  0.391] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.041 -0.266  0.539  0.147 -0.315  0.412  0.013 -0.128  0.292 -0.030
 -0.099  0.391] applied_action=[-2.697e-02 -2.395e-01  5.767e-01  1.529e-01 -2.482e-01  3.553e-01
  3.111e-04 -1.146e-01  3.106e-01 -7.054e-02 -1.094e-01  4.430e-01] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.021  0.586 -1.191  0.044  0.585 -1.268 -0.085  0.614 -1.260  0.072
  0.625 -1.212] commanded_leg_q=[-0.021  0.586 -1.191  0.044  0.585 -1.268 -0.085  0.614 -1.260  0.072
  0.625 -1.212] current_leg_q=[-0.018  0.632 -1.327  0.019  0.631 -1.336 -0.088  0.636 -1.339  0.087
  0.638 -1.323] leg_q_error=[-0.003 -0.046  0.135  0.025 -0.046  0.069  0.002 -0.022  0.079 -0.015
 -0.014  0.110] current_leg_dq=[ 0.012  1.500 -0.568 -0.535 -0.236  0.006  0.345  1.023  0.918 -0.221
  0.023  1.555] current_tau_est=[ 1.707 -2.004  6.686  0.074  6.457  4.315  6.086 -0.025  8.582 -7.199
  3.637  8.487] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.021  0.586 -1.188  0.040  0.591 -1.285 -0.085  0.617 -1.253  0.074
  0.623 -1.209] lowcmd_leg_q_hw=[-0.021  0.586 -1.188  0.040  0.591 -1.285 -0.085  0.617 -1.253  0.074
  0.623 -1.209] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[54.000 50.000 46.000 52.000]
2026-06-04 21:51:42,366 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 1.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.100 -0.000 -0.000] safe_cmd=[-0.100  0.000  0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:42,509 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.019  0.000 -0.006] commands=[-0.090  0.000  0.000] raw_action=[-0.090 -0.203  0.590  0.177 -0.269  0.386 -0.002 -0.110  0.321 -0.066
 -0.134  0.451] clipped_action=[-0.090 -0.203  0.590  0.177 -0.269  0.386 -0.002 -0.110  0.321 -0.066
 -0.134  0.451] startup_limited_action=[-0.090 -0.203  0.590  0.177 -0.269  0.386 -0.002 -0.110  0.321 -0.066
 -0.134  0.451] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[-0.090 -0.203  0.590  0.177 -0.269  0.386 -0.002 -0.110  0.321 -0.066
 -0.134  0.451] applied_action=[-0.067 -0.277  0.539  0.169 -0.362  0.466 -0.001 -0.120  0.286 -0.030
 -0.104  0.363] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[-0.029  0.574 -1.203  0.047  0.548 -1.232 -0.085  0.613 -1.268  0.079
  0.626 -1.238] commanded_leg_q=[-0.029  0.574 -1.203  0.047  0.548 -1.232 -0.085  0.613 -1.268  0.079
  0.626 -1.238] current_leg_q=[-0.020  0.634 -1.328  0.016  0.635 -1.342 -0.086  0.637 -1.343  0.086
  0.641 -1.333] leg_q_error=[-0.009 -0.060  0.124  0.031 -0.087  0.110  0.000 -0.024  0.075 -0.006
 -0.015  0.096] current_leg_dq=[-0.012 -1.139 -0.018 -0.143 -1.884 -0.315  0.081 -0.469 -0.281  0.585
 -0.419 -0.305] current_tau_est=[ 2.029 -1.905 22.332 -0.891 -3.711  6.306  8.436  0.074  5.879 -6.778
  0.940  5.310] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[-0.029  0.582 -1.212  0.050  0.548 -1.233 -0.086  0.614 -1.262  0.080
  0.628 -1.228] lowcmd_leg_q_hw=[-0.029  0.582 -1.212  0.050  0.548 -1.233 -0.086  0.614 -1.262  0.080
  0.628 -1.228] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[45.000 48.000 44.000 47.000]
2026-06-04 21:51:42,866 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[-0.000 -0.000 -0.000] safe_cmd=[0.000 0.000 0.000] valid=True inhibited=False reason= gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:43,049 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.020  0.001 -0.002] commands=[0.000 0.000 0.000] raw_action=[ 0.102 -0.167  0.616  0.097 -0.174  0.312 -0.082 -0.092  0.231 -0.070
 -0.106  0.482] clipped_action=[ 0.102 -0.167  0.616  0.097 -0.174  0.312 -0.082 -0.092  0.231 -0.070
 -0.106  0.482] startup_limited_action=[ 0.102 -0.167  0.616  0.097 -0.174  0.312 -0.082 -0.092  0.231 -0.070
 -0.106  0.482] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.102 -0.167  0.616  0.097 -0.174  0.312 -0.082 -0.092  0.231 -0.070
 -0.106  0.482] applied_action=[ 0.100 -0.145  0.611  0.101 -0.160  0.315 -0.066 -0.091  0.231 -0.090
 -0.110  0.447] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[ 0.001  0.616 -1.181  0.035  0.613 -1.281 -0.097  0.622 -1.285  0.069
  0.625 -1.211] commanded_leg_q=[ 0.001  0.616 -1.181  0.035  0.613 -1.281 -0.097  0.622 -1.285  0.069
  0.625 -1.211] current_leg_q=[-0.018  0.641 -1.325  0.017  0.643 -1.341 -0.087  0.643 -1.339  0.089
  0.643 -1.319] leg_q_error=[ 0.019 -0.025  0.144  0.018 -0.030  0.061 -0.010 -0.020  0.054 -0.020
 -0.018  0.108] current_leg_dq=[ 0.000  0.279 -0.809 -0.271  0.240 -0.061  0.349 -0.089 -0.192 -0.186
 -0.085 -0.542] current_tau_est=[ 2.301  0.421  7.112 -0.025  0.346  4.836  6.481  2.424  5.263 -7.941
  2.548  5.405] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[ 2.083e-04  6.086e-01 -1.191e+00  3.412e-02  6.142e-01 -1.312e+00
 -9.737e-02  6.235e-01 -1.289e+00  7.413e-02  6.284e-01 -1.209e+00] lowcmd_leg_q_hw=[ 2.083e-04  6.086e-01 -1.191e+00  3.412e-02  6.142e-01 -1.312e+00
 -9.737e-02  6.235e-01 -1.289e+00  7.413e-02  6.284e-01 -1.209e+00] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[56.000 45.000 42.000 48.000]
2026-06-04 21:51:43,385 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:43,589 INFO Policy diag | handover=1.000 est_lin_vel=[ 0.011 -0.001  0.001] commands=[0.000 0.000 0.000] raw_action=[ 0.124 -0.181  0.560  0.112 -0.161  0.275 -0.070 -0.099  0.197 -0.085
 -0.086  0.466] clipped_action=[ 0.124 -0.181  0.560  0.112 -0.161  0.275 -0.070 -0.099  0.197 -0.085
 -0.086  0.466] startup_limited_action=[ 0.124 -0.181  0.560  0.112 -0.161  0.275 -0.070 -0.099  0.197 -0.085
 -0.086  0.466] startup_limiter_active=False startup_abs_clipped=False startup_delta_clipped=False timed_action=[ 0.124 -0.181  0.560  0.112 -0.161  0.275 -0.070 -0.099  0.197 -0.085
 -0.086  0.466] applied_action=[ 0.141 -0.154  0.498  0.074 -0.161  0.284 -0.070 -0.091  0.207 -0.053
 -0.106  0.508] startup_kick=[0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000] target_leg_q=[ 0.009  0.613 -1.217  0.030  0.613 -1.290 -0.098  0.622 -1.293  0.075
  0.626 -1.192] commanded_leg_q=[ 0.009  0.613 -1.217  0.030  0.613 -1.290 -0.098  0.622 -1.293  0.075
  0.626 -1.192] current_leg_q=[-0.016  0.641 -1.320  0.019  0.644 -1.338 -0.086  0.645 -1.337  0.090
  0.647 -1.316] leg_q_error=[ 0.025 -0.028  0.103  0.012 -0.031  0.047 -0.012 -0.023  0.045 -0.014
 -0.021  0.125] current_leg_dq=[ 0.202 -0.008  0.111  0.012 -0.275  0.107  0.132 -0.554  0.055  0.019
 -0.597  1.322] current_tau_est=[ 0.767 -4.230 19.772 -2.251 -4.181  7.255  9.722 -0.445  7.871 -9.574
  0.445 13.229] motor_mode=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.] lowcmd_leg_q_policy=[ 0.004  0.623 -1.211  0.034  0.615 -1.297 -0.089  0.626 -1.296  0.067
  0.629 -1.186] lowcmd_leg_q_hw=[ 0.004  0.623 -1.211  0.034  0.615 -1.297 -0.089  0.626 -1.296  0.067
  0.629 -1.186] lowcmd_kp=[200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000 200.000
 200.000 200.000 200.000] lowcmd_kd=[10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000 10.000
 10.000 10.000] arm_target=[0.000 0.000 0.000 0.000 0.000 0.000] arm_current=[0.000 0.000 0.000 0.000 0.000 0.000] arm_smoothed_cmd=[0.000 0.500 0.300 0.000 0.000 0.000] sim2sim_delay=1 hold_prob=0.050 foot_force=[60.000 41.000 38.000 56.000]
2026-06-04 21:51:43,905 INFO Joystick base command | dry_run=False raw_axes={'lx': -0.0, 'ly': 0.0, 'rx': -0.0, 'ry': -0.0} raw_cmd=[0.000 0.000 0.000] safe_cmd=[0.000 0.000 0.000] valid=False inhibited=False reason=wirelesscontroller_stale gate=BaseCommandGate(standup_done=True, policy_running=True, lowlevel_align_done=True, emergency_stop=False)
2026-06-04 21:51:44,089 INFO Emergency stop
