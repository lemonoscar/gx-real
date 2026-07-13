# Policy startup jitter reward notes

## Log symptoms

The June 4 deployment log shows that jitter starts immediately after policy
rollout, even when the commanded base velocity is zero.

Key observations:

- At policy start, `commands=[0.000 0.000 0.000]`, but the policy outputs large
  leg actions immediately. The first raw action has calf terms near `0.98`,
  `0.85`, `0.51`, `0.65`; the startup limiter clips only the first step.
- Within the next second, commanded calf targets move far away from the live
  standing pose. Several `leg_q_error` terms reach about `0.10-0.17 rad`.
- Joint velocities and torque estimates spike during the handover window,
  especially on thigh/calf joints.
- Foot force readings fluctuate while the robot should be standing still.
- The current config already has `action_rate_l2`, `joint_acc_l2`,
  `joint_torques_l2`, `stand_still`, `feet_slide`, and contact penalties, but
  these do not specifically protect the policy handover / zero-command phase.

## Reward gaps to fill

### 1. Zero-command stance preservation

Add or strengthen rewards that are active when the velocity command norm is
below a small threshold, especially for the first rollout seconds.

Recommended terms:

- `zero_cmd_stance_tracking_l2`: penalize leg joint deviation from the runtime
  ready stance / default standing pose when command is near zero.
- `zero_cmd_action_l2`: penalize nonzero policy action when command is near zero.
- `zero_cmd_foot_contact`: reward all four feet staying in contact at zero
  command.
- `zero_cmd_xy_yaw_drift`: penalize base XY drift and yaw drift while command is
  zero.

Reason: the log shows large action output before any nonzero command is present.
The policy should learn that zero command means a quiet stance, not a gait.

### 2. Handover/startup-specific smoothness

General `action_rate_l2` is not enough because startup begins from a known
standing target and the first few policy actions matter most.

Recommended terms:

- `startup_action_l2`: penalize absolute action magnitude for the first
  `0.5-2.0 s` after rollout.
- `startup_action_delta_l2`: penalize action change from the previous deployed
  action, initialized to zero or to the handover target.
- `startup_action_jerk_l2`: penalize second difference
  `a_t - 2*a_{t-1} + a_{t-2}`.
- `startup_joint_target_delta_from_ready_l2`: penalize large target joint
  displacement from the live ready pose during the initial handover.

Reason: runtime already has a 3 s action limiter. Training should expose and
reward the same smooth handover behavior, otherwise the limiter hides a policy
that still wants to jump.

### 3. Joint tracking and actuator realism

The log shows target/current mismatch and high velocity/torque during startup.

Recommended terms:

- `joint_target_tracking_error_l2`: penalize `q_target - q_measured` for leg
  joints. This discourages targets that the real PD loop cannot track cleanly.
- Increase `joint_acc_l2` and `joint_torques_l2` modestly for calf/thigh joints.
- Add `joint_power_abs` or strengthen current `joint_power`, with per-joint
  weighting on calves.
- Add `joint_vel_l2` instead of leaving it null, again with thigh/calf emphasis.

Reason: the policy currently finds aggressive calf/thigh targets acceptable,
but the real robot responds with visible lag and oscillation.

### 4. Base motion smoothness

The current config penalizes base linear acceleration but does not appear to
directly penalize roll/pitch angular acceleration or orientation jerk.

Recommended terms:

- `base_ang_acc_xy_l2`: penalize roll/pitch angular acceleration.
- `projected_gravity_rate_l2`: penalize rapid changes in projected gravity.
- Strengthen `flat_orientation_l2` during zero command and startup.
- Keep `lin_vel_z_l2`, but consider a startup-specific version with higher
  weight.

Reason: startup jitter is primarily a stability problem, not a command-tracking
problem.

### 5. Contact stability at zero command

Current foot-force logs fluctuate during zero-command startup.

Recommended terms:

- `zero_cmd_feet_slide`: stronger foot slip penalty while command is zero.
- `zero_cmd_contact_force_variance`: penalize large imbalance or rapid changes
  in foot contact force.
- `stance_feet_air_penalty`: penalize feet leaving contact during zero command.

Reason: a quiet standing policy should not create a gait-like contact sequence.

### 6. Arm-conditioned base stability

The deployment consumes external arm state/target. Rewards for arm-conditioned
base robustness are currently disabled/null in the active reward section.

Recommended terms:

- Enable `arm_joint_pos_tracking_l2`, `arm_joint_vel_l2`, and `arm_action_rate_l2`
  when training with the arm present.
- Enable an arm-conditioned stability penalty such as
  `arm_pose_conditioned_base_stability`.
- Randomize arm pose around the real working range while enforcing base
  flatness and zero-command stance.

Reason: even if X5 is externally controlled, the locomotion policy observes arm
state and must be robust to arm mass/pose changes.

## Training setup changes that support the rewards

These are not rewards, but they are needed for the rewards to matter.

- Add reset states that start from the real live-ready pose, not only the nominal
  training default pose.
- Add a short zero-command phase at episode start before velocity commands.
- Randomize the policy handover pose within the measured real offsets.
- Randomize PD gains, motor strength, friction, payload/arm pose, action delay,
  observation delay, and action hold probability. Current deployment reports
  fixed action delay `(1, 1)`, hold probability `0.05`, and no action noise.
- Train with command ramps rather than instant command changes.

## Priority order

1. Add zero-command stance/action rewards.
2. Add startup action delta and action jerk penalties.
3. Add joint target tracking error and thigh/calf velocity penalties.
4. Add zero-command contact stability rewards.
5. Enable arm-conditioned base stability rewards.
6. Broaden startup/domain randomization after the above terms are stable.

The immediate target is not higher walking speed. The first target is a policy
that can accept `commands=[0, 0, 0]` and remain quiet for several seconds after
handover without runtime action clipping doing all the work.
