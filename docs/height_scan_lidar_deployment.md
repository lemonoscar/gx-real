# LiDAR Height-Scan Deployment Validation

This runbook is for the robot-side host. Do not run low-level policy control until the perception-only checks pass.

## 1. Perception-Only Tests

Allowed first:

```bash
cd ~/gx-real
source scripts/setup_env.sh
ros2 topic list | grep -Ei "lidar|unilidar|cloud|point|imu"
ros2 topic info /unilidar/cloud
ros2 topic hz /unilidar/cloud

scripts/run_height_scan_monitor.sh \
  --contract policies/height_scan_contract.yaml \
  --topic /unilidar/cloud \
  --base-frame base \
  --lidar-frame unilidar_lidar \
  --print-rate 5
```

Perception acceptance:

- `/unilidar/cloud` or the actual LiDAR topic exists.
- Topic type is `sensor_msgs/msg/PointCloud2`.
- Topic frequency is at least `4 Hz`.
- Monitor output reports `shape=187`.
- Flat-ground `valid_ratio >= 0.60`.
- `age_s` p95 is below `0.25 s`.
- Fallback ratio is below `10%`.
- No persistent flat-ground spikes larger than `0.30 m`.
- A `0.10 m` board or step produces a matching grid response with error below `0.05 m`.

## 2. Policy Zero-Speed Tests

Only after perception passes:

Use a policy/export pair whose `env.yaml` declares a real `height_scan`
observation function. Do not enable height scan with the default
`policies/policy.onnx` / `policies/env.yaml` bundle, which is the flat
zero-height-scan policy.

```bash
scripts/run_leg12_real.sh \
  --policy_path policies/rough/policy.onnx \
  --enable-height-scan \
  --height-scan-contract policies/rough/height_scan_contract.yaml \
  --height-scan-topic /unilidar/cloud \
  --cmd-vx 0.0 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --disable-arm \
  --standup-mode internal
```

Zero-speed acceptance:

- `30 s` stable standing.
- `height_scan_ok` remains true.
- Fallback ratio is below `10%`.
- `leg_q_error` is generally below `0.08 rad`.
- No continuous abnormal motor command.
- Emergency stop works.

## 3. Low-Speed Flat Tests

Only after zero-speed policy tests pass:

```bash
scripts/run_leg12_real.sh \
  --policy_path policies/rough/policy.onnx \
  --enable-height-scan \
  --height-scan-contract policies/rough/height_scan_contract.yaml \
  --height-scan-topic /unilidar/cloud \
  --cmd-vx 0.05 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --disable-arm \
  --standup-mode internal
```

Low-speed flat acceptance:

- `30 s` stable low-speed walking.
- No obvious shaking, limping, or falling.
- `age_s` p95 is below `0.25 s`.
- Fallback ratio is below `10%`.

## 4. Rough Terrain Tests

Only after low-speed flat tests pass, repeat the same opt-in height-scan policy command at conservative speeds on the intended rough terrain. Stop immediately if `height_scan_ok` drops persistently, fallback ratio exceeds `10%`, terrain spikes exceed the perception thresholds, or the base shows unstable motion.
