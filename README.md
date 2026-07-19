# gx-real：Go2-X5 真机部署仓库

`gx-real` 用于在 Unitree Go2 + ARX5 X5 + Jetson 上部署 `Go2-X5-lab`
导出的 12 维腿部策略。当前生产路径分为互斥的 Flat 和 Rough 两类；本分支重点是
Rough 策略、固定机械臂输入和 LiDAR elevation map。

> 安全状态：仓库中的 Flat/Rough artifact manifest 仍是 `UNRELEASED`，Rough
> perception contract 仍是 `UNVERIFIED`。真实 LowCmd/CAN 输出会被发布门禁拒绝。
> 只有完成目标硬件校准、证据采集和 manifest 发布后才允许上机；不要通过改字符串
> 或关闭校验绕过门禁。

真机操作者从 [上机使用指南](docs/上机使用指南.md) 开始；LiDAR 安装、外参和
验收按 [LiDAR 校准指南](docs/lidar_calibration.md) 执行；开发者继续阅读
[开发与仓库指南](docs/developer_onboarding.md)。

## 1. 当前系统边界

当前 actor 是 `260D observation -> 12D leg action`，不是 18 维 whole-body
actor。Go2 与 X5 分属两个互斥 writer：

```text
Go2 lowstate + joystick + rough GridMap
                 |
                 v
       WBC / 260D observation
                 |
          policy.onnx (12D)
                 |
                 v
            Go2 lowcmd

X5 CAN <-> x5_fixed_hold node
             |       |
             |       +-> fault -> /safety/estop -> WBC stop
             +-> /arm/state, /arm/target_state (physical safety only)
```

必须同时满足以下不变量：

- WBC 只写 Go2；`x5_fixed_hold` 只写 X5 CAN。
- X5 物理目标固定为 `[0.0, 0.3, 0.5, 0.0, 0.0, 0.0]`，夹爪目标为 `0.0`。
- actor 中的机械臂位置、速度、历史动作、命令和夹爪输入都由固定合同生成；真实
  `/arm/state` 数值不会复制进 actor。
- 实测 arm topic 仍用于物理 fixed-hold freshness/tracking 安全门。它们失效会停腿，
  但不会改变固定 actor 输入。
- Rough actor 只接受 `GridMap/elevation` 产生的 187 维 live scan；zero、last-valid、
  Unitree HeightMap 和直接点云都不能获得运动许可。
- 同时只能存在一个 Go2 LowCmd writer 和一个 X5 CAN writer。

## 2. 260 维策略观测合同

当前 Rough 发布包位于 `policies/rough/current/`，任务为
`RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnlyHard-v0`。部署从发布包内的
`env.yaml`、`height_scan_contract.yaml` 和精确 `grid_xy` 读取合同，而不是凭
手写维度猜测。

| Slice | 维度 | 内容 | 真机来源 |
| --- | ---: | --- | --- |
| `[0:3]` | 3 | base linear velocity | Go2 状态估计 |
| `[3:6]` | 3 | base angular velocity | Go2 IMU |
| `[6:9]` | 3 | projected gravity | Go2 IMU quaternion |
| `[9:12]` | 3 | velocity command | 固定命令或 Go2 手柄 |
| `[12:30]` | 18 | joint position relative | 12 腿实测 + 6 臂固定训练姿态 |
| `[30:48]` | 18 | joint velocity | 12 腿实测 + 6 臂零速度 |
| `[48:66]` | 18 | previous action | 12 腿历史动作 + 6 个零 |
| `[66:253]` | 187 | height scan | Rough GridMap；Flat 为精确零常量 |
| `[253:259]` | 6 | arm joint command | 固定 `[0, 0.3, 0.5, 0, 0, 0]` |
| `[259:260]` | 1 | gripper command | 固定 `0.0` |

arm position 在拼接前使用固定训练姿态，再减去训练 default pose，因此 actor
看到的最后 6 个 joint-position-relative 值是精确零。`env.yaml` 若不再满足固定
default pose、六组 `[0, 0]` command offset、zero action padding 或固定 gripper，
离线检查和真机节点都会拒绝加载。

## 3. Rough 地图与坐标系

生产拓扑固定为：

```text
LiDAR PointCloud2 + IMU
  -> deskew / localization
  -> self-filter
  -> elevation mapper
  -> grid_map_msgs/msg/GridMap, layer=elevation
  -> gx-real 17 x 11 sampler
  -> observation[66:253]
```

坐标合同：

- map frame 默认为 `odom`；`/localization/pose` 必须是同一 frame 下的
  `base_link` 原点姿态，不是 LiDAR pose。
- `base_link` 使用 x 向前、y 向左、z 向上；训练扫描网格是
  `base_yaw_aligned`，只使用 base yaw，不使用 roll/pitch 旋转采样平面。
- 发布包保存的 187 个 `grid_xy` 原顺序就是 actor flatten 顺序：17 个 x 点
  先变化，11 个 y 行后变化。
- 对每个 base 网格点 `(x_b, y_b)`，查询世界坐标为：

  ```text
  x_m = x_robot + cos(yaw) * x_b - sin(yaw) * y_b
  y_m = y_robot + sin(yaw) * x_b + cos(yaw) * y_b
  ```

- actor 高度值为
  `clip(base_z - elevation(x_m, y_m) - 0.5, -1.0, 1.0)`。
- GridMap 按 Eigen 列主序解码，矩阵轴为 `[x_buffer_index, y_buffer_index]`，
  并应用 `outer_start_index/inner_start_index` circular-buffer 偏移。
- map/pose frame 不同、四元数无效、source 过期、时间差超过 30 ms、关键区未知、
  几何/存储顺序不符或 fallback 均 fail-closed。
- 创建控制节点前还会把 Rough `env.yaml` 中的 0.1 m resolution、1.6×1.0 m size、
  `xy` ordering、yaw/world-down 射线和 observation clip/scale 与运行时 contract
  逐项交叉验证。

这些公式由 `tests/test_height_scan_core.py` 中的 circular-buffer/yaw 测试和保存的
Isaac Lab ray-hit reference 逐元素验证。实际 LiDAR 外参和 mapper 仍必须按目标硬件
校准，不能由单元测试替代。

## 4. 目录与职责

```text
config/
  deployments/{flat,rough}.yaml       部署类、固定臂和高度源合同
  artifact_manifest.yaml              Flat 发布清单
  go2_leg_safety_contract.yaml         最终腿命令安全合同
  hardware_writer_allowlist.yaml       硬件 writer 清单

policies/
  policy.onnx + env.yaml               Flat 发布包
  rough/current/                       Rough 模型、checkpoint、env、reference、manifest

real-wbc/modules/
  wbc_node_leg12_arm_passthrough.py    260D 拼接、12D 推理、Go2 控制与运行时门禁
  deployment_profile.py               Flat/Rough 互斥合同和固定 arm 训练契约
  height_scan_core.py                  纯 NumPy GridMap/height scan 坐标转换
  height_scan_provider.py              ROS2 map/pose 订阅、时序与 frame 检查
  spacemouse_arm_node.py               legacy SpaceMouse 与生产 fixed-hold X5 owner
  artifact_manifest.py                发布资产、commit 和 perception gate
  final_command_safety.py              Go2 最终命令边界
  safety_state.py / safety_lease.py    锁存状态机与跨节点 heartbeat

real-wbc/scripts/
  run_wbc_{flat,rough}.py              部署类 Python 入口
  run_x5_fixed_hold_{flat,rough}.py    X5 fixed-hold Python 入口
  run_height_scan_monitor.py           无 actuator 的感知监视器

scripts/
  prepare_{flat,rough}_run.sh           构建、环境、writer、CAN、topic、sport-mode preflight
  run_leg12_{flat,rough}_real.sh        Go2 操作入口
  run_x5_fixed_hold_{flat,rough}.sh     X5 操作入口
  height_scan/                          发布包合同、reference 和 parity 工具
  policy/                               checkpoint actor 导出工具

unitree_ros2/                           Unitree ROS2/CycloneDDS 与消息
unitree_sdk2/                           SDK2、CRC、sport-mode 工具
arx5-sdk/                               X5 SDK、绑定、URDF 和库
tests/                                  纯离线合同/安全/地图回归测试
docs/                                   当前指南与历史审计材料
```

`real-wbc/modules/wbc_node.py`、`real-wbc/scripts/run_wbc.py`、SpaceMouse 动态控臂、
iPhone/MoCap 和 EEF trajectory 属于 legacy/研究路径，不是当前 Rough 生产入口。

## 5. 标准入口与启动顺序

构建和 preflight：

```bash
cd ~/gx-real
conda deactivate 2>/dev/null || true
scripts/prepare_rough_run.sh \
  --network-iface eth0 \
  --can-if can0 \
  --check-joystick-motion
```

preflight 通过后使用三个终端：

1. 启动 `scripts/run_x5_fixed_hold_rough.sh`。节点先在 damping/STANDBY，目标已固定。
2. 启动 `scripts/run_leg12_rough_real.sh`。入口先通过 MotionSwitcher 检查并释放 MCF，
   确认成功后才启动 WBC；WBC 收到第一帧 LowState 后立即进入当前姿态
   `Kp=0, Kd=3` 的 Passive。
3. WBC heartbeat 正常后发布 `/arm/fixed_hold/enable=true`，等待 X5 进入目标姿态和
   tracking gate；最后由操作者按 R1 执行内部 FixStand，再按 L2 验姿并直接进入策略。

FixStand 的最终腿姿态与策略 ready/action offset 完全一致，不依赖已关闭的 MCF 或
其他外部起身控制器；L2 不再插入第二段 1.5 秒姿态对齐。

完整命令、停止顺序、手柄按键和故障处理见
[上机使用指南](docs/上机使用指南.md)。旧的 `scripts/run_leg12_real.sh` 会主动退出，
因为未指定 Flat/Rough 的入口是不安全的。

## 6. 离线构建与验证

开发机可执行：

```bash
python3 -m py_compile \
  real-wbc/modules/deployment_profile.py \
  real-wbc/modules/height_scan_core.py \
  real-wbc/modules/height_scan_provider.py \
  real-wbc/modules/wbc_node_leg12_arm_passthrough.py \
  real-wbc/modules/spacemouse_arm_node.py \
  real-wbc/scripts/run_wbc_leg12.py

bash -n scripts/setup_env.sh scripts/prepare_real_run.sh \
  scripts/run_leg12_rough_real.sh scripts/run_x5_fixed_hold_rough.sh

pytest -q tests

python3 scripts/height_scan/check_policy_height_scan_contract.py \
  --policy-kind rough \
  --policy policies/rough/current/policy.onnx \
  --env-yaml policies/rough/current/env.yaml \
  --contract policies/rough/current/height_scan_contract.yaml

git diff --check
```

Jetson 的完整 C++/ROS2 构建由 `prepare_rough_run.sh` 执行。开发机缺少 ROS2、
`onnxruntime` 或目标架构库时，应记录为环境限制；不要把 import 失败误写成真机通过。

## 7. 替换 Rough 策略

替换模型必须作为完整不可拆分发布包处理：

- checkpoint、`agent.yaml`、`env.yaml`；
- `policy.pt`、`policy.onnx`、`export_metadata.json`；
- `height_scan_contract.yaml/.npz` 和仿真 reference；
- Torch 生成的 `policy_reference.npz`；
- `perception_contract.yaml` 与 artifact manifest；
- 每个文件的 SHA-256、训练 commit、Python/ONNX Runtime 和 SDK snapshot。

禁止只替换 `policy.onnx`。任何 observation slice、joint order、height grid、固定 arm
pose、action scale 或输出维度变化都必须先修改部署代码与测试，再生成新发布包。

## 8. 文档导航

- [上机使用指南](docs/上机使用指南.md)：唯一生产操作手册。
- [LiDAR 校准指南](docs/lidar_calibration.md)：安装、6DoF 外参、时间同步、map 验收。
- [开发与仓库指南](docs/developer_onboarding.md)：构建、修改点、验证和交付。
- [LiDAR/height-map 后端决策](docs/lidar_height_backend_decision_2026-07-16.md)：为什么生产选择 GridMap。
- [Rough 发布包说明](policies/rough/README.md)：模型资产和 release gate。
- [文档索引](docs/README.md)：当前文档与历史审计材料边界。
