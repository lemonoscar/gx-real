# gx-real 开发与仓库指南

本文面向维护 `rough-policy` 真机分支的开发者。生产系统是
`260D observation -> 12D Go2 leg action`，X5 由独立 fixed-hold 节点控制；它不是
一个 18D whole-body action 策略，也不是 SpaceMouse 在线控臂系统。

操作者应阅读 [上机使用指南](上机使用指南.md)，LiDAR/外参人员应阅读
[LiDAR 校准指南](lidar_calibration.md)。根目录 [README](../README.md) 是仓库总览，
本文重点解释代码边界、构建、修改和交付流程。

## 1. 系统合同

```text
Go2 LowState + WirelessController + GridMap(Rough only)
                         |
                         v
              WBC observation builder
                         |
                  policy.onnx
                         |
                         v
                 12D Go2 LowCmd

X5 CAN <-> x5_fixed_hold -> /arm/state, /arm/target_state
              |
              +-> /safety/estop
```

核心不变量：

- WBC 是唯一 Go2 writer，fixed-hold 是唯一 X5 writer；
- X5 物理目标固定为 `[0, 0.3, 0.5, 0, 0, 0]`；
- actor 的 arm observation 在进程整个生命周期固定，绝不读取实测关节值；
- 实测 `/arm/state` 和 `/arm/target_state` 仍参与独立的 freshness、producer、tracking
  和 lease 安全门；
- Rough `[66:253]` 必须是 live GridMap 187D scan，fallback 只能诊断，不能放行；
- artifact、训练合同、writer inventory 或最终腿命令安全不满足时 fail-closed。

这一区分很重要：**观测去耦不等于物理安全去耦**。修改 arm 代码时，既要证明 actor
不受真实 arm 数值影响，也要证明 arm feedback fault 仍会停止 Go2。

## 2. 260D observation

| Slice | 维度 | 语义 |
| --- | ---: | --- |
| `[0:3]` | 3 | base linear velocity |
| `[3:6]` | 3 | base angular velocity |
| `[6:9]` | 3 | projected gravity |
| `[9:12]` | 3 | base velocity command |
| `[12:30]` | 18 | 12 腿 relative position + 6 臂固定 relative position |
| `[30:48]` | 18 | 12 腿 velocity + 6 臂零 velocity |
| `[48:66]` | 18 | 12 腿 previous action + 6 个零 padding |
| `[66:253]` | 187 | Flat 精确零；Rough live height scan |
| `[253:259]` | 6 | 固定 arm command |
| `[259:260]` | 1 | 固定 gripper command |

Rough 发布包的 `env.yaml`、`height_scan_contract.yaml` 和
`height_scan_contract.npz` 是运行时权威合同。加载器会验证训练 joint order、固定 arm
default pose、六个 `[0,0]` arm command range、`use_default_offset`、零 action padding、
固定 gripper 以及 260/12 shape。不要只看模型文件名推断兼容性。

## 3. 仓库结构

```text
gx-real/
  README.md                          仓库总览、架构和主入口
  config/
    deployments/flat.yaml           Flat 签名部署配置
    deployments/rough.yaml          Rough 签名部署配置
    artifact_manifest.yaml          Flat artifact release 清单
    go2_leg_safety_contract.yaml     最终腿命令安全合同
    hardware_writer_allowlist.yaml   writer 源码清单和 legacy 边界

  policies/
    policy.onnx, policy.pt, env.yaml Flat 策略包
    rough/current/                   Rough 完整发布包、reference、合同、manifest

  real-wbc/modules/
    wbc_node_leg12_arm_passthrough.py  观测拼接、推理、Go2 状态机/输出
    deployment_profile.py             Flat/Rough 配置与训练固定臂合同
    arm_observation.py                 arm state cache 与固定观测值
    height_scan_core.py                纯 NumPy GridMap/坐标/采样逻辑
    height_scan_provider.py            ROS2 provider、时序/frame/coverage
    height_scan_policy_validation.py   ONNX height 敏感性检查
    spacemouse_arm_node.py             fixed-hold owner；另含隔离的 legacy 模式
    artifact_manifest.py               发布状态、hash、commit/perception 校验
    final_command_safety.py             最终 Go2 position/PD/状态合同
    safety_state.py                     锁存安全状态机
    safety_lease.py                     WBC <-> X5 heartbeat lease
    runtime_safety.py                   运行时 freshness/producer 检查
    can_owner_lock.py                   X5 CAN 进程互斥
    hardware_ownership.py               writer inventory 扫描
    base_command_provider.py            固定/手柄底盘命令

  real-wbc/scripts/
    run_wbc_flat.py, run_wbc_rough.py   签名 Python 入口
    run_wbc_leg12.py                     两类共享参数和启动实现
    run_x5_fixed_hold_{flat,rough}.py   签名 X5 入口
    run_height_scan_monitor.py           无 actuator 的感知监视器

  scripts/
    setup_env.sh                         ROS/Python/SDK 环境
    check_env.py, check_env.sh           import/type-support 检查
    prepare_{flat,rough}_run.sh           生产 preflight 外层入口
    prepare_real_run.sh                   共享构建和硬件检查
    run_leg12_{flat,rough}_real.sh        Go2 shell 入口
    run_x5_fixed_hold_{flat,rough}.sh     X5 shell 入口
    height_scan/                          合同/reference/parity 工具
    policy/                               RSL-RL actor 导出工具
    ci/                                   writer inventory 检查

  real-wbc/ros2/robot_state/             自定义 arm/teleop ROS2 消息
  unitree_ros2/                           Unitree ROS2 messages/CycloneDDS
  unitree_sdk2/                           SDK2、CRC、sport-mode C++ tool
  arx5-sdk/                               ARX5 SDK、库、模型和 Python binding
  tests/                                  不接硬件的合同与安全回归
  docs/                                   当前手册和历史审计证据
```

`real-wbc/modules/wbc_node.py`、`real-wbc/scripts/run_wbc.py`、SpaceMouse 动态控臂、
MoCap/iPhone/EEF trajectory 是 legacy/研究代码，不是 Rough 生产链。保留它们是为了
追溯，而不是允许从生产脚本调用。

## 4. 开发环境

### 4.1 开发机

地图、合同和大多数安全测试是纯 Python/NumPy，可在 x86 开发机运行。先检查：

```bash
cd /path/to/gx-real
git status --short --branch
python3 --version
```

如果默认 Python 没有 pytest/onnxruntime，可使用项目批准的隔离环境；不要改变真机脚本
默认的 `/usr/bin/python3`。测试不会创建硬件 writer。

### 4.2 Jetson

真机必须使用系统 Python，且先退出 Conda：

```bash
conda deactivate 2>/dev/null || true
cd ~/gx-real
export GX_REAL_NETWORK_IFACE=eth0
export GX_REAL_PERCEPTION_SETUP=/absolute/path/to/perception_ws/install/setup.bash
source scripts/setup_env.sh
scripts/check_env.sh --rough
```

`setup_env.sh` 会：

- source Foxy 或 Humble；
- source Unitree colcon workspace 的标准 `install/setup.bash`，不硬编码 Python 版本；
- source `robot_state`、可选 perception workspace；
- 选择对应架构 ARX5 shared library；
- 把本仓库模块、ARX5 binding 和本地 Unitree message install 加入路径；
- 固定使用 CycloneDDS/指定网络接口（若 package 可用）；
- 拒绝缺失 policy、CRC module 或 X5 URDF。

不要把 `unitree_sdk2/python` 整目录加入 `PYTHONPATH`；那里与 ROS2 生成的
`unitree_go` 同名，可能导入错误 module/type support。环境脚本会主动移除该路径，只按
绝对路径加载 `crc_module.so`。

## 5. 构建

生产 preflight 默认执行所有必需构建：

```bash
scripts/prepare_rough_run.sh \
  --network-iface eth0 \
  --can-if can0 \
  --check-joystick-motion
```

等价的构建部分为：

Unitree messages 还需要当前 ROS 发行版的 `rosidl_generator_dds_idl` 开发包；例如
Humble 的 Debian 包名为 `ros-humble-rosidl-generator-dds-idl`。应由机器镜像/依赖
清单预装，不要在上机当天临时改变系统环境。

preflight 的 colcon 构建会清理旧 CMake cache 并显式使用 `/usr/bin/python3`。若
`unitree_sdk2/build/CMakeCache.txt` 来自另一份移动过的源码路径，脚本会保留它并改用
忽略目录 `unitree_sdk2/build-gx-real/`；也可用 `GX_REAL_SDK_BUILD_DIR` 显式选择目录。

```bash
source /opt/ros/foxy/setup.bash   # Humble 机器改为对应路径

cd ~/gx-real/unitree_ros2/cyclonedds_ws
colcon build --packages-select unitree_api unitree_go unitree_hg

cd ~/gx-real/real-wbc/ros2
colcon build --packages-select robot_state

cd ~/gx-real/unitree_sdk2
cmake -S . -B build
cmake --build build --target disable_sports_mode_go2 -j "$(nproc)"
```

ARX5 Python binding/目标架构库由 `arx5-sdk` 提供；安装方式与 JetPack、ROS 和厂商 SDK
版本相关，应在目标 Jetson 的构建记录中固定版本。不要把开发机 x86 library 当作
aarch64 构建成功。

## 6. 离线验证

每次相关修改至少执行：

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

python3 scripts/height_scan/check_policy_height_scan_contract.py \
  --policy-kind flat \
  --policy policies/policy.onnx \
  --env-yaml policies/env.yaml

git diff --check
```

测试重点包括：

- actor 固定 arm observation 在收到不同 live state 后仍逐元素不变；
- live arm feedback fault 在 fixed observation 模式下仍触发安全门；
- fixed-hold 内部 fault 发布 `/safety/estop`；
- Flat/Rough 配置互斥和固定训练合同；
- 260D/12D ONNX shape、height sensitivity 和 reference parity；
- GridMap 列主序、x/y 轴、circular start index、yaw、frame、stamp、coverage；
- 保存的 Isaac Lab 187D reference 精确重放；
- writer inventory、manifest、lease、最终腿命令和启动脚本 fail-closed。

单元测试通过不等于目标硬件发布通过。Jetson 构建、ROS type support、LiDAR 外参、
latency、X5 CAN 和 Go2 低能量试验必须另行记录。

## 7. 修改工作流

### 7.1 替换策略

策略必须成套替换，至少包括：

- checkpoint、`agent.yaml`、`env.yaml`；
- `policy.pt`、`policy.onnx`、导出 metadata；
- height contract `.yaml/.npz`、仿真 height reference；
- Torch policy reference；
- perception contract 和 artifact manifest；
- 所有文件 SHA、训练 commit、导出环境和目标 runtime 版本。

先运行导出/契约检查，再更新部署。禁止只复制一个 `policy.onnx`，也禁止从 Flat 模型
改名制造 Rough 包。obs/action slice、joint order、default pose、grid、scale、clip、
action scale 中任一变化，都需要同步修改部署代码和测试。

### 7.2 修改机械臂逻辑

生产约束分两层：

1. `ArmObservationCache.get_fixed_initial()` 生成 actor 所需固定 pose、零速度/力矩、
   固定 target/gripper；
2. live arm callback 只更新 safety cache，验证 producer/freshness/tracking，不能覆盖
   actor cache。

任何改动都应增加两类测试：注入极端不同的 live arm 值后 actor slice 完全不变；让
live state/target stale、producer 错误或 tracking 超限后 motion 被拒绝。

物理 fixed-hold 目标来自 `config/deployments/{kind}.yaml`，训练合同来自对应
`env.yaml`。两者不一致必须拒绝加载，不能在 CLI 上临时补偿。

### 7.3 修改 Rough 地图

坐标和数值合同：

```text
x_m = x_r + cos(yaw) * x_b - sin(yaw) * y_b
y_m = y_r + sin(yaw) * x_b + cos(yaw) * y_b
v   = clip(base_z - elevation(x_m, y_m) - 0.5, -1, 1)
```

`grid_xy` 顺序来自发布包，x-fast/y-outer。GridMap 数据来自 Eigen 列主序，轴为
`[x_buffer_index, y_buffer_index]`，必须应用 outer/inner start index。不要用 NumPy
默认直觉重排，也不要忽略 circular buffer。

纯数学、message parsing 和 ROS subscription 分别留在 `height_scan_core.py` 与
`height_scan_provider.py`，这样坐标转换可离线测试。新增后端不能自动获得生产资格；
只有 `grid_map` 由当前 Rough deployment 允许，其他 source 保持 diagnostic-only。

### 7.4 修改安全或 writer

新增任何 LowCmd/CAN 写点时，必须先决定唯一 owner，并更新：

- `config/hardware_writer_allowlist.yaml`；
- `scripts/ci/check_hardware_writer_inventory.py` 与测试；
- writer lock、safety lease、manifest 和操作文档。

不要在 callback 异常中继续输出 last command。生产 fault 应进入锁存安全状态，撤销
motion output，并要求完整重启/复检。

### 7.5 修改 CLI 或启动脚本

共享实现位于 `run_wbc_leg12.py` 和 `prepare_real_run.sh`，Flat/Rough 外层入口负责签名
类别。新增参数时同步更新：

- Python parser 和 profile validation；
- `prepare_real_run.sh` 打印的精确 next-step 命令；
- `docs/上机使用指南.md`；
- entrypoint/source 回归测试。

默认值必须是生产安全值。目前 arm observation 默认且强制为 `fixed_initial`，物理
fixed-hold safety 必须显式开启。

## 8. Rough 发布流程

Rough 在发布前至少完成：

1. 模型/reference/合同离线验证；
2. 目标 Jetson 完整构建和 import/type-support 检查；
3. 按 [LiDAR 校准指南](lidar_calibration.md) 完成外参、mapper、latency、coverage 和
   held-out parity；
4. X5 fixed-hold freshness/tracking/lease 故障注入；
5. Go2 最终指令合同、架空 shadow、零速站立和分阶段低能量试验；
6. 独立复核全部证据。

manifest 使用两次提交解决自引用问题：

1. 提交所有代码、模型、合同和证据，得到 source commit；
2. 只修改一个目标 manifest，填入 source commit、真实版本/hash 和 `RELEASED`；
3. 创建 manifest-only release commit；
4. 运行时验证 release commit 的父提交正是 source commit，且差异只包含该 manifest；
5. 工作树必须干净，所有发布资产逐项校验 SHA。

若 release commit 同时改变代码、模型或第二个 manifest，运行时会拒绝。perception
contract 也必须先为真实 `VERIFIED`，只改 artifact manifest 不够。

## 9. Review 清单

提交前确认：

- 改动是否保持 actor arm 输入固定、live arm 仅安全用途；
- 是否保持 Flat/Rough 互斥和唯一 writer；
- map frame、yaw、flatten、column-major/circular semantics 是否有测试；
- 所有 invalid/stale/fallback 路径是否 fail-closed；
- 是否误改或伪造 `UNRELEASED/UNVERIFIED/UNSET`；
- 是否包含模型、合同和配置 SHA；
- Python、shell、pytest、flat/rough policy contract、C++/ROS 构建是否有记录；
- 当前文档是否与实际默认参数和启动顺序一致；
- 是否保留用户已有的无关工作树改动。

## 10. 常见误区

- **“机械臂锁住，所以直接读实测值没问题”**：不对。策略训练合同要求固定 actor
  输入；实测值只属于安全门。
- **“GridMap 是二维数组，reshape 就行”**：不对。必须处理 Eigen 列主序、x/y 轴和
  circular start index。
- **“TF 能连通就说明外参正确”**：不对。还要通过平地、墙、台阶、左右/前后和 yaw
  held-out 验收。
- **“fallback 能提高鲁棒性”**：对 Rough 生产不成立。fallback 地图与真实环境脱节，
  必须撤销 motion permit。
- **“pytest 通过就能上机”**：不对。发布门还要求目标架构构建、感知校准、真实硬件
  故障注入和独立复核。
- **“旧 SpaceMouse 入口仍在，所以可以并行启动”**：不对。legacy 文件只用于研究和
  追溯，与 fixed-hold 并发会违反 X5 唯一 writer。
