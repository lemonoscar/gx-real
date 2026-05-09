# gx-real 真机开发文档

这份文档面向第一次接触本仓库的人，目标是把 `real` 目录下分散的上机、网络、硬件和策略替换说明整理成一条完整开发路径。默认部署环境是机器狗机身上的 Jetson Orin NX 开发板，路径按 `~/gx-real` 书写。当前主线不是原始 UMI-on-Legs 的完整末端轨迹控制链，而是 `Go2 + X5/ARX5` 真机上的 `12D 腿部 policy + 6D 机械臂姿态直通 + 可选 SpaceMouse 遥操作` 部署链。

## 1. 当前系统做什么

当前仓库用于在真机上运行：

- 运行端：Go2 机身上的 Jetson Orin NX，预期架构是 `aarch64`，默认使用系统 `/usr/bin/python3`。
- Go2 腿部：读取低层状态，通过 `policy.onnx` 输出 12 维腿部动作。
- X5/ARX5 机械臂：默认保持启动参数指定的 6 维关节姿态，`A` 键可把机械臂交给 SpaceMouse Arm teleop，`X` 键回复位姿态。
- Go2 通信：ROS2 `lowstate/lowcmd`，并使用 Unitree DDS/ROS2 消息包。
- X5 通信：SocketCAN `can0` + `arx5_interface`。
- 控制流程：先关闭 Go2 `sport_mode`，再由 `R1` 起身，最后由 `L2` 进入低层 policy rollout。
- SpaceMouse：单独运行 teleop 节点，只发布 `/teleop/*` ROS2 topic；硬件下发仍然只有 WBC 主节点负责。

主入口链路：

```text
scripts/run_leg12_real.sh
  -> scripts/setup_env.sh
  -> real-wbc/scripts/run_wbc_leg12.py
  -> real-wbc/modules/wbc_node_leg12_arm_passthrough.py
  -> policies/policy.onnx + policies/env.yaml

scripts/run_spacemouse_teleop.sh      # 可选，另开终端
  -> real-wbc/scripts/run_teleop.py
  -> /teleop/mode + /teleop/base_cmd + /teleop/eef_delta + /teleop/gripper_cmd
```

一句话理解：这个仓库保留了 UMI-on-Legs 的真机通信、状态读取、手柄流程、起身流程和急停框架，但把原来的 18 维 whole-body actor 换成了当前的 12 维腿部 ONNX policy；机械臂目标由固定关节姿态、复位按键或 SpaceMouse 末端增量生成，最后仍统一由 WBC 主节点下发到 Go2 和 X5。

## 2. 目录结构

```text
real/
  README.md                         # 本文档，新人入口

  doc/
    上机使用指南.md                  # 真机操作细节和故障处理
    260维输入设计.md                 # policy 观测拼接契约
    小替换代码清单.md                # 从 UMI WBC 到 leg12 版本的改造说明
    替换思路.md                      # 替换网络的路线选择
    README.md                       # 文档导航

  scripts/
    setup_env.sh                    # 设置 GX_REAL_ROOT、PYTHONPATH、ROS2、SDK 动态库路径
    check_env.sh                    # 调用 check_env.py 做环境预检
    check_env.py                    # 检查 policy、env.yaml、CRC、ROS2 type support 和 Python import
    setup_arx_can.sh                # 配置 ARX5 SocketCAN can0
    disable_sports_mode_go2.sh      # 编译/调用 Unitree SDK 工具关闭 sport mode
    run_leg12_real.sh               # 当前推荐启动入口
    run_spacemouse_teleop.sh        # 可选 SpaceMouse teleop 发布节点
    run_arm_spacemouse_test.sh      # 只测 X5 + SpaceMouse，不启动 Go2/policy

  policies/
    policy.onnx                     # 当前真机推理模型
    policy.pt                       # 训练侧导出的 PyTorch 模型或备份
    env.yaml                        # 训练环境导出的部署配置

  real-wbc/
    modules/
      wbc_node_leg12_arm_passthrough.py
      common.py
      velocity_estimator.py
      spacemouse_shared_memory.py
      shared_memory/
    scripts/
      run_wbc_leg12.py
      run_teleop.py
    ros2/
      robot_state/                  # EEF 历史消息和当前 Teleop ROS2 消息
    docs/                           # 硬件、网络、开发环境细分说明

  arx5-sdk/                         # X5/ARX5 机械臂 SDK 和 Python 绑定
  unitree_sdk2/                     # Unitree SDK2、CRC 模块、关闭 sport mode 工具
  unitree_ros2/                     # Unitree ROS2/CycloneDDS 消息工作区
  logs/                             # 每次运行的日志目录
```

当前优先维护的是 `scripts/run_leg12_real.sh` 到 `wbc_node_leg12_arm_passthrough.py` 这条链，以及可选的 `scripts/run_spacemouse_teleop.sh` 到 `/teleop/*` 这条遥操作输入链。`real-wbc/scripts/run_wbc.py`、`real-wbc/modules/wbc_node.py`、EEF trajectory、iPhone/MoCap 等内容主要属于原 UMI-on-Legs 历史链路或后续扩展。

## 3. 硬件和外部依赖

必需硬件：

- Unitree Go2。
- Go2 机身 Jetson Orin NX 开发板，作为默认部署主机。
- X5/ARX5 机械臂。
- USB-CAN 转接器，当前默认接口名是 `can0`。
- Go2 网络连接，通常是 `192.168.123.xxx` 网段。
- X5 供电链路，建议通过 DC 降压模块输出稳定 24V。
- Go2 手柄，用于 `R1/L2/L1/R2/A/X/Y` 操作。

可选或历史外设：

- iPhone/MoCap：原任务空间位姿估计链路需要，当前默认 `--pose_estimator none`。
- GoPro、采集卡、fin-ray gripper：原 UMI 数据采集链路需要，当前 leg12 行走调试不依赖。
- 3Dconnexion SpaceMouse Wireless：可用于机械臂末端和底盘 teleop；不是走路 policy 的必需项。

软件环境默认假设机器人端路径为：

```bash
~/gx-real
```

核心依赖：

- Jetson Orin NX 上的 Ubuntu + ROS2 Foxy 或 Humble，架构应为 `aarch64`。
- `/usr/bin/python3` 下可 import 的 `onnxruntime`。不要只装到 conda 环境里。
- 本仓库内的 `unitree_ros2/cyclonedds_ws/install`。
- 本仓库内的 `unitree_sdk2/python/crc_module.so`。
- `arx5_interface` Python 接口。
- 可选 SpaceMouse 依赖：系统包 `spacenavd`、`libspnav-dev`，Python 包 `spnav`、`atomics`。

Jetson Orin NX 上机基线检查：

```bash
uname -m
cat /proc/device-tree/model 2>/dev/null || true
cat /etc/nv_tegra_release 2>/dev/null || true
which python3
/usr/bin/python3 --version
ls /opt/ros/foxy/setup.bash /opt/ros/humble/setup.bash 2>/dev/null
```

预期重点：

- `uname -m` 是 `aarch64`。如果是 `x86_64`，说明你在开发电脑上，不是狗身上的 Jetson。
- `setup_env.sh` 会根据 `uname -m` 自动优先使用 `arx5-sdk/lib/aarch64`。
- 上机运行默认用 `/usr/bin/python3`。如果 `python3` 指到 conda，不要直接用它跑真机主节点。
- ROS2 版本由机器系统决定；本仓库脚本会优先 source Foxy，找不到时再 source Humble。

## 4. 第一次部署

先退出 conda，避免 Python/ROS2 包路径冲突：

```bash
conda deactivate
cd ~/gx-real
```

确认你在 Jetson Orin NX 机器人端，而不是开发电脑：

```bash
uname -m
cat /proc/device-tree/model 2>/dev/null || true
```

如果 `uname -m` 不是 `aarch64`，不要把这个 shell 当作上机环境判断依据。

确认 policy 文件存在：

```bash
ls policies/policy.onnx
ls policies/env.yaml
```

确认系统 Python 能加载核心推理依赖：

```bash
/usr/bin/python3 -c "import onnxruntime; print('onnxruntime ok')"
```

如果失败，需要给 Jetson 的系统 Python 安装匹配 `aarch64`/JetPack/Ubuntu 的 ONNX Runtime 包。不要只在 conda 里安装，否则 `scripts/run_leg12_real.sh` 仍然会失败。

安装 ARX5 Python 接口：

```bash
cd ~/gx-real/arx5-sdk
source /opt/ros/foxy/setup.bash
/usr/bin/python3 -m pip install --user --no-build-isolation .
```

编译 Unitree ROS2 消息包：

```bash
cd ~/gx-real/unitree_ros2/cyclonedds_ws
source /opt/ros/foxy/setup.bash
colcon build
```

编译本仓库的 `robot_state` 消息包。SpaceMouse teleop 新增的 `/teleop/*` 消息也在这里生成：

```bash
cd ~/gx-real/real-wbc/ros2
source /opt/ros/foxy/setup.bash
colcon build --packages-select robot_state
```

如果使用 Humble，把 `/opt/ros/foxy/setup.bash` 换成对应的 Humble 环境。当前脚本会优先加载 Foxy，找不到时再加载 Humble。

如果要使用 SpaceMouse：

```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
/usr/bin/python3 -m pip install --user atomics
/usr/bin/python3 -m pip install --user https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
```

不要安装 PyPI 默认的 `spnav==0.9`。它在 Jetson Python3/aarch64 上可能报 `undefined symbol: PyCObject_AsVoidPtr`，需要先卸载后再安装上面的 GitHub 固定版本。

## 5. 每次上机前检查

进入 Jetson 上的仓库并加载环境：

```bash
conda deactivate
cd ~/gx-real
git pull
source scripts/setup_env.sh
```

基础环境预检：

```bash
scripts/check_env.sh
```

通过时应看到：

```text
[gx-real] python imports OK
```

如果要使用 SpaceMouse，再做可选依赖检查：

```bash
scripts/check_env.sh --spacemouse
```

这会额外检查 `spacenavd`、`libspnav`、`spnav`、`atomics` 和 Teleop ROS2 type support。通过只说明依赖存在，不代表接收器已经插好或 daemon 已经收到设备事件。

确认当前 shell 真的在 Jetson 上：

```bash
uname -m
which python3
echo "${GX_REAL_PYTHON_BIN}"
```

上机时 `uname -m` 应该是 `aarch64`，`GX_REAL_PYTHON_BIN` 应该是 `/usr/bin/python3`。

检查 Go2 网络和 ROS2 topic：

```bash
ip a
ip route
ros2 topic list
ros2 topic echo /lowstate --once
ros2 topic echo /wirelesscontroller --once
ros2 topic echo lf/sportmodestate --once
```

如果 topic 没有数据，先修网络和 CycloneDDS，不要继续进入低层 rollout。

检查 `can0`：

```bash
ip -details link show can0
```

如果没有 `can0`，或状态不是 `UP`：

```bash
scripts/setup_arx_can.sh
ip -details link show can0
```

关闭 Go2 `sport_mode`。这里的 `eth0` 要换成 Jetson 上实际连接 Go2 的网卡，可先用 `ip a` 找到 `192.168.123.xxx` 所在接口：

```bash
scripts/disable_sports_mode_go2.sh eth0
```

`sport_mode` 没关掉时不要进入低层 rollout，因为 Go2 原厂高层控制和低层 `lowcmd` 会抢控制权。

## 6. 标准启动流程

典型启动命令：

```bash
cd ~/gx-real
scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --cmd-vx 0.5 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0 \
  --arm-reset-pose 0.0 0.5 0.3 0.0 0.0 0.0
```

第一次验证建议把 `--cmd-vx` 降到 `0.0` 或 `0.1`，先确认低层接管、关节顺序和机械臂保持姿态正常，再逐步提高速度。

如果要启用 SpaceMouse，另开一个 Jetson 终端：

```bash
cd ~/gx-real
source scripts/setup_env.sh
scripts/run_spacemouse_teleop.sh --initial-mode arm --deadzone 0.3 --max-value 500
```

SpaceMouse 节点只发布 `/teleop/*`，不直接写 `lowcmd` 或 `can0`。不要同时运行 `arx5-sdk/python/examples/spacemouse_teleop.py`，那个示例会直接控制 ARX5，和本仓库 WBC 主节点抢 `can0`。

如果只想单独测试机械臂和 SpaceMouse，不启动机器狗、不启动 policy：

```bash
cd ~/gx-real
source scripts/setup_env.sh
scripts/setup_arx_can.sh              # can0 已经 UP 时可跳过
scripts/run_arm_spacemouse_test.sh    # 默认 X5_umi can0
```

这个命令会直接调用 ARX5 SDK 的 Cartesian SpaceMouse 示例，只控制 X5，不订阅 Go2，不发布 `lowcmd`，也不会运行 ONNX policy。它会拒绝在 `run_leg12_real.sh` 或 `run_wbc*.py` 已经运行时启动，避免两个进程同时抢 `can0`。如果需要显式指定模型和 CAN 接口：

```bash
scripts/run_arm_spacemouse_test.sh X5_umi can0
```

单独机械臂测试的默认速度按正常遥操作设置：末端平移 `0.10 m/s`，旋转 `0.30 rad/s`，夹爪 `0.03 m/s`。默认不再额外限制 home 附近工作空间，位置约束交给 ARX5 SDK 的 IK、关节限位和电流保护。首次测试仍然只轻推 SpaceMouse，不要长时间顶住一个方向。如果需要临时收窄，可以透传参数：

```bash
scripts/run_arm_spacemouse_test.sh X5_umi can0 --workspace-xyz 0.08 0.08 0.06 --workspace-rpy 0.25 0.25 0.25
```

看到：

```text
Deploy node ready
```

按键顺序：

1. 按 `R1`：启动 internal 起身。如果当前姿态已经接近站姿，程序会尽量跳过预蹲阶段，并用当前站姿校准本次运行的 policy ready/action/obs offset。
2. 等狗稳定站住。
3. 按 `L2`：启动低层对齐，然后进入 RL rollout。

手柄按键：

- `R1`：启动 internal 起身。
- `L2`：起身完成后启动低层对齐和 policy；rollout 中再次按下会恢复配置的移动命令。
- `A`：把机械臂控制权交给 SpaceMouse，切到 Arm teleop 并用当前机械臂姿态重置 EEF anchor。
- `X`：机械臂回 `--arm-reset-pose`。
- `Y`：底盘 command 平滑切到 `0 0 0`，policy 保持运行。
- `R2`：停止 policy。
- `L1`：紧急停止并退出。
- `B`：切换 SpaceMouse teleop 的 Arm/Base 模式，属于扩展功能。

## 7. 控制架构

运行时主要数据流：

```text
Go2 LowState
  -> lowlevel_state_cb
  -> 260D obs
  -> ONNX policy
  -> 12D leg action
  -> action scale + offset
  -> 12D leg target

arm_pose / SpaceMouse teleop arm target
  -> 6D arm target

[12D leg target, 6D arm target]
  -> set_motor_position(...)
  -> Go2 LowCmd + ARX5 joint command
```

关键文件：

- `real-wbc/scripts/run_wbc_leg12.py`：解析启动参数、配置日志、创建 ROS2 节点。
- `real-wbc/modules/wbc_node_leg12_arm_passthrough.py`：当前主控制节点。
- `real-wbc/modules/common.py`：Go2 腿部关节顺序和 reorder 工具。
- `real-wbc/modules/velocity_estimator.py`：复用 UMI 风格的 base linear velocity 估计。
- `scripts/setup_env.sh`：环境变量、ROS2、PYTHONPATH 和动态库路径。
- `scripts/check_env.py`：检查部署前最容易错的依赖。

当前主节点保留的真机功能：

- Go2 `LowState` 读取。
- Go2 `LowCmd` 下发和 CRC。
- X5 `get_state()` / `set_joint_cmd()`。
- 手柄启动、停止、急停。
- internal 起身和低层对齐。
- policy 日志和运行日志落盘。

## 8. Policy 契约

当前部署模型来自：

```text
policies/policy.onnx
policies/env.yaml
```

`env.yaml` 不是普通运行参数文件，而是训练环境导出的完整配置。部署代码会读取：

- `joint_names`。
- `dog_joint_names`。
- `arm_joint_names`。
- 训练初始关节位置。
- 观测缩放。
- 动作缩放和动作 clip。
- 仿真 `dt/render_interval`，用于推导 policy 频率。
- actuator stiffness/damping，用于日志和对齐训练配置。

当前观测拼接顺序：

```text
260 = 3 + 3 + 3 + 3 + 18 + 18 + 18 + 187 + 6 + 1
```

对应：

| 范围 | 名称 | 来源 |
|---:|---|---|
| `[0:3)` | `lin_vel` | IMU、足端接触、关节状态估计 |
| `[3:6)` | `ang_vel` | IMU gyroscope，乘训练缩放 |
| `[6:9)` | `gravity_vec` | IMU quaternion 投影重力 |
| `[9:12)` | `commands` | `--cmd-vx/--cmd-vy/--cmd-yaw`，启动后从 0 ramp 到目标 |
| `[12:30)` | `dof_pos` | 12 腿 + 6 臂真实关节位置，减 offset |
| `[30:48)` | `dof_vel` | 12 腿 + 6 臂真实关节速度，乘 scale |
| `[48:66)` | `actions` | 上一拍动作历史 |
| `[66:253)` | `height_scan` | 当前为全 0 |
| `[253:259)` | `arm_joint_command` | 当前机械臂目标姿态 |
| `[259:260)` | `gripper_command` | 固定 gripper command |

输出契约：

- ONNX 输出必须是 12 维腿部 action。
- action 先按 `env.yaml` 中的 scale 和 offset 映射到腿部目标关节位置。
- 机械臂 6 维目标不由 ONNX 输出，而是由 `--arm_pose`、`X` 复位按键或 SpaceMouse teleop 缓存提供；`A` 用于进入 SpaceMouse Arm teleop。
- 最终下发仍然是 18 维：

```text
full_action[0:12]  = leg_policy_target
full_action[12:18] = arm_passthrough_pose
```

关节顺序要特别注意。硬件接口默认顺序是：

```text
FR, FL, RR, RL
```

部署代码会根据 `env.yaml` 中的 `dog_joint_names` 建立策略顺序和接口顺序之间的映射。只要换 policy 或重新导出 `env.yaml`，必须重新检查 `dog_joint_names`、`joint_names[:12]` 和动作维度。

## 9. 修改代码时怎么入手

如果只是换一版腿部 policy：

1. 替换 `policies/policy.onnx`。
2. 同步替换对应训练导出的 `policies/env.yaml`。
3. 确认 ONNX 输入维度和输出维度，输出必须是 12。
4. 执行 `scripts/check_env.sh`。
5. 先用 `--cmd-vx 0.0` 或 `pose_test` 做低风险验证。

如果要改观测：

- 主要改 `wbc_node_leg12_arm_passthrough.py` 的 `lowlevel_state_cb(...)`。
- 同步更新 `doc/260维输入设计.md`。
- 保持 `obs.shape[0] == ONNX input_dim`。
- 不要只改代码不改 `env.yaml`，否则训练和部署契约会漂。

如果要改动作映射：

- 主要看 `init_policy(...)` 和 `map_leg_action_to_targets(...)`。
- 重点核对 `leg_action_scale`、`leg_action_offset`、`real_deploy_leg_offset`。
- 真机上先用静态或低速命令验证 `current_leg_q` 是否跟随 `lowcmd_leg_q_policy`。

如果要改 SpaceMouse teleop：

- 输入读取在 `real-wbc/modules/spacemouse_shared_memory.py`。
- ROS2 发布在 `real-wbc/scripts/run_teleop.py`。
- WBC 消费在 `wbc_node_leg12_arm_passthrough.py` 的 `teleop_*` 回调。
- ROS2 消息在 `real-wbc/ros2/robot_state/msg/Teleop*.msg`，改消息后必须重新 `colcon build --packages-select robot_state`。
- 保持 SpaceMouse 节点只发布 `/teleop/*`，不要让它直接写 Go2 `lowcmd` 或 ARX5 `can0`。

如果要恢复 UMI 原始任务空间链路：

- 参考 `real-wbc/modules/wbc_node.py` 和 `real-wbc/scripts/run_wbc.py`。
- 需要重新梳理 `EEFState/EEFTraj`、pose estimator、历史 trajectory teleop 和 whole-body actor。
- 不建议把这条链和当前 leg12 主链直接混在同一个节点里，先用独立入口验证。

## 10. 开发验证命令

Jetson 上请优先使用系统 Python 做检查：

```bash
/usr/bin/python3 -m py_compile \
  real-wbc/modules/wbc_node_leg12_arm_passthrough.py \
  real-wbc/scripts/run_wbc_leg12.py \
  real-wbc/scripts/run_teleop.py \
  scripts/check_env.py
```

Shell 脚本语法检查：

```bash
bash -n scripts/setup_env.sh
bash -n scripts/check_env.sh
bash -n scripts/setup_arx_can.sh
bash -n scripts/disable_sports_mode_go2.sh
bash -n scripts/run_leg12_real.sh
bash -n scripts/run_spacemouse_teleop.sh
bash -n scripts/run_arm_spacemouse_test.sh
```

格式/空白检查：

```bash
git diff --check
```

真机前环境检查：

```bash
source scripts/setup_env.sh
scripts/check_env.sh
scripts/check_env.sh --spacemouse    # 只在使用 SpaceMouse 时需要
uname -m
cat /proc/device-tree/model 2>/dev/null || true
ip -details link show can0
ros2 topic list
```

常用 ROS2 检查：

```bash
ros2 topic echo /lowstate
ros2 topic echo /wirelesscontroller
ros2 topic echo lf/sportmodestate
```

## 11. 日志和故障定位

每次运行会在 `logs/YYYYMMDD_HHMMSS/run.log` 下保存日志。重点看这些日志：

- `Runtime targets`：启动参数是否被正确读取，尤其是 `arm_hold_pose`、`arm_reset_pose`、`commanded_leg_kp/kd`。`button_arm_pose` 可能仍在旧启动参数里打印，但 `A` 键当前不再发送它。
- `Runtime leg offset update`：`R1` 后是否使用当前站姿做 runtime offset。
- `Policy diag`：policy 输出、clip、命令、低层目标、真实关节、误差和足端力。
- `Arm diag`：机械臂目标、当前状态和平滑命令。
- `Pose test diag`：只验证关节目标跟踪时使用。

常见问题：

- `check_env.sh` 失败在 `onnxruntime`：确认是在 Jetson 的 `/usr/bin/python3` 下安装，而不是 conda。用 `/usr/bin/python3 -c "import onnxruntime"` 复查。
- `robot_state.msg.Teleop*` import 失败：重新编译 `real-wbc/ros2`，然后重新 `source scripts/setup_env.sh`。
- `Could not import 'rosidl_typesupport_c' for package 'robot_state'`：通常是 Jetson 上 `robot_state` 的生成消息还没按最新代码 clean rebuild，或当前 shell 还在 conda `base`。先 `conda deactivate`，再删除 `real-wbc/ros2/build/robot_state` 和 `real-wbc/ros2/install/robot_state` 后重新 `colcon build --packages-select robot_state`。
- `check_env.sh --spacemouse` 失败在 `spnav` 或 `atomics`：安装 SpaceMouse Python 依赖；失败在 `spacenavd` 或 `libspnav`：安装/启动系统服务。
- `undefined symbol: PyCObject_AsVoidPtr`：卸载 PyPI 版 `spnav`，安装 README 中固定的 Cheng Chi fork。
- 单独机械臂测试失败在 `Error document empty` / `Failed to get chain from kdl tree`：ARX5 Python 扩展可能从 pip 安装目录加载，默认找不到仓库里的 URDF。更新到最新代码后，`scripts/run_arm_spacemouse_test.sh` 会显式把 `arx5-sdk/models` 传给示例。
- `Inverse kinematics failed: E_EXCEED_JOINT_LIMIT` 或 `Over current detected`：目标末端位姿太快或太远，已经触到 IK/关节/电流保护。立即松开 SpaceMouse 或 `Ctrl+C` 停止；必要时临时降低速度 `--pos-speed 0.03 --ori-speed 0.10`，或加 home 附近工作空间限制。
- SpaceMouse 没反应：确认接收器插在 Jetson 上，`spacenavd` 正在运行，且没有直接运行 ARX5 SDK 的 SpaceMouse 示例抢设备或抢 `can0`。
- 单独机械臂测试前不要运行 `run_leg12_real.sh`。`scripts/run_arm_spacemouse_test.sh` 会直接控制 X5，和 WBC 主节点互斥。
- 程序启动后机器人不动：看到 `Deploy node ready` 后还需要确认 sport mode 已关闭，按 `R1` 起身，等起身完成后按 `L2`。
- `sport_mode state has not been received`：ROS2 sport state 链路不可用，优先查网络和消息包。只有受控诊断时才加 `--allow-unknown-sport-mode`。
- `sport_mode is still active`：先运行 `scripts/disable_sports_mode_go2.sh eth0`。
- `None of the motors are initialized`：`can0` 可能存在，但 X5 电机没有反馈。检查电源、急停、CAN-H/CAN-L/GND、终端电阻、CANable 是否接到 X5 总线、波特率是否为 1Mbps，以及是否误用 `--disable-arm`。用 `ip -s -d link show can0` 看 SDK 运行后是否只有 TX 没有 RX。
- `commands` 非零但狗不动：看 `Policy diag` 里的 `lowcmd_kp`、`lowcmd_leg_q_policy`、`current_leg_q` 和 `leg_q_error`，优先排查低层控制权、sport mode、力矩限制、电池和关节顺序。

## 12. 安全规则

真机调试时遵守：

- Jetson 上只运行一个 WBC 主节点。不要同时启动 `run_wbc.py`、`run_wbc_leg12.py` 或任何直接控制 `can0` 的 ARX5 示例。
- 第一次跑新 policy 时，先 `--cmd-vx 0.0` 或 `0.1`。
- 机械臂运动空间内不要放手、线缆和工具。
- 改 X5 电源线前先断电，降压模块输出先用万用表确认约 24V。
- 不要在 sport mode 未关闭时强行跑 lowcmd。
- `max_leg_error` 长期大于 `0.08 rad` 时不要跑动态 policy。
- SpaceMouse 只作为 `/teleop/*` 输入源。真正下发 Go2/X5 的只能是 `wbc_node_leg12_arm_passthrough.py`。
- 按 `A` 后才把机械臂交给 SpaceMouse Arm teleop；按 `Y` 会清零底盘并回 Arm mode；`R2` 停 policy；`L1` 急停退出。
- 不要一开始就同时改 policy、obs、动作 scale、起身流程和 teleop，先保持变量可控。
- 手柄 `L1` 是第一急停手段，旁边必须有人能及时按下。

## 13. Demo 区

这里预留给后续放 demo。建议每个 demo 用同一套模板，便于复现实验。

### Demo 1: 待补充

- 日期：
- 硬件配置：
- policy 版本：
- commit：
- 启动命令：
- 操作步骤：
- 运行现象：
- 关键日志：
- 视频/图片：
- 结论：

### Demo 2: 待补充

- 日期：
- 硬件配置：
- policy 版本：
- commit：
- 启动命令：
- 操作步骤：
- 运行现象：
- 关键日志：
- 视频/图片：
- 结论：

## 14. 参考文档

- [上机使用指南](doc/上机使用指南.md)：最细的真机操作步骤。
- [260维输入设计](doc/260维输入设计.md)：当前 policy obs 契约。
- [小替换代码清单](doc/小替换代码清单.md)：leg12 + arm passthrough 的改造思路。
- [替换思路](doc/替换思路.md)：如果后续继续换网络，如何选切入层。
- [real-wbc 开发文档索引](real-wbc/docs/README.md)：硬件、网络、装配和开发环境细分说明。
- [网络与通信配置](real-wbc/docs/network.md)：Go2 网络、ROS2、sport mode 和 `can0`。
- [硬件装配说明](real-wbc/docs/assembly.md)：X5 供电、安装、USB-CAN 和外设。
- [3D 打印说明](real-wbc/docs/3d_printing.md)：安装板和历史外设打印件。
