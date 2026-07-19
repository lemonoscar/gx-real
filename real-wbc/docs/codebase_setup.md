# 开发环境与代码结构

这份文档说明 `gx-real/real-wbc` 的开发入口和运行依赖。实际真机操作步骤请以 [上机使用指南](../../doc/上机使用指南.md) 为准。

## 1. 当前部署目标

当前仓库面向 `Go2 + X5/ARX5` 真机部署：

- Go2 腿部：12 维 RL policy 输出目标关节位置。
- X5 机械臂：默认保持指定关节姿态，运行中可通过手柄切换目标。
- Go2 通信：ROS2 `lowstate/lowcmd`。
- X5 通信：SocketCAN `can0` + `arx5_interface`。
- policy 文件：`policies/policy.onnx`。
- policy 配置：`policies/env.yaml`。

主入口链路：

```text
scripts/run_leg12_real.sh
  -> scripts/setup_env.sh
  -> real-wbc/scripts/run_wbc_leg12.py
  -> real-wbc/modules/wbc_node_leg12_arm_passthrough.py
```

## 2. 目录说明

```text
gx-real/
  scripts/
    setup_env.sh                 # 设置 PYTHONPATH、ROS2、SDK 动态库路径
    check_env.sh                 # 环境预检
    setup_arx_can.sh             # 配置 ARX5 SocketCAN
    disable_sports_mode_go2.sh   # 验证并释放 Go2 MCF
    run_leg12_real.sh            # 当前推荐启动入口

  policies/
    policy.onnx                  # 真机推理模型，本地文件，默认不进 Git
    env.yaml                     # policy 导出配置，会被部署代码读取

  real-wbc/
    scripts/run_wbc_leg12.py
    modules/wbc_node_leg12_arm_passthrough.py
    modules/common.py
    modules/velocity_estimator.py

  arx5-sdk/
    python/                      # arx5_interface Python 绑定
    models/X5_umi.urdf           # 当前 X5 模型

  unitree_ros2/
    cyclonedds_ws/               # Unitree ROS2 消息和 CycloneDDS 工作区

  unitree_sdk2/
    python/crc_module.so         # lowcmd CRC 计算模块
    build/disable_sports_mode_go2
```

## 3. 机器人端环境

推荐在机器人端使用仓库根目录：

```bash
cd ~/gx-real
source scripts/setup_env.sh
scripts/check_env.sh
```

`setup_env.sh` 会做这些事：

- 设置 `GX_REAL_ROOT`。
- 设置默认 policy：`GX_REAL_POLICY_PATH=~/gx-real/policies/policy.onnx`。
- 使用 `/usr/bin/python3`，避免 conda 环境污染真机 ROS2。
- 加入 `real-wbc`、`arx5-sdk/python` 到 `PYTHONPATH`。
- 加载 `/opt/ros/foxy/setup.bash` 或 `/opt/ros/humble/setup.bash`。
- 加载本仓库内 `unitree_ros2/cyclonedds_ws/install` 下的消息包。

`check_env.sh` 会检查：

- `policy.onnx` 和 `env.yaml` 是否存在。
- `crc_module.so` 是否存在。
- `X5_umi.urdf` 是否存在。
- `onnxruntime`、`arx5_interface`、`unitree_go`、`unitree_api` 能否 import。
- `LowCmd/LowState/WirelessController` 的 ROS2 type support 是否可用。

## 4. 推荐开发流程

本地或机器人上修改 Python 代码后，至少执行：

```bash
python3 -m py_compile \
  real-wbc/modules/wbc_node_leg12_arm_passthrough.py \
  real-wbc/scripts/run_wbc_leg12.py \
  scripts/check_env.py
git diff --check
```

如果修改了 shell 脚本，手动检查：

```bash
bash -n scripts/setup_env.sh
bash -n scripts/run_leg12_real.sh
bash -n scripts/setup_arx_can.sh
bash -n scripts/disable_sports_mode_go2.sh
```

真机运行前：

```bash
conda deactivate
cd ~/gx-real
git pull
source scripts/setup_env.sh
scripts/check_env.sh
scripts/setup_arx_can.sh
scripts/disable_sports_mode_go2.sh eth0
```

## 5. 当前启动命令

典型启动：

```bash
scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --cmd-vx 0.5 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --gripper-cmd 0.0 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0 \
  --arm-reset-pose 0.0 0.5 0.3 0.0 0.0 0.0 \
  --button-arm-pose 0.4 2.8 1.5 1.3 0.4 0.4
```

启动后手柄流程：

1. 看到 `Deploy node ready`。
2. 节点收到 LowState 后会持续发送 `Kp=0, Kd=3` 的 Passive 命令；按 `R1` 进入 internal FixStand，并平滑运动到 policy ready pose。
3. 等待起身结束。
4. 机器人稳定后按 `L2`：以当前实测 FixStand 姿态进入 1.2 秒零速度 policy handover，完成后再平滑启用配置速度。

按键说明：

- `L1`：紧急停止。
- `R1`：从 Passive 进入 internal FixStand。
- `L2`：FixStand 稳定后从当前实测姿态启动零速度 policy handover；rollout 中再次按下可恢复配置的速度命令。
- `A`：机械臂去 `--button-arm-pose`。
- `X`：机械臂回 `--arm-reset-pose`。
- `Y`：底盘 command 平滑切到 `0 0 0`，policy 保持运行。
- `R2`：停止 policy 并保持最后姿态。

## 6. 关键配置

### `policies/env.yaml`

部署代码主要读取：

- `dog_joint_names`
- `arm_joint_names`
- `joint_names`
- 初始关节位置
- 动作缩放 `action_scale`
- 观测缩放
- action clip
- policy 频率相关字段
- actuator stiffness/damping 的训练配置

这个文件是训练环境导出的完整配置，包含大量仿真字段。不要随意删除字段；如果后续要精简，建议新增真机专用 `deploy.yaml`，确认日志完全一致后再切换。

### 腿部增益

腿部 PD 不提供命令行覆盖，并按状态分开：

- Passive：`Kp=0, Kd=3`。
- internal FixStand：采用 Unitree RL Lab Go2 配置，每条腿髋/大腿/小腿分别为 `Kp=[60,80,80]`、`Kd=[5,4,4]`。
- policy rollout：从配套 `policies/env.yaml` actuator stiffness/damping 逐关节读取；当前固定模型包为 `Kp=40, Kd=1`。
- handover：保持零速度，在 1.2 秒内把关节目标和 PD 从实测 FixStand 状态平滑切换到策略输出和训练 PD。

启动日志 `Unitree RL Lab Go2 FixStand PD` 和 `Training leg PD loaded` 会分别打印两组实际值。

### 机械臂初始化

默认行为：

- ARX5 初始化成功：正常下发 arm pose。
- ARX5 初始化失败：打印错误并继续 body-only，`A/X` 机械臂按键被忽略。
- 需要机械臂不在线就中止：加 `--require-arm`。
- 明确不启用机械臂：加 `--disable-arm`。

如果看到：

```text
None of the motors are initialized. Please check the connection or power of the arm.
```

说明 `can0` 已打开，但 X5 电机没有反馈。优先检查 X5 电源、电机初始化、CAN 线和 `can0`。

## 7. 历史链路

原 UMI-on-Legs 代码仍保留了一部分：

- `real-wbc/scripts/run_wbc.py`
- `real-wbc/modules/wbc_node.py`
- `real-wbc/scripts/run_teleop.py`
- `real-wbc/ros2/robot_state`
- SpaceMouse / EEF trajectory / iPhone / MoCap 任务空间轨迹链路

这些不是当前 `leg12 + arm passthrough` 主流程。除非明确要恢复原始 whole-body trajectory tracking，不建议在当前上机流程里使用。
