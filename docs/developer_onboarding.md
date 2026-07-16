1. 项目总述
gx-real 是 Go2 + X5/ARX5 + SpaceMouse 的真机部署仓库。
项目仓库：https://github.com/lemonoscar/gx-real
主链路如下：
Go2 lowstate / wirelesscontroller
        |
scripts/run_leg12_real.sh
        |
real-wbc/scripts/run_wbc_leg12.py
        |
real-wbc/modules/wbc_node_leg12_arm_passthrough.py
        |
policies/policy.onnx + policies/env.yaml
        |
Go2 lowcmd


SpaceMouse + can0
        |
scripts/run_spacemouse_arm.sh
        |
real-wbc/scripts/run_spacemouse_arm.py
        |
real-wbc/modules/spacemouse_arm_node.py
        |
ARX5 SDK + /arm/state + /arm/target_state

2. 推荐先读哪些文件

第一次接触项目建议按这个顺序阅读：

1. README.md
总览当前系统、目录结构和主流程。

2. docs/上机使用指南.md
真机操作、启动命令、安全约束、常见故障。

3. real-wbc/docs/codebase_setup.md
开发环境、Python/ROS2 路径、历史代码说明。

4. real-wbc/scripts/run_wbc_leg12.py
Go2/WBC 运行参数入口。

5. real-wbc/modules/wbc_node_leg12_arm_passthrough.py
Go2 主控制节点，包含手柄按键、policy rollout、安全门控、日志。
  
6. real-wbc/scripts/run_spacemouse_arm.py
X5/SpaceMouse 运行参数入口。
  
7. real-wbc/modules/spacemouse_arm_node.py
SpaceMouse 轴映射、夹爪、回零、退出保护、CAN owner lock。
  
3. 仓库结构
gx-real/
  README.md

  docs/
    developer_onboarding.md          # 本文档
    上机使用指南.md                   # 详细真机操作指南
    README.md                        # 文档导航

  scripts/
    setup_env.sh                     # 设置 ROS2/PYTHONPATH/LD_LIBRARY_PATH
    check_env.sh                     # 环境预检
    check_env.py                     # Python 侧环境检查
    prepare_real_run.sh              # 真机前置准备
    setup_arx_can.sh                 # 配置 X5 SocketCAN can0
    disable_sports_mode_go2.sh       # 验证并释放 Unitree MCF
    run_leg12_real.sh                # Go2/WBC 主入口
    run_spacemouse_arm.sh            # X5/SpaceMouse 主入口

  policies/
    policy.onnx                      # 部署 policy
    env.yaml                         # 训练导出的部署配置
    height_scan_contract.yaml        # 可选高度扫描契约

  real-wbc/
    modules/
      wbc_node_leg12_arm_passthrough.py
      spacemouse_arm_node.py
      base_command_provider.py
      arm_observation.py
      can_owner_lock.py
      runtime_safety.py
    scripts/
      run_wbc_leg12.py
      run_spacemouse_arm.py
      run_teleop.py
    ros2/
      robot_state/                   # /arm/state 等消息定义
    docs/

  arx5-sdk/                          # X5/ARX5 SDK 和 Python 绑定
  unitree_sdk2/                      # Unitree SDK2 和 CRC 模块
  unitree_ros2/                      # Unitree ROS2 消息工作区
  logs/                              # 运行日志

当前优先维护的是真机主线：

- scripts/run_leg12_real.sh
- real-wbc/scripts/run_wbc_leg12.py
- real-wbc/modules/wbc_node_leg12_arm_passthrough.py
- scripts/run_spacemouse_arm.sh
- real-wbc/scripts/run_spacemouse_arm.py
- real-wbc/modules/spacemouse_arm_node.py

4. 硬件和软件前提

必需硬件：

- Unitree Go2。
- 狗身上的 Jetson Orin NX，架构是 aarch64。
- X5/ARX5 机械臂和稳定 24V 供电。
- USB-CAN 转接器，推荐使用 /dev/serial/by-id/usb-Openlight_Labs_CANable2*。
- Go2 手柄。
- 3Dconnexion SpaceMouse。
  
必需软件：

- Ubuntu + ROS2 Foxy 或 Humble。
- 系统 Python /usr/bin/python3。
- onnxruntime 安装在系统 Python 下。
- colcon、cmake。
- arx5_interface Python 绑定。
- spacenavd、libspnav-dev、Python spnav 和 atomics。
- Unitree ROS2 消息包已编译。
- 本仓库 robot_state ROS2 消息包已编译。
  
上机默认不要使用 conda Python。每个真机终端都先执行：
conda deactivate
cd ~/gx-real

确认你在机器人端 Jetson：
uname -m
cat /proc/device-tree/model 2>/dev/null || true
which python3
/usr/bin/python3 --version
预期 uname -m 是 aarch64。如果是 x86_64，说明你在开发电脑或服务器上，只适合读代码、改代码和做离线检查，不适合执行真机命令。

5. 第一次安装和构建

5.1 准备代码
机器人端：
cd ~
git clone <repo-url> gx-real
cd ~/gx-real
conda deactivate
真机仓库实际已经存在：
cd ~/gx-real
git pull origin main
conda deactivate

5.2 检查 policy 文件
cd ~/gx-real
ls policies/policy.onnx
ls policies/env.yaml
默认部署 policy 路径由 scripts/setup_env.sh 设置：
GX_REAL_POLICY_PATH=~/gx-real/policies/policy.onnx
如果要临时使用其他 policy：
export GX_REAL_POLICY_PATH=/abs/path/to/policy.onnx
source scripts/setup_env.sh

5.3 安装系统 Python 依赖
检查 ONNX Runtime：
/usr/bin/python3 -c "import onnxruntime; print('onnxruntime ok')"
如果失败，需要给 Jetson 的系统 Python 安装匹配 JetPack/Ubuntu/aarch64 的 ONNX Runtime。不要只装到 conda 里。

5.4 安装 ARX5 Python 接口
cd ~/gx-real/arx5-sdk
source /opt/ros/foxy/setup.bash
/usr/bin/python3 -m pip install --user --no-build-isolation .
如果系统是 Humble：
source /opt/ros/humble/setup.bash
安装后检查：
cd ~/gx-real
source scripts/setup_env.sh
/usr/bin/python3 -c "import arx5_interface as arx5; print(arx5.__file__)"

5.5 编译 Unitree ROS2 消息
cd ~/gx-real/unitree_ros2/cyclonedds_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select unitree_api unitree_go unitree_hg
5.6 编译本仓库 robot_state 消息

cd ~/gx-real/real-wbc/ros2
source /opt/ros/foxy/setup.bash
colcon build --packages-select robot_state

5.7 安装 SpaceMouse 依赖
sudo apt install libspnav-dev spacenavd
sudo systemctl enable spacenavd.service
sudo systemctl start spacenavd.service
/usr/bin/python3 -m pip install --user atomics
/usr/bin/python3 -m pip install --user https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
如果 import spnav 报 PyCObject_AsVoidPtr，先卸载错误版本：
/usr/bin/python3 -m pip uninstall -y spnav
再安装上面的固定 GitHub 版本。

5.8 一键预检
cd ~/gx-real
source scripts/setup_env.sh
scripts/check_env.sh
scripts/check_env.sh --spacemouse
正常情况下应该看到：
[gx-real] environment ready
[gx-real] python imports OK
[gx-real] spacemouse imports OK

6. 修改仓库与开发流程

6.1 开始工作
cd ~/gx-real
git status --short
git pull origin main
conda deactivate
source scripts/setup_env.sh

6.2 修改代码前先定位主线
常见需求和对应文件：
Go2 policy 启动、按键、低层命令、安全门控:
  real-wbc/modules/wbc_node_leg12_arm_passthrough.py

Go2 运行参数:
  real-wbc/scripts/run_wbc_leg12.py
  scripts/run_leg12_real.sh

X5 SpaceMouse 轴映射、夹爪、回零、退出保护:
  real-wbc/modules/spacemouse_arm_node.py
  real-wbc/scripts/run_spacemouse_arm.py

手柄速度映射:
  real-wbc/modules/base_command_provider.py

arm state/target observation:
  real-wbc/modules/arm_observation.py

通用输入安全检查:
  real-wbc/modules/runtime_safety.py

CAN 写进程互斥:
  real-wbc/modules/can_owner_lock.py

真机前置脚本:
  scripts/prepare_real_run.sh
  scripts/setup_arx_can.sh
  scripts/setup_env.sh

6.3 本地静态检查
每次改 Python 后至少运行：
cd ~/gx-real
python3 -m py_compile real-wbc/scripts/run_wbc_leg12.py real-wbc/scripts/run_spacemouse_arm.py
python3 -m py_compile real-wbc/modules/wbc_node_leg12_arm_passthrough.py real-wbc/modules/spacemouse_arm_node.py
改 shell 脚本后运行：
bash -n scripts/setup_env.sh
bash -n scripts/prepare_real_run.sh
bash -n scripts/run_leg12_real.sh
bash -n scripts/run_spacemouse_arm.sh
提交前检查空白错误：
git diff --check

6.4 提交和推送
git status --short
git diff
git add <changed-files>
git commit -m "Describe the real change"
git push origin main

7. 真机前置准备

每次上机都建议先跑前置脚本：
cd ~/gx-real
conda deactivate

scripts/prepare_real_run.sh \
  --network-iface eth0 \
  --can-dev auto \
  --can-if can0 \
  --spacemouse
如果 USB-CAN 自动识别失败，使用 by-id 路径：
cd ~/gx-real
conda deactivate

CAN_DEV=$(ls /dev/serial/by-id/usb-Openlight_Labs_CANable2* 2>/dev/null | head -1)
echo "$CAN_DEV"

scripts/prepare_real_run.sh \
  --network-iface eth0 \
  --can-dev "$CAN_DEV" \
  --can-if can0 \
  --spacemouse
如果你已经确认代码构建过，只想快速检查：
scripts/prepare_real_run.sh \
  --network-iface eth0 \
  --can-dev "$CAN_DEV" \
  --can-if can0 \
  --spacemouse \
  --no-build

前置脚本会做这些事情：

- 检查是否退出 conda。
- 检查是否在 Jetson/aarch64 上。
- 编译 Unitree ROS2 消息、robot_state 消息和 sport-mode 工具。
- source scripts/setup_env.sh。
- 检查 policy、Python import、SpaceMouse 依赖。
- 检查是否有已有 WBC/X5 写控制进程。
- 配置 can0。
- 检查 Go2 ROS2 topic。
- 通过 `MotionSwitcherClient::ReleaseMode()` 释放 Go2 MCF，并用 `CheckMode()` 复核。
  
8. 单独测试 X5

在启动 SpaceMouse 节点之前，先确认 X5 SDK 能读到机械臂状态。

8.1 配置 CAN
cd ~/gx-real
conda deactivate

CAN_DEV=$(ls /dev/serial/by-id/usb-Openlight_Labs_CANable2* 2>/dev/null | head -1)
echo "$CAN_DEV"

scripts/setup_arx_can.sh "$CAN_DEV" can0 8
ip -s -d link show can0
can0 应该存在且是 UP,LOWER_UP。

8.2 使用 ARX5 SDK 读取关节
cd ~/gx-real
source scripts/setup_env.sh

cd ~/gx-real/arx5-sdk
/usr/bin/python3 python/examples/calibrate.py X5 can0
正常时会持续打印 6 个关节角。更可信的状态是这些值不是长期全零，并且移动机械臂后数值会变化。

如果出现：
ImportError: libhardware.so: cannot open shared object file

说明没有 source 仓库环境，重新执行：
cd ~/gx-real
source scripts/setup_env.sh
cd ~/gx-real/arx5-sdk
/usr/bin/python3 python/examples/calibrate.py X5 can0

如果一直全零或报 missing feedback，不要启动 SpaceMouse Arm 节点。优先检查：
- X5 是否上电。
- 急停是否释放。
- CAN H/L 是否接反。
- CAN 终端电阻是否正确。
- USB-CAN 是否变成了新的 /dev/ttyACM*，建议始终使用 /dev/serial/by-id/...。
- 是否有其他进程占用 can0。
  
查看当前可能占用 X5 的进程：
pgrep -af "run_spacemouse_arm|calibrate.py|test_joint_control|arx5|candump"

9. 启动 X5 SpaceMouse 节点

确认第 8 节 X5 SDK 测试通过后，打开一个终端运行：
cd ~/gx-real
conda deactivate
unset ARX5_REQUIRE_INIT_FEEDBACK
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_spacemouse_arm.sh \
  --model X5 \
  --can-interface can0 \
  --safety-topic /safety/estop

当前默认 SpaceMouse 参数已经保存在代码里：
raw frame: true
translation axis map: z, x, y
translation signs: 1, -1, 1
rotation axis map: rx, ry, rz
rotation signs: 1, 1, 1
position speed: 0.05
rotation speed: 0.15
deadzone: 0.10
watchdog: 0.25 sec

如果第一次上机想更慢，可以显式降速：
scripts/run_spacemouse_arm.sh \
  --model X5 \
  --can-interface can0 \
  --safety-topic /safety/estop \
  --sm-pos-speed 0.01 \
  --sm-rot-speed 0.04 \
  --sm-deadzone 0.12 \
  --sm-watchdog-sec 0.25

如果觉得太慢，可以使用较快参数：
scripts/run_spacemouse_arm.sh \
  --model X5 \
  --can-interface can0 \
  --safety-topic /safety/estop \
  --sm-pos-speed 0.05 \
  --sm-rot-speed 0.15 \
  --sm-deadzone 0.10 \
  --sm-watchdog-sec 0.25

SpaceMouse 按键约定：

- 左键：夹爪打开。
- 右键：夹爪闭合。
- 左右键同时按一次：夹爪不动，机械臂回到关节目标 [0, 0.3, 0.5, 0, 0, 0]。
- 节点退出时：先尝试让 X5 回到 home，再切 damping。
- 收到 /safety/estop=True：触发 X5 安全保护。
  
注意事项：
- 真机当前用 --model X5，不要用旧的 X5_umi。
- 不要同时运行两个会写 can0 的 X5 进程。
- can0 UP 只说明接口存在，不说明 SDK 已经正确解析电机状态。
  
10. 启动 Go2/WBC

Go2/WBC 需要另开一个终端。先确认 SpaceMouse Arm 节点已经发布 /arm/state，如果你使用 --require-arm-state-for-rl，WBC 会要求机械臂状态可用后才允许 policy rollout。

10.1 固定速度模式

固定速度模式适合测试 policy 是否能稳定启动。建议第一次先从 --cmd-vx 0.20 开始，确认稳定后再改 0.50。

cd ~/gx-real
conda deactivate
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --base-command-source fixed \
  --cmd-vx 0.50 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --arm-control-owner external_spacemouse \
  --arm-state-topic /arm/state \
  --arm-target-topic /arm/target_state \
  --safety-topic /safety/estop \
  --require-arm-state-for-rl \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0

固定速度模式下的手柄流程：

1. 按 R1：内部起身。
2. 等起身完成。
3. 第一次按 L2：进入 policy 对齐/启动，初始速度为零或接管速度。
4. policy 已经启动后，再按 L2：把速度目标切到 --cmd-vx --cmd-vy --cmd-yaw，例如 0.50 0.0 0.0。
5. 按 Y：底盘速度归零。
6. 按 R2：停止 policy。
7. 按 L1：急停，并发布 /safety/estop 给 X5 节点。
  
10.2 手柄摇杆速度模式

摇杆模式适合人工实时控制底盘速度：

cd ~/gx-real
conda deactivate
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --base-command-source wireless_joystick \
  --joy-vx-axis ly \
  --joy-vx-sign 1 \
  --joy-vy-axis lx \
  --joy-vy-sign -1 \
  --joy-yaw-axis rx \
  --joy-yaw-sign -1 \
  --joy-deadzone 0.12 \
  --joy-max-vx 0.50 \
  --joy-max-vy 0.0 \
  --joy-max-yaw 0.0 \
  --arm-control-owner external_spacemouse \
  --arm-state-topic /arm/state \
  --arm-target-topic /arm/target_state \
  --safety-topic /safety/estop \
  --require-arm-state-for-rl \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0

摇杆模式下：

- 左摇杆前后控制 vx。
- 左摇杆左右当前被配置为 vy，但上面命令把 --joy-max-vy 设为 0.0，等于关闭侧向速度。
- 右摇杆左右当前被配置为 yaw，但上面命令把 --joy-max-yaw 设为 0.0，等于关闭转向。
- L2 启动 policy 后，速度一直来自摇杆，不再做固定速度切换。
- Y 会抑制摇杆速度，直到摇杆回中。
  
11. 不开 SpaceMouse Arm 节点时能不能走

可以，但取决于你的参数。

如果 WBC 启动时带了：

--require-arm-state-for-rl

就必须先启动 SpaceMouse Arm 节点，并且 /arm/state 可用。否则 WBC 会阻止 policy rollout，这是为了避免 policy observation 中机械臂状态无效。

如果只是想单独测试狗腿，不接 X5，可以去掉这个参数，并设置：

--arm-control-owner none

如果想让 policy 网络继续收到机械臂输入，但不要实时读取 /arm/state，可以使用固定初始值模式：

--arm-observation-mode fixed_initial

这个模式会把 policy observation 里的机械臂当前关节、目标关节都固定为 --arm_pose，机械臂速度和力矩固定为 0，夹爪目标固定为 --gripper-cmd。该模式不会订阅 /arm/state 和 /arm/target_state，即使命令里保留 --require-arm-state-for-rl，也不会等待实时机械臂状态。

12. 安全机制

当前真机链路里已经有这些安全约束：

- prepare_real_run.sh 会拒绝 conda 环境、检查重复 WBC/X5 写进程、配置 CAN、检查 topic。
- CanOwnerLock 防止多个 X5 writer 同时打开 can0。
- WBC 低层控制有 lowstate watchdog 和 MCF 重新激活检测。
- WBC policy 启动前 3 秒有 action abs limit 和 action delta limit。
- WBC 发现运行时安全错误会触发 safety stop。
- L1 会发布 /safety/estop=True，SpaceMouse Arm 节点收到后保护 X5。
- SpaceMouse Arm 节点退出时会尝试回 home，再切 damping。
- SpaceMouse sample 超时会 hold X5。
- 输入、action、arm state、height scan 等关键向量有 finite 检查。
  
真机上不要绕过这些保护：

- 不要用 --allow-missing-can 做真实运动。
- 不要在 policy 运行时启动 ARX5 SDK 示例去抢 can0。
- 不要在 X5 状态全零或 missing feedback 时启动 SpaceMouse Arm 节点。
- 不要直接上高速，先用 --cmd-vx 0.0 或 0.20 验证启动稳定性。
  
13. Policy 和 env.yaml 契约

部署 policy 由两部分组成：
policies/policy.onnx
policies/env.yaml

替换 policy 时必须保证：
- ONNX 输入名、输入维度与部署代码一致。
- env.yaml 中 observation term 与训练导出一致。
- 腿部关节顺序与真机 Go2 关节顺序一致。
- 输出 action 是当前部署链路期望的 12 维腿部 action。
- action scale、default pose、clip 逻辑与训练一致。
- 如果 policy 训练时包含 arm state/target，真机必须提供 /arm/state 和 /arm/target_state，或明确使用等价占位。
  
替换后至少运行：
cd ~/gx-real
source scripts/setup_env.sh
scripts/check_env.sh
python3 -m py_compile real-wbc/modules/wbc_node_leg12_arm_passthrough.py

真机第一次测试新 policy：
1. 固定速度设为 0.0 或 0.20。
2. 确认 R1 起身稳定。
3. 第一次 L2 只观察 policy 启动抖动，不立即给大速度。
4. 日志里检查 action、limited action、leg q error、foot force。
5. 再提高 --cmd-vx。
  
14. 日志位置和排查方法

WBC 每次运行会创建：
logs/YYYYMMDD_HHMMSS/
  run.log

SpaceMouse Arm 每次运行会创建：
logs/YYYYMMDD_HHMMSS_spacemouse_arm/
  run.log

查看最新日志：
cd ~/gx-real
ls -td logs/* | head
tail -n 200 logs/<latest>/run.log

抓 CAN 原始数据：
mkdir -p ~/gx-real/logs/manual
candump -tz can0 | tee ~/gx-real/logs/manual/can_x5_check.log

检查 Go2 topic：
source scripts/setup_env.sh
ros2 topic list | sort
ros2 topic echo /lowstate --once
ros2 topic echo /wirelesscontroller --once

检查 arm topic：
ros2 topic echo /arm/state --once
ros2 topic echo /arm/target_state --once

给别人分析问题时，优先提供：
- 完整启动命令。
- 终端完整报错，不要只截最后一行。
- logs/<timestamp>/run.log。
- logs/<timestamp>_spacemouse_arm/run.log。
- ip -s -d link show can0。
- pgrep -af "run_spacemouse_arm|run_leg12_real|run_wbc|calibrate.py|arx5|candump"。
- 必要时提供 candump 前几秒日志。
  
15. 常见问题

15.1 CAN interface 'can0' does not exist

说明 SocketCAN 没配置或 USB-CAN 设备号变了。执行：
cd ~/gx-real
conda deactivate
CAN_DEV=$(ls /dev/serial/by-id/usb-Openlight_Labs_CANable2* 2>/dev/null | head -1)
echo "$CAN_DEV"
scripts/setup_arx_can.sh "$CAN_DEV" can0 8
ip -s -d link show can0
不要硬编码 /dev/ttyACM0，重启后经常会变成 /dev/ttyACM1。

15.2 Device not found: /dev/ttyACM0

使用 by-id 路径：
ls -l /dev/serial/by-id/
然后把完整路径传给 setup_arx_can.sh。

15.3 Missing feedback from joint motor IDs
这通常不是 Python 参数问题，而是 X5 SDK 没拿到有效电机反馈。检查：
- X5 电源。
- X5 急停。
- CAN H/L。
- 终端电阻。
- 是否只有一个 X5 writer。
- candump 是否有 001,002,004,005,006,007,008 等电机帧。
- calibrate.py X5 can0 是否能打印非零关节状态。
如果 calibrate.py 都不正常，不要继续启动 SpaceMouse Arm 节点。

15.4 ImportError: libhardware.so

没有加载 ARX5 动态库路径。执行：
cd ~/gx-real
source scripts/setup_env.sh
然后重新运行命令。

15.5 /wirelesscontroller 没 sample

检查 Go2 网络和 DDS：
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh
ros2 topic list | sort
ros2 topic echo /wirelesscontroller --once
如果 /lowstate 有、/wirelesscontroller 没有，重点查手柄连接和 Go2 DDS topic。

15.6 机械臂有 SpaceMouse 日志但不动

按顺序查：
1. 是否用 --model X5。
2. calibrate.py X5 can0 是否能读到关节变化。
3. 是否有其他进程占用 can0。
4. SpaceMouse Arm 日志是否出现 X5 position hold enabled。
5. 是否出现 Inverse kinematics failed、Missing feedback、invalid X5 state。
6. 当前末端目标是否已经接近关节/IK 限制。
  
15.7 policy 一启动抖动大

先不要直接提高速度。按以下顺序排查：
- 固定速度设为 --cmd-vx 0.0。
- 确认 R1 起身后腿部姿态稳定。
- 确认第一次 L2 对齐阶段没有明显跳变。
- 检查 startup-action-limit-sec、startup-action-abs-limit、startup-action-delta-limit 是否启用。
- 对比训练环境里的初始姿态、action scale、default joint pose、history buffer 和部署一致性。
- 查看 run.log 中 action、limited action、leg_q_error、base command、foot force。
  
16. 推荐启动命令速查

16.1 前置准备
cd ~/gx-real
conda deactivate

CAN_DEV=$(ls /dev/serial/by-id/usb-Openlight_Labs_CANable2* 2>/dev/null | head -1)

scripts/prepare_real_run.sh \
  --network-iface eth0 \
  --can-dev "$CAN_DEV" \
  --can-if can0 \
  --spacemouse \
  --no-build

16.2 终端 A: X5/SpaceMouse
cd ~/gx-real
conda deactivate
unset ARX5_REQUIRE_INIT_FEEDBACK
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_spacemouse_arm.sh \
  --model X5 \
  --can-interface can0 \
  --safety-topic /safety/estop

16.3 终端 B: Go2/WBC 固定 0.5 m/s
cd ~/gx-real
conda deactivate
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --base-command-source fixed \
  --cmd-vx 0.50 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --arm-control-owner external_spacemouse \
  --arm-state-topic /arm/state \
  --arm-target-topic /arm/target_state \
  --safety-topic /safety/estop \
  --require-arm-state-for-rl \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0

16.4 终端 B: Go2/WBC 手柄速度模式
cd ~/gx-real
conda deactivate
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh

scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --base-command-source wireless_joystick \
  --joy-vx-axis ly \
  --joy-vx-sign 1 \
  --joy-vy-axis lx \
  --joy-vy-sign -1 \
  --joy-yaw-axis rx \
  --joy-yaw-sign -1 \
  --joy-deadzone 0.12 \
  --joy-max-vx 0.50 \
  --joy-max-vy 0.0 \
  --joy-max-yaw 0.0 \
  --arm-control-owner external_spacemouse \
  --arm-state-topic /arm/state \
  --arm-target-topic /arm/target_state \
  --safety-topic /safety/estop \
  --require-arm-state-for-rl \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0

17. 最小上机检查清单

上电前：
- Go2 周围安全。
- X5 周围安全。
- 硬件急停可用。
- 手柄 L1 可用。
- X5 24V 供电正常。
- CAN H/L 和终端电阻确认。
  
上机前：
- conda deactivate。
- source scripts/setup_env.sh。
- prepare_real_run.sh 通过。
- calibrate.py X5 can0 正常。
- SpaceMouse Arm 节点正常启动。
- WBC 看到 /arm/state。
  
开始运动：
- 先固定 --cmd-vx 0.0 或 0.20。
- R1 起身。
- 等站稳。
- 第一次 L2 启动 policy。
- 观察 3 秒。
- 再按 L2 或调速度进入移动。
  
出现异常：
- 先 L1。
- 再看日志。
- 不要重复启动多个 X5/WBC 进程抢控制。
