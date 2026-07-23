# gx-real 真机开发文档

这份文档面向第一次接触本仓库的人。默认部署环境是机器狗机身上的 Jetson Orin NX，路径按 `~/gx-real` 书写。当前推荐主线是纯 SportMode 组合启动：无线手柄摇杆只控制 Go2 的速度和转向，独立 SpaceMouse Arm 节点独占 X5/ARX5，两个节点由一个 supervisor 命令按顺序启动。旧的 WBC/ONNX/`lowcmd` 链仍保留用于历史实验，但不会由纯 SportMode 入口启动。

当前推荐的真机部署流程见本 README 第 4 节，更细的控制约定见 [纯 SportMode 上机指南](docs/纯SportMode上机指南.md)。

## 1. 当前系统做什么

当前仓库用于在真机上运行：

- 运行端：Go2 机身上的 Jetson Orin NX，预期架构是 `aarch64`，默认使用系统 `/usr/bin/python3`。
- Go2 腿部：保持原厂 SportMode，只接收 `Move(vx, 0, yaw)` 和 `StopMove`。
- X5/ARX5 机械臂：由独立 SpaceMouse Arm 节点控制，与四足控制解耦。
- Go2 通信：订阅 ROS2 `/wirelesscontroller`，向 Unitree `/api/sport/request` 发命令；不发布 `lowcmd`。
- X5 通信：SocketCAN `can0` + `arx5_interface`，只允许 SpaceMouse Arm 节点打开写控制。
- 控制流程：保持 Go2 SportMode；启动时关闭避障并查询确认，然后关闭原厂手柄直通，只解释摇杆轴。
- SpaceMouse：独立 Arm 节点由组合入口自动启动，默认使用 raw SpaceMouse 输入，经显式 axis/sign/scale 参数映射后直接控制 X5，并发布 arm state/target topic。

主入口只需一个终端；supervisor 会等待狗进入 `SPORTMODE_ACTIVE` 后再启动机械臂：

```text
scripts/run_sportmode_with_arm.sh eth0 can0
  -> sportmode_wireless_node -> Unitree Sport Move/StopMove
  -> wait SPORTMODE_ACTIVE
  -> spacemouse_arm_node      -> ARX5 can0 command
```

一句话理解：四足只走 Unitree 高层 SportMode，机械臂只走独立 X5 节点；运行时没有策略推理，也没有低层腿部写控制。

纯 SportMode 运行约束：

- 当前主线是两个硬件写控制节点：SportMode 节点只写 Go2 高层速度，SpaceMouse Arm 节点只写 X5。
- Go2 手柄按键不映射任何软件功能；`ly` 生成前后速度，`rx` 生成转向，默认关闭侧移。
- 避障关闭必须收到 Unitree 查询确认，否则节点 fail-closed，不进入运动状态。
- ROS2/DDS 是真机通信前提，runtime shell 必须显示 `rmw=rmw_cyclonedds_cpp` 和正确的 `cyclonedds_iface`。
- `can0 UP` 只说明 SocketCAN 存在；`None of the motors are initialized` 说明 ARX5 SDK 没解析到可用电机状态，需继续查当前 CAN 回包、电源/急停/接线或 `--model` 配置。

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
    prepare_real_run.sh             # 封装真机前置构建、接口检查、CAN 和 sport mode 处理
    setup_arx_can.sh                # 配置 ARX5 SocketCAN can0
    disable_sports_mode_go2.sh      # 编译/调用 Unitree SDK 工具关闭 sport mode
    run_sportmode_wireless.sh       # 纯 SportMode 四足入口
    run_sportmode_with_arm.sh       # 推荐：一个命令按顺序启动狗与机械臂
    run_leg12_real.sh               # legacy Go2/WBC 入口
    run_spacemouse_arm.sh           # 推荐 SpaceMouse + X5 独立控制入口
    run_spacemouse_teleop.sh        # legacy teleop topic 发布节点
    run_arm_spacemouse_test.sh      # 只测 X5 + SpaceMouse，不启动 Go2/policy

  policies/
    policy.onnx                     # 当前真机推理模型
    policy.pt                       # 训练侧导出的 PyTorch 模型或备份
    env.yaml                        # 训练环境导出的部署配置

  real-wbc/
    modules/
      wbc_node_leg12_arm_passthrough.py
      base_command_provider.py
      arm_observation.py
      sportmode_wireless.py
      spacemouse_arm_node.py
      common.py
      velocity_estimator.py
      spacemouse_shared_memory.py
      shared_memory/
    scripts/
      run_wbc_leg12.py
      run_sportmode_wireless.py
      run_spacemouse_arm.py
      run_teleop.py
    ros2/
      robot_state/                  # Teleop、ArmState、ArmTargetState ROS2 消息
    docs/                           # 硬件、网络、开发环境细分说明

  arx5-sdk/                         # X5/ARX5 机械臂 SDK 和 Python 绑定
  unitree_sdk2/                     # Unitree SDK2、CRC 模块、关闭 sport mode 工具
  unitree_ros2/                     # Unitree ROS2/CycloneDDS 消息工作区
  logs/                             # 每次运行的日志目录
```

当前优先维护的是单命令 supervisor 启动的纯 SportMode + 独立 X5 链。`run_leg12_real.sh`、`run_wbc*.py`、EEF trajectory、iPhone/MoCap 等内容属于旧的策略实验链路或后续扩展。

## 3. 硬件和外部依赖

必需硬件：

- Unitree Go2。
- Go2 机身 Jetson Orin NX 开发板，作为默认部署主机。
- X5/ARX5 机械臂。
- USB-CAN 转接器，当前默认接口名是 `can0`。
- Go2 网络连接，通常是 `192.168.123.xxx` 网段。
- X5 供电链路，建议通过 DC 降压模块输出稳定 24V。
- Go2 手柄；纯 SportMode 节点只读取摇杆轴，忽略全部按键。

可选或历史外设：

- iPhone/MoCap：原任务空间位姿估计链路需要，当前默认 `--pose_estimator none`。
- GoPro、采集卡、fin-ray gripper：原 UMI 数据采集链路需要，当前 leg12 行走调试不依赖。
- 3Dconnexion SpaceMouse Wireless：用于独立 X5/ARX5 机械臂控制；不是走路 policy 的必需项。

软件环境默认假设机器人端路径为：

```bash
~/gx-real
```

核心依赖：

- Jetson Orin NX 上的 Ubuntu + ROS2 Foxy 或 Humble，架构应为 `aarch64`。
- 本仓库内的 `unitree_ros2/cyclonedds_ws/install`。
- `/usr/bin/python3` 下可 import 的 `rclpy`、`unitree_api` 和 `unitree_go`。
- 只有 legacy WBC/`lowcmd` 链才需要 `onnxruntime`、policy 文件和 `crc_module.so`。
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

## 4. 纯 SportMode 真机部署（当前推荐）

本节是 `Go2 + X5 + SpaceMouse` 当前主线的可执行上机流程。这条链不加载 policy、不运行 WBC、不发布 `/lowcmd`。不要在这条链上执行 `prepare_real_run.sh`、`disable_sports_mode_go2.sh` 或 `run_leg12_real.sh`；它们属于后文的 legacy 低层链路。

### 4.1 安全前提

上电前必须同时满足：

- Go2 在平整、无人的开阔场地，足端接地稳定，机身周围没有线缆和障碍物。
- 硬件急停可随时触达，操作员不站在机械臂旋转半径内。
- Go2 保持原厂 SportMode；本链路只发高层 `Move/StopMove/StandDown`。
- 不得同时运行 `run_wbc*.py`、`run_leg12_real.sh`、ARX5 SDK 示例或其他 X5 CAN 写进程。
- 启动前松开所有手柄和 SpaceMouse 输入。纯 SportMode 运行期间原厂手柄按键被关闭，不能把它当成本软件的急停。

### 4.2 首次拉取与后续更新

在 Go2 机身 Jetson 上新建目录并只拉取当前分支：

```bash
conda deactivate 2>/dev/null || true
cd ~
git clone \
  --branch agent/pure-sportmode-runtime \
  --single-branch \
  git@github.com:lemonoscar/gx-real.git \
  gx-real
cd ~/gx-real
git status -sb
```

如果 Jetson 没有配置 GitHub SSH key，可将 clone URL 换成 `https://github.com/lemonoscar/gx-real.git`。如果目录已经存在，不要再 clone：

```bash
cd ~/gx-real
git status --short
git fetch origin
git switch agent/pure-sportmode-runtime
git pull --ff-only origin agent/pure-sportmode-runtime
```

`git status --short` 有本地修改时先处理或保存这些修改，不要盲目覆盖。

### 4.3 首次构建运行环境

确认运行主机和系统 Python：

```bash
cd ~/gx-real
uname -m
/usr/bin/python3 --version
ls /opt/ros/foxy/setup.bash /opt/ros/humble/setup.bash 2>/dev/null
```

真机 Jetson 应显示 `aarch64`。不要用 conda Python 运行 ROS2 真机节点。

首次部署需要编译 Unitree 和本仓库的 ROS2 消息；以 Foxy 为例：

```bash
cd ~/gx-real/unitree_ros2/cyclonedds_ws
source /opt/ros/foxy/setup.bash
colcon build

cd ~/gx-real/real-wbc/ros2
source /opt/ros/foxy/setup.bash
colcon build --packages-select robot_state
```

系统是 Humble 时将两处 `foxy` 换成 `humble`。Unitree SportMode 配置工具会由启动脚本使用 CMake 自动编译。

安装 X5 Python 接口和 SpaceMouse 依赖：

```bash
cd ~/gx-real/arx5-sdk
source /opt/ros/foxy/setup.bash
/usr/bin/python3 -m pip install --user --no-build-isolation .

sudo apt update
sudo apt install -y can-utils libspnav-dev spacenavd
sudo systemctl enable --now spacenavd.service
/usr/bin/python3 -m pip install --user atomics
/usr/bin/python3 -m pip install --user \
  https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
```

如果是 Humble，同样替换 ROS 环境。不要安装 PyPI 默认的 `spnav==0.9`；它在某些 Jetson Python3 环境会出现 `PyCObject_AsVoidPtr` 错误。

### 4.4 每次上机的基础检查

先找到与 Go2 相连、带有 `192.168.123.*` 地址的网卡：

```bash
conda deactivate 2>/dev/null || true
cd ~/gx-real
ip -br address
ip route
```

下文以 `eth0` 为例；如果实际是 `enP8p1s0` 或其他名称，所有 `eth0` 都必须替换。

在一个临时终端加载纯 SportMode 环境并检查 ROS2：

```bash
cd ~/gx-real
export GX_REAL_NETWORK_IFACE=eth0
export GX_REAL_REQUIRE_POLICY=0
export GX_REAL_REQUIRE_CRC=0
source scripts/setup_env.sh

ros2 topic list
timeout 3s ros2 topic echo /wirelesscontroller
ros2 topic info /lowcmd
```

Foxy 的 `ros2 topic echo` 不一定支持 `--once`，因此文档统一使用 `timeout 3s`。确认 `/wirelesscontroller` 有数据，并且 `/lowcmd` 不存在或 publisher count 为 `0`。如果没有数据，先修复网络、CycloneDDS 或 ROS2 消息包，不得继续启动运动。

再检查没有互斥的写控制进程：

```bash
pgrep -af 'run_wbc|run_leg12_real|disable_sports_mode_go2|spacemouse_teleop' || true
```

### 4.5 当前没有机械臂时的联动验证

使用组合入口的 `--dry-run`：狗节点真实运行，但机械臂节点不打开 CAN、也不依赖 SpaceMouse。三个速度上限默认全部为 `0`，因此只验证 SportMode 预检、心跳和退出链路：

```bash
cd ~/gx-real
scripts/run_sportmode_with_arm.sh --dry-run eth0 can0
```

必须看到：

```text
[pure-sportmode] required configuration checks passed; readable states and light brightness confirmed at 0
Pure SportMode ready (SPORTMODE_ACTIVE)
```

同时确认日志没有 `detected a lowcmd publisher`。supervisor 随后会自动显示 `SPORTMODE_ACTIVE; arm pid=... dry_run=1`。在这个终端按一次 `Ctrl-C`，预期顺序是：

1. 机器狗发布 `STOPPING` 并等待机械臂节点。
2. dry-run 机械臂节点自动退出。
3. 机器狗调用 SportMode `StandDown` 后退出。

这只验证 ROS2 门控和进程联动，不验证 X5 CAN、电机反馈、回位或 damping 的真实硬件动作。

### 4.6 安装 X5 后的正式单命令启动

首先上电 X5，松开硬件急停，确认机械臂初始姿态不会与 Go2 或地面碰撞，然后配置 CAN：

```bash
cd ~/gx-real
scripts/setup_arx_can.sh auto can0 8
ip -details link show can0
timeout 2s candump can0
```

`can0 UP` 不等于电机已正常反馈；`candump` 完全无数据或 ARX5 报 `None of the motors are initialized` 时，先检查 24V 供电、急停、CAN-H/CAN-L、波特率和型号，不要继续使能。

首次真机用一个命令启动，并保持狗速度为 `0`：

```bash
cd ~/gx-real
scripts/run_sportmode_with_arm.sh eth0 can0 0.0 0.0 0.0
```

supervisor 会先启动狗、等待 `SPORTMODE_ACTIVE`，然后自动启动机械臂。任何时候都不需要第二个终端。组合入口默认速度也是 `0 0 0`；上面的三个显式零值用于提醒首次上机不能直接运动。

机械臂启动后先保持 SpaceMouse 两个按键都松开，再同时按下两键一次执行显式使能。只有收到 `SPORTMODE_ACTIVE` 且安全心跳有效时才会接受使能。第一次只做小幅度、单轴、短时间测试；确认方向后再逐步提高速度。

两个 SpaceMouse 按键松开后再次同时按下，会请求回到固定关节位置 `[0, 0.3, 0.5, 0, 0, 0]`。回位前必须确认整条轨迹没有碰撞风险。

机械臂验证通过后，退出并重新执行组合命令，再将机器狗速度从低值开始：

```bash
scripts/run_sportmode_with_arm.sh eth0 can0 0.10 0.00 0.10
```

不得超过程序硬限制：`vx <= 0.30 m/s`、`vy <= 0.20 m/s`、`yaw <= 0.30 rad/s`。

### 4.7 停机顺序和故障语义

正常联动停机只在组合启动终端按一次 `Ctrl-C`，然后等待两个进程自行退出；不要立即再按第二次或使用 `kill -9`。

| 事件 | 机械臂行为 | 机器狗行为 |
|---|---|---|
| 组合入口正常 `Ctrl-C`/`SIGTERM` | 优先回 `[0, 0.3, 0.5, 0, 0, 0]`，然后 damping 并退出 | 等机械臂退出后请求 `StandDown` |
| 机械臂节点自行正常退出 | 优先回固定位置，然后 damping 并退出 | supervisor 保持狗运行 |
| X5 CAN/反馈/SpaceMouse/门控异常 | 不主动回位，立即 damping 并非零退出 | 保持运行 |
| 机器狗节点故障或心跳消失 | 不主动回位，立即 damping 并退出 | 停止速度输出；故障路径不保证 `StandDown` |

正常回位只会在安全心跳健康且底盘状态为 `SPORTMODE_ACTIVE` 或 `STOPPING` 时执行。当前 Cartesian controller 用固定关节目标的 FK 位姿生成命令，并用真实关节反馈确认结果。最长运动时间为 3 秒，再加 0.5 秒收敛窗口，反馈误差阈值为 `0.05 rad`。无法在窗口内确认时会放弃主动运动并进入 damping。

机器狗正常退出时最多等待机械臂节点 5 秒；超时会记录 warning 并继续请求 `StandDown`。Go2 `StandDown` 的过渡速度由固件决定，SDK 没有速度参数。

`SIGKILL`、断电、CAN 硬件中断或进程崩溃无法保证回位或趴卧，必须依赖可触达的硬件急停和现场防护。

### 4.8 首次真机验收清单

只有下列项目全部通过，才可以从零速逐步增加命令：

- [ ] Jetson 为 `aarch64`，运行 Python 为 `/usr/bin/python3`。
- [ ] `setup_env.sh` 显示 `rmw=rmw_cyclonedds_cpp` 和正确的 `cyclonedds_iface`。
- [ ] `/wirelesscontroller` 持续有数据，`/lowcmd` 不存在或 publisher count 为 `0`。
- [ ] 所有必需 SDK 配置检查通过；允许重复 `StopMove()` 和 `Pose(false)` 显示幂等 `-1` warning，避障、UWB 跟随、自动恢复和 VUI 亮度读回值必须符合预期。
- [ ] supervisor 显示 `SPORTMODE_ACTIVE` 后才启动机械臂，速度为零时机器狗不移动。
- [ ] 无机械臂 dry-run 中，狗退出能让臂节点先退出，再请求 `StandDown`。
- [ ] X5 上电后 `can0` 有有效反馈，不存在第二个 CAN 写进程。
- [ ] 机械臂只在 `SPORTMODE_ACTIVE` 后能显式使能，小幅单轴方向正确。
- [ ] 机械臂单独正常退出不影响机器狗。
- [ ] 狗正常退出时，机械臂回位、damping、退出、Go2 `StandDown` 的顺序正确。

关于灯光：软件会将 VUI brightness 设为 `0` 并读回确认，但固件高优先级系统指示灯可能仍亮。`GetBrightness()==0` 不等于物理指示灯必然熄灭，需要在当前 Go2 固件上目视验证。

## 5. Legacy WBC 第一次部署（非纯 SportMode 主线）

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

编译本仓库的 `robot_state` 消息包。`/teleop/*`、`/arm/state`、`/arm/target_state` 消息都在这里生成：

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

## 6. Legacy WBC 每次上机前检查

推荐把所有可重复的前置构建和接口检查交给脚本：

```bash
conda deactivate
cd ~/gx-real
scripts/prepare_real_run.sh --network-iface eth0 --can-dev auto --can-if can0 --spacemouse
```

这个脚本会顺序完成：

- 编译/刷新 `unitree_ros2/cyclonedds_ws`、`real-wbc/ros2` 的 `robot_state` 消息和 `unitree_sdk2/build/disable_sports_mode_go2`。
- 加载 `scripts/setup_env.sh` 并执行 `scripts/check_env.sh --spacemouse`。
- 检查 `spacenavd`、互斥的 WBC/X5 写控制进程、`can0`、Go2 ROS2 topic。
- 调用 `scripts/disable_sports_mode_go2.sh eth0` 关闭 sport mode。

如果连接 Go2 的网卡不是 `eth0`，只改 `--network-iface`；如果 USB-CAN 不能自动识别，只改 `--can-dev`。下面的单项命令主要用于失败后的定位。

如果要排查 Go2 手柄摇杆轴输入，在前置脚本后面加 `--check-joystick-motion`，并按提示在采样窗口内拨动摇杆；脚本会检查 `/wirelesscontroller` 的 `lx/ly/rx/ry` 是否真的变化。

如果需要同步代码，先手动执行 `git pull`，再运行前置脚本；脚本本身不会隐式修改 Git 工作区。

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
timeout 3s ros2 topic echo /lowstate
timeout 3s ros2 topic echo /wirelesscontroller
timeout 3s ros2 topic echo lf/sportmodestate
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

## 7. Legacy WBC 标准启动流程

典型 WBC 启动命令。默认 `--arm-control-owner external_spacemouse`，WBC 不会打开 `can0` 或下发 X5 command：

```bash
cd ~/gx-real
scripts/run_leg12_real.sh \
  --device cpu \
  --pose_estimator none \
  --standup-mode internal \
  --cmd-vx 0.5 \
  --cmd-vy 0.0 \
  --cmd-yaw 0.0 \
  --base-command-source fixed \
  --arm-control-owner external_spacemouse \
  --gripper-cmd 0.0 \
  --leg-kp 200 \
  --leg-kd 10 \
  --arm_pose 0.0 0.5 0.3 0.0 0.0 0.0 \
  --arm-reset-pose 0.0 0.5 0.3 0.0 0.0 0.0
```

第一次验证建议把 `--cmd-vx` 降到 `0.0` 或 `0.1`，先确认低层接管和关节顺序，再逐步提高速度。`--arm_pose` 现在只是 WBC observation fallback，不会由 WBC 下发给 X5。

推荐的 X5 控制方式是另开一个 Jetson 终端启动 SpaceMouse Arm 节点：

```bash
cd ~/gx-real
source scripts/setup_env.sh
scripts/run_spacemouse_arm.sh \
  --can-interface can0 \
  --sm-use-raw-frame true \
  --sm-pos-speed 0.03 \
  --sm-rot-speed 0.10 \
  --sm-deadzone 0.12
```

SpaceMouse Arm 节点是唯一 X5 写控制进程，会打开 `can0`，直接下发 ARX5 command，并持续发布 `/arm/state` 和 `/arm/target_state`。不要同时运行 `arx5-sdk/python/examples/spacemouse_teleop.py`、`scripts/run_arm_spacemouse_test.sh` 或 WBC legacy arm write 模式，避免两个进程同时抢 `can0`。

如果只想单独测试机械臂和 SpaceMouse，不启动机器狗、不启动 policy：

```bash
cd ~/gx-real
source scripts/setup_env.sh
scripts/setup_arx_can.sh              # can0 已经 UP 时可跳过
scripts/run_arm_spacemouse_test.sh    # 默认 X5_umi can0
```

这个命令会直接调用 ARX5 SDK 的 Cartesian SpaceMouse 示例，只控制 X5，不订阅 Go2，不发布 `lowcmd`，也不会运行 ONNX policy。它是 legacy 单臂测试入口；新联合架构优先使用 `scripts/run_spacemouse_arm.sh`。如果需要显式指定模型和 CAN 接口：

```bash
scripts/run_arm_spacemouse_test.sh X5_umi can0
```

单独机械臂测试的默认速度按正常遥操作设置：末端平移 `0.10 m/s`，旋转 `0.30 rad/s`，夹爪 `0.03 m/s`。默认不再额外限制 home 附近工作空间，位置约束交给 ARX5 SDK 的 IK、关节限位和电流保护。首次测试仍然只轻推 SpaceMouse，不要长时间顶住一个方向。如果需要临时收窄，可以透传参数：

```bash
scripts/run_arm_spacemouse_test.sh X5_umi can0 --workspace-xyz 0.08 0.08 0.06 --workspace-rpy 0.25 0.25 0.25
```

如果只想看 warning/error，减少 SDK 的周期 debug 输出：

```bash
scripts/run_arm_spacemouse_test.sh X5_umi can0 --log-level warning
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
- `L2`：起身完成后启动低层对齐和 policy；fixed 模式 rollout 中再次按下会恢复配置的移动命令。
- `Y`：底盘 command 平滑切到 `0 0 0`，policy 保持运行；joystick 模式下会 inhibit 到摇杆全部回中。
- `R2`：停止 policy。
- `L1`：紧急停止并退出。
- `A/B/X/↑/↓`：默认 no-op，不再影响机械臂。机械臂动作只来自独立 SpaceMouse Arm 节点。

底盘 command 来源：

- 默认 `--base-command-source fixed`，继续使用 `--cmd-vx/--cmd-vy/--cmd-yaw`。
- 可选 `--base-command-source wireless_joystick`，左摇杆/右摇杆映射由 `--joy-*-axis` 和 `--joy-*-sign` 显式配置，并带 deadzone、速度上限、加速度限制、watchdog 和 `Y` inhibit。

## 8. Legacy WBC 控制架构

运行时主要数据流：

```text
Go2 LowState
  -> lowlevel_state_cb
  -> 260D obs
  -> ONNX policy
  -> 12D leg action
  -> action scale + offset
  -> 12D leg target

/arm/state + /arm/target_state
  -> 6D arm state/target observation only

12D leg target
  -> set_motor_position(...)
  -> Go2 LowCmd only

SpaceMouse raw input
  -> explicit axis/sign/scale mapping
  -> standalone SpaceMouse Arm Node
  -> ARX5 can0 command + /arm/state + /arm/target_state
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
- `/arm/state` 和 `/arm/target_state` 订阅，用于机械臂 observation。
- 手柄启动、停止、急停。
- internal 起身和低层对齐。
- policy 日志和运行日志落盘。

## 9. Policy 契约

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
| `[9:12)` | `commands` | fixed 模式来自 `--cmd-vx/--cmd-vy/--cmd-yaw`；joystick 模式来自 Go2 手柄摇杆安全滤波 |
| `[12:30)` | `dof_pos` | 12 腿真实关节位置 + `/arm/state.joint_pos`，减 offset |
| `[30:48)` | `dof_vel` | 12 腿真实关节速度 + `/arm/state.joint_vel`，乘 scale |
| `[48:66)` | `actions` | 上一拍动作历史 |
| `[66:253)` | `height_scan` | 当前为全 0 |
| `[253:259)` | `arm_joint_command` | 优先 `/arm/target_state.joint_target`，target stale 时回退 `/arm/state.joint_pos` |
| `[259:260)` | `gripper_command` | 优先 `/arm/target_state.gripper_target`，target stale 时回退 `/arm/state.gripper_pos` |

输出契约：

- ONNX 输出必须是 12 维腿部 action。
- action 先按 `env.yaml` 中的 scale 和 offset 映射到腿部目标关节位置。
- 机械臂 6 维目标不由 ONNX 输出，也不由 WBC 生成；默认由独立 SpaceMouse Arm 节点发布 `/arm/target_state`。
- WBC 最终只向 Go2 下发 12 维腿部目标：

```text
full_action[0:12]  = leg_policy_target
full_action[12:18] = observation target cache only; WBC 默认不下发给 X5
```

关节顺序要特别注意。硬件接口默认顺序是：

```text
FR, FL, RR, RL
```

部署代码会根据 `env.yaml` 中的 `dog_joint_names` 建立策略顺序和接口顺序之间的映射。只要换 policy 或重新导出 `env.yaml`，必须重新检查 `dog_joint_names`、`joint_names[:12]` 和动作维度。

## 10. 修改代码时怎么入手

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

如果要改 SpaceMouse Arm 控制：

- 输入读取在 `real-wbc/modules/spacemouse_shared_memory.py`，默认使用 raw `get_motion_state()`。
- 控制节点在 `real-wbc/modules/spacemouse_arm_node.py`，启动入口是 `real-wbc/scripts/run_spacemouse_arm.py`。
- ROS2 消息在 `real-wbc/ros2/robot_state/msg/Arm*.msg`，改消息后必须重新 `colcon build --packages-select robot_state`。
- 保持 SpaceMouse Arm 节点是唯一 X5 写控制进程；WBC 默认只消费 `/arm/state` 和 `/arm/target_state`。

如果要恢复 UMI 原始任务空间链路：

- 参考 `real-wbc/modules/wbc_node.py` 和 `real-wbc/scripts/run_wbc.py`。
- 需要重新梳理 `EEFState/EEFTraj`、pose estimator、历史 trajectory teleop 和 whole-body actor。
- 不建议把这条链和当前 leg12 主链直接混在同一个节点里，先用独立入口验证。

## 11. 开发验证命令

Jetson 上请优先使用系统 Python 做检查：

```bash
/usr/bin/python3 -m py_compile \
  real-wbc/modules/base_command_provider.py \
  real-wbc/modules/arm_observation.py \
  real-wbc/modules/spacemouse_arm_node.py \
  real-wbc/modules/wbc_node_leg12_arm_passthrough.py \
  real-wbc/scripts/run_wbc_leg12.py \
  real-wbc/scripts/run_spacemouse_arm.py \
  real-wbc/scripts/run_teleop.py \
  scripts/check_env.py
```

Shell 脚本语法检查：

```bash
bash -n scripts/setup_env.sh
bash -n scripts/check_env.sh
bash -n scripts/prepare_real_run.sh
bash -n scripts/setup_arx_can.sh
bash -n scripts/disable_sports_mode_go2.sh
bash -n scripts/run_leg12_real.sh
bash -n scripts/run_spacemouse_arm.sh
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

## 12. 日志和故障定位

每次运行会在 `logs/YYYYMMDD_HHMMSS/run.log` 下保存日志。重点看这些日志：

- `Runtime targets`：启动参数是否被正确读取，尤其是 `base_command_source`、`arm_control_owner`、`arm_hold_pose`、`commanded_leg_kp/kd`。
- `Runtime leg offset update`：`R1` 后是否使用当前站姿做 runtime offset。
- `Policy diag`：policy 输出、clip、命令、低层目标、真实关节、误差和足端力。
- `Arm observation stale`：WBC 是否收到新鲜 `/arm/state` 和 `/arm/target_state`。
- `Pose test diag`：只验证关节目标跟踪时使用。

常见问题：

- `check_env.sh` 失败在 `onnxruntime`：确认是在 Jetson 的 `/usr/bin/python3` 下安装，而不是 conda。用 `/usr/bin/python3 -c "import onnxruntime"` 复查。
- `robot_state.msg.Teleop*` 或 `robot_state.msg.Arm*` import 失败：重新编译 `real-wbc/ros2`，然后重新 `source scripts/setup_env.sh`。
- `Could not import 'rosidl_typesupport_c' for package 'robot_state'`：通常是 Jetson 上 `robot_state` 的生成消息还没按最新代码 clean rebuild，或当前 shell 还在 conda `base`。先 `conda deactivate`，再删除 `real-wbc/ros2/build/robot_state` 和 `real-wbc/ros2/install/robot_state` 后重新 `colcon build --packages-select robot_state`。
- `unitree_api` 类型支持报缺少 `libpython3.x.so`：Unitree ROS2 消息曾在错误的 Conda Python 下编译。先 `conda deactivate`，清除继承的 ROS/Python 环境，重新 source 系统 ROS，再用 `colcon build --cmake-clean-cache --packages-select unitree_api unitree_go unitree_hg --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3` 重建仓库内的 `unitree_ros2/cyclonedds_ws`。
- `check_env.sh --spacemouse` 失败在 `spnav` 或 `atomics`：安装 SpaceMouse Python 依赖；失败在 `spacenavd` 或 `libspnav`：安装/启动系统服务。
- `undefined symbol: PyCObject_AsVoidPtr`：卸载 PyPI 版 `spnav`，安装 README 中固定的 Cheng Chi fork。
- 单独机械臂测试失败在 `Error document empty` / `Failed to get chain from kdl tree`：ARX5 Python 扩展可能从 pip 安装目录加载，默认找不到仓库里的 URDF。更新到最新代码后，`scripts/run_arm_spacemouse_test.sh` 会显式把 `arx5-sdk/models` 传给示例。
- `Background send_recv task is running too slow`：这是 ARX5 SDK 的 DEBUG 级通信周期提示。偶发 `2-4 ms` 且机械臂运动平滑、无 `warning/error` 时可以忽略；默认日志级别已改为 `info`，需要排查底层周期时再加 `--log-level debug`。
- `Inverse kinematics failed: E_EXCEED_JOINT_LIMIT` 或 `Over current detected`：目标末端位姿太快或太远，已经触到 IK/关节/电流保护。立即松开 SpaceMouse 或 `Ctrl+C` 停止；必要时临时降低速度 `--pos-speed 0.03 --ori-speed 0.10`，或加 home 附近工作空间限制。
- SpaceMouse 没反应：确认接收器插在 Jetson 上，`spacenavd` 正在运行，且没有直接运行 ARX5 SDK 的 SpaceMouse 示例抢设备或抢 `can0`。
- 单独机械臂测试前不要运行 `run_leg12_real.sh`。`scripts/run_arm_spacemouse_test.sh` 会直接控制 X5，和 WBC 主节点互斥。
- 程序启动后机器人不动：看到 `Deploy node ready` 后还需要确认 sport mode 已关闭，按 `R1` 起身，等起身完成后按 `L2`。
- `sport_mode state has not been received`：ROS2 sport state 链路不可用，优先查网络和消息包。只有受控诊断时才加 `--allow-unknown-sport-mode`。
- `sport_mode is still active`：先运行 `scripts/disable_sports_mode_go2.sh eth0`。
- `None of the motors are initialized`：`can0` 可能存在，但 X5 电机没有反馈。检查电源、急停、CAN-H/CAN-L/GND、终端电阻、CANable 是否接到 X5 总线、波特率是否为 1Mbps，以及是否误用 `--disable-arm`。用 `ip -s -d link show can0` 看 SDK 运行后是否只有 TX 没有 RX。
- `commands` 非零但狗不动：看 `Policy diag` 里的 `lowcmd_kp`、`lowcmd_leg_q_policy`、`current_leg_q` 和 `leg_q_error`，优先排查低层控制权、sport mode、力矩限制、电池和关节顺序。

## 13. Legacy WBC 安全规则

真机调试时遵守：

- Jetson 上只运行一个 WBC 主节点。X5 写控制也只能有一个进程：默认只允许 `scripts/run_spacemouse_arm.sh` 打开 `can0`。
- 第一次跑新 policy 时，先 `--cmd-vx 0.0` 或 `0.1`。
- 机械臂运动空间内不要放手、线缆和工具。
- 改 X5 电源线前先断电，降压模块输出先用万用表确认约 24V。
- 不要在 sport mode 未关闭时强行跑 lowcmd。
- `max_leg_error` 长期大于 `0.08 rad` 时不要跑动态 policy。
- WBC 默认不写 X5；不要用 `--arm-control-owner wbc`，除非明确做 legacy 回退测试。
- `Y` 会清零底盘，joystick 模式下会 inhibit 到摇杆回中；`R2` 停 policy；`L1` 急停退出。
- 不要一开始就同时改 policy、obs、动作 scale、起身流程和 teleop，先保持变量可控。
- 仅在 legacy WBC 链路中，手柄 `L1` 才是软件急停输入；纯 SportMode 链路忽略所有手柄按键，必须依赖硬件急停。

## 14. Demo 区

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

## 15. 参考文档

- [上机使用指南](doc/上机使用指南.md)：最细的真机操作步骤。
- [260维输入设计](doc/260维输入设计.md)：当前 policy obs 契约。
- [小替换代码清单](doc/小替换代码清单.md)：leg12 + arm passthrough 的改造思路。
- [替换思路](doc/替换思路.md)：如果后续继续换网络，如何选切入层。
- [real-wbc 开发文档索引](real-wbc/docs/README.md)：硬件、网络、装配和开发环境细分说明。
- [网络与通信配置](real-wbc/docs/network.md)：Go2 网络、ROS2、sport mode 和 `can0`。
- [硬件装配说明](real-wbc/docs/assembly.md)：X5 供电、安装、USB-CAN 和外设。
- [3D 打印说明](real-wbc/docs/3d_printing.md)：安装板和历史外设打印件。
