# 纯 SportMode 上机指南

这条链路是当前推荐的 `Go2 + X5` 手动控制模式。它不加载 ONNX 策略、不运行 WBC、不发布 `lowcmd`，四足和机械臂由两个独立节点控制：

- `sportmode_wireless_node`：只读取 `/wirelesscontroller` 的摇杆轴，并向 `/api/sport/request` 发送 `Move(vx, vy, yaw)` 或 `StopMove`。
- `spacemouse_arm_node`：独占 `can0`，使用 SpaceMouse 控制 X5，并发布 `/arm/state` 与 `/arm/target_state`。

## 控制约定

- 左摇杆前后 `ly` → 前后速度 `vx`，默认上限 `0.30 m/s`。
- 右摇杆左右 `rx` → 偏航速度 `yaw`，默认上限 `0.30 rad/s`。
- 侧向速度 `vy` 默认上限为 `0`，因此左摇杆左右不会产生运动。
- 无线手柄的所有按键都被忽略，不触发起身、趴下、姿态、步态、特技或机械臂动作。
- 启动和无线消息超时后，必须先把所有摇杆回中，节点才会再次接受运动命令。
- 无线消息超过 `0.25 s` 未更新时，节点持续发送 `StopMove`。

启动脚本先通过仓库内的 Unitree C++ SDK 同步执行以下设置；任一 SDK 返回码非零都会拒绝启动：

1. 前后各执行一次 `StopMove`。
2. `obstacles_avoid=false`，再读回并确认为 `false`。
3. `Utrack/UWB follow=false`，读回开关并确认 `IsTracking=false`。
4. `SwitchJoystick(false)` 和 `Pose(false)`。
5. 扩展 Sport API 的 `HandStand`、`FreeBound`、`FreeJump`、`FreeAvoid`、`ClassicWalk`、`WalkUpright`、`CrossStep` 全部设为 `false`。
6. `AutoRecovery=false`，再读回并确认为 `false`；这避免跌倒后固件自主起身，但也意味着不再具备自动恢复能力。
7. 通过 VUI 将灯光亮度设置值设为 `0`，再读回并确认为 `0`。

随后 ROS 2 节点会再次关闭并读回避障，同时重复发送 `SwitchJoystick(false)`。节点还会每秒通过 VUI 重新设置亮度为 `0` 并读回确认，防止其他服务改回该设置值。避障未确认关闭或 VUI 亮度未确认为 `0` 时，不会进入可运动状态；全部确认完成且摇杆回中过一次后才接受速度。SDK 没有为其余布尔开关提供 Get API，因此这些项只能检查同步调用成功；避障、UWB 跟随、自动恢复和 VUI 亮度则是显式读回确认。

注意：`GetBrightness()==0` 只能证明 VUI 设置值已归零，不能证明固件高优先级状态灯必然物理熄灭。关闭避障时的蓝色指示是否受 VUI 亮度控制，必须在对应固件的真机上目视确认。如果读回为 `0` 但蓝灯仍亮，公开 SDK 没有能够可靠关闭该状态灯的接口。

`StaticWalk`、`TrotRun`、`EconomicGait`、`FreeWalk` 和 `SwitchAvoidMode` 是无 `false` 参数的动作/模式或 toggle API，不能安全地“盲调一次来关闭”；启动前后的 `StopMove` 用于终止正在进行的运动。已在新版 SDK 删除的旧 `ContinuousGait` API 也不会发送。

运行中一旦发现 `/lowcmd` 存在发布者，机器狗节点会立即 fail-closed 并停止 SportMode 运动。机器狗节点正常退出时发布 `STOPPING`，机械臂收到后先回到固定关节位置 `[0, 0.3, 0.5, 0, 0, 0]`，再进入 damping 并退出。机械臂自身异常只会让机械臂阻尼并退出，不会停止机器狗。

## 启动

上机前确认 Go2 已处于 SportMode 并稳定站立，场地清空，机械臂供电和 `can0` 正常，所有摇杆已松开。不要同时运行 `run_wbc*.py`、`run_leg12_real.sh` 或任何其他 X5 CAN 写进程。

启动脚本要求 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`；如果本机没有安装/构建该 RMW，会直接退出，不会退回到可能无法连接 Go2 的其他 DDS 实现。

必须使用两个终端分别启动，且先启动机器狗：

```bash
# 终端 A：机器狗
cd ~/gx-real
export GX_REAL_NETWORK_IFACE=eth0
scripts/run_sportmode_wireless.sh

# 等终端 A 显示 SPORTMODE_ACTIVE 后，打开终端 B：机械臂
cd ~/gx-real
export GX_REAL_NETWORK_IFACE=eth0
scripts/run_spacemouse_arm.sh --can-interface can0
```

`scripts/run_sportmode_with_arm.sh` 已停用，调用时只会打印双终端指令并退出。机械臂只有在底盘心跳为 `SPORTMODE_ACTIVE` 时才接受 SpaceMouse 双键显式使能；机械臂硬件初始化完成后 5 秒仍未收到首个机器狗心跳时，会进入 damping 并异常退出。

## 可调参数

默认只建议调整速度上限、死区和方向符号：

```bash
scripts/run_sportmode_wireless.sh \
  --joy-max-vx 0.30 \
  --joy-max-yaw 0.30 \
  --joy-deadzone 0.12
```

硬限制为 `vx <= 0.30 m/s`、`vy <= 0.20 m/s`、`yaw <= 0.30 rad/s`，超过限制时节点拒绝启动。保持 `--joy-max-vy 0.0` 即为“只前后速度 + 转向”模式。

## 停止与恢复

正常停机时，只需在终端 A 对机器狗节点按 `Ctrl-C`：

1. 机器狗停止速度命令并发布 `STOPPING`。
2. 机械臂先回到 `[0, 0.3, 0.5, 0, 0, 0]`，进入 damping 后退出。
3. 机器狗等待机械臂退出，然后调用 SportMode `StandDown` 进入趴卧姿态；过渡速度由 Go2 固件控制，SDK 接口本身没有速度参数。

单独在终端 B 停止机械臂时，机械臂正常回固定位置后退出，机器狗保持运行。机械臂自身发生 CAN、反馈、SpaceMouse 或安全门控异常时，不执行主动回位，而是立即 damping 并退出；机器狗不受影响。

机器狗节点故障或心跳消失时，机械臂会按故障路径立即 damping 并退出，不会在底盘状态不可信时主动回位。

`SIGKILL`、断电、CAN 中断或进程崩溃无法保证回位或趴卧动作。纯 SportMode 进程运行期间原厂手柄直通保持关闭，因此不要依赖手柄按键作为本软件链路的急停；必须保证硬件急停始终可触达。
