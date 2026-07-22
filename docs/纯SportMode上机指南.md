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

随后 ROS 2 节点会再次关闭并读回避障，同时重复发送 `SwitchJoystick(false)`，直到摇杆回中过一次才接受速度。SDK 没有为其余布尔开关提供 Get API，因此这些项只能检查同步调用成功；避障、UWB 跟随和自动恢复则是显式读回确认。

`StaticWalk`、`TrotRun`、`EconomicGait`、`FreeWalk` 和 `SwitchAvoidMode` 是无 `false` 参数的动作/模式或 toggle API，不能安全地“盲调一次来关闭”；启动前后的 `StopMove` 用于终止正在进行的运动。已在新版 SDK 删除的旧 `ContinuousGait` API 也不会发送。

运行中一旦发现 `/lowcmd` 存在发布者，节点会立即 fail-closed、停止 SportMode 运动并联动机械臂软件急停，防止高低层控制同时存在。

## 启动

上机前确认 Go2 已处于 SportMode 并稳定站立，场地清空，机械臂供电和 `can0` 正常，所有摇杆已松开。不要同时运行 `run_wbc*.py`、`run_leg12_real.sh` 或任何其他 X5 CAN 写进程。

启动脚本要求 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`；如果本机没有安装/构建该 RMW，会直接退出，不会退回到可能无法连接 Go2 的其他 DDS 实现。

一个命令同时启动底盘节点和机械臂节点：

```bash
cd ~/gx-real
scripts/run_sportmode_with_arm.sh eth0 can0
```

参数一是 Go2 网络接口，参数二是 X5 CAN 接口。任一节点退出时，启动脚本会终止另一个节点；底盘节点退出前重复发送 `StopMove`，机械臂节点退出时切入 damping。

如需分开调试，可用两个终端：

```bash
# 终端 A：纯 SportMode 底盘
export GX_REAL_NETWORK_IFACE=eth0
scripts/run_sportmode_wireless.sh

# 终端 B：独立 X5 节点
export GX_REAL_NETWORK_IFACE=eth0
export GX_REAL_REQUIRE_POLICY=0
export GX_REAL_REQUIRE_CRC=0
scripts/run_spacemouse_arm.sh --can-interface can0
```

机械臂保持原有安全约定：先等待底盘节点的 `/safety/heartbeat`，再按 SpaceMouse 双键执行显式使能；异常或安全心跳消失时进入 damping。

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

按 `Ctrl-C` 停止两个节点。纯 SportMode 进程运行期间原厂手柄直通保持关闭，因此不要依赖手柄按键作为本软件链路的急停；必须保证硬件急停始终可触达。若要恢复 Unitree 原厂手柄全部功能，退出本程序后重启 SportMode/机器人控制服务。
