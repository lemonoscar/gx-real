# 网络与通信配置

这份文档覆盖 Go2 局域网、机器人外网、ROS2 通信和 ARX5 `can0`。当前上机流程里，网络问题通常会表现为：

- 收不到 `lowstate`。
- `sport_mode` 状态不可用。
- `disable_sports_mode_go2` 找不到机器人。
- ARX5 初始化失败，或者 `can0` 不存在。

## 1. Go2 局域网

Unitree Go2 默认使用 `192.168.123.xxx` 网段。连接方式通常是：

```text
开发电脑/Jetson 网口 <-> Go2 机身网络
```

检查网卡：

```bash
ip a
ip route
```

你需要确认连接 Go2 的网卡处于同一网段，例如：

```text
192.168.123.10/24
```

如果无法发现机器人，先确认：

- 网线连接正确。
- 网卡 IP 在 `192.168.123.xxx`。
- 没有被 WiFi/其他网卡路由覆盖。
- ROS2 使用的是 Unitree 对应的 CycloneDDS 配置。

## 2. ROS2 环境

进入仓库后：

```bash
cd ~/gx-real
export GX_REAL_NETWORK_IFACE=eth0
source scripts/setup_env.sh
scripts/check_env.sh
```

如果 ROS2 消息正常，`check_env.sh` 应该能通过 `unitree_go` 和 `unitree_api` 的 import/type support 检查。

常用检查：

```bash
ros2 topic list
ros2 topic echo /lowstate
ros2 topic echo /wirelesscontroller
```

如果 topic 不存在或没有数据：

- 先检查 Go2 局域网连接。
- 确认 `scripts/setup_env.sh` 输出 `rmw=rmw_cyclonedds_cpp`，并且 `GX_REAL_NETWORK_IFACE` 是连接 Go2 的网卡。
- 确认没有错误地从 `unitree_sdk2/python` 导入旧消息包。

## 3. 关闭 sport mode

Go2 原厂高层控制和低层 `lowcmd` 不能同时控制电机。真机上机前先执行：

```bash
scripts/disable_sports_mode_go2.sh eth0
```

`eth0` 要换成连接 Go2 的实际网卡。脚本会：

- 检查 `unitree_sdk2`。
- 如果缺少 `build/disable_sports_mode_go2`，自动编译。
- 设置 SDK 动态库路径。
- 调用 Unitree SDK 工具关闭 sport mode。

如果失败，优先检查：

- 网卡名是否正确。
- Go2 是否在 `192.168.123.xxx` 链路上。
- `unitree_sdk2/build` 是否能正常编译。
- 机器人是否已经进入可通信状态。

## 4. 外网连接

Go2 的机身网络经常占用默认路由，例如：

```text
default via 192.168.123.1
```

如果同时接了实验室网线或 USB WiFi，但仍然不能访问外网，可以临时删除 Go2 局域网默认路由：

```bash
sudo route del -net 0.0.0.0 netmask 0.0.0.0 gw 192.168.123.1
```

确认外网：

```bash
ip route
ping 8.8.8.8
```

如果需要每次开机自动处理，可以把删除路由逻辑放到 systemd 服务里。生产上机前建议先手动确认，避免误删实际需要的路由。

## 5. ARX5 SocketCAN

X5 机械臂通过 USB-CAN 转接到 SocketCAN。当前代码默认使用：

```text
can0
```

推荐配置：

```bash
scripts/setup_arx_can.sh
```

如果自动识别失败，显式指定设备：

```bash
scripts/setup_arx_can.sh /dev/ttyACM0 can0 8
scripts/setup_arx_can.sh /dev/serial/by-id/usb-XXXX can0 8
```

检查 `can0`：

```bash
ip -details link show can0
```

正常情况下应该能看到 `can0` 已经 `UP`。如果不存在：

- USB-CAN 没插好。
- 设备名不是 `/dev/ttyACM0` 或 `/dev/ttyUSB0`。
- `slcand` 没启动成功。
- 当前用户没有足够权限执行 `sudo modprobe` / `sudo ip link`。

## 6. 机械臂初始化错误

如果启动时出现：

```text
None of the motors are initialized. Please check the connection or power of the arm.
```

含义是：

- `can0` 已经被 ARX5 SDK 打开。
- 但是 SDK 没有收到任何 X5 电机初始化反馈。

优先检查：

- X5 是否供电。
- X5 急停/开关是否处于可用状态。
- CAN 线是否连接到 X5。
- `can0` 是否对应正确的 USB-CAN。
- 机械臂电机是否完成初始化。

当前代码默认会继续 body-only 运行。若希望机械臂不在线就直接中止，启动时加：

```bash
--require-arm
```

## 7. 常见问题

### `ros2 topic list` 没有 Unitree topic

处理顺序：

1. `source scripts/setup_env.sh`
2. 检查 Go2 网卡 IP。
3. 检查 CycloneDDS 配置。
4. 重启 ROS2 shell 后重试。

### `sport_mode` 状态没收到

当前控制器默认会拒绝低层 rollout。先修复状态链路；只有受控诊断时才使用：

```bash
--allow-unknown-sport-mode
```

### ARX5 的 `can0` 正常但机械臂不动

`can0 UP` 只代表 SocketCAN 接口存在，不代表电机已经在线。继续检查 X5 电源、电机初始化和 CAN 线。
