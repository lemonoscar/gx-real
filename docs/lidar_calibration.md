# Rough LiDAR 安装、外参与验收指南

本文定义 `gx-real` Rough 生产链的 LiDAR 校准和放行流程。它适用于
`PointCloud2 + IMU -> deskew/localization -> self-filter -> elevation mapper -> GridMap`
拓扑，不把 Unitree HeightMap 或直接点云分箱提升为生产输入。

> 当前状态：`policies/rough/current/perception_contract.yaml` 仍为
> `UNVERIFIED`，型号、固件、外参、自滤波和 mapper 配置仍为 `UNSET`。
> 本文给出如何取得这些证据，不代表当前硬件已经通过校准。没有目标机器的实测
> rosbag 和独立复核，不得把合同改为 `VERIFIED`。

## 1. 要校准的对象

需要分别处理三类参数，不能混为一个“LiDAR 已校准”：

1. **传感器内部标定**：LiDAR 激光通道、LiDAR 内置 IMU 偏置/轴向和厂家固件。
   只能使用对应型号、固件的厂家流程；升级固件后需要重新核对。
2. **安装外参**：LiDAR 刚性安装坐标系 `lidar` 到机器人 `base_link` 的 6DoF
   变换。这是本指南的核心。
3. **链路参数**：时间戳/去畸变、self-filter、LIO 和 elevation mapper 配置。
   外参正确但时间不同步或自滤波错误，同样会生成错误的 187D 输入。

安装完成后，支架不得松动、弯曲或有可感知振动。拆装传感器、移动支架、修改
URDF/static TF、升级固件或改变去畸变/mapper 配置，都视为校准失效。

## 2. 坐标系合同

本仓库采用 ROS REP-103 方向：

- `base_link`：x 向前，y 向左，z 向上；原点必须与定位输出所代表的机器人原点一致；
- `lidar`：使用驱动发布点云 header 中的真实 frame 名；如果名称不是 `lidar`，应在
  感知配置和合同中统一修改，不能在多处设置未声明别名；
- `odom`：局部世界/地图 frame；
- `/localization/pose`：`geometry_msgs/msg/PoseStamped`，表示 `base_link` 原点在
  `odom` 中的姿态，不是 LiDAR 在 `odom` 中的姿态；
- `/terrain/elevation_map`：`grid_map_msgs/msg/GridMap`，header frame 与 pose header
  frame 必须相同，生产默认均为 `odom`。

本文将外参记作 `T_base_lidar`，其方向严格定义为：

```text
p_base = T_base_lidar * p_lidar
```

也就是把 LiDAR frame 中的点变换到 `base_link`。用 tf2 检查时，命令为：

```bash
ros2 run tf2_ros tf2_echo base_link lidar
```

不要同时从 URDF、static transform publisher 和 mapper 配置发布三份相同 TF。
整棵 TF 树只能有一个权威父子关系；重复或反向发布常会表现为偶发跳变，而不是
稳定报错。

生产 `grid_map` 路径不会使用 WBC 参数 `--height-scan-extrinsic`。外参必须已经被
deskew/localization/self-filter/mapper 链正确消费；该 WBC 参数只对
`pointcloud2` 诊断 provider 有意义，不能修复生产 GridMap。

## 3. 标定前准备

### 3.1 场地和量具

准备以下条件：

- 可确认水平的硬质平面，至少覆盖机器人前后左右约 2 m；
- 两个相互垂直的竖直平面或墙角；
- 实测高度的刚性台阶，建议使用 0.10 m 标称台阶并记录量具实测值；
- 卷尺/卡尺、水平仪或可信的姿态基准；
- Go2 固定支撑或吊架，确保采集过程中不会自行运动；
- 足够磁盘空间记录原始点云、IMU、TF、定位和 GridMap。

验收数据必须包含一组不参与求解的 held-out rosbag，避免只证明拟合数据本身。

### 3.2 固件、频率和 frame 盘点

先记录 LiDAR 型号、序列号、固件、驱动 commit、mapper commit 和配置文件。然后在
只启动感知、所有 actuator 禁能的状态下执行：

```bash
source scripts/setup_env.sh
ros2 topic info --verbose /lidar/points_deskewed
ros2 topic info --verbose /lidar/imu_raw
ros2 topic info --verbose /terrain/elevation_map
ros2 topic info --verbose /localization/pose

ros2 topic hz /lidar/points_deskewed
ros2 topic hz /lidar/imu_raw
ros2 topic hz /terrain/elevation_map
ros2 topic hz /localization/pose
```

Foxy 的 `ros2 topic echo` 不支持后续发行版的 `--once/--field` 组合。另一个终端用
有限超时检查 frame 和 TF：

```bash
timeout -s INT 3s ros2 topic echo --no-arr /lidar/points_deskewed
timeout -s INT 3s ros2 topic echo --no-arr /terrain/elevation_map
timeout -s INT 3s ros2 topic echo /localization/pose
ros2 run tf2_ros tf2_echo base_link lidar
ros2 run tf2_ros tf2_echo odom base_link
```

也可以一次生成含系统、ROS graph、Unitree LiDAR 状态/固件、topic 频率、样本和 TF 的
只读报告：

```bash
scripts/rough_real_ops.sh probe
```

所有生产消息必须使用传感器/算法产生的有效时间戳。用接收时刻冒充采样时刻会让
deskew 和 30 ms map/pose skew 检查失去意义。

### 3.3 机械初值

机器人保持标准站立几何，测量 LiDAR 原点相对 `base_link` 的 x/y/z，并按支架 CAD
或量具估计 roll/pitch/yaw，形成 `T_base_lidar` 初值。记录：

- 测量基准点和照片；
- 平移单位必须是米、角度在配置中使用弧度；
- 旋转约定和变换方向；
- 测量人员、日期、传感器/支架序列号。

不要把这个机械初值直接当成最终标定值。几毫米平移或不足一度的姿态误差也会在
机器人旋转、坡面和台阶边缘形成系统性高度误差。

## 4. 数据采集

### 4.0 MCF 和执行器边界

正式几何校准在 **MCF 保持活动** 的标准站立姿态下进行。先用 Unitree 官方控制器让
Go2 站立，再增加独立吊架或防倾倒支撑；X5 必须由机械支撑或经过批准的 commissioning
流程固定在 `[0, 0.3, 0.5, 0, 0, 0]`。校准期间禁止启动 Go2 LowCmd writer、X5 CAN
writer，禁止运行 `preflight`、`arm`、`legs` 或手工执行
`disable_sports_mode_go2.sh`。

`calibration-init` 和 `calibration-capture` 使用 MotionSwitcher `CheckMode` 在采集前后
确认仍有活动 motion mode，但不会调用 `ReleaseMode`。它们同时拒绝已存在的 WBC/X5
writer。该检查只能证明采集边界正确，不能替代独立物理支撑。

### 4.1 建议场景

在 actuator 禁能、机器人可靠固定时，至少记录以下独立片段：

1. 水平地面，机器人朝向 0°；
2. 同一位置 yaw 约 +90° 和 -90°；
3. 已知竖直墙分别位于前、后、左、右；
4. 已知高度台阶分别落在扫描网格前/后/左/右区域；
5. 支架允许的轻微 roll/pitch 姿态变化；
6. 机械臂保持 `[0, 0.3, 0.5, 0, 0, 0]`，用于检查 self-filter 是否清除机身/X5
   回波且没有误删地面；
7. 一组独立的 held-out 平地、墙面和台阶片段。

动态采集只能在静态标定通过后进行，用于验证 deskew；不要用动态误差反向掩盖静态
外参错误。

### 4.2 rosbag

根据实际 ROS2 版本确认 `ros2 bag` 可用后，用统一入口记录原始和派生数据：

若 Foxy mapper 尚未确定，可先保存 Unitree onboard LIO 原始/派生数据，不会要求
GridMap：

```bash
GX_REAL_ROUGH_RECORD_DURATION=30 \
  scripts/rough_real_ops.sh record-raw prone_inventory
```

这份趴卧数据可用于盘点 topic、频率、固件和静态噪声，不能替代标准站立几何下的
LiDAR 外参标定。

原装 Unitree 服务可能同时发布 `utlidar_lidar` 下的 `/utlidar/cloud` 和已经内部转换到
`base_link` 的 `/utlidar/cloud_base`，但不发布对应 ROS TF。标准站立平地 bag 可直接用
同时间戳、同点序的两种点云反推出服务实际使用的 `T_base_lidar`，并拟合平地做初检：

```bash
/usr/bin/python3 real-wbc/scripts/analyze_utlidar_extrinsic_bag.py \
  logs/lidar_calibration/20260722-210736_raw_flat_extrinsic_check
```

分析器会在点数、点序或配对残差不能证明对应关系时失败，不会用 ICP 猜测外参；输出的
平地结果仍需结合机器人是否水平、机械量测和独立墙面/台阶数据人工复核。

mapper 启动并通过四个生产 topic/两条 TF 检查后，再记录完整数据：

```bash
GX_REAL_ROUGH_RECORD_DURATION=30 \
  scripts/rough_real_ops.sh record flat_yaw0
```

首次完整校准不要使用散落的普通 `record` 命令，而应建立一个绑定 Git commit、策略、
checkpoint 和感知合同哈希的校准会话：

```bash
cd ~/gx-real-rough-candidate
export GX_REAL_ROUGH_PERCEPTION_SETUP="$HOME/rough_perception_ws/install/setup.bash"
export GX_REAL_OPERATOR_CONFIRM_CALIBRATION_STAND=YES
export GX_REAL_ROUGH_RECORD_DURATION=30

scripts/rough_real_ops.sh calibration-init first_rough_calibration
```

只有在 MCF 正常保持标准站立、机器人有独立支撑、X5 已固定到生产姿态后才设置上面的
确认变量。每次改变场景布置或机器人 yaw 后，等待机器人、LIO 和地图重新稳定，再采集：

```bash
scripts/rough_real_ops.sh calibration-capture first_rough_calibration flat_yaw0
scripts/rough_real_ops.sh calibration-capture first_rough_calibration flat_yaw_p90
scripts/rough_real_ops.sh calibration-capture first_rough_calibration flat_yaw_m90

scripts/rough_real_ops.sh calibration-capture first_rough_calibration wall_front
scripts/rough_real_ops.sh calibration-capture first_rough_calibration wall_rear
scripts/rough_real_ops.sh calibration-capture first_rough_calibration wall_left
scripts/rough_real_ops.sh calibration-capture first_rough_calibration wall_right

scripts/rough_real_ops.sh calibration-capture first_rough_calibration step_front 0.100
scripts/rough_real_ops.sh calibration-capture first_rough_calibration step_rear 0.100
scripts/rough_real_ops.sh calibration-capture first_rough_calibration step_left 0.100
scripts/rough_real_ops.sh calibration-capture first_rough_calibration step_right 0.100

scripts/rough_real_ops.sh calibration-capture first_rough_calibration x5_self_filter
scripts/rough_real_ops.sh calibration-capture first_rough_calibration heldout_flat
scripts/rough_real_ops.sh calibration-capture first_rough_calibration heldout_step 0.100
```

`0.100` 必须替换为量具测得的真实台阶高度，不能只填写标称值。held-out 场景不能参与
外参或 mapper 参数求解。完成后检查采集集合：

```bash
scripts/rough_real_ops.sh calibration-status first_rough_calibration
```

该命令要求每个场景都存在已正常结束、且采集后 MCF 仍活动的 rosbag。它只报告
**采集完整性**，不会自动把 `geometry_review_status` 改为通过，也不会修改
`perception_contract.yaml`。若采集中 MCF 状态丢失，保留该 bag 作为故障证据，但不能
纳入校准通过集合。

普通 `record` 仅用于临时诊断；正式校准使用上面的 session/capture 命令。脚本自动
附加时间戳、有限时停止并执行 `ros2 bag info`。若驱动还提供未去畸变原始点云，建议
一并记录并写明 topic。需要时再执行：

```bash
sha256sum path/to/lidar_extrinsic.yaml \
  path/to/self_filter.yaml \
  path/to/mapper.yaml
```

不要提交大型 rosbag 到 Git；将只读存储位置、bag 哈希、配置哈希和分析报告写入
发布证据。

## 5. 求解 6DoF 外参

使用团队选定、可复现的标定工具或离线优化程序，以机械测量为初值。仓库不绑定某个
尚未随代码交付的第三方标定器，因此不能在文档中假定某条不存在的“一键命令”。求解
至少应约束：

- 水平地面的法向在 `base_link` 中接近 +z；
- 前后墙面法向与 base x 轴一致，左右墙面法向与 base y 轴一致；
- 已知台阶的高度、前后/左右位置和符号正确；
- yaw 改变后，同一世界几何仍在 `odom` 中重合；
- LiDAR IMU 与机器人姿态的时间偏移不会被误吸收到 roll/pitch 外参中。

输出必须保存完整 4×4 齐次矩阵，或平移加规范化四元数；同时保存人可读的
x/y/z/roll/pitch/yaw、单位和变换方向。用新参数重放 held-out bag，不能只报告优化
残差。

## 6. 将标定应用到生产链

1. 在 URDF/static TF 或感知栈唯一配置点写入 `T_base_lidar`；
2. 确认 deskew、LIO、self-filter、mapper 使用同一 TF 树和消息时间戳；
3. self-filter 使用 X5 固定姿态对应的机器人几何，过滤机身/机械臂但不侵蚀地面；
4. mapper 发布 `/terrain/elevation_map` 的 `elevation` layer，frame 为 `odom`；
5. localization 发布同一 `odom` frame 下的 `base_link` pose；
6. 重启整条感知链，避免旧 TF 或旧地图残留；
7. 运行下节验收，全部通过后才更新 perception contract。

不要通过调大 WBC timeout、降低 coverage 或允许 fallback 来“校准”感知。这些参数只会
隐藏问题，并不会改变地图几何。

## 7. 验收

### 7.1 运行时硬门

以下是当前代码直接执行的 Rough motion-permit 合同：

- GridMap source age 不超过 `0.25 s`；
- pose/map stamp 差不超过 `0.03 s`；
- map 与 pose frame 完全相同；
- 全局有效率至少 `0.60`；
- 关键落足区有效率至少 `0.95`；
- 关键区 sentinel cell 为 `0`；
- 连续 `5` 帧有效后才可能获得 motion permit；
- zero、last-valid 或任何 fallback 永远不能给 Rough actor 放行。

用无 actuator 的监视器检查：

```bash
/usr/bin/python3 real-wbc/scripts/run_height_scan_monitor.py \
  --source grid_map \
  --topic /terrain/elevation_map \
  --pose-topic /localization/pose \
  --map-layer elevation \
  --contract policies/rough/current/height_scan_contract.yaml \
  --timeout 0.25 \
  --min-valid-ratio 0.60 \
  --min-critical-valid-ratio 0.95 \
  --max-critical-sentinel-cells 0
```

日志应持续显示 `shape=187`、`ok=True`、`fallback=False`、`height_source=grid_map`、
正确的 map/pose frame 和满足合同的 age/coverage。一次 `ok=True` 不算通过，应在每个
标定场景与 held-out bag 中统计分布和故障次数。

### 7.2 几何放行建议

下列数值是本项目建议的首次发布验收门，不是当前仓库已取得的实测结果。若硬件团队
采用更严格阈值，应在发布报告中记录；若要放宽，必须给出风险分析和独立复核：

| 测试 | 建议门限 |
| --- | --- |
| 平地中心区域高度偏差 | 绝对中位数不超过 0.02 m |
| 平地中心区域起伏 | p95 - p05 不超过 0.03 m |
| 实测约 0.10 m 台阶 | 重建高度误差不超过 0.02 m |
| 台阶方向 | 前/后/左/右与真实布置一致，不允许镜像或转置 |
| yaw ±90° 重复采集 | 世界几何保持重合，不能随机器人产生固定旋转偏移 |
| 静止地图 | 无周期性随 LiDAR 转动的高度波纹或机身/X5 ghost |
| held-out 数据 | 同样满足全部门限，且未参与外参求解 |

还必须人工检查以下符号测试：

- 前方台阶只影响正 x 对应网格；后方只影响负 x；
- 左侧台阶只影响正 y；右侧只影响负 y；
- 机器人原地 yaw 后，map/world 几何不旋转漂移，而 actor 网格随 base yaw 正确取样；
- actor 数值遵循 `clip(base_z - elevation - 0.5, -1, 1)`：更高地形使数值更小；
- GridMap layer 不发生 x/y 转置，circular-buffer 滚动后场景不跳格。

### 7.3 时间和动态验收

静态几何通过后，在低速人工推行或安全平台上验证 deskew。报告至少包含 map/pose
频率、source age 和 skew 的 p50/p95/p99，以及丢帧、覆盖率和 CPU/GPU 峰值。运行时
硬门只是单帧上限；发布报告应证明长时间运行有余量，而不是刚好碰线。

## 8. 更新 perception contract 与发布证据

验收完成后，将真实值写入
`policies/rough/current/perception_contract.yaml`：

- `calibration.lidar_model`：精确型号；
- `calibration.lidar_firmware`：真实固件版本；
- `calibration.lidar_to_base_extrinsic`：外参配置路径、变换方向和 SHA-256，或稳定的
  证据标识；
- `calibration.self_filter`：实现、配置路径/commit 和 SHA-256；
- `mapping.implementation`：LIO/elevation mapper 实现与 commit；
- `mapping.configuration_hash`：参与生产的完整配置 SHA-256；
- `evidence`：bag 标识/哈希、验收报告、测试机器和复核人。

只有独立复核确认上述数据和 held-out 结果后，才把 `verification_status` 改为
`VERIFIED`。随后重新计算 perception contract 的 SHA-256，更新 Rough artifact
manifest，并按仓库 README 的 manifest-only release 流程发布。修改任何校准或 mapper
配置后，旧哈希和旧 `VERIFIED` 状态立即失效。

## 9. 何时必须重新标定或回滚

出现以下任一条件立即撤销 Rough 放行，恢复 `UNVERIFIED` 或回滚到最后一套经过复核的
完整感知发布包：

- LiDAR/支架拆装、碰撞、松动或明显振动；
- 型号、序列号、固件、驱动、时间同步方式改变；
- URDF、static TF、self-filter、LIO 或 mapper 配置改变；
- 平地出现系统偏差、墙面倾斜、台阶镜像/转置或 yaw 相关漂移；
- age/skew/coverage 经常触发门限；
- X5 固定姿态产生未过滤 ghost，或 self-filter 误删关键落足区；
- 不能复现合同中记录的配置 SHA、bag 或 held-out 报告。

回滚必须成套恢复外参、TF、self-filter、mapper 和 perception contract，不能只替换其中
一个文件。
