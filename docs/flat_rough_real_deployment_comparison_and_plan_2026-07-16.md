# gx-real Flat / Rough 双真机部署方案

> 日期：2026-07-16
> 范围：参考 `phase_guided_terrain_traversal` 的真机感知与联调链路，修正 `gx-real` 的腿部策略部署边界、height scan、X5/传感器接入、通信拓扑和安全门。
> 本文是实施与验收方案，不代表当前硬件已经获准输出。

## 1. 结论

`gx-real` 应当分成两个互斥的真机部署类，而不是继续用同一个节点上的 `--enable-height-scan` 开关来区分策略：

- `FlatDeployment`：只接受 flat 策略资产；260 维观测中的 `[66:253]` 必须由代码直接生成 187 个精确的 `0`；不创建、不读取 height provider；LiDAR 或建图故障不得影响 flat 策略。
- `RoughDeployment`：只接受 rough 策略资产；`[66:253]` 必须来自与训练 RayCaster 契约等价的实时 187 维高度观测；LiDAR、位姿、TF、局部高程图、新鲜度和覆盖率全部进入运动许可条件；不得在感知失效时悄悄退化成全 0。

两类部署共享同一个经过安全审计的 WBC/LowCmd 核心，但使用两个独立入口、两份发布清单、两套观测合同和不同的设备前置条件。运行中禁止热切换；切换策略类别必须先停止、释放硬件 writer、重新校验资产并建立新的 session。

当前仓库还不能直接部署 rough 策略，原因不是只有“height scan 没接上”：当前策略包本身是 flat 环境快照，rough 合同与它混放；现有检查器仍会返回 `PASS`；rough 感知链未形成可启动、可监控、可 fail-closed 的闭环；X5 运行方式与固定机械臂训练条件也可能不一致。同时，发布清单是 `UNRELEASED`，Go2 最终指令安全合同是 `UNVERIFIED`，当前生产入口会正确拒绝硬件输出。

## 2. 已确认的当前事实

### 2.1 当前 `policies/` 是 flat 资产，不是可发布的 rough 资产

当前 [`policies/env.yaml`](../policies/env.yaml) 明确包含：

- `terrain_type: plane`；
- `height_scanner: null`；
- policy 和 critic 的 height term 都是 `_zero_height_scan`；
- 观测维度仍为 260，其中 `[66:253]` 只是为了维持接口尺寸而保留的 187 个零。

但同目录 [`height_scan_contract.yaml`](../policies/height_scan_contract.yaml) 又把任务标成 `RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0`。这不是完整 rough 发布包，而是两种语义被拼在了一起。

本地验证还发现：

- ONNX 接口是 `obs [1,260] -> actions [1,12]`，模型没有足以辨别策略类别的元数据；
- 现有 `check_policy_height_scan_contract.py` 对这组“flat env + rough 标签”仍返回 `PASS`；它目前只证明形状、基础数值和推理时延可用，不能证明策略类别或高度语义正确；
- 对当前策略强行注入非零高度会显著改变动作。全 1 高度切片的动作变化接近 1 rad 量级，合成 0.15 m 台阶也产生约 0.17 rad 的变化。即使 flat 训练时高度恒零，网络中未受训练约束的相关权重仍可能对非零输入任意响应，所以绝不能把 live scan 接到 flat policy 上试跑。

### 2.2 当前硬件输出被两道门正确阻断

[`config/artifact_manifest.yaml`](../config/artifact_manifest.yaml) 是 `UNRELEASED`，且其训练/运行版本信息没有形成完整可追溯发布；[`config/go2_leg_safety_contract.yaml`](../config/go2_leg_safety_contract.yaml) 是 `UNVERIFIED`，关节范围、每周期步长、速度、加速度、jerk 和时效阈值均未填写。

实际调用验证得到：

- 发布清单校验报错：`artifact manifest is not RELEASED`；
- 最终腿部安全合同校验报错：`hardware output remains blocked`。

这两处 fail-closed 行为应保留。修复 rough 感知不能绕过发布和最终指令安全门。

### 2.3 当前基础测试状态

与本方案直接相关的 height scan、策略合同、最终指令安全和发布清单测试为 `30 passed, 1 skipped`；hardware writer inventory 检查通过。它说明现有基础模块可作为重构起点，但不证明 rough 真机链闭环已经成立。

## 3. 与 phase_guided_terrain_traversal 的完整对比

| 项目 | phase_guided_terrain_traversal | gx-real 当前状态 | gx-real 应采取的做法 |
|---|---|---|---|
| 策略接口 | PGTT actor 153D，动作 12D | 260D -> 12D | 保持 GX 的 260D，不复用 PGTT 观测拼接 |
| 高度尺寸 | 11×9 = 99 | rough 合同 17×11 = 187；当前 flat 为 187 个 0 | Flat 固定 0；Rough 严格 17×11 和导出坐标表 |
| 高度语义 | 世界/局部高程减去窗口最小值，旧部署再乘 1.5；高台阶趋向正值 | Isaac RayCaster 等价的垂直距离减 offset；高台阶通常使对应值更负 | 只复用感知拓扑，不复制 PGTT 的减最小值、符号或倍率 |
| 网格顺序 | C-order 11×9 | 合同声明 17×11、`xy`/row-major，但必须由仿真样本实证 | 用导出的 `x/y` 坐标数组作为唯一真值，做方向和 flatten 测试 |
| 感知设备 | Unitree LiDAR + LiDAR IMU | GX 文档/BOM 尚未把 LiDAR 链定义为 rough 的必选设备 | Rough 明确加入 LiDAR、IMU、安装支架、供电和标定资产 |
| 定位/建图 | transform -> Point-LIO -> elevation_mapping -> filter -> heightmap | 有 pointcloud/HeightMap provider 和监视器，但没有完整生产启动链 | 采用独立的 LIO + 2.5D elevation map 进程，再由 sampler 生成 187D |
| 时间处理 | 部分节点使用 latest TF；PointCloud2 header 和 PID 管理存在缺口 | provider 主要以回调到达时间判断新鲜，TF 也取 latest | 统一用消息 header stamp，在同一时刻查询 TF，校验 map/pose skew |
| 无效格处理 | 越界/NaN 被写成 0，0 与真实平地不可区分 | rough provider 也可在失败时返回 last-valid/0 | Rough 内部保留 validity mask；不满足条件时撤销运动许可，不把未知伪装成 0 |
| 原始点云采样 | 先建高程图再采样 | GX 支持直接把点云按格子取 20% 分位 | 直接点云仅保留为诊断；生产 rough 使用重力对齐、时序累计的高程图 |
| Go2 通信 | 直接 Unitree SDK 写 `rt/lowcmd`，状态 `rt/lowstate` | ROS2/CycloneDDS WBC，有 writer lock、状态机、estop、最终门 | 保留 GX 的控制/安全骨架，不复制 phase 的直接 writer |
| PD 参数 | 旧真机代码约 60/3；新 PGTT 导出合同又是 40/0.5 | GX 启动默认约 200/10；仿真 actuator 约 32/0.8 | 每个策略发布包记录并验证自己的实机增益，禁止从 phase 抄数值 |
| 动作尺度 | 默认腿姿 + action×0.5 | 当前 flat env 是 hip 0.18、thigh/calf 0.32；rough 源码通常为 0.25 | Flat/Rough 各自从确切 checkpoint 的 env 快照读取，不共享猜测值 |
| X5 | PGTT Go2 策略明确不把 X5 放入 actor | GX 的 260D 观测包含 X5 状态/目标，输出仍只有 12 条腿动作 | X5 必须有独立 writer，并严格复现各策略训练时的固定姿态/命令 |
| 启动方式 | 两终端先感知后策略，带 RViz 检查 | `prepare_real_run.sh` 只检查 Go2/arm 主链，未把 rough 感知设为启动门 | Flat 和 Rough 两个 launcher；Rough 增加感知预检、稳定窗口和运动许可 |
| 进程健壮性 | 脚本运行时编译、后台 PID 捕获不完整、异常传播不足 | GX 控制链更完整，但 rough 感知没有统一 supervisor | 借鉴顺序，不复制脚本；构建和运行分离，任一必需进程退出即撤销许可 |

### 3.1 phase 中值得借鉴的部分

- 把 LiDAR 驱动、运动补偿/定位、高程融合、固定尺寸采样和策略执行拆成可单独观察的阶段；
- 策略启动前先在 RViz/监控器中确认局部地图随机器人稳定移动；
- 由机器人位姿把固定 body-yaw 网格投影到世界高程图，而不是把瞬时稀疏点直接当策略输入；
- 感知终端先启动、稳定后再允许策略进入主动控制。

### 3.2 phase 中不能直接复用的部分

- `deploy_real.py` 的 153D 拼接、phase/frequency 项、11×9 网格、`height - min(height)` 和 `×1.5`；
- 硬编码的传感器姿态、用户目录标定文件、LiDAR 外参和 topic；
- 无 timestamp/shape/finite/freshness 完整约束的 PointCloud2 flatten；
- 越界或 NaN 直接填 0 的处理；
- 旧 Kp/Kd、默认腿姿、动作尺度和直接 `LowCmd` writer；
- `run_elevation.sh` 的运行时编译、后台 PID 管理及 latest-TF 用法。

换言之，复用的是系统分层和联调经验，不是其神经网络合同或硬件参数。

## 4. 双部署类设计

### 4.1 不复制两份 WBC

建议保留一个共享的 `Leg12WbcCore`，用组合方式注入两个不可变部署类：

```python
class DeploymentProfile(Protocol):
    kind: Literal["flat", "rough"]

    def validate_bundle(self, bundle: ReleaseBundle) -> None: ...
    def required_inputs(self) -> tuple[InputRequirement, ...]: ...
    def height_observation(self, now: float) -> HeightObservation: ...
    def motion_permit(self, now: float) -> PermitResult: ...


class FlatDeployment(DeploymentProfile): ...
class RoughDeployment(DeploymentProfile): ...
```

这样能满足“两个真机部署类”，同时避免把 LowCmd、急停、关节顺序、arm 状态和 watchdog 复制成两套逐渐分叉的实现。

### 4.2 FlatDeployment 的硬约束

- 只加载 `policy_kind: flat` 的发布包；
- 合同必须声明训练 height term 为 `_zero_height_scan`，terrain/scene 中 scanner 为空；
- `height_observation()` 每个周期新建或返回只读的 `np.zeros(187, float32)`；
- 启动时对 187 个值做 bit-exact/`count_nonzero == 0` 断言；
- 不实例化 `HeightScanProvider`，即使系统中存在 LiDAR topic 也不订阅；
- 如果用户向 flat 入口传入 height source、map topic 或 rough contract，直接拒绝启动；
- LiDAR/LIO 故障不进入 flat 的运动许可，但 Go2、X5、arm 观测、policy deadline 和通用安全门仍必须健康。

### 4.3 RoughDeployment 的硬约束

- 只加载 `policy_kind: rough` 的发布包；
- env 快照必须声明 live height observation，不能包含 `_zero_height_scan`；
- 必须加载 17×11 合同、仿真参考样本、LiDAR 外参、高程图配置和感知时序合同；
- 未得到连续 N 帧健康 scan 前，状态机不能从 READY 进入 ACTIVE；
- 每帧必须验证 source stamp、TF stamp、pose/map skew、finite、形状、网格顺序、全局覆盖率和关键落足区覆盖率；
- 感知短暂断流时只能在一个经实测确认的很短窗口内使用 last-valid，同时将速度命令平滑降到 0；超过窗口进入 STOPPING/FAULT；
- 永远不允许用 187 个 0 作为 rough 的“正常 fallback”；
- `pointcloud2` 直接分箱和 Unitree HeightMap 都只能用于 monitor/offline comparison；生产模式固定为 `grid_map_msgs/msg/GridMap` 的 `elevation` layer。

### 4.4 两个独立入口

建议新增：

```text
scripts/run_leg12_flat_real.sh
  -> real-wbc/scripts/run_wbc_flat.py
  -> FlatDeployment + shared Leg12WbcCore

scripts/run_leg12_rough_real.sh
  -> scripts/prepare_rough_perception.sh
  -> real-wbc/scripts/run_wbc_rough.py
  -> RoughDeployment + shared Leg12WbcCore
```

生产入口不再暴露通用 `--enable-height-scan`。模式由入口和签名发布包共同决定，不能由一个容易误触的布尔参数改变。

## 5. 策略资产和发布目录必须拆开

推荐目录：

```text
policies/
  flat/<release-id>/
    policy.onnx
    env.yaml
    observation_contract.yaml
    action_contract.yaml
    artifact_manifest.yaml
    reference_observations.npz
  rough/<release-id>/
    policy.onnx
    env.yaml
    observation_contract.yaml
    action_contract.yaml
    height_scan_contract.yaml
    height_scan_reference.npz
    perception_contract.yaml
    artifact_manifest.yaml
```

每份 manifest 至少新增并 hash：

- `policy_kind: flat | rough`；
- 训练仓库 commit、checkpoint 路径/哈希、导出脚本 commit；
- ONNX、原始 env 快照、260D slice 表、动作缩放、默认关节姿态和关节顺序；
- X5 训练模式、固定姿态、arm command/gripper 的生成方式；
- 实机 Kp/Kd、ready pose offset 及其验证证据；
- rough 专属的网格坐标数组、clip、offset、flatten 顺序、仿真 reference NPZ；
- rough 专属的 LiDAR 型号、固件、外参、LIO/elevation/sampler 配置和 topic/frame；
- CycloneDDS、Unitree/ARX SDK 和最终腿部安全合同；
- Python/ONNX Runtime/RMW 版本。

当前 `.gitignore` 会忽略多数 `policies/*` 文件，而主 manifest 又没有 hash height contract/reference/extrinsic。实现时应选择一种明确发布机制：Git LFS 或独立只读 artifact store 均可，但发布清单和所有 hash 必须进入版本控制，不能依赖某台 Jetson 上的散落文件。

## 6. 260D 观测合同

两类策略共享外层形状，但不能因此认为它们语义相同：

| Slice | 维度 | 含义 |
|---|---:|---|
| `[0:3]` | 3 | base linear velocity |
| `[3:6]` | 3 | base angular velocity |
| `[6:9]` | 3 | projected gravity |
| `[9:12]` | 3 | velocity command |
| `[12:30]` | 18 | Go2 12 + X5 6 joint position |
| `[30:48]` | 18 | Go2 12 + X5 6 joint velocity |
| `[48:66]` | 18 | previous actions/对应 padding |
| `[66:253]` | 187 | Flat: 全 0；Rough: live 17×11 height scan |
| `[253:259]` | 6 | arm target/command |
| `[259:260]` | 1 | gripper |

部署导出器不应继续把这些索引硬编码后直接声称合同成立，而应从确切 checkpoint 的 env manager term 顺序、维度和 modifier/scale 中导出，并用一批仿真运行时观测逐项回放验证。

特别需要注意：当前工作区中的 rough 环境源码已有默认姿态/动作参数演化，而现有 policy/env 资产来自更早版本。重新导出 rough policy 时，必须使用 checkpoint 当时保存的 env 快照，不能拿今天工作树中的配置推断旧 checkpoint 合同。

## 7. Rough height map 的正确处理方案

### 7.1 目标语义

Rough 的 187 维值必须与 Isaac Lab 训练时的 RayCaster observation 等价，而不是“看起来像地形高度”即可。其核心可表达为：在 base-yaw 网格各射线位置，取重力方向的地表交点，计算射线原点/机器人参考高度到地面的垂直距离，再按训练 offset、scale、clip 处理。

在当前合同下，平地且机器人参考高度约 0.5 m 时应接近 0；前方地面抬高 0.10 m 时，相应格通常应约为 -0.10，而不是 +0.10。最终公式、参考原点和符号必须由训练 observation 函数与导出的仿真样本共同锁定。

### 7.2 生产感知流水线

```text
Unitree LiDAR points + LiDAR IMU
        |
        v
timestamp / calibration / self-filter / deskew
        |
        v
Point-LIO or equivalent gravity-aligned odometry
        |                         |
        |                         +--> map/odom -> base_link pose at source stamp
        v
local 2.5D elevation map + variance/validity layers
        |
        v
17×11 base_yaw sampler
        |
        +--> values[187] + valid_mask[187] + source_stamp + diagnostics
        |
        v
RoughDeployment motion-permit gate
        |
        v
shared 260D observation builder
```

建议从 phase 工程迁移/重写 LiDAR 驱动、Point-LIO 和 elevation mapping 的配置，但把它们放入 `gx-real` 自己的 perception workspace/容器并固定版本。构建在镜像或安装阶段完成，生产启动脚本只负责运行，不现场 `colcon build`。

### 7.3 坐标系合同

至少定义并校验以下 frame：

- `map` 或 `odom`：重力对齐的高程世界坐标；
- `base_link`：Go2 机体全姿态坐标；
- `base_yaw`：仅保留 yaw 的虚拟采样坐标；
- `lidar`：标定后的 LiDAR 坐标；
- LiDAR IMU frame：只用于 LIO，不能与 Go2 policy IMU 无说明地混用。

每个 height scan 都必须携带原始 map stamp，并在同一个 stamp 查询 `map -> base_link`。生产代码禁止用 `Time(0)`/latest TF 代替同步。若 TF 插值失败、map 与 pose 时间差超限、时钟倒退或 header stamp 过旧，则该帧无效。

LiDAR 到 base 的 6DoF 外参必须来自可复现标定文件并纳入 manifest；phase 中硬编码旋转、桌面路径 calibration 和异常后静默使用默认值的方式不能进入生产。

### 7.4 网格与采样

- 网格固定 17×11，覆盖 1.6 m × 1.0 m，分辨率 0.1 m；
- 不能只根据 shape 猜测轴顺序。发布包应保存 187 对 `(x_i, y_i)`，sampler 按该数组逐点采样；
- 高程图分辨率应优于策略网格；当前生产 sampler 有意采用 phase `getIndex`/grid_map_core 的 cell-index 语义，以避免引入未经训练和验证的插值。若未来改用双线性或受方差约束的局部平面插值，必须作为新的感知合同重新做 reference/rosbag parity；
- 局部高程图范围必须覆盖旋转后的完整网格并留滤波边界，phase 的 1.5 m 地图不足以直接承载 1.6 m 的 GX 网格；
- 保留 `valid_mask`、方差、观测年龄和每格来源。只在有明确半径和方差上限时插值，不能把未知格直接写成 0。

必须增加四类方向性单元测试：X 正坡、Y 正坡、左/右单侧台阶、yaw 90°。这些测试同时锁定 x/y、左右、前后、符号和 flatten 顺序。

### 7.5 为什么不把瞬时原始点云作为生产输入

当前 `points_to_height_scan()` 对每格使用 z 的 20% 分位值。一个格子同时含有低处地面、台阶踏面、立面和 X5/腿部回波时，这个统计量不等价于从上向下的 RayCaster 地表交点，可能忽略抬高的踏面。瞬时点云还缺少重力对齐、运动去畸变、时间累计和完整覆盖。

因此：

- raw point cloud 模式用于安装检查、外参调试和与 elevation-map 的对照；
- rough 发布入口只接受 2.5D elevation-map sampler；
- 必须加入 Go2 腿、X5、线缆和 LiDAR 支架的 self-filter/遮挡 mask；
- 若未来确实要发布 direct-cloud 版本，它应成为第三种独立感知合同并重新训练/验证，而不是替换 rough 的输入源。

### 7.6 感知失效行为

Flat 的 0 是训练语义；Rough 的 0 是“机器人位于标准高度的平地”语义。对 Rough 来说，用 0 表示未知会把传感器故障伪装成确定的平地。

建议状态机行为：

1. 启动阶段：连续若干帧满足 age、skew、coverage、finite 和姿态条件后才建立 `rough_perception_ready`；
2. 运行中短丢帧：在经实测确定的极短窗口内使用 last-valid，同时将速度命令斜坡降到 0；
3. 超过短窗口：进入 `STOPPING`，保持受控姿态并禁止新运动命令；
4. 持续失效或关键落足区未知：锁存 fault，需要操作者确认后才能重新 READY；
5. 物理急停始终独立，不依赖 ROS2 或策略进程。

现有约 0.5 s 的 last-valid 窗口对台阶运动可能过长：0.5 m/s 时机器人已移动 0.25 m。建议先以 100 ms 量级作为测试起点，并根据 rosbag 的 p95/p99 时延和受控停车试验确定最终阈值；最终值写入 rough perception contract，而不是散落在代码默认参数里。

## 8. 其他设备的接入

### 8.1 Rough 必选设备

- Unitree L1/L2 或最终选定型号的 3D LiDAR；
- 与该 LiDAR 配套、可用于去畸变/LIO 的 IMU；
- 刚性支架、防松紧固、已知视场和可复测外参；
- Jetson Orin NX/当前计算机及稳定供电；
- Go2 Ethernet/CycloneDDS；
- X5 的 24V 电源、USB-CAN/can0 和独立 owner；
- 无线手柄作为 enable/软件急停输入；
- 独立的物理急停/驱动电源切断回路。

MoCap/iPhone 可以在标定和 ground-truth 评估阶段使用，但不应成为 rough 生产运行的隐藏依赖。

### 8.2 X5 的正确角色

当前 260D policy 输出只有 12 条腿命令，但输入包含 6 个 X5 关节状态、速度、目标和 gripper。因此 X5 不是可以忽略的“附加设备”：实际姿态偏离训练分布会改变腿策略动作。

DogOnly rough 训练通常锁定机械臂。对应真机应优先新增 `x5_fixed_hold` owner：

- 独占 `can0`；
- 把 X5 保持在该发布包记录的精确训练姿态；
- 发布新鲜的 `/arm/state` 和 `/arm/target_state`；
- WBC 只读，不写 can0；
- arm 状态过旧、误差超限或 CAN owner 丢失时，撤销腿部运动许可。

当前生产链要求 SpaceMouse arm writer，这与固定臂训练条件存在冲突。除非某个具体策略在相同的机械臂运动分布下训练并发布了对应合同，否则 SpaceMouse 只能作为调试/人工回收工具，不能在 rough 行走时任意移动 X5。Flat 也应从它自己的 env 快照决定 fixed-hold 还是其他 arm mode，不能全局假设。

### 8.3 两套 IMU 的用途必须分开

- Go2 lowstate IMU：给 policy 的 angular velocity/projected gravity，并参与机体状态估计；
- LiDAR IMU：给点云去畸变和 LIO；
- 两者需要时间基准和安装外参，但不能把一个 topic 直接替换另一个而不重新验证观测合同。

## 9. 完整真机通信拓扑

```text
                         +----------------------+
Go2 lowstate ----------->|                      |
wirelesscontroller ----->|  shared Leg12 WBC    |---- single writer ----> Go2 lowcmd
sportmodestate ---------->|  + Flat/Rough profile|
/arm/state -------------->|                      |
/arm/target_state ------->|                      |
                         +----------^-----------+
                                    |
                        Rough only: values + validity + stamp
                                    |
LiDAR points/IMU -> deskew/LIO -> elevation map -> 17×11 sampler
        |                 |              |               |
        + raw health      + pose/TF      + variance      + scan diagnostics

X5 USB-CAN <---- exclusive x5_fixed_hold owner ----> /arm/state, /arm/target_state

physical E-stop ---------------------------> actuator power / independent stop
ROS estop + safety heartbeat + owner locks -> WBC and X5 state machines
```

建议标准化 rough ROS2 topic：

| 功能 | 建议 topic/frame | 必须检查 |
|---|---|---|
| LiDAR 原始点 | `/lidar/points_raw`, frame `lidar` | rate、stamp、finite |
| LiDAR IMU | `/lidar/imu_raw` | rate、stamp、饱和/跳变 |
| 去畸变点云 | `/lidar/points_deskewed` | frame、latency |
| LIO 位姿 | `/localization/odom` + TF | gravity alignment、covariance |
| 局部高程图 | `/terrain/elevation_map` (`grid_map_msgs/msg/GridMap`) | `elevation` layer、列主序、circular index、resolution、stamp、coverage |
| Rough scan | `/terrain/height_scan` | 187 values + valid mask + source stamp |
| Rough 状态 | `/terrain/height_scan_status` | age、skew、coverage、fault reason |
| X5 状态/目标 | `/arm/state`, `/arm/target_state` | age、mode、tracking error |
| 运动许可 | `/safety/motion_permit` | mode/session/reason |

如果沿用 phase 的旧 topic（如 `/utlidar/cloud`），也必须在单一配置中显式映射；禁止同一链路同时出现 `/unilidar/cloud`、`/utlidar/cloud`、`base`、`base_link` 等未声明别名。当前 gx-real 不同脚本的默认 topic/frame 已经存在这种不一致，应在 rough contract 中收口。

## 10. 启动、运行和停止顺序

### 10.1 Flat

1. 验证 flat manifest、ONNX/env hash、260D/12D、零高度合同和 flat 专属 PD/动作合同；
2. 建立 Go2/X5 单 writer 锁，检查 can0、lowstate tick、sport mode、无线手柄和物理急停；
3. 启动 per-policy 的 X5 owner，等待 arm 状态和 tracking error 稳定；
4. 启动 `run_wbc_flat.py`，先处于输出禁止状态；
5. 完成 ready pose 和最终指令安全合同检查后，由操作者显式 enable；
6. 任一通用安全条件失败，进入受控停止；LiDAR 状态不参与该类判断。

### 10.2 Rough

1. 完成 Flat 的所有通用硬件检查，但加载 rough 发布包；
2. 启动 LiDAR/IMU，验证型号、固件、外参 hash 和原始 topic；
3. 启动 deskew/LIO，确认重力方向、位姿连续性和 TF timestamp；
4. 启动 elevation mapping 和 17×11 sampler；
5. 运行 perception preflight：静止平地、人工放置已知 10 cm 台阶、yaw/左右方向检查；
6. 连续健康窗口建立 `rough_perception_ready`；
7. 启动 X5 fixed-hold 并验证训练姿态；
8. 启动 `run_wbc_rough.py`，manifest、策略类型、height reference、PD/动作合同全部通过后仍保持禁止输出；
9. 操作者显式 enable，先零速度站立，再按分阶段试验限幅放开；
10. 任一必需感知进程退出、stamp/coverage 超限或 X5 状态失效，立即撤销运动许可并执行受控停止。

生产 supervisor 应监视所有子进程。停止时先把速度命令降为 0/进入安全姿态，再停止 policy writer，最后停止感知；不能先杀感知而让 rough policy继续运行。

## 11. 观察到的修改项与优先级

### P0：上机前必须解决

1. **拆分策略发布包和两个部署类**：禁止当前 flat env 与 rough contract 混放；移除生产入口的通用布尔切换。
2. **重新导出真实 rough checkpoint**：从 checkpoint 对应的 env 快照导出 260D 合同、live 187D 样本、动作尺度、默认姿态和 X5 模式。
3. **强化合同检查器**：flat 必须验证 zero term/scanner null；rough 必须验证 live term/scanner/grid；策略类别和运行类不一致必须硬失败。
4. **建立 production rough 感知链**：LiDAR/IMU -> LIO -> elevation map -> stamped 17×11 scan，不以 raw pointcloud 分箱作为正式源。
5. **删除 rough 的 zero fallback**：未知值使用 validity mask 和运动许可表达，不能伪装成平地。
6. **完成并独立复核最终腿部安全合同**：保留 `UNVERIFIED` 阻断，直到有可追溯的 Go2 硬件证据和低能量测试。
7. **分别发布 Flat/Rough manifest**：当前 `UNRELEASED` 不能改字即上线，必须把新增合同、标定和版本 hash 全部纳入。
8. **锁定 X5 训练状态**：为固定臂策略部署 fixed-hold owner，禁止无训练依据的 SpaceMouse 运动。
9. **校准动作和执行器合同**：当前 flat 的 0.18/0.32、rough 源码的 0.25、phase 的 0.5 和实机 200/10 不能混用；每份 bundle 独立记录、台架验证。

### P1：首次地形行走前解决

1. provider 改用 header stamp 和同 stamp TF，校验时钟倒退、map/pose skew 和队列延迟；
2. 高程图采样由 nearest 改为经测试的插值/方差规则，加入 self-filter 和关键落足区 mask；
3. `prepare_real_run.sh` 拆成 flat/rough preflight，rough 检查完整 topic、frame、rate、latency、coverage 和进程存活；
4. 将 ONNX 推理与最终硬件 watchdog 的调度隔离，避免单线程 executor 中推理阻塞 LowCmd/安全定时器；
5. lowstate 健康检查加入 tick 单调前进、CRC/错误状态、电池/温度和数据冻结检测；
6. 将 per-policy ready pose offset、arm pose、PD、限幅和速度上限全部从代码默认值移入签名合同；
7. 修正文档/BOM 中 SpaceMouse 是否必需、LiDAR 缺失、topic/frame 和启动入口不一致。

### P2：提高可维护性与复现性

1. perception 镜像固定依赖版本，启动时不编译；
2. 每次真机运行自动保存 manifest、参数、topic rate、诊断、scan 和关键状态 rosbag；
3. 提供只读 dashboard，显示当前部署类、bundle ID、motion permit、map age/coverage 和 fault reason；
4. 加入标定到期/支架移动检测、外参复测工具和 LiDAR 遮挡回归数据集；
5. 用硬件在环回放 recorded scan/lowstate，持续跑 Flat/Rough 互斥矩阵。

## 12. 建议的文件级改造

### gx-real

- `real-wbc/modules/deployment_profile.py`：定义共享协议、`FlatDeployment`、`RoughDeployment` 和互斥校验；
- `real-wbc/modules/wbc_node_leg12_arm_passthrough.py`：由布尔开关改为注入 profile；共享控制逻辑不复制；
- `real-wbc/modules/height_scan_provider.py`：只服务 Rough；使用 source stamp、同 stamp TF、显式 validity 和 fail-closed 状态；
- `real-wbc/modules/height_scan_core.py`：以导出的坐标数组采样，增加方向/符号测试；生产禁用 direct-cloud；
- `real-wbc/modules/height_scan_policy_validation.py`：把 `policy_kind/env term/runtime class/source` 四者一致性设为硬条件；
- `real-wbc/scripts/run_wbc_flat.py`、`run_wbc_rough.py`：两个不可混用的入口；
- `scripts/run_leg12_flat_real.sh`、`run_leg12_rough_real.sh`：两个 operator 入口；
- `scripts/prepare_flat_run.sh`、`prepare_rough_run.sh`：不同前置条件和检查清单；
- `scripts/height_scan/check_policy_height_scan_contract.py`：增加策略类别、真实 env observation、reference parity 和 sensitivity 检查；
- `config/deployments/flat.yaml`、`rough.yaml`：topic/frame/device/fault policy 的唯一配置源；
- `policies/flat/...`、`policies/rough/...`：独立签名 bundle；
- `tests/test_deployment_profiles.py`：完整互斥矩阵；
- `docs/README.md`、BOM、网络和上机指南：更新双入口和 Rough 必选设备。

### Go2-X5-lab

- [`scripts/height_scan/export_height_scan_contract.py`](../../Go2-X5-lab/scripts/height_scan/export_height_scan_contract.py)：停止硬编码任务/shape，直接从运行时 env manager 和 checkpoint env 快照导出；release 模式禁止 static placeholder；
- [`train_route_env_cfg.py`](../../Go2-X5-lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/go2_x5/train_route_env_cfg.py)：flat/rough task 的 live/zero 语义作为导出检查来源；
- 导出多组平面、斜坡、单侧台阶、yaw 和随机 rough reference NPZ，而不是只保留一个样本。

### phase_guided_terrain_traversal 参考源

- 可参考 [`run_elevation.sh`](../../phase_guided_terrain_traversal/ros_scripts/run_elevation.sh)、[`transform_everything.py`](../../phase_guided_terrain_traversal/ros_ws/src/transform_sensors/transform_sensors/transform_everything.py) 和 [`heightmap_node.cpp`](../../phase_guided_terrain_traversal/ros_ws/src/heightmap_node/src/heightmap_node.cpp) 的拓扑；
- 不直接拷贝 [`deploy_real.py`](../../phase_guided_terrain_traversal/deploy/deploy_real.py) 的观测、归一化、PD、writer 或硬编码外参。

## 13. 分阶段实施计划

### 阶段 A：资产和类别边界，不接硬件输出

- 建立 `flat/<release-id>` 和 `rough/<release-id>`；
- 实现两个 profile 和两个入口；
- flat 的 187D 全零做 bit-exact 测试；
- 从确切 checkpoint 重新导出 rough bundle；
- 合同检查器必须拒绝当前“flat env + rough contract”组合；
- ONNX 在仿真 reference 观测上的动作与训练侧导出结果逐样本一致。

**退出标准**：下面的互斥矩阵全部自动化通过，且没有 ROS/硬件 writer 参与。

### 阶段 B：感知离线与台架

- 固定 LiDAR 型号、外参和 perception 镜像；
- 在 rosbag/静止台架上验证平地、10 cm 台阶、左右/yaw 和遮挡；
- 测量 header age、pose/map skew、coverage、CPU/GPU 占用和 p99 延迟；
- 由测量结果确定 stale/last-valid/STOPPING 阈值。

**退出标准**：方向、符号、顺序与仿真合同一致；关键区域没有 silent zero；拔掉 LiDAR 会撤销 permit。

### 阶段 C：无行走硬件联调

- Go2 架空或安全支撑，X5 fixed-hold；
- WBC shadow mode 只计算不发布 LowCmd；
- 校验关节顺序、ready pose、arm obs、policy deadline 和所有 writer locks；
- 完成 Go2 最终指令安全合同证据和独立复核。

**退出标准**：无多 writer，所有故障注入都到达预期状态，manifest 才具备转为 `RELEASED` 的候选条件。

### 阶段 D：平地低能量

- 先 Flat：零速度站立、极低速度、限时/限距；
- 再 Rough 但仍在平地：确认 live scan 近零且不是由 fallback 产生；
- 比较 Flat/Rough 动作、温度、跟踪误差和停车距离；
- 验证感知断流、TF 延迟、arm fault 和 operator estop。

**退出标准**：两类分别稳定，任何混搭均拒绝，rough 失去感知时受控停止。

### 阶段 E：地形递进

按低矮单台阶、缓坡、连续规则台阶、目标 rough 场景递进；每一级单独限制速度、步数、区域和试验时长。每次只改变一个变量，保存完整运行 bundle 与 rosbag，评审通过后再升级。

## 14. 必须通过的验收矩阵

| 发布包 | 启动类 | 感知 | 预期结果 |
|---|---|---|---|
| Flat | FlatDeployment | 无 | 允许进入 READY；height `[66:253]` 精确全 0 |
| Flat | FlatDeployment | 有 | 仍只用全 0；不得订阅或受其影响 |
| Flat | RoughDeployment | 健康 | 启动失败：policy kind/env term 不匹配 |
| Rough | FlatDeployment | 无 | 启动失败：policy kind 不匹配 |
| Rough | RoughDeployment | 无/未就绪 | 保持禁止输出，不能 READY |
| Rough | RoughDeployment | 健康 | 合同与连续健康窗口通过后才可 READY |
| Rough | RoughDeployment | stale/coverage 失败 | 降速并转 STOPPING/FAULT，绝不切换全 0 |
| 任意 | 任意 | 任意 | manifest 非 RELEASED 或 leg safety 非 VERIFIED 时硬件输出阻断 |

额外数值验收：

- Flat 在随机外部点云、NaN map、错误 TF 注入下仍输出相同 187 个 0，但通用安全条件仍有效；
- Rough 对解析平面/台阶的 sampler 与仿真参考在约定容差内一致；
- +10 cm 台阶的格子符号、位置和幅度正确；
- x/y 互换、左右镜像、row/column 转置、yaw 延迟和旧 stamp 均会被测试捕获；
- policy 输出 shape/finite 只是最低条件，还要验证 reference action parity；
- 推理阻塞、LiDAR 断流、LIO 跳变、map 冻结、X5 CAN 丢失、lowstate tick 冻结和 writer 冲突均有故障注入测试。

## 15. 实施前仍需从实物或训练记录确认的参数

这些项目不改变双类架构，但会阻止 Rough 最终发布：

- 真正要部署的 flat 和 rough checkpoint 路径、训练 commit 和原始 env snapshot；
- LiDAR 最终型号、安装位置、视场被 X5 遮挡的范围及 6DoF 外参；
- Go2/X5 的准确硬件版本、关节限制证据、可接受的 PD/温度/电流范围；
- rough 训练时 X5 固定姿态、gripper 值和 arm target 生成逻辑；
- elevation map 在目标速度下的实测 rate、p95/p99 age、skew 和关键区域 coverage；
- 失去 perception 后最安全的受控停止姿态与最大停车距离。

这些值应通过导出记录、标定和低能量实验获得，不能从 phase 的数值或当前工作区的最新源码推测。

## 16. 最终建议

第一步不是给现有入口加更多 `if enable_height_scan`，而是先完成资产拆分和双部署类。只有当合同检查器能够稳定拒绝当前这组混搭资产，且 Rough 的 live scan 在离线参考上证明了符号、方向、时间和未知值语义，才值得继续搭建真机感知链。

建议实施顺序为：**Flat 类收口 -> Rough checkpoint 重新导出 -> Rough perception 离线闭环 -> X5 fixed-hold -> 最终安全合同 -> shadow/低能量真机 -> 递进地形**。任何阶段都不应通过关闭 fail-closed 检查来推进。
