# gx-real：Go2-X5 model_13493 平地部署

当前默认模型是robust续训完成后的 **`model_13493.pt` Actor**，483维输入、12维腿动作输出。**187维高度输入默认全部为0**。固定速度与Unitree手柄入口使用同一模型；机械臂仍由独立SpaceMouse Arm节点控制，WBC只写腿。

## 当前模型与配置

| 项目 | 设置 |
|---|---|
| 模型目录 | `policies/unified483_model13493/` |
| 通用模型路径 | `policies/policy.onnx`等五个文件为上述版本的相对符号链接 |
| Actor | 483→512→256→128→12，ELU；仅Actor参与部署 |
| 策略频率 | 50Hz；电机发布定时器500Hz |
| 腿动作 | `q_default + 0.25 × action`，随后执行启动/关节限位等保护 |
| 腿目标PD | Kp40、Kd1 |
| 高度观测 | 默认 `obs[66:253]=0`，不订阅点云 |
| 臂收拢姿态 | `[0, 0.3, 0.5, 0, 0, 0] rad` |
| 依赖 | Go2 LowState、臂状态和目标、连续base原点世界位姿、原有MCF/急停链 |

零高度版仍需要base定位来计算站立锚点，默认位姿话题为`/utlidar/robot_pose`。需要确认它表示base原点，并与ROS时钟一致。零高度模式不等于无定位运行。

完整观测、历史、坐标、动作和时间定义见[Unified部署说明](UNIFIED483_DEPLOYMENT.zh-CN.md)。当前权重来自带地形观测的robust训练，零输入是部署选择，未另行宣称完成盲走训练或实机稳定性验收。

## 启动入口

先在独立终端按实际硬件配置启动臂控制与状态发布。腿节点不打开机械臂CAN。

```bash
# 固定速度模式：先以零速度进入策略
bash scripts/run_unified483_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0

# 手柄模式：默认零高度
bash scripts/run_unified483_joystick_real.sh

# 原通用入口也已经切到13493，默认零速度、零高度
bash scripts/run_leg12_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

以上是实际硬件入口，会校验模型后释放MCF并启动低层控制；本次代码切换没有执行这些命令。以往Shell中设置过`GX_REAL_POLICY_PATH`、`GX_REAL_POLICY_BUNDLE_PATH`或`GX_REAL_UNIFIED_BUNDLE`的用户，应先检查是否仍显式指向旧版本；环境变量是保留的有意覆盖通道。

## 速度与手柄

独立死区：vx±0.1m/s、vy±0.05m/s、wz±0.05rad/s内分别归零。输出范围vx−0.5～0.75、vy±0.2、wz±0.3，多轴指令采用训练椭球限幅。手柄入口保留回中解锁、速度斜坡及0.25s失联检查。

| 操作 | 功能 |
|---|---|
| 左摇杆上下 / 左右 | vx / vy |
| 右摇杆左右 | wz |
| R1 | 起身到FixStand |
| L2 | 稳定后进入策略；完成零速交接后放行摇杆 |
| Y | 速度清零并要求摇杆回中 |
| L1 | 首次受控停机，再次硬急停 |
| R2 | 立即停止策略，不作正常下机流程 |

机械臂目标/实测状态收拢并稳定后才放行行走；臂展开时底盘命令归零。WBC不会强制覆盖外部臂控制器的轨迹。

## 离线核验

以下命令不启动机器人：

```bash
python scripts/check_policy_bundle.py \
  --policy policies/policy.onnx --manifest policies/policy_bundle.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

13493 checkpoint SHA256：`f93ec57e3509c591e2fb34b37233db3d9a670c02f68bf1a27a951c597a6c8aea`。

ONNX SHA256：`8094409179153a2054420c5426c64e17dd2a2677e828d070f00ed55e508a0988`。

导出时batch1/8/32与PyTorch对照通过（rtol/atol各2e-5）；另有240帧结构化观测对照。模型、配置、扫描契约和网格均有哈希绑定。文件/CPU测试不能代替真实传感器、调度时延和硬件动作验收。

## 点云与旧模型

当前工作重点为零高度平地部署。`run_unified483_pointcloud_real.sh`保留为实验入口，也使用13493，但坐标转换及采集时间检查的已知缺口尚未修复，不作为正式验收版本。

显式回退到11494：

```bash
GX_REAL_UNIFIED_BUNDLE="$PWD/policies/unified483_model11494" \
  bash scripts/run_unified483_real.sh --cmd-vx 0
```

旧19000保存在`policies/legacy_flat19000/`，不会被默认加载；若确需旧Flat链路，需显式设置`GX_REAL_POLICY_PATH`与`GX_REAL_POLICY_BUNDLE_PATH`到该目录，并使用`run_leg12_real.sh`。旧模型契约是260维，不能替换进483维入口。

硬件安装、网络与历史控制说明见[历史Flat文档](docs/legacy_flat19000_README.md)；其中旧模型、旧观测和旧放行结论只适用于旧版本。[上机使用指南](docs/上机使用指南.md)保留硬件操作资料。
