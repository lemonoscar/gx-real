# Go2-X5 Unified 483 维策略部署

本分支提供 `model_11494.pt` 的 Actor ONNX 及独立入口。当前约 2,000 轮续训仍在 lab-server 运行，未把未完成训练的结果当作已验收模型。默认 `run_leg12_real.sh` 仍使用原 Flat 260 维模型；新入口是 `run_unified483_real.sh`（默认零高度版）和 `run_unified483_pointcloud_real.sh`（真实点云版）。两者使用同一套Actor权重，区别仅为187维地形输入及对应扫描检查。

## 与原部署的区别

| 项目 | 原 Flat | 当前 Unified |
|---|---|---|
| Actor 输入/输出 | 260 / 12 | 483 / 12 |
| 地形 | 零高度占位 | 默认187点全零；点云入口使用真实扫描 |
| 附加状态 | 无 | 步态相位、站立标记、站立锚点误差、4 帧历史 |
| 默认关节角 | 部署偏置 | 训练 `env.yaml` 的默认关节角 |
| 机械臂 | 固定姿态启动检查 | 保留启动检查；运行时读实测六轴角/速度与六轴目标 |
| 行走门控 | 旧指令流程 | 目标收拢且实测收拢后才放行 |
| 控制周期 | 原逻辑 | `sim.dt × decimation = 0.0025 × 8 = 0.02 s` |

只运行 Actor。560 维 Critic 的负载质量、摩擦、接触等特权数据不进入真机推理。无需精确负载标签。

## 输入顺序（零起始、右端不包含）

| 范围 | 内容 | 缩放 |
|---|---|---|
| 0:3 | 本体坐标系线速度 | ×2 |
| 3:6 | 本体坐标系角速度 | ×0.25 |
| 6:9 | 投影重力 | ×1 |
| 9:12 | 生效 vx、vy、wz 指令 | ×1 |
| 12:30 | 12 腿＋6 臂关节角减训练默认角 | ×1 |
| 30:48 | 18 关节速度 | ×0.05 |
| 48:66 | 上次腿位置残差动作12＋补零6 | ×1 |
| 66:253 | 高度扫描187 | clip[-1,1] |
| 253:259 | 机械臂绝对目标角6 | ×1 |
| 259:260 | 训练常量0 | 不填夹爪开度 |
| 260:267 | sin相位、cos相位、站立标记、锚点本体系dx/dy、sin偏航差、cos偏航差−1 | ×1 |
| 267:483 | 前4帧，每帧54，最旧在前 | 内部状态不使用外部观测缩放 |

每帧54维为角速度3、重力3、关节角偏差18、关节速度18、上次腿动作12。历史只在50 Hz推理时前进，初始化用首帧填满；低层状态回调不推进历史。重新进入策略时重置。

关节顺序读取绑定的 `env.yaml`，按关节名映射硬件顺序。腿输出为 `default_q + action_scale × action`，仍经过启动动作限制、位置限位与PD交接；下一帧记录最终位置指令对应的残差。沿用配置的启动零残差保持期（当前1秒），同时保留原PD交接。硬件通信本身已有延迟，真机入口不再额外注入训练时的随机动作延迟、保持或噪声。

## 指令、定位与机械臂

各轴独立归零：`|vx| ≤ 0.10 m/s`、`|vy| ≤ 0.05 m/s`、`|wz| ≤ 0.05 rad/s`。随后按训练椭球限幅，vx范围−0.5～0.75、vy±0.2、wz±0.3；多轴同时达到最大值会共同缩放，缩放后再次应用死区。原遥控器的上游限幅和斜坡仍生效。

收臂姿态 `[0, 0.3, 0.5, 0, 0, 0] rad`。放行条件：目标最大误差<0.03 rad、实测最大误差<0.10 rad、实测最大速度<0.15 rad/s。放行后使用0.20 rad实测误差迟滞；目标离开收拢姿态立即将底盘指令归零。机械臂仍由外部单独进程控制，WBC不接管CAN。门控不能阻止外部控制器突然甩臂，也不替代轨迹碰撞检查。

站立锚点需要连续的**base原点世界位姿**，默认从 `/utlidar/robot_pose` 读取 `PoseStamped`。必须确认该话题实际是base位姿；若是LiDAR位姿，应先按标定外参转换再发布。不得把LiDAR原点或累计积分漂移替代base原点。时间戳与ROS时钟一致；消息过期、世界frame改变、超过25 cm的定位跳变会阻断或停止策略。默认新鲜度0.25 s。

187点按yaw对齐，x∈[-0.8,0.8]、y∈[-0.5,0.5]、间距0.1 m，`xy` meshgrid顺序。数值语义为`base_z − ground_z − 0.5 m`，遵循现有高度模块及其有效覆盖规则。点云版订阅 `/unilidar/cloud`（PointCloud2），不接受零回退或旧扫描回退。默认零高度版不创建点云订阅，直接将 `obs[66:253]` 全部置零。两版仍要求新鲜的base位姿与机械臂反馈。点云外参、base位姿含义、关节零位及力矩能力仍需在实际设备上核对。

## 使用

纯离线核对，不触碰硬件：

```bash
python scripts/check_policy_bundle.py \
  --policy policies/unified483_model11494/policy.onnx \
  --manifest policies/unified483_model11494/policy_bundle.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

实际设备上，在机械臂状态/目标、base位姿和原有急停链准备好后，使用默认零高度入口（此命令会释放MCF并启动低层控制，本次开发没有执行）：

```bash
bash scripts/run_unified483_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

真实点云版还要求有效点云与标定后的坐标转换：

```bash
bash scripts/run_unified483_pointcloud_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

启动日志会打印 `Unified height input mode=zero/pointcloud`。零高度是主动屏蔽输入的对照版本，并不意味着当前权重已另行完成盲走训练；输入0也不等同于所有姿态下的平地高度。当前服务器训练继续保留地形观测，未因增加部署入口而改变。

保留原R1/L2接管流程、MCF检查、低层状态看门狗、关节限位、急停与机械臂单写者限制。启动时机械臂需处于收拢姿态；站稳后由外部臂控制器执行连续轨迹。具体遥控操作沿用主README。

续训完成后，使用实际最终checkpoint与同一次训练的`params/env.yaml`重新导出到新目录：

```bash
python scripts/export_unified_actor.py \
  --checkpoint /path/to/model_13493.pt \
  --env /same/run/params/env.yaml --output policies/unified483_model13493
GX_REAL_UNIFIED_BUNDLE="$PWD/policies/unified483_model13493" \
  bash scripts/run_unified483_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

真实点云版还要求有效点云与标定后的坐标转换：

```bash
bash scripts/run_unified483_pointcloud_real.sh --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

启动日志会打印 `Unified height input mode=zero/pointcloud`。零高度是主动屏蔽输入的对照版本，并不意味着当前权重已另行完成盲走训练；输入0也不等同于所有姿态下的平地高度。当前服务器训练继续保留地形观测，未因增加部署入口而改变。

导出器拒绝覆盖已有目录，生成Actor、配置、高度扫描网格、SHA256清单与CPU等价性记录。先完成仿真回归及实际传感器输入核对，再替换部署版本。

## 当前验证范围

- 原始Actor与ONNX在batch 1/8/32随机输入上的CPU输出一致至`rtol=2e-5, atol=2e-5`，最大绝对差约3.05e-5（结合相对容差通过）。
- 离线测试覆盖死区、正负轴限幅、收臂迟滞、相位、四帧历史顺序与缩放、锚点坐标变换、失效输入停止、配置函数和文件哈希绑定。
- 补齐主分支缺失的旧260维高度扫描契约文件，原模型/配置未替换。
- 尚未在ROS真机链路或实体Go2-X5上验收。原11494在此前YZY两级下楼对照中仅2/3成功，不能宣称已达到真机楼梯通过标准。
