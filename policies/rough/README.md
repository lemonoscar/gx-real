# Rough policy releases

Rough 发布包必须由真实 live-height checkpoint 导出，不能从 flat policy 复制或仅修改标签。除通用资产外，每个版本还必须包含并在 manifest 中哈希：

- `height_scan_contract.yaml`：17×11 网格、187D、`[66:253]`、坐标顺序、offset/scale/clip；
- `height_scan_contract.npz`：运行时实际读取的精确 `grid_xy`，必须单独哈希；
- `height_scan_reference.npz`：仿真运行时参考观测；
- `policy_reference.npz`：由训练 checkpoint/Torch 对仿真参考观测产生的期望动作；
- `policy.pt`、`policy.onnx`、checkpoint、agent/env 和 `export_metadata.json`；
- `perception_contract.yaml`：LiDAR、frame、topic、外参、时间和覆盖率合同；
- env 中 live `height_scan` 和非空 `scene.height_scanner`。

`RoughDeployment` 只接受 Unitree `height_map_array` 作为生产源，并严格解析公开
IDL 的 x-major 索引、origin cell 和米制单位。adapter 使用 map/pose 时间门、短期
world cache 和受约束局部平面补全；实测高度原样保留，关键非机身区域未知绝不补全。
GridMap 与直接 pointcloud 都是 diagnostic-only；
缺失、过期、时间不同步、覆盖不足或 fallback scan 都不会获得运动许可。

actor 的机械臂相关输入也属于发布合同：训练 default pose 与真机固定目标均为
`[0.0, 0.3, 0.5, 0.0, 0.0, 0.0]`，arm velocity、18D previous-action 的臂部
padding 和 gripper 均为零，六个 arm command range 均为 `[0, 0]`。真机实测
`/arm/state` 只用于 fixed-hold safety，不能替换 actor 中的固定值。离线合同检查会在
这些字段不一致时拒绝发布包。

后端选择及其与 Go2-X5-lab、phase 工程的逐项依据见
[`docs/lidar_height_backend_decision_2026-07-16.md`](../../docs/lidar_height_backend_decision_2026-07-16.md)。
目标硬件的安装、6DoF 外参、时间同步、已知台阶和 held-out rosbag 验收按
[`docs/lidar_calibration.md`](../../docs/lidar_calibration.md) 执行。

`perception_contract.yaml` 必须是 `VERIFIED`，且 LiDAR 型号/固件、6DoF 外参、self-filter、mapping 实现和配置哈希均已填写；仅把 manifest 改成 `RELEASED` 仍会被启动门拒绝。

## 当前候选模型

`current/` 已更新为 Go2-X5-lab 的 R1 regular-ascent repair `model_37500.pt`
（iteration 37500，SHA-256
`abf4404d717e19436479f467fa1f39dad8a9f29e7d3624897861d0cc360beb3b`）。
源文件位于
`Go2-X5-lab/logs/rsl_rl/r1_best_regular_ascent_37500/`，配套 agent/env 与导出的
TorchScript、ONNX、policy reference 均纳入 manifest 哈希。

该候选只完成本机离线合同验证，manifest 保持 `UNRELEASED`；在 LiDAR 感知合同、Go2
腿部安全合同和目标 Jetson 版本/库哈希关闭前，不能用于真机放行。

## 真机起身链路

`scripts/run_leg12_rough_real.sh` 会先通过 Unitree MotionSwitcher 检查并释放 MCF；只有
再次检查确认没有活动 motion mode，WBC 才会启动。收到首帧 LowState 后，WBC 以当前
腿姿态连续输出 `Kp=0, Kd=3` Passive。R1 由本节点执行内部 FixStand，最终目标直接使用
该 rough policy 的 ready/action offset；L2 只验证跟踪误差并直接进入 rollout，不再
调用 MCF/外部起身，也不再增加第二段姿态对齐。
