# Rough policy releases

Rough 发布包必须由真实 live-height checkpoint 导出，不能从 flat policy 复制或仅修改标签。除通用资产外，每个版本还必须包含并在 manifest 中哈希：

- `height_scan_contract.yaml`：17×11 网格、187D、`[66:253]`、坐标顺序、offset/scale/clip；
- `height_scan_contract.npz`：运行时实际读取的精确 `grid_xy`，必须单独哈希；
- `height_scan_reference.npz`：仿真运行时参考观测；
- `policy_reference.npz`：由训练 checkpoint/Torch 对仿真参考观测产生的期望动作；
- `policy.pt`、`policy.onnx`、checkpoint、agent/env 和 `export_metadata.json`；
- `perception_contract.yaml`：LiDAR、frame、topic、外参、时间和覆盖率合同；
- env 中 live `height_scan` 和非空 `scene.height_scanner`。

`RoughDeployment` 只接受 `grid_map_msgs/msg/GridMap` 的 `elevation` layer
作为生产源，并严格解析 GridMapRosConverter 的列主序和 circular-buffer
索引。Unitree `height_map_array` 与直接 pointcloud 都是 diagnostic-only；
缺失、过期、时间不同步、覆盖不足或 fallback scan 都不会获得运动许可。

后端选择及其与 Go2-X5-lab、phase 工程的逐项依据见
[`docs/lidar_height_backend_decision_2026-07-16.md`](../../docs/lidar_height_backend_decision_2026-07-16.md)。

`perception_contract.yaml` 必须是 `VERIFIED`，且 LiDAR 型号/固件、6DoF 外参、self-filter、mapping 实现和配置哈希均已填写；仅把 manifest 改成 `RELEASED` 仍会被启动门拒绝。
