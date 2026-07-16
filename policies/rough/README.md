# Rough policy releases

Rough 发布包必须由真实 live-height checkpoint 导出，不能从 flat policy 复制或仅修改标签。除通用资产外，每个版本还必须包含并在 manifest 中哈希：

- `height_scan_contract.yaml`：17×11 网格、187D、`[66:253]`、坐标顺序、offset/scale/clip；
- `height_scan_reference.npz`：仿真运行时参考观测；
- `perception_contract.yaml`：LiDAR、frame、topic、外参、时间和覆盖率合同；
- env 中 live `height_scan` 和非空 `scene.height_scanner`。

`RoughDeployment` 只接受 `height_map_array` 生产源；缺失、过期、时间不同步、覆盖不足或 fallback scan 都不会获得运动许可。
