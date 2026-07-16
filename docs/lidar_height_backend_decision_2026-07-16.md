# Rough LiDAR / height-map 后端决策（2026-07-16）

## 结论

Rough 真机生产链固定为：

```text
Unitree LiDAR PointCloud2 + IMU
  -> deskew / Point-LIO
  -> elevation_mapping
  -> grid_map_msgs/msg/GridMap（elevation layer）
  -> gx-real 17×11 sampler
  -> observation[66:253]
```

Unitree `unitree_go/msg/HeightMap` 只保留作抓包、可视化和同 rosbag A/B
诊断，不能建立 rough motion permit。`pointcloud2` 直接分箱同样只用于诊断。

## 依据

### Go2-X5-lab 定义的是策略输入语义，不是真机建图后端

- `velocity_env_cfg.py` 使用 yaw-aligned `RayCasterCfg`，`GridPatternCfg`
  为 0.1 m、1.6 m × 1.0 m；导出的 `grid_xy` 是 17×11、x-fast、
  y-outer 的 187 点。
- Rough policy 使用 Isaac Lab `mdp.height_scan`；保存的真值满足
  `base_z - ray_hit_z - 0.5`，然后 scale 1、clip `[-1, 1]`，写入
  observation `[66:253]`。
- Go2-X5-lab 的 `rl_sim.cpp` 只订阅已经排成二维数组的
  `sensor_msgs/Image` 并执行 `depth - 0.5`；它没有从点云构造地形图。
- `rl_real_go2_x5.cpp` 没有 HeightMap、GridMap 或 PointCloud2 的订阅逻辑。
  因此不能把 Go2-X5-lab 解读为“官方 HeightMap 是既有真机实现”。

### Unitree HeightMap 目前不能证明轴、原点和尺度

Unitree 公开 IDL 只给出 stamp、frame、resolution、width、height、origin 和
data 字段，没有定义 data 的 x/y 展开顺序、origin 是边界还是 cell center、
高度单位/无效值以及地图如何随机器人移动。已有两份真机快照说明 topic 可读，
但关键区域 sentinel 仍触发 fail-closed，也不足以证明方向和尺度。

公开依据：Unitree SDK2 的
[`HeightMap_` IDL 生成接口](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/include/unitree/idl/go2/HeightMap_.hpp)
及 Unitree ROS2 对 `/utlidar/cloud` 原始点云的
[`README`](https://github.com/unitreerobotics/unitree_ros2/blob/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88/README.md)。

因此当前 `height_map_array` 解析器只用于诊断；在同一 rosbag 中完成已知台阶、
左右、前后、yaw 90°、尺度和无效值对照之前，不得提升为生产源。

### phase_guided_terrain_traversal 给出了可审计的生产拓扑

phase 工程使用 self-filter/坐标变换、Point-LIO、`elevation_mapping`、
`grid_map_msgs/GridMap`，再由 `heightmap_node.cpp` 按机器人 pose/yaw 逐点
`getIndex` 取 elevation。gx-real 复用的是这套拓扑和操作顺序，不复制它的
9×7 网格、数值缩放、latest-TF 或 NaN 写零行为。

GridMap wire contract 可从 `GridMapRosConverter` 完整还原：Eigen 列主序，
`column_index`/`row_index` 标签，矩阵轴为 x/y buffer index，且必须应用
`outer_start_index`/`inner_start_index` circular-buffer 偏移。gx-real 对任何
标签、stride、geometry、layer 或非 identity map pose 异常都 fail-closed。

## gx-real 的映射合同

- source：`grid_map`
- ROS type：`grid_map_msgs/msg/GridMap`
- topic：`/terrain/elevation_map`
- layer：`elevation`
- pose：`/localization/pose`，`geometry_msgs/msg/PoseStamped`
- sampling：加载发布包内精确的 187 对 `grid_xy`，只使用 base yaw 变换到
  map frame；按 grid_map_core 的 cell-index/circular-buffer 语义取值。
- value：`clip(base_z - elevation - 0.5, -1, 1)`；顺序原样写入 `[66:253]`。
- validity：非 finite/sentinel、关键区越界、ground band 异常、age/skew/frame
  异常均撤销许可；生产 GridMap 路径不填补机器人足迹内的未知 cell。
- release：只有 GridMap wire contract、mapper 配置哈希、LiDAR/外参/self-filter
  和真机 rosbag parity 全部写入已复核的 perception contract 后才可标记
  `VERIFIED`。

## 已加入的离线证据

回归测试覆盖 GridMapRosConverter 列主序、x/y 轴、circular-buffer start index、
yaw/符号/flatten 顺序、关键 footprint unknown fail-closed，以及保存的
Isaac Lab 187D reference 逐元素重放。目标 Jetson 的 rosbag 和硬件实验仍是
发布必需证据，单元测试不能替代。
