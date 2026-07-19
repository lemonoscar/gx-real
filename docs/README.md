# gx-real 文档索引

当前生产文档以本页列出的“现行文档”为准。仓库还保留设计评审和交接材料用于追溯，
但其中的旧命令、SpaceMouse 主链或历史 release 状态不能覆盖现行手册。

## 现行文档

1. [仓库 README](../README.md)：系统边界、260D 合同、Rough 地图、目录、入口和验证。
2. [真机上机使用指南](上机使用指南.md)：唯一生产操作手册；包含 X5 -> WBC ->
   fixed-hold enable 的启动顺序、停止和故障处理。
3. [LiDAR 校准指南](lidar_calibration.md)：刚性安装、`T_base_lidar`、时间同步、
   rosbag、GridMap 几何/时序验收和 perception release。
4. [开发与仓库指南](developer_onboarding.md)：代码职责、构建、测试、修改与交付流程。
5. [Rough 发布包说明](../policies/rough/README.md)：模型资产、height/reference 和
   perception contract 要求。
6. [LiDAR/height-map 后端决策](lidar_height_backend_decision_2026-07-16.md)：为什么
   Rough 生产只接受 GridMap/elevation，Unitree HeightMap/直接点云只用于诊断。

当前生产拓扑为：

```text
scripts/run_x5_fixed_hold_{flat,rough}.sh
  -> X5 fixed-hold owner -> physical safety topics

scripts/run_leg12_{flat,rough}_real.sh
  -> run_wbc_{flat,rough}.py
  -> run_wbc_leg12.py
  -> wbc_node_leg12_arm_passthrough.py
  -> fixed actor arm observation + 12D Go2 action
```

Rough 额外使用：

```text
/terrain/elevation_map + /localization/pose
  -> height_scan_provider.py
  -> height_scan_core.py
  -> observation[66:253]
```

## 现行安全结论

- manifest 仍为 `UNRELEASED`，Rough perception contract 仍为 `UNVERIFIED`；
- actor arm 输入固定为 `[0, 0.3, 0.5, 0, 0, 0]`，真实 arm 值只参与 safety；
- X5 生产 owner 是 fixed-hold，不是 SpaceMouse；
- Rough 只接受 live GridMap；fallback、Unitree HeightMap 和直接点云不得放行；
- 不得通过改状态字符串、降低 coverage 或关闭校验绕过发布门。

## 历史设计与审计材料

以下文件解释某个时间点的设计、缺口或评审结论。阅读时以文件日期为边界，并用现行
文档核对命令和状态：

- [2026-07-16 Flat/Rough 部署对比与计划](flat_rough_real_deployment_comparison_and_plan_2026-07-16.md)
- [2026-07-16 真机收口交接](gx-real-handoff-2026-07-16.md)
- [2026-07-12 Phase A 独立评审摘要](gx_real_phase_a_independent_review_summary_2026-07-12.md)
- [2026-07-12 Phase A 实现记录](gx_real_safety_phase_a_implementation_2026-07-12.md)
- [2026-07-12 安全评审](gx_real_safety_review_2026-07-12.md)
- [Phase A 最小阻塞项](safety_phase_a_minimal_blockers.md)
- [策略启动抖动/reward 记录](policy_startup_jitter_reward_notes.md)
- [旧替换思路](替换思路.md)

历史材料中若出现 `run_leg12_real.sh`、`external_spacemouse`、动态机械臂观测或
Unitree HeightMap 生产输入，均不再是当前 Rough 生产做法。

## 代码入口

- [部署配置](../config/deployments)
- [Rough 发布包](../policies/rough/current)
- [WBC/感知/安全模块](../real-wbc/modules)
- [Python 入口](../real-wbc/scripts)
- [shell/preflight/契约工具](../scripts)
- [离线回归测试](../tests)
