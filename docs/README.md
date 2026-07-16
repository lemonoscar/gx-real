# real 文档导航

如果是第一次使用这个仓库，先看根目录的完整开发文档：

- [gx-real 真机开发文档](../README.md)
- [gx-real 新手开发使用手册](developer_onboarding.md)

这份目录保留更细的专题文档，用于查具体实现细节、上机步骤和后续改造思路。

## 当前主链路

当前推荐维护和上机的链路是：

```text
scripts/run_leg12_real.sh
  -> real-wbc/scripts/run_wbc_leg12.py
  -> real-wbc/modules/wbc_node_leg12_arm_passthrough.py
  -> policies/policy.onnx + policies/env.yaml
```

控制逻辑：

- 前 `12` 维腿动作来自 RL policy。
- X5/ARX5 机械臂由独立 SpaceMouse Arm 节点写 `can0`；WBC 只消费 `/arm/state` 和 `/arm/target_state`。
- Go2/X5 的底层通信、状态读取、手柄流程、起身流程和急停框架尽量沿用原真机链路。

## 专题文档

- [Flat / Rough 双真机部署方案](flat_rough_real_deployment_comparison_and_plan_2026-07-16.md)：对比 phase 工程，定义两类互斥策略部署、Rough height map、X5/传感器接入、完整通信拓扑和分阶段验收。
- [上机使用指南](上机使用指南.md)：真机操作步骤、启动命令、按键流程和常见故障。
- [260维输入设计](260维输入设计.md)：当前 `260D obs -> 12D action` 的观测拼接契约。
- [小替换代码清单](小替换代码清单.md)：把原 UMI WBC 改成 `leg12 + arm passthrough` 的最小改造说明。
- [替换思路](替换思路.md)：当前真机部署后的理论修正，以及后续继续替换上层任务网络、whole-body policy 或底层控制链时的路线选择。

## 相关目录

- [real-wbc](../real-wbc)：真机控制主体。
- [scripts](../scripts)：环境、CAN、sport mode 和启动脚本。
- [policies](../policies)：policy 模型和训练导出的部署配置。
- [arx5-sdk](../arx5-sdk)：X5/ARX5 机械臂 SDK。
- [unitree_sdk2](../unitree_sdk2)：Go2 SDK、CRC 模块和 sport mode 工具。
- [unitree_ros2](../unitree_ros2)：Unitree ROS2/CycloneDDS 通信栈。

## 原始 UMI 链路

以下文件仍保留，但不是当前标准上机入口：

- [run_wbc.py](../real-wbc/scripts/run_wbc.py)
- [wbc_node.py](../real-wbc/modules/wbc_node.py)
- [run_teleop.py](../real-wbc/scripts/run_teleop.py)
- [robot_state ROS2 消息](../real-wbc/ros2/robot_state)

除非明确要恢复原 UMI-on-Legs 的任务空间轨迹控制，不建议把这些文件作为当前 leg12 上机主流程。
