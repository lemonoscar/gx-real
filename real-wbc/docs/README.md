# real-wbc 开发文档索引

这个目录放的是 `real-wbc` 子模块的开发和硬件说明。当前 `gx-real` 真机主链路已经不是原始 UMI-on-Legs 的完整任务空间轨迹链路，而是：

```text
scripts/run_leg12_real.sh
  -> real-wbc/scripts/run_wbc_leg12.py
  -> real-wbc/modules/wbc_node_leg12_arm_passthrough.py
  -> policies/policy.onnx + policies/env.yaml
```

上机操作以仓库根目录下的 [上机使用指南](../../doc/上机使用指南.md) 为准；本目录主要解释开发环境、网络、硬件和历史链路。

## 文档划分

- [codebase_setup.md](codebase_setup.md)：开发环境、仓库结构、当前控制入口、调试命令。
- [network.md](network.md)：Go2 局域网、外网、ARX5 `can0` 和常见网络问题。
- [assembly.md](assembly.md)：Go2 + ARX5 的硬件装配、电源、CAN 和外设连接。
- [bill_of_materials.md](bill_of_materials.md)：核心物料、可选物料、历史 UMI 外设。
- [hardware_design_choices.md](hardware_design_choices.md)：Go2、ARX5、iPhone/MoCap、GoPro 的取舍说明。
- [3d_printing.md](3d_printing.md)：3D 打印件、材料和切片注意事项。

## 当前主链路与历史链路

当前推荐维护的主链路是 `leg12 + arm passthrough`：

- 腿部由 12 维 RL policy 控制。
- 机械臂默认保持 `--arm_pose`，也可以通过手柄 `A/X` 切换目标。
- 底盘速度命令由 `--cmd-vx/--cmd-vy/--cmd-yaw` 指定，手柄 `Y` 可以将 command 置零。
- 默认起身流程是 `--standup-mode internal`。

以下内容属于原 UMI-on-Legs 历史链路，除非明确要恢复任务空间轨迹控制，否则不要作为当前上机主流程：

- `real-wbc/scripts/run_wbc.py`
- `real-wbc/modules/wbc_node.py`
- `real-wbc/scripts/run_teleop.py`
- `real-wbc/ros2/robot_state`
- SpaceMouse / EEF trajectory / diffusion policy 相关说明

## 修改文档时的约定

- 上机命令和故障处理优先更新 [上机使用指南](../../doc/上机使用指南.md)。
- 开发环境和代码入口说明更新 [codebase_setup.md](codebase_setup.md)。
- 硬件接线、CAN、电源问题更新 [assembly.md](assembly.md) 或 [network.md](network.md)。
- 不要在文档里写本机绝对路径，除非是机器人上固定路径，例如 `~/gx-real`。
