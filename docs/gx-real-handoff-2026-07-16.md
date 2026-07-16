# gx-real 真机收口交接（2026-07-16）

## 当前结论

- GitHub 分支：`rough-policy`
- Draft PR：<https://github.com/lemonoscar/gx-real/pull/1>
- 首个可追溯快照：`609522459c93049e2049fbbb8dc080285fc96b6d`
- 本机：`x86_64` 开发机；本轮未连接 Go2、未创建 LowCmd publisher、未打开 X5 CAN controller、未运行真实 ROS2 控制节点。
- 当前真机放行：**NO**。Flat/Rough manifest 均为 `UNRELEASED`，Rough perception contract 和 Go2 leg safety contract 均未验证。

## 已完成的软件侧收口

1. Flat/Rough 使用不同 Python、Shell、配置、policy bundle 和 manifest 入口；旧 `run_leg12_real.sh` 固定拒绝策略类别不明确的启动。
2. Flat 的 187D height slice 由代码直接生成精确零，且不创建 height provider。
3. Rough 只接受 `height_map_array`，验证 source age、pose/map stamp skew、frame、finite、coverage、critical sentinel 和连续健康帧；fallback 永远不能获得 motion permit。
4. 感知 stale/fallback 会清零连续健康帧，恢复后必须重新累计 5 帧。
5. Rough checkpoint 29500 已重新导出；ONNX 输入/输出为 260/12，10 cm 台阶造成非零动作响应。
6. 新增 `policy_reference.npz`：期望动作来自 checkpoint/Torch，ONNX 重放最大误差约 `5.36e-7`。
7. Rough manifest 逐项哈希 ONNX、TorchScript、checkpoint、agent/env、height YAML/NPZ、仿真 reference、policy reference、perception contract 和 export metadata。
8. Rough 发布门要求 perception contract 为 `VERIFIED`，且 LiDAR/固件/外参/self-filter/mapping 字段不能为 `UNSET/UNKNOWN/UNVERIFIED`。
9. X5 生产 owner 改为策略训练姿态的 `x5_fixed_hold`；生产 WBC 拒绝 SpaceMouse motion、错误 arm pose、无 freshness 和错误 producer。
10. fixed-hold 必须收到 WBC safety lease 和显式 operator enable；feedback 无效、tracking error、lease fault 或 ESTOP 会锁存并进入 damping。
11. 生产 fixed-hold 禁止非 dry-run 的 `--allow-missing-can`；preflight 会识别 fixed-hold 冲突进程。
12. preflight 默认不再要求 SpaceMouse，并打印与实际生产门一致的 Flat/Rough + fixed-hold 启停命令。
13. manifest 发布采用可实现的两提交协议：source commit 后只能创建一个 manifest-only release commit。
14. 最新唯一操作步骤已写入 `docs/上机使用指南.md`；旧 README/onboarding 顶部已明确标记历史入口不可用于生产。

## 当前离线验证

使用临时 `/tmp` ONNX Runtime 环境，不修改 Jetson/系统依赖：

```text
pytest: 168 passed, 0 failed, 0 skipped
Flat contract: PASS, input 260, output 12
Rough contract: PASS, input 260, output 12
Rough 10 cm sensitivity: max_action_delta=0.249644
Torch/ONNX reference parity: max_abs_error≈0.00000054
hardware writer inventory: 32 candidate files, all classified
Python compile / bash -n / git diff --check: PASS
```

复现命令见 `docs/上机使用指南.md` 第 4 节。测试必须限制在 `tests/`，否则 pytest 会收集 vendor 的硬件示例。

## 仍阻止真机发布的证据

### P0：任何真实输出前必须关闭

- `config/go2_leg_safety_contract.yaml` 仍为 `UNVERIFIED`，关节位置、step/velocity/acceleration/jerk、command age、lowstate age 均为空。
- Flat manifest 仍含旧 commit、x86_64 库哈希和 `onnxruntime_version: UNRESOLVED`。
- Rough manifest 的 commit、SDK snapshot、Python/ORT 版本、CycloneDDS、Go2 safety hash和 Jetson shared libraries 均为 `UNSET` 或空。
- Rough perception contract 的 LiDAR 型号/固件、刚性安装、6DoF 外参、self-filter、mapping implementation/config hash 均为 `UNSET`。
- 没有目标 Jetson 上的完整 clean build、ROS message type-support、topic/frame/rate/latency/coverage 证据。
- 没有 Go2 架空 shadow、关节顺序、ready pose、PD、电流/温度和最终命令边界的硬件证据与独立复核。
- 没有 X5 fixed-hold 在禁能/支撑条件下的 motor ID、起始误差、跟踪误差、ESTOP 和停机验证。

### 代码无法替代的已知风险

- ARX SDK 当前 `JointState.timestamp` 来自 controller loop，不是逐电机 CAN RX 时间；仅比较该 timestamp 不能证明总线仍收到新反馈。必须由 driver RX counter/per-motor freshness 或可追溯外部监测关闭此项。
- ONNX 推理与 Go2 watchdog 仍在同一进程/executor；永久卡死不能由同进程 timer 自救，需要独立进程 watchdog/硬件接收端证据。
- elevation map 当前 nearest sampling、footprint unknown fill、self-filter 和 mapper 方差规则尚无目标硬件 rosbag parity。
- 感知失效后的安全停车距离和 X5 damping 的物理行为必须实测，不能由单元测试推断。

## 建议继续顺序

1. 在目标 Jetson 固定 ROS/Python/ORT、SDK 和 shared-library 版本，完成 clean build，但保持 manifest `UNRELEASED`。
2. 固定 LiDAR 与支架，完成外参、自滤波和 mapper 配置；录制平地、10 cm 台阶、左右单侧、yaw 90°、遮挡/断流 rosbag。
3. 离线回放并记录 rate、p95/p99 age/skew、coverage、方向/符号/顺序和 CPU/GPU；通过独立复核后才能把 perception contract 改为 `VERIFIED`。
4. 为 X5 增加或取得真实 CAN RX freshness 证据，完成 fixed-hold 禁能/支撑故障注入。
5. 从 Go2 硬件合同和架空低能量试验填写最终腿部安全合同并独立复核。
6. 提交所有代码/资产得到 source commit；再只改一个目标 manifest 创建 manifest-only release commit。
7. 先 Flat shadow/零速/低速限时，再 Rough 平地 live scan，最后按单台阶、缓坡和规则地形递进；每步只改变一个变量并保存完整日志/rosbag。

## 不要做

- 不要因 manifest 阻断而直接改成 `RELEASED`；
- 不要把 perception contract 的 `UNSET` 只改成字符串占位；
- 不要在 Rough 感知失效时喂全零或 last-valid scan；
- 不要同时运行 SpaceMouse/fixed-hold/ARX 示例等多个 X5 writer；
- 不要在这台 `x86_64` 开发机执行真实部署命令。
