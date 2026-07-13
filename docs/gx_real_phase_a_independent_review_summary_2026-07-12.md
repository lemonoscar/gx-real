# gx-real Phase A 独立安全复审总结

审查日期：2026-07-12  
审查基线：`main@264f10cc7c65be5af7f0d88a6db0043113265c97`  
审查范围：当前工作树、Git diff、生产启动路径、硬件写入调用链、ARX vendor 源码及安全测试  
审查方式：严格只读；未启动 ROS 2/DDS、LowCmd publisher、ARX controller、CAN、SpaceMouse 或任何真实硬件

## 1. 最终判定

```text
PHASE A REJECTED
NO-GO FOR HARDWARE
```

当前默认配置具有两道有效的临时阻断：

- `config/artifact_manifest.yaml` 为 `UNRELEASED`，推荐入口会在 `rclpy.init()` 前退出；
- `config/go2_leg_safety_contract.yaml` 为 `UNVERIFIED`，WBC 会在创建 LowCmd publisher 前退出。

这说明当前默认启动是 fail-closed 的，但不能证明 Phase A 已正确完成。复审发现多个仍属于 Phase A 范围的代码级 P0，包括 ARX constructor 提前写 CAN、artifact/contract 路径未可靠绑定、writer inventory 漏检，以及 legacy writer 可通过环境变量恢复。

下一步只能是：

```text
FIX PHASE A DEFECTS
```

## 2. Git 与改动基线

复审开始时：

```text
HEAD:   264f10cc7c65be5af7f0d88a6db0043113265c97
branch: main
index:  无 staged 修改
worktree: dirty
```

当前 HEAD 与原始审查基线完全相同，因此：

- 基线到当前 HEAD 的提交差异为零；
- 所有 tracked 修改均位于未提交工作树；
- 另有大量 untracked Phase A 文件，普通 `git diff` 和 `git diff --stat` 不会显示它们；
- 当前 Git 状态无法独立证明每项修改发生在 Phase A 之前还是期间。

可按实现语义识别的 Phase A 内容包括：

- `SafetyStateMachine`、`SafetyLeaseMonitor`；
- `FinalLegCommandSafety`；
- arm observation session/sequence/freshness gate；
- Go2/X5 ownership lock；
- artifact manifest 与 Go2 safety contract；
- legacy/vendor writer gate；
- 对应安全测试及 production wiring。

README、height-scan、文档迁移和 onboarding 等修改不属于 Phase A 最小安全封堵，但同样处于当前未提交工作树中。

## 3. 已验证有效的实现

以下性质可从当前生产代码得到较强证据：

1. 推荐 WBC 和 Arm 主入口中的 ESTOP/shutdown 已不再调用 `reset_to_home()`。
2. WBC 的 R2 会撤销输出许可、清除 `start_time`、alignment、policy 和 pose-test 标志，发布有界 passive LowCmd，并立即从 joystick callback 返回。
3. FAULT、ESTOP、arm producer session 变化及 heartbeat session 变化均为 latch，不会因消息恢复自动进入 ACTIVE。
4. `/arm/state` 与 `/arm/target_state` 分别维护 freshness、source、session 和 sequence；运动状态下任一路 stale/invalid 都会触发 latched fault。
5. 推荐 WBC 路径在 LowCmd publisher 创建前获取 `go2-lowcmd` flock。
6. 推荐 Arm 路径按固定顺序获取 `x5-can`、`x5-gripper` flock，并在 ARX controller 构造前完成获取。
7. X5 CLI 和 Arm Node 都仅接受精确型号 `X5`。
8. ESTOP topic 和 safety heartbeat 使用 reliable、depth 1、transient-local QoS；`false` ESTOP 不会解除 latch。
9. 正常 Python exception 路径有 `finally` cleanup；主 shutdown 方法具有幂等保护。

这些结论不覆盖 SIGKILL、`os._exit`、进程崩溃、外部非合作 writer 或真实驱动器断联行为。

## 4. 阻断性发现

### 4.1 ARX constructor 在 preflight 前写硬件

Python 层设计的顺序是：

```text
ownership lock
→ ARX constructor
→ explicit damping
→ feedback preflight
→ operator ARM
→ position gain/output
```

但 ARX vendor C++ 源码证明 constructor 内部已经：

- 向各 motor 发送 clear frame；
- 对 DM motor 发送 enable frame；
- 读取并复制 motor telemetry；
- 设置 `kp=0`、`kd=default_kd` 的 damping gain；
- 发送多轮 joint/gripper command；
- 启动 `controller_dt=0.002` 的后台 send/recv 线程。

因此“feedback preflight 和 operator ARM 前没有硬件输出”不成立。该结论属于 `VENDOR-SOURCE-PROVEN`，不是 mock-only 推断。

此外，Python preflight 使用的 feedback timestamp 是 SDK 在复制当前缓存后写入的本地 controller time，并不证明对应 motor 刚刚收到新 CAN frame。全零位置检查也可能拒绝合法零附近姿态。

### 4.2 Artifact manifest 没有绑定实际加载物

当前 manifest 仍存在以下缺口：

- `--artifact-manifest` 可指向仓库外 manifest；
- `--final-command-contract` 可加载与 manifest 中 contract hash 不同的文件；
- manifest 哈希 x86_64 ARX 库，而目标 Jetson 实际加载 aarch64 库；
- 未哈希实际安装和 import 的 `arx5_interface` pybind extension；
- Unitree/ARX SDK snapshot 只要求非空字符串，不是可验证 hash；
- 被哈希的 CycloneDDS XML 不等于 `setup_env.sh` 实际使用的内联 `CYCLONEDDS_URI`；
- `dirty_worktree_policy=ALLOW_EXPLICIT` 没有文件级 explicit allowlist；
- 验证与实际加载之间仍有 TOCTOU。

当前 `UNRELEASED` 状态能够临时阻止启动，但只把问题推迟到 release manifest 被填写时，并未解决上述绑定缺陷。

### 4.3 Writer inventory 不完整

当前 inventory 只扫描：

```text
.py .cpp .cc .cxx .sh
```

并只识别有限的 publisher/controller/cansend/set command 正则。它没有覆盖：

- `.h`、`.hpp` 中的 `ChannelPublisher`；
- Unitree SDK2 native `rt/lowcmd` writers；
- CMake、launch、Docker/compose 和 install wiring；
- `set_gain`、`set_gripper`、`reset_to_home` 等硬件动作；
- 动态调用和 topic 宏。

因此“32 candidate files, all classified”是扫描器自身定义下的结果，不代表仓库 writer 完整分类。`unitree_sdk2` 中的 `low_level`、`stand_example_go2` 和 state-machine writer 均未被发现。

### 4.4 Legacy writer 仍有参数绕过

`real-wbc/scripts/run_wbc.py` 只要求：

```text
GX_REAL_HARDWARE_MODE=offline
--offline-legacy-only
```

随后仍会调用 `rclpy.init()`、构造 legacy WBC、创建 LowCmd publisher 和 ARX controller。环境变量名不能证明进程处于离线环境，因此该路径不是硬阻断。

legacy `wbc_node.py` 仍包含：

- LowCmd publisher；
- `Arx5JointController`；
- `reset_to_home()`；
- `set_gain()`；
- `set_joint_cmd()`。

### 4.5 FinalLegCommandSafety 尚未证明全部不变量

生产 non-passive chain 已形成：

```text
ONNX output
→ clip/scale/offset
→ policy joint order
→ hardware joint order
→ FinalLegCommandSafety
→ motor_cmd
→ CRC
→ LowCmd publish
```

但仍有两个关键缺陷：

1. R2、FAULT、ESTOP 和 shutdown 不会调用 limiter `reset()`。停止一段时间后重新 ARM，下一次 validate 很可能因 `dt > max_dt_sec` 直接进入 FAULT。
2. limiter 先施加 `max_step`，随后根据 jerk/acceleration/velocity 重新积分位置；最终结果没有再次断言 `abs(command - previous) <= max_step`，因此在已有非零速度时不能保证最终 step limit。

`source` 和 `session_id` 也由同一调用点同时写入 expected/actual context，当前主要是内部一致性断言，不构成对上游 producer 的认证。

### 4.6 ESTOP 与 ARX 后台线程不同步

Arm ESTOP callback 的显式 Python 顺序是：

```text
latch ESTOP
→ output_enabled=False
→ arm_position_control_enabled=False
→ set_to_damping()
```

但 ARX 500 Hz background thread 不读取 Python `_safety_lock` 或 output permit。ESTOP callback 执行期间，后台线程仍可能发送前一 position target/gain，直到 `set_to_damping()` 更新 SDK 内部 gain/interpolator。

因此只能证明“第一条显式 Python controller call 是 damping”，不能证明“ESTOP 到达后没有并发旧命令写入”。

## 5. 原始问题状态摘要

| 范围 | CLOSED | PARTIALLY CLOSED | OPEN |
|---|---|---|---|
| GX-P0-01 ～ GX-P0-12 | P0-05、P0-07、P0-09 | P0-01、02、03、04、06、08、10、12 | P0-11 |
| GX-P1-01 ～ GX-P1-09 | P1-06 | P1-09 | P1-01、02、03、04、05、07、08 |

重点说明：

- P0-05：R2后的WBC非passive命令生成已关闭；
- P0-07：arm state/target持续freshness gate已建立；
- P0-09：错误X5型号入口已关闭；
- P0-11：legacy writer仍可通过环境变量恢复，保持OPEN；
- P1-05：ONNX和所有Go2 watchdog仍在同一executor，永久阻塞问题保持OPEN。

## 6. 测试审计

在静态确认测试不会初始化 ROS、LowCmd、ARX、CAN 或 SpaceMouse 后，实际执行：

```text
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest \
  -p no:cacheprovider -q -rs tests
```

环境与结果：

```text
Python:      /usr/bin/python3 3.10.12
pytest:      6.2.5
numpy:       1.21.5
PyYAML:      5.4.1
onnxruntime: 未安装
passed:      146
failed:      0
skipped:     1
warnings:    0
```

安全相关 skip：

```text
tests/test_policy_height_scan_contract.py:53
could not import 'onnxruntime'
```

主要测试质量问题：

- WBC safety wiring 多为 `source.index()` 静态字符串测试；
- X5 测试使用 fake controller，未执行 vendor constructor；
- QoS 测试只搜索枚举文字，没有实例化实际 QoS 对象；
- 没有 ROS message clean build；
- 当前 `real-wbc/ros2/install` 生成文件早于 `.msg` 修改，且不含新增字段；
- 没有 SIGINT、SIGTERM、spin exception 子进程测试；
- 没有 callback 并发、ARX后台线程、CAN bus-off 或永久阻塞 fault injection；
- 没有 R2→等待→重新ARM的 limiter 测试；
- writer inventory allowlist过宽且扫描范围不完整。

因此 `146 passed, 1 skipped` 只能证明当前单元测试断言通过，不能证明生产安全声明成立。

## 7. 未关闭且不得误判的问题

| 项目 | 状态 |
|---|---|
| WBC SIGKILL后Go2 receiver行为 | BENCH-TEST-REQUIRED |
| executor/ONNX永久卡死 | OPEN |
| Arm Node SIGKILL后X5驱动器行为 | BENCH-TEST-REQUIRED |
| ARX constructor内部发送 | OPEN；vendor源码已证明会发送 |
| CAN bus-off/error-passive/down/recover | BENCH-TEST-REQUIRED |
| 外部非合作DDS writer | OPEN |
| 外部非合作CAN writer | OPEN |
| 独立硬件ESTOP/动力切断 | BENCH-TEST-REQUIRED |
| Go2物理安全合同 | OPEN |
| ONNX Runtime版本和实际P99延迟 | BENCH-TEST-REQUIRED |
| ROS message rebuild和DDS联调 | BENCH-TEST-REQUIRED |

正常退出与进程强杀必须分开评价：

- SIGINT、SIGTERM和Python exception：生产入口已有 `finally`，但仍缺目标ROS环境的signal测试；
- SIGKILL、`os._exit`、进程崩溃：不会运行Python cleanup或ARX destructor，本轮没有解决。

## 8. Phase A最小修复顺序

### P0-1：禁止ARX constructor提前写硬件

- 将 controller 创建拆成 receive-only preflight 和显式 enable/start；
- constructor阶段不得clear、enable、发送motor command或启动发送线程；
- fake CAN记录必须证明operator ARM前TX数量为零。

### P0-2：原子绑定manifest、contract与实际加载文件

- production入口固定repo内manifest路径；
- final contract只能来自已验证manifest；
- 哈希实际import的pybind extension、aarch64 backend及解析后的ELF依赖；
- 验证实际`CYCLONEDDS_URI`；
- production拒绝所有dirty worktree。

### P0-3：真正阻断legacy/vendor writer

- legacy production入口无条件退出；
- 离线分析代码不得import或构造hardware writer；
- Unitree SDK2 writers增加默认OFF CMake gate；
- production install/build tree不包含writer examples。

### P0-4：重写writer inventory

- 扫描Python、C/C++、headers、shell、CMake、launch、Docker和compose；
- 识别LowCmd/ChannelPublisher、ARX controller、set command/gain/gripper/home、CAN调用；
- allowlist按文件和明确原因分类，不允许整个vendor目录前缀放行。

### P1：修复final limiter生命周期和不变量

- R2、FAULT、ESTOP和shutdown时reset；
- 每次重新获得output permission后只从fresh lowstate prime；
- validate返回前重新断言position、step、velocity、acceleration和jerk全部满足合同；
- 增加可控时钟和property tests。

### 验证补齐

- clean build `robot_state` messages；
- 实际QoS compatibility和late-joiner测试；
- SIGINT/SIGTERM/spin exception子进程测试；
- ARX constructor零TX测试；
- 跨host/container真实flock inode测试；
- 清除ONNX安全相关skip。

## 9. Phase B进入条件

只有以下条件全部满足后才能进入Phase B：

1. 独立复审无新增P0；
2. 所有Phase A代码级P0关闭；
3. 安全相关测试无skip；
4. ROS message可clean build，生成接口与源码一致；
5. 所有production入口在真实publisher/controller/CAN初始化前fail-closed；
6. ARX constructor副作用已通过receive-only改造消除，或明确形成经批准的Phase B隔离方案；
7. writer inventory没有未分类候选；
8. 当前完整diff再次经过独立复审。

在满足以上条件前，不得连接真机、创建LowCmd publisher、打开can0或构造真实ARX controller。
