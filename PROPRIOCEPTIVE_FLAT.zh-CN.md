# 无外部XY定位的平地策略

从model_13493独立续训2000轮，Actor483/Critic560、12个腿动作不变。部署入口为`scripts/run_proprioceptive_flat_real.sh`。旧13493及原入口保留；新模型未训练完成前不切换默认权重。

## 观测契约

- 高度66:253始终全0，站立位置误差263:265始终全0；Actor不读取外部XY定位。
- 相对航向265:267使用IMU四元数相对于站立起点的yaw，不要求全球朝向。训练模拟±0.001rad/s航向漂移。
- 线速度0:3来自IMU角速度、腿关节角/速度与足端接触的运动学估计，缩放×2。
- 角速度、投影重力、关节状态、臂目标、上次动作、相位和四帧历史保留。训练关节速度噪声幅度改为0.1rad/s。
- Critic和奖励仍读取仿真真实位置/速度；不把特权量传给Actor。

此版本使用Go2已有足端力信息（阈值20N），并非严格仅IMU和关节编码器。没有外部定位，也不把IMU加速度双积分当作可靠XY位置。

## 共用速度估计

训练和部署共用`proprioceptive_math.py`，分别以Torch/NumPy执行。按URDF计算四足FK及速度，使用`-J*dq - omega×r`补偿机身转动；选取相互一致（差≤0.2m/s）的支撑腿，再以60ms时间常数低通。训练增加传感器噪声及3%接触丢失。

没有接触时暂时保持上次估计；部署连续失去支撑观测超过0.25s则停止。支撑腿同时滑动不能靠一致性过滤发现，因此无条件5cm绝对位置保证不成立。

## 启动与版本检查

先将**本次续训产生的checkpoint及同次env.yaml**用`export_unified_actor.py`导出到新的独立目录。导出器写入`proprioceptive_flat_v1`标识，拒绝把原13493直接重标为新策略。

```bash
python scripts/export_unified_actor.py \
  --checkpoint /path/to/model_15492.pt \
  --env /same/run/params/env.yaml \
  --output policies/proprioceptive_flat_model15492

# 默认手柄，只有明确指定已训练的新包才允许启动
GX_REAL_PROPRIO_BUNDLE="$PWD/policies/proprioceptive_flat_model15492" \
  bash scripts/run_proprioceptive_flat_real.sh

# 固定零速度站立
GX_REAL_PROPRIO_BUNDLE="$PWD/policies/proprioceptive_flat_model15492" \
  bash scripts/run_proprioceptive_flat_real.sh --base-command-source fixed --cmd-vx 0 --cmd-vy 0 --cmd-yaw 0
```

包校验在MCF释放前执行。旧包会被明确拒绝，不会静默退回依赖外部定位的策略。当前旧默认入口仍指向13493，可用于回退。

运行时不创建外部base位姿订阅，改用LowState中的IMU四元数；仍检查LowState、机械臂状态/目标、接触速度估计、MCF以及急停。动作限制、关节限位、启动保持和收臂门控保留。外部机械臂节点仍是CAN唯一写入者。

## 训练与验证边界

4096平地环境，seed20261028，固定lr5e-5；六类正负纯轴、混合命令及45%站立采样，25条完整动臂路径、随机空载/0～450g与现有摩擦/质量/COM/PD扰动。旧训练任务不修改。

离线测试覆盖NumPy/Torch估计器一致性、FK有限差分、旋转补偿、无外部pose输入、IMU相对航向、零高度/零XY通道和支撑丢失停止；不能替代真机验收。完整验收应同时报告位移、速度、姿态、脚滑、跌倒/碰撞和任务完成率，尤其复测右移。
