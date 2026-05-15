# 物料清单

这份清单按当前 `gx-real` 真机链路重新整理。价格和链接来自原始文档，主要作为采购参考，不保证实时准确。

## 1. 当前主链路必需

| 类别 | 物料 | 用途 | 备注 |
|---|---|---|---|
| 机器人 | Unitree Go2 Edu Plus | 四足底盘 | 原文参考价 `$12500`，联系 `sales_global@unitree.cc` |
| 机械臂 | ARX5 / X5 | 机载机械臂 | 原文参考价 `$10000`，联系 `contact@arx-x.com` |
| 通信 | USB-CAN adapter | X5 `can0` 通信 | 当前代码默认 `can0` |
| 通信 | 连接 Go2 的网线/网卡 | ROS2 `lowstate/lowcmd` | 网段通常是 `192.168.123.xxx` |
| 电源 | DC voltage converter | 将 Go2 电池侧电压降到 X5 可用范围 | 建议输出 24V |
| 电源 | XT30/XT60 接头、开关、线材 | X5 供电链路 | 接线前必须测正负极和输出电压 |
| 固定 | Go2-X5 安装板 | 将机械臂固定到 Go2 | 见 [3D 打印说明](3d_printing.md) |
| 工具 | M3 螺丝、剥线钳、接线端子、绝缘材料 | 装配和维护 | 真机现场建议常备 |

## 2. 当前调试建议准备

| 物料 | 用途 |
|---|---|
| USB WiFi 或 USB 网口 | 给 Jetson 提供外网，方便 `git pull`、安装依赖和远程调试 |
| 短网线 | 本地直连 Go2 网络 |
| 万用表 | 检查 X5 供电、电压和极性 |
| 扎带/魔术贴 | 固定 CAN 线、电源线和 USB 线 |
| 备用 USB-CAN | 排查 CAN 适配器故障 |
| 冰袋/散热辅助 | Go2 长时间站立和高增益调试时可能发热 |

## 3. 原始链接参考

### Go2 外设

| 物料 | 原始链接 | 原参考价 |
|---|---|---:|
| 3Dconnexion SpaceMouse | https://a.co/d/06NFKwu8 | 169.99 |
| USB3.0 Extension Dock | https://a.co/d/03erfsS2 | 19.99 |
| USB2.0 WiFi Module | https://a.co/d/0ciXDmGC | 6.99 |
| USB3.0 WiFi Module | https://a.co/d/0hogevhr | 19.97 |
| USBC Full Speed Extension Cable | https://a.co/d/006Kr2JI | 16.99 |
| USBC to Ethernet Adapter | https://a.co/d/0i7nzTLV | 23.99 |
| Ethernet Cable 2ft | https://www.amazon.com/dp/B094Z4SR6S | 7.99 |

### ARX5 电源和通信

| 物料 | 原始链接 | 原参考价 |
|---|---|---:|
| XT30 Male Connectors | https://a.co/d/0cCPW4Ci | 8.98 |
| XT30 Female Connectors | https://a.co/d/0hFC3DkJ | 8.98 |
| XT60 Switch | https://a.co/d/04wBpfON | 11.99 |
| 480W DC Voltage Converter | https://a.co/d/0aqB0RTf | 33.99 |
| USBA cable extender | https://a.co/d/08jO0Mdc | 7.99 |
| USB to CAN adapter | https://www.amazon.com/dp/B0CRB8KXWL | 22.99 |

### 原 UMI 视觉/采集外设

这些不是当前 `leg12 + arm passthrough` 主链路必需项。

| 物料 | 用途 | 原始链接 | 原参考价 |
|---|---|---|---:|
| iPhone 15 Pro | 原 iPhone VIO | https://www.apple.com/iphone-15-pro | 999 |
| GoPro Hero9 | 原 UMI 数据采集 | https://www.amazon.com/dp/B09J713ZS7 | 219.99 |
| GoPro Media Mod | GoPro 扩展 | https://a.co/d/01eDlS1u | 79.99 |
| GoPro Max Lens Mod | 广角镜头 | https://a.co/d/0bXdEG0X | 68.69 |
| Elgato Capture Card | GoPro 视频输入 | https://www.amazon.com/dp/B08FRSB1CM | 147.34 |
| Micro HDMI to HDMI Cable | GoPro 到采集卡 | https://a.co/d/0fD7mgUK | 12.99 |
| XT60 to USBC Cable | GoPro/外设供电 | https://www.amazon.com/dp/B0C7KYC64S | 12.89 |

## 4. 采购优先级

优先买齐：

1. Go2、X5/ARX5。
2. USB-CAN、X5 供电降压链路、安装板。
3. Go2 网络连接件。
4. 万用表、接线工具、备用线材。

暂缓：

- GoPro、采集卡、SpaceMouse、iPhone 支架。
- 这些只有在恢复原 UMI 任务空间 teleop 或数据采集时才需要。
