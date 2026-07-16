# Flat policy releases

Flat 真机入口按用户指定继续使用仓库当前资产：

- `policies/policy.onnx`
- `policies/env.yaml`
- `config/artifact_manifest.yaml`

运行时只接受：

- manifest 的 `policy_kind: flat`；
- env 中 `_zero_height_scan`、`scene.height_scanner: null` 和 plane terrain；
- 260D 输入、12D 输出；
- `[66:253]` 由 `FlatDeployment` 生成的 187 个精确零。

模型、checkpoint 和 NPZ 使用外部 artifact store 或 Git LFS；YAML 合同和 manifest 应进入版本控制。当前 flat manifest 仍保持 `UNRELEASED`，直到最终腿部安全合同和独立复核全部完成。
