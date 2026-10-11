# VLA 训练工程

## 定位与边界

在固定 Microduck 身体上，用既有运动策略生成专家示范并微调 SmolVLA。训练使用 LeRobot 原生数据集、训练循环和 checkpoint；推理适配既有 `Policy` 协议。这里不修改 robotd 的控制权或硬件路径。

## 代码地图

```text
src/duckmind_training/
  contract.py     # 状态/动作语义、任务采集参数、成功判据和 Episode 划分
  simulation.py   # 复用上游 MuJoCo + BAM + ONNX 专家，提供离线采集和评估身体
  collect.py      # 动作前观测、专家目标和物理结果的原始证据
  dataset.py      # 原生 LeRobotDataset 转换，分别保存训练/评估数据
  train.py        # 调用 LeRobot 原生训练器
  learned.py      # SmolVLA checkpoint 到既有 Policy 的适配
  evaluate.py     # 留出动作误差和同步物理评估
tests/           # 关节映射、数据边界和失败轨迹过滤
```

## 约定

- `TASKS` 只驱动专家采集，不能用于学习策略的推理分派。
- 每帧先观察再生成和执行动作；action 保存物理单位的关节目标，不是实际位置或 BAM 电机力矩。
- 训练/评估按完整轨迹与初始条件分离；标准化统计来自训练集。
- 当前状态是 15 个关节位置、3 个角速度、3 个投影重力；图像来自头部相机。新增模态必须同时修改采集与推理。
- 原始数据、模型、缓存和运行日志放在 `artifacts/` 或外部数据目录，不能进入 Git；不得默认上传 Hugging Face 或 W&B。
- 训练依赖固定于 Linux amd64 / CUDA 环境。`make fix`、`make lint`、`make test` 使用独立 CPU 检查环境；涉及数据或模型的改动还需在 Linux 运行小规模采集/训练验证。
- 仿真暂停等待模型的同步评估不代表真实时间控制；执行协议通过、离线误差下降、物理成功必须分别报告。

## References

- [操作说明](README.md)
- [数据与训练设计](../docs/design/vla-training.md)
- [模型侧接口](../docs/design/policy-interface.md)
