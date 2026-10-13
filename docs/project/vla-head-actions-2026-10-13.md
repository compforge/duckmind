# 头部动作课程：从向前走扩展到多任务语言策略

## 目标与基线

在固定 Microduck 构型上，让已有学生学习保持站立、向左看、向右看、抬头、低头，
并保留向前走能力。仍使用黑色固定图像，研究的是语言条件运动策略；不证明视觉能力。
起点是 [v17](vla-long-walking-2026-10-13.md) mixed step1000，
该模型长时平衡有所改善，但严格直线行走尚未与专家对齐。

PR #4 已合并；本轮工作树 `vla-head-actions` 从 `origin/main` 的 `0d332fd` 开始。
实验入口为 `training/experiments/head_curriculum.py`，启动命令保存在同目录
`run-head-curriculum.sh`；LeRobot 原生训练循环及 SmolVLA 架构保持不变。

## 数据与预算

- 保留 v17 全部向前走数据：30 episodes、10,600 frames，包含已有重复前缀，不能视为全新轨迹。
- 新增五类动作各 10 条候选：seeds 600..609，每条 200 ticks / 4 秒；
  相同 seed 的物理状态、36 维观测和上一执行目标在不同任务间必须完全一致。
- 每类交替使用两种训练表达；完整保存成功与失败轨迹，仅接受合格示范。
  每类至少 8 条通过才继续，全部通过时训练集为 80 episodes / 20,600 frames。
- 独立采集 seeds 620..622 的六类教师验收轨迹，不参与训练或归一化统计。
- state51 = body36 + actual previous target15；action 为 15 维绝对关节目标 rad。
  H=10、E=5、50 Hz，训练与实际推理图像均为 uint8 零值 256×256×3。
- 2 步 smoke 丢弃，正式从 v17 mixed 权重重新开始 2000 步；batch8、optimizer seed42，
  优化器重启。总新增优化预算 2002，不自动追加。

样本仍由原生 LeRobot 按帧采样。向前走约占一半帧，五个新任务分配另一半。
这同时改变任务覆盖、数据比例与归一化，不能声称隔离了单一原因。
2000×8/20600≈0.78 次平均帧暴露；如果失败，需要区分学习不足、遗忘和实现问题。

## 验收

baseline 使用 v17 原统计；zero 使用相同权重与新训练统计；final 使用训练后权重。
三者分别在相同 3 个身体初态 × 6 任务 × 2 说法上测试，保持相同推理噪声。
36 条 rollout 只有 3 个身体初态，一次优化器训练，不是 36 个独立训练重复。

预先声明的新表达：站在原地保持不动、脑袋朝左边转一点、脑袋朝右边转一点、
抬起脑袋看看上面、垂下脑袋看看下面、朝着正前方迈步前行。
这些文本必须不出现在本轮训练集；历史模型是否见过其他近义表达不作无依据保证。

头部动作运行 4 秒。要求全程稳定、最大水平漂移 < 0.08 m、yaw < 0.5 rad，
末一秒 head_pitch/head_yaw 对目标的 MAE < 0.15 rad，有指令的关节还需沿正确方向超过 0.1 rad。
颈部和roll误差完整报告；未命令时不作为零目标的主要验收轴。除此之外，还从保存的完整物理
状态重算校正后相机光轴相对躯干的方向：相对初始中性姿态，受命轴需沿正确方向 >0.1 rad，
另一轴平均绝对变化 <0.15 rad。站立要求两轴变化均小于0.15 rad。教师和学生共用此规则。
`*_delta` 是相对默认姿态，而不是相对本次初态。该复合判据是本实验的操作性定义；
原有误差尺度与漂移门限被保留，新增持续性与视线要求在教师、站立和错误目标上标定；不把关节零值等同于自然语言目标。
同一轨迹按五种目标评分，报告混淆矩阵，不能只看是否保持平衡。

向前走运行 12 秒，沿用 `recovery_outcome` strict，并单列 4 秒前缀和平衡稳定率。
保存每次实际模型输入、chunk、完整物理 trace 与延迟；同步等待推理会暂停物理时间，
不构成实时 robotd 部署证据。短 smoke 只检验真实数据/训练/推理/执行通路；
10 ticks 下 strict=False 是预期，不能当作完整任务评分。

## 实验前判断

| 选择 | 预期影响 | 本轮决策 |
|---|---|---|
| 五类静态任务 + 行走 replay | 增加可区分语言任务，检查行走遗忘 | 执行 |
| 一次加入所有技能 | 难以定位多种失败，还涉及其他教师模型 | 暂不执行 |
| 改结构、loss 或移除 previous target | 可能改变表达能力，但尚无本轮证据 | 固定现有实现 |

主观先验而非结论：多任务学习并保持行走 45%；任务干扰/采样失衡 25%；忽略文本 15%；
数据或评估 bug 5%；未知 10%。若同初态输出近似相同、离线拟合却很好，重点查语言启动
与 previous-target 维持捷径；若 zero 已全面退步，应先区分统计变化；若只在训练表达有效，
不宣称语言泛化。目标是新增能力验证，非证明某训练配方优于所有替代方案。

独立审查指出：跨关节或先时间平均会掩盖错动作，现按每关节 tail MAE；
原 collector 仅存成功轨迹的限制由本入口保留全部 Trace 解决；smoke 中无效目标或不完整
执行必须阻止正式训练。本次使用同模型新上下文独立审查，未进行不同模型家族交叉审查。

## 运行与证据

远端代码保留 home 相对路径 `~/rustprojects/compforge/duckmind/.worktrees/vla-head-actions`。
依赖复用原工作树虚拟环境，但 `PYTHONPATH` 明确指向新工作树 training/src 与 policy/src，
已核对两个实际 import 路径；不修改旧环境安装。工件写入根盘
`/srv/duckmind-training/v18-owner/run`，启动前根盘约 53 GiB，data 盘约 5.2 GiB。
不删除已有数据或模型。

首次 prepare 在第一条轨迹写报告时出现 `TypeError: Object of type bool is not JSON serializable`，
是 NumPy bool 标量，零训练。原始目录保留为 `prepare-attempt1`，完整日志为 `prepare.log`；
修复为内置 bool 并增加 JSON 序列化和错误动作测试，重新采集日志为 `prepare2.log`。

第二次 prepare 完成50条训练和18条教师验收轨迹后停在抬头门禁：10/10抬头都稳定，
但neck约−0.229rad越过原先对零目标的0.15rad误差。原命令确实包含neck=0，
不能说四轴判据本身没有依据；这里要学的是文本任务，不是精确复制每个命令寄存器。

独立审查提醒两pitch轴方向相反，可能互相抵消。使用`calibrate-gaze`重算校正后相机光轴：
三个初态抬头的视线仰角分别由−0.1774/−0.1808/−0.1748变为+0.0375/+0.0520/+0.0394rad，
变化+0.2149/+0.2329/+0.2142rad。站立尾段约−0.19rad，低头约−0.40rad。
因此在学生验收前锁定上文的文本任务判据，并保留原四关节误差。
第二次完整数据和校准结果保留于`prepare-attempt2`，不作为一次成功prepare报告。

第三次prepare已完成，完整日志`prepare3.log`共1767行，无新异常。
独立审计`independent-preflight-audit.json`核验68份raw、80episodes/20600frames，
旧10600帧及文本不变、新10000帧逐帧对应raw、eval和新措辞排除，
episode/global float64统计与6个zero processor张量相等，权重/数据SHA一致，
80个episode末帧的原生H10padding及黑图均通过。

教师五类任务训练50/50、独立初态15/15通过；错误目标误通过0/60。
前进教师4秒strict3/3，12秒strict0/3，仅heading失败，三条均稳定600ticks。
这再次说明不能把长时平衡与严格直线行走混为一谈。

已启动launcher PID4163605（先smoke，随后baseline/zero/正式2000/final），
实时状态见`run.pid`、`run.stage`、`run.log`、`run.exit-code`。源代码提交`eed6fd9`，
新PR为 https://github.com/compforge/duckmind/pull/5 。学生新动作结果尚待验收。
代码规范通过，CPU测试13 passed / 2 skipped；这不代替GPU smoke和闭环任务评分。
依赖记录见`runtime-provenance.json`，输入源哈希见`run-source-sha256.json`。

独立校准也使用620..622，因此这些是未进入训练的校准/评估初态，不能称完全未触碰的盲测。
自动跟进已切换到本轮，预算耗尽并交付结果后暂停，不自动开启下轮实验。

## ml-debug 运行表（准备阶段）

| 检查 | 当前证据 / 待验 |
|---|---|
| 完整日志与配置 | 首次 prepare 完整 traceback 已读；最终prepare3完整1767行无新异常，config.json固定任务/种子/预算 |
| SHOULD 与实测 | 同初态、黑图、stats与native边界独立通过；训练预算实际值待完成 |
| null 与尺度 | 教师静态任务15/15、错目标0/60；行走12s strict0/3但全部稳定 |
| 初始 demo | baseline / zero 待执行 |
| persistence | 入口已包含每任务起点/20tick的12chunks对照；运行结果待填，不代替闭环 |
| baseline / held-out | 待执行；新初态620..622不进入训练 |
| warmup / lr | 原生 warmup100、decay2000；实际日志待填 |
| 完整 sample | 第一条教师 trace 已落盘；模型实际输入/chunk 待 smoke |
| 最差 step / grad | 未训练，unknown；需要原生训练日志 |
| 意外日志 | NumPy bool 序列化失败；explained: 已修复并测试 |
| 尚缺证据 | 真实GPU smoke、正式训练与最终物理效果 |
| 竞争解释 | 见上方先验，均尚未作为结果归因 |
| 新上下文审查 | 发现平均误差与smoke gate风险，已修正；剩余需实际产物验证 |
| 最便宜区分实验 | 同初态不同文本/同噪声的首个chunk + 短闭环，区分文本启动与姿态维持 |
| 耗时与GPU | prepare 使用 ONNX CPU 与 MuJoCo 渲染；正式峰值显存待原生日志 |

## 2026-10-13 04:28 UTC 阶段记录

smoke实际完成2次更新（原生training_step=2），六任务各完整执行10ticks且稳定；
严格判据False符合短轨迹定义。baseline和zero各36条已完成，正式训练已开始并保存step500。
当前没有final结果，不能声称已学会新动作或发生训练后遗忘；源文件哈希检查仍一致。

| 任务 | baseline任务通过 / 稳定 | zero任务通过 / 稳定 |
|---|---|---|
| stand | 1/6 / 2/6 | 3/6 / 6/6 |
| look_left | 0/6 / 0/6 | 0/6 / 0/6 |
| look_right | 0/6 / 1/6 | 0/6 / 0/6 |
| look_up | 0/6 / 1/6 | 0/6 / 3/6 |
| look_down | 0/6 / 6/6 | 3/6 / 6/6 |
| forward（12秒） | 0/6 / 4/6 | 0/6 / 0/6 |

两组权重SHA均为`4f5e7dca40aa806de5c1aa5bd3b5fe489f1263679ce1c3afad49e10bd6e71936`。
每行6条仍只是3body×2措辞。零更新就出现能力变化，证明不能把final相对baseline的全部差值
归因于参数学习。保持既定流程，不因为这个预先设计的对照结果更改预算或判据。

当前完整日志快照1206行/78392字符无Traceback，step10 loss0.997、step530 loss0.076，
记录的mem_gb约2.33；只是优化过程，不代替闭环指标。后续保存的checkpoint以实际training_step为准。
训练集12个chunk中，静态任务在20tick后的persistence误差很小，也说明离线拟合可被维持姿态
捷径满足，仍需同初态文本启动和完整任务验收。
