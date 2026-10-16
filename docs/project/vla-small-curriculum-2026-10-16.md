# 小课程语言拟合与闭环诊断

用户授权继续“站立、前进、后退”小数据充分拟合。先用 v26 零更新诊断区分文本条件拟合与闭环执行，之后再冻结小课程预算。

## 已有证据

v25 已结束，独立最终审计 3749 项零不匹配。baseline/control/augmented 后退严格成功均 0/12；各有 9/12 全程稳定，但同义表达通常仅移动毫米。原话会后退后倒地。旧任务成功分别 7/10、8/10、8/10。1000 更新只抽取 8000 个样本，约占复制后的 42000 训练帧 0.19 遍；这不证明欠拟合是唯一原因，也不足以否定语言增强。

原始证据：devbox `/srv/duckmind-training/v25-owner/independent-final-audit.json` 与 `.md`，完整训练和物理记录在同目录 `run/`。仅同步仿真，无实时或视觉能力结论。

## v26：六种训练说法是否拟合

复用 head_curriculum.py 的诊断入口，保持 v24 参数和路径兼容。v21 原始教师 700/702 与第一条已接受纠偏轨迹各取 0/20/60/140 tick，12 个已见训练状态。六种 v25 后退文本、站立和前进两个异目标对照，2 个相同 flow noise，比较 v25 初始模型与 augmented 最终模型，共 386 次推理（含每模型一次精确重复），零训练、零物理推进。

保存物理 state、实际 prior、教师目标、所有 native inputs、RNG、预测、时延。分原始/纠偏、起步/持续报告腿关节 E5/H10 误差、persistence、同义文本与随机噪声差异。两个对照文本没有对应教师目标，仅评估动作分离，不能把其后退误差当任务分数。模型统计仍为 v21 full train 统计。

预测：若同义表达的教师状态误差仍明显分化，小课程优先加强语言绑定；若教师状态相近但闭环失败，则后续应检查状态偏离。两者可并存，不能将离线误差等同走路成功。

## 小课程约束

下一阶段仅三任务，匹配身体初态，完整轨迹按训练/评估拆分。图像保持黑图，state51/H10/E5、原生 LeRobot 训练器不变。所有文本的首段、持续段与教师接管状态分别检查，防止 previous_target 的惯性续写掩盖语言无效。训练预算、统计选择与 checkpoint 存储在启动前固定。当前尚未启动新训练。


## v27 冻结方案

采用v25 augmented最终权重。训练初态700/730，各采stand/forward/backward200tick；三任务共享相同reset physics/state/prior。6条真实轨迹各配6文本，36episode/7200帧；每task同曝光权重。老师须严格通过且不被误判为另外两任务；失败保留raw并中止，不调阈值。评估770/771不进训练、不提前教师筛选，旧报告body/seed字段检索未命中。

LeRobot0.6.2 fresh warmstart原生会重建processor统计，没有独立normalization-reference选项。使用新dataset真实float64统计，zero只换统计、保持v25权重；zero和正式模型共享新统计。不能把v25到zero变化当学习效果。

预算2步smoke丢弃+4000正式，batch8/H10/E5/seed42，optimizer重启，不追加；原生save_freq1000，保留1000/2000/3000/4000。4个正式+smoke+zero约9.2GiB，启动前root18GiB可用，不删除旧工件。4000步约4.44次采样遍历，不预称充分拟合。

zero与1000/2000/4000分别对每条轨迹六个时刻0/5/10/20/60/140、全部六文本、两个flow noise进行432次训练状态推理；每次保存state/prior/target/prediction与文本/噪声索引。这是分期anchor诊断，不是全帧验收。zero/1000/2000各训练初态三任务原话6条，final四初态三任务原话/第六种训练说法/留出说法36条；全部200tick。新初态只有final，不能报告pre/post改进。不同阶段冻结物理噪声；一optimizer seed不构成训练重复。smoke为首训练初态三任务各10tick。

教师采集和学生推理均复用现有Simulation/Trace/outcome，学生仅使用文本条件模型，无规则分派。独立设计审查未发现启动阻断；源码检查training lint/test为17 passed/2 skipped，Linux真实训练链路待smoke。完整新数据审计之后才启动训练。入口run-small-curriculum.sh，工件 `/srv/duckmind-training/v27-owner/run`。


## v26 已完成

run.stage=complete、exit0，386推理，零更新/零物理推进。独立CPU审计5874项通过、零差异，报告 `/srv/duckmind-training/v26-owner/independent-final-audit.json/.md`。六种训练文本在原轨迹起步/持续、纠偏接管早/晚四组的腿部E5与H10教师误差全部低于baseline。汇总E5 MSE（12状态×2噪声，rad²）：

| 文本 | baseline | augmented |
|---|---:|---:|
| 向后走 | .001458 | .000908 |
| 往后退一点 | .001291 | .000767 |
| 朝身后退着走 | .002637 | .001065 |
| 向后迈步 | .001626 | .000873 |
| 往身后走 | .001252 | .000826 |
| 往后退步 | .001650 | .000935 |

残差集中在原轨迹起步：augmented六说法E5 MSE约.001365–.002182；持续段约.000372–.000423。朝身后退着走起步.003671→.002182，持续.001624→.000423，纠偏早段.003520→.000680。不能据此断言同义表达完全没有学到；下一步检查共同初态中的语言选择与闭环恢复。

这里没有v25等预算control，只能归因整套继续训练配方，不能独立证明文本增强收益。12个状态均来自训练轨迹，纠偏early是接管后早段，不是静止起步。未测新语言/新初态，未推进身体，也没有实时结论。Native输入快照是同路径独立preprocess调用，非predict内部hook。

v27准备已完成：6/6教师严格成功，错误目标无误通过；36ep/7200帧已生成，zero权重不变。正在独立preflight，尚未启动正式训练。代码提交78e9b5644c663740732be57ccf0ca53aa062b920已推到PR5，remote一致。devloop完成提交推送后因过期action-chunks路径postcheck退出1，不能称整个脚本成功；training与policy验证已通过。


## v27 运行记录

独立preflight最终8440检查通过；CPU native loader首8与全部36episode尾部、6教师raw、36副本、全部7200黑图、episode/global统计、6zero有效张量与哈希已核实。审计自身前两次sampler调用/epoch使用错误保留attempt1/2，修正的是审计脚本，未修改数据或训练实现。准备后root可用16.11GiB。

2026-10-16 16:08（Asia/Shanghai）启动唯一launcher PID74446，run.pid已匹配，smoke完成2次实际更新，155个optimizer参数step均2；真实num_workers2通过。三任务各完整稳定10tick，实际输入/previous_target/chunk前5步与关节执行逐项吻合，独立smoke88checks通过。报告在owner `independent-smoke-audit.json/.md`。随后进入zero-fit；正式4000更新在zero-fit/zero-eval后自动开始，尚不能报正式完成。

完整日志：`/srv/duckmind-training/v27-owner/run/run.log`。状态看同目录run.stage/run.exit-code，禁止重复启动。源码快照在owner/source-snapshot；当前仅smoke2更新已证实，总预算4002。代码与teacher/data冻结，不自动扩新实验或重新启用自动跟进。正式结果完成后仍需独立审核，不能从loss或smoke推断成功。
