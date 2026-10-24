# v29：早期后退纠偏是否真的可恢复

v28证明教师轨迹拟合和学生访问状态上的误差不同，但单步教师查询没有证明身体能够恢复。本轮直接验证恢复能力，零新VLA调用、零优化更新，不改变既有成功门限。

固定v27 step4000的两条原话后退轨迹：seed700失稳、730稳定但偏航。分别从原始reset精确回放20/40步，再调用已有resume_teacher恢复实际previous target对应的专家历史，持续由专家执行200步。两个初态各另跑一条200步纯教师正控。全部前缀/后缀、失败与原始物理状态保留；接管不重设origin或yaw，原任务方向始终有效。

每条候选用原locomotion_outcome严格判定：完整200步、全程稳定、后退进展、横移约束、原航向、尾段持续行走和直立。只将通过候选写入accepted清单，示范标签仅来自专家后缀，学生前缀不进入标签。4条候选只有2个已知初态，窗口相互重叠，不是4个独立身体或新泛化测试。

原学生轨迹剩余时间不足200步：700为91/71步，730为180/160步。因此单独报告相同时间长度的物理比较，不把截断学生后缀的complete=False当作完整200步因果对照；原门限至少需要100步，较短窗口要读取平衡/进展等分项。

原教师/上游和v27原始轨迹SHA需与v28证据相同；逐帧核验prefix state、prior、integration、actuator targets、metrics；接管瞬间物理与实际目标不变。逆映射有效性已有v28独立原生标签校准（最大差1.79e-7rad），本轮继续复用相同Simulation实现。

入口`head_curriculum.py collect-early-recovery`与`run-early-recovery-curriculum.sh`；输出`/srv/duckmind-training/v29-owner/run`。本轮先输出候选原始数据及accepted清单，不生成新的LeRobot训练集、不启动训练。通过独立复核后，再决定数据混合与配对训练预算。旧结果与模型不覆盖、不删除。

## 采集完成与数据准备

唯一launcher3818103已退出0，2/2纯教师正控、4/4纠偏候选通过。四条后缀均完整稳定200tick，后退进展分别0.592/0.656/0.587/0.668m；原航向约束保持。候选来自700/730各20/40两个前缀，仅新增800个物理帧。这是教师恢复可行性证据，不是学生模型改善。

下一步复用原生prepare_data整理独立`data-preparation/dataset`：保留v27旧36episode/7200帧精确数值与文本，再将4条后缀各配原有6种训练说法，新增24episode/4800帧，合计60episode/12000帧。只复制语言标注，不称为新增24条独立示范；不引入未见说法、不包含前缀标签。图像保持黑，state51/H10/50Hz不变，episode/global统计用真实float64计算。

复用prepare_data同时保存不更新权重、仅换新统计的zero checkpoint，供未来归一化对照使用。后退采样占比由1/3升至3/5，原数据曝光随之下降；之后的配对实验须明确这个配方变化并回归站立/前进。当前新增预算仍为零优化更新。

独立采集审计172项通过：1200个纯教师/恢复动作标签均由CPU ONNX重算，最大差2.98e-7rad；四个prefix与接管首帧、actual previous target、物理时序、原方向和既有strict判据一致。原学生seed700在相同91/71步窗口末失稳，而教师保持稳定并后退约.217/.221m；seed730两个共同窗口中学生/教师均稳定，教师保持原方向，完整4秒纠偏亦通过。不能因此宣称学生训练已经改善。

采集源码在owner/source-snapshot，数据准备在采集完成和独立复核后新增，另存data-source-snapshot与data-source.sha256；后续复核采集应使用其冻结快照，不能要求当前driver永远等于旧版本。独立报告为owner/independent-final-audit.json/.md及ml-debug-form.md。

## 数据准备完成

prepare入口PID3830352已退出0。独立数据审计1772项通过：旧7200帧的数值/文本/索引精确保留，新24episode逐帧对应四条专家后缀的六种语言副本；全部12000帧解码为黑图；60episode及global的mean/std/min/max/count按float64复核；240个原生H10样本包含边界padding校验；6个有效processor张量与dataset统计一致，zero权重SHA仍与v27 step4000相同。未新增VLA调用或优化更新。

新数据集：`/srv/duckmind-training/v29-owner/data-preparation/dataset`。仅换统计的零更新对照：同目录`zero-checkpoint`。映射表`recovery-mapping.json`明确物理轨迹和语言副本的关系。独立报告`/srv/duckmind-training/v29-owner/independent-data-audit.json/.md`及ml-debug-form.md附录。

下一轮可用同一初始模型、共享统计与相等更新预算比较原36episode和混合60episode，明确后退采样比例和原数据曝光同时变化；评估必须包括原来两个后退失败、站立/前进回归和预先保留的新初态。当前只完成可用纠偏数据的准备，尚未启动这轮配对训练。
