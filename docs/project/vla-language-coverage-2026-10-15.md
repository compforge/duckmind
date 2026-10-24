# v25：后退文本覆盖配对实验

v24 在固定教师状态和采样噪声下发现，旧说法“朝身后退着走”产生的动作差异大于模型自身的采样波动，而两句新说法没有同样明显的偏差。这支持检验文本覆盖，但不能解释原话闭环也会失败的问题。本轮只比较文本覆盖，不改变身体观测、动作空间或模型结构。

## 配对与预算

使用 v21 的 134 条源轨迹、31400 帧，原生 LeRobot merge 生成六份相同轨迹副本。对照组保留每条轨迹原文；增广组通过原生 modify_tasks 给每条后退轨迹分别配六种表达：向后走、往后退一点、朝身后退着走、向后迈步、往身后走、往后退步。修改后重新加载 parquet，并修正 task/episode/index 的统计。

两组所有非 task 数值、视频字节、episode 顺序和物理统计相同。归一化使用完整 v21 训练数据的 canonical float64 数值，重复六次只改变 count。804 条副本不等于 804 条独立示范。实际采样选取全部 15 条后退源轨迹及其它十类动作各两条，共 35 条源轨迹的六份副本、42000 个 anchor。两组都用原生 EpisodeAwareSampler，seed=42；保存前 8000 anchor 的确定性重建序列与文本曝光。这是预期采样证据，不冒充训练 hook 的实录。

初始权重为 v19 step4000，使用 v21 zero-checkpoint 的归一化。state51=身体36+上一实际执行目标15；固定黑图；H10/E5、50Hz、batch8、optimizer seed42。每组 2 步冒烟丢弃、1000 步正式训练，fresh optimizer，总计最多 2004 更新。训练通过已有 duckmind_training.train 调用 LeRobot，不另写 trainer。两组都保留其它动作回放，但不预先假定不会遗忘。

## 验收

初始模型和两组最终模型各运行 22 个预定 case：

- 后退：旧开发身体730及预声明新初态760/761，交叉原话、已转入增广训练的旧说法、两句新说法“向身后迈几步”“往后方走一段”，共12条，每条200tick。
- 旧动作：其它十类各在身体730运行原话，共10条；forward600tick并另报4秒前缀，其它200tick。

新初态需先检查旧工件库存，再执行教师可达性校准；它们经过教师检查后不称完全未触碰环境。真正语言与身体同时留出的只有两句新话×两个新初态=4格/模型。单一优化器种子不足以证明一般性显著优势。无效 moving-start 单列，保留分母，不更换 seed、不修改成功阈值。smoke 通过新增 labeled-text 参数执行10tick，仅检验运行链路，不要求200tick任务判据。

保存完整实际输入、chunk、执行动作和物理轨迹，独立重算任务成功、平衡、语言差异、旧动作回归及延迟。固定黑图不证明视觉能力，暂停物理等待推理不证明实时部署。不得将关键词分派或专家接管当学生能力。

## 运行记录

入口：`training/experiments/head_curriculum.py prepare-language-pair / train-language-pair`，启动脚本 `run-language-curriculum.sh`。工件目录 `/srv/duckmind-training/v25-owner/run`。先准备并独立审计，再启动唯一训练进程。代码提交 `c9989ce95b47fc565c5c3c4f0b651000ca95cdc0`；冻结 driver SHA256 `6b9364a7898f9a97103c00709850b3ea0aa74d71e5084b6de90e6ea90bcbebcd`。准备进程1076120已退出0；2026-10-15 20:37（Asia/Shanghai）启动唯一训练编排进程1100371，当前实际进入smoke-control。正式两组1000步会在smoke和baseline之后自动开始，尚不能称训练完成。


准备阶段的证据：

- 两组各804副本/188400帧，所有非 task 数值相等，视频SHA多重集合相等；源数据未修改。源episode计数不是统计独立样本数，尤其纠偏窗口来自重叠状态。
- 教师730/760/761各完整200tick且严格通过（3/3），后退进展分别约0.570/0.560/0.537m；不使用这些校准轨迹训练。
- 采样顺序重建的8000anchors两组相同，其中后退3407（42.59%）；增广六说法曝光分别550、556、561、618、573、549。其余4593为十类回放。
- 库存初查542份旧报告/config/audit的结构化seed/body字段未命中760/761；独立复核会记录完整扫描范围。
- 本机training lint/test通过（17 passed, 2 skipped），managed提交同时验证policy lint/test。devloop提交推送成功后仍因旧action-chunks路径postcheck退出1，已核实remote SHA一致。
- native merge尝试加载torchcodec时报告缺少libavutil，随后成功回退pyav；显式数据读入也用pyav。准备无学生推理、无优化更新。

独立preflight已通过：parent目录 `independent-preflight-audit.json/.md/.py` 保存96071项检查（72.48秒），完整数值、episode/global统计、有效processor和源哈希通过；真实CPU DataLoader每组前16个anchor通过，全部210 selected episode末尾H10 padding通过。独立旧库存扫描788个JSON，760/761身份字段及两句fresh文本均无命中。扫描清单和哈希已保存；源episode数量不是统计独立性。

唯一launcher PID1100371，进度以 `run/run.stage`、`run/run.log`、`run/run.exit-code` 为准。正式结果待填，不把准备或冒烟通过当作学会后退。没有重启自动跟进。

20:41 中途快照：两组 native step2 和155个optimizer参数step均为2；真实num_workers2链路通过。两条smoke均完整稳定10tick，实际观测/previous_target/两个预测块与10个执行目标逐项对齐；六有效归一化张量一致。`independent-interim-audit.json/.md` 的134项检查通过（passed-snapshot）。smoke严格任务不通过是因为只跑10tick，不满足200tick门槛，不能当学会或训练失败。

初始模型快照仅前两条：身体730原话99tick失稳；旧增广说法200tick稳定，但未达到后退任务判据。两条已独立重算。此时正式训练尚未开始；编排将在baseline完整后分别从同一初始checkpoint训练两组，结果仍待完成，不能从局部baseline推断文本增广收益。


## 最终结果（取代以上运行中快照）

2026-10-15 21:07:21（Asia/Shanghai）全流程 complete/exit0，实际2+2冒烟、1000+1000正式，共2004更新。独立最终审计 `/srv/duckmind-training/v25-owner/independent-final-audit.json/.md/.py` 3749检查、零不匹配，覆盖全部66主验收和2冒烟、实际输入/执行/初态/六统计/完整日志与哈希。

| 模型 | 后退严格通过 | 后退完整稳定 | 其它十任务通过 | 前进4秒 | 前进12秒 |
|---|---:|---:|---:|---|---|
| baseline | 0/12 | 9/12 | 7/10 | 失败 | 318tick失稳 |
| control | 0/12 | 9/12 | 8/10 | 通过 | 600tick，航向失败 |
| augmented | 0/12 | 9/12 | 8/10 | 通过 | 561tick失稳 |

三组原话在三个初态都发生后退后失稳；其它三说法完整稳定，但只移动约0.9–3毫米，不能当后退成功。增加同义表达的这个1000更新配方未改善严格成功；不能推断所有语言增强无效。每组只有一个optimizer seed。

后退训练状态两个原话anchor MSE：baseline .001509/.001914，control .001346/.002380，augmented .001297/.001697 rad²；persistence .012062/.023630。该诊断没有覆盖全部六说法，也不证明闭环。推理中位约 .25秒，超过E5的 .1秒执行窗口，暂停物理验收不能称实时。

后续转入 `vla-small-curriculum-2026-10-16.md`：先诊断六说法拟合，再做匹配初态的小课程。旧模型和全部失败证据保留。
