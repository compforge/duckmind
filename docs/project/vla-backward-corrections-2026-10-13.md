# 后退专项纠偏配对实验（v21）

v20表明两个训练说法在专家状态上拟合相近，但训练初态也会失稳。
本轮问：相同更新预算下，集中拟合后退时，加入学生偏离状态的专家纠偏是否改善闭环？
这是后退专项诊断模型；通过旧任务回归检查遗忘，不默认替换v19或部署到robotd。

## 数据与对照

- 固定v19step4000初始权重，模型与state51、H10/E5、50Hz、黑图契约不变。
- 重放v20两个已知失败（body700/730）的全部执行动作，要求物理数组逐tick精确相等。
- 从每条的20/50/80tick接管，共6个候选。同步专家的previous action，再执行200tick。
  只保存接管后的专家动作为训练标签，学生前缀不作标签；所有候选原始结果保留。
- 原backward严格判据不变；至少4条接受且覆盖两个来源身体才进入训练。窗口重叠，不能当6个独立身体。
- 老师对700/730及新评估身体740..742先做200tick正控；必须严格通过。新身体因此已做教师检查，
  不能称完全未触碰，但尚未根据学生结果调参。
- 新native dataset完整保留v19的129eps/30400frames，再追加接受的纠偏。
  新归一化由完整训练union计算；没有评估状态参与统计。旧数据、模型不改、不删。

两组都从v19相同权重、共享union有效processor开始；control只通过原生episodes选择10条旧后退轨迹，
mixed选择相同10条加接受的纠偏窗口。两组均fresh optimizer、batch8、seed42、1000正式更新，
另各2步smoke丢弃不用作初始化，总预算2004更新。更新预算相同，但原数据曝光改变，
只能比较“集中后退训练的混合配方”，不能归因纯额外纠偏样本的独立效应。
旧课程其他家族只影响共享归一化，不被这两组采样训练。

纠偏沿用两个训练文本交替标注；不加入朝身后退着走或往后退步，不将看过的失败表达偷换成新holdout。
这里没有文本规则分派、专家接管的学生验收、新增RL损失或模型架构改变。

## 验收

zero只替换union统计、0更新；control/mixed各自训练后采用相同22case：

- 后退body700/730（开发）与740..742（新身体），交叉原话/旧留出说法，共10case，每条200tick。
- 旧HEAD_TASKS六任务在body730各两说法，共12case；forward600tick另报4秒，其余200tick。
- smoke每组在700做10tick后退；只要求完整、有限、稳定，短长度strictFalse不作失败。

相同初态physics/state/prior，噪声seed=body*10000+tick。保留每次实际黑图/state/prior/chunk/latency和trace。
旧留出说法已使用，不是新语言测试；只有1 optimizer seed，不能把22case当22次独立训练。
物理暂停等待推理，无实时/视觉/硬件能力声明。指标包括strict、平衡、方向、语言、遗忘和延迟。

## 预期与研究判断

| 解释/方案 | 预期 | 可区分结果 |
|---|---|---|
| 纠偏状态覆盖不足 | mixed比control在新身体上更稳定且后退 | 仅开发身体改善不足以支持泛化 |
| 原数据拟合精度不足 | control也可能明显改善 | 两组均改善不证明纠偏必要 |
| 纠偏曝光或量不足 | mixed仍失败 | 不凭一轮否定纠偏，报告实际样本曝光 |
| 多任务干扰/遗忘 | 专项后退变好，旧任务变差 | 不能当全课程平替结果 |
| 语言起步映射不足 | 原话改善而旧留出仍不动 | 后续单独设计语言监督，不扩大本轮 |

采集/准备先审计，再唯一启动训练。任何门禁失败保留证据，不降门槛、不自动增加预算。
工件在devbox /srv/duckmind-training/v21-owner/run；入口run-backward-curriculum.sh。
结果交付后暂停自动跟进，不自动追加下一轮或合并PR。

## 采集进展

prepare进程1852532已完成6候选、5个教师正控和原生数据转换，随后在记录上游Git版本时退出。
5个纠偏窗口通过，共1000帧；700prefix20全程稳定但净后退和尾部后退未达标，
保留原始轨迹、不纳入训练。5个teacher-only正控均通过，完整学生失败回放与接管点物理精确一致。
实际union134eps/31400frames，control10eps/2000frames，mixed15eps/3000frames。
实际以prepare.exit-code=0、data-audit和独立审计为准，当前尚未正式训练。

代码d57ecd376e2d712f5d2c62f3db8c666193f2c4a9已推送并核验远端一致，训练组件
make fix/lint/test通过（17passed、2skipped）。devloop提交推送后仍因旧action-chunks路径的
postcheck退出1，不能称整个工具全成功；PR5实验描述已更新，未自动合并。
独立审计重点是非连续episode选择的全局/本地索引映射、实际future-action/padding、
两组fullunion归一化以及纠偏raw到dataset时序。审计通过才可唯一启动2004更新流程。

## 准备收尾恢复

首次prepare在数据和zero checkpoint保存后，调用上游`git rev-parse HEAD`失败；devbox上的
上游是无`.git`的同步源码目录。原prepare.log与prepare-attempt1-source.py保留，未进行训练。
修复把收尾独立为`finalize-backward-preparation`：读取已保存manifest，核验源/数据/初始权重
哈希及6个有效processor，重建两组选集，不重新采集、编码视频或更新参数。
Git revision记录为null并保留原错误，teacher、推理源码、scene的文件哈希作为已验证身份。
元数据收尾退出0，prepare.exit-code=0；这不抹去首次prepare失败。
独立数据审核未完成前，不启动训练。恢复代码make fix/lint/test通过（17passed、2skipped）。

## 独立审核与启动

独立CPU审核11项通过（工件parent independent-preflight-audit.json/.md）：旧129eps逐帧保持，
新5窗口与raw精确对应，6候选和5正控严格评分重算一致；2global+268episode数值统计、
6zero有效张量与manifest哈希一致。原生sampler完整索引检查确认control2000、mixed3000帧，
没有其他任务混入；逐episode首中尾75样本和真实DataLoader48样本的全局索引、H10未来动作
与padding均准确。CPU审核workers0，真实训练workers2由smoke覆盖。
全31400黑图解码复用prepare完整结果及哈希，并独立抽检；未使用的近似quantile未重算。

唯一训练launcher PID1897903已启动，入口train-backward-pair，冻结源码da1ede7b3f4f74dd7ba5211277773e04252483e3。
预算仍为每组2smoke+1000正式，共2004；需等zero/control/mixed所有物理验收后比较能力。
原准备失败日志与attempt1审计继续保留；本次只续元数据收尾，未重复采集或训练。

## 最终结果与研究决策

runner已完成，run.stage=complete、run.exit-code=0、PID1897903退出。四次原生训练为
2、2、1000、1000更新，所有checkpoint的155个优化器step计数与native training_step一致；
总2004预算已用完，没有重训或追加。两组正式模型均从同一v19权重开始，smoke权重丢弃。

这次等预算纠偏配方没有改善严格后退成功率。混合组部分轨迹更稳定，但偏航或原地不动，
不能算任务完成；两组旧任务回归均下降。这些诊断模型不替换v19、不部署robotd。

| 模型 | 后退开发初态原话 | 后退新初态原话 | 后退旧留出措辞 | 后退严格合计 | 后退全程稳定 | 旧任务严格回归 |
|---|---:|---:|---:|---:|---:|---:|
| zero：相同初始权重、新统计、无更新 | 0/2 | 0/3 | 0/5 | 0/10 | 6/10 | 9/12 |
| control：原后退数据 | 0/2 | 1/3 | 0/5 | 1/10 | 6/10 | 7/12 |
| mixed：原数据加纠偏 | 0/2 | 0/3 | 0/5 | 0/10 | 7/10 | 7/12 |

- mixed在开发body730完成200tick且稳定，但偏航−0.620rad，超过原判据，仍失败。
- mixed在新body742只后退0.00452m；control同初态后退约0.284m并通过。稳定站着不是后退成功。
- 旧站立/头部任务zero9/10、control7/10、mixed7/10；前进12秒三者均0/2，4秒前缀分别0/2、0/2、1/2。
- 三者推理中位数约248/253/251ms；mixed p95约295ms，939次均超过E5的100ms执行窗口。
  评估仍暂停物理等待推理，不具备实时执行证据。

独立最终审核位于devbox /srv/duckmind-training/v21-owner：independent-final-audit.json/.md、
independent-final-ml-debug-form.json、independent-final-ml-debug.md。核验68条轨迹（66正式比较+2smoke）、
14235次执行动作、2852份实际输入，重算68项strict、30项gaze、6项前进4秒、212项交叉目标评分、
176条fit误差，全部与记录一致。同初态、每phase6个有效统计张量、source/data/external/model哈希一致。
完整run.log共3337行257561字节，原始失败均保留。独立审计曾把日志step1K解析成1，已仅修审计解析器，
不改运行代码，不重训；原prepare的Git元数据失败及零训练收尾恢复也继续保留。

限制：一个优化器seed；新初态仅740..742三个，且经过教师筛查；旧留出措辞此前已用过；
纠偏5窗口仅来自两个相关初态，混合也改变原数据曝光。因此不能从1次与0次成功推断control普遍更好，
也不能据此否定纠偏数据方法。没有视觉、真实硬件、连续任务切换或新语言泛化证据。

下一步最便宜的诊断是检查**新增纠偏标签本身是否被模型拟合**。当前22个fit探针只取每个family
最早旧episode的t0/t20，完全没有覆盖129..133新纠偏episode；这个证据缺口现在明确记录。
可先固定state/text/noise，对比zero/control/mixed在新纠偏状态上的解码动作误差；若能拟合却闭环失败，
再单独比较E1/E5执行窗口。以上为后续建议，本轮未执行、未自动增加预算。
结果交付后暂停duckmind-vla自动跟进，PR5保持open、不合并。
