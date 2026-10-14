# 纠偏标签拟合诊断（v22）

v21混合纠偏未提高严格后退成功率，但其离线探针没有覆盖新增纠偏episode。
本轮固定现有三个checkpoint，检查“新增纠偏训练状态上的目标序列是否更接近教师”。
只诊断拟合，不更新参数、不运行物理、不扩大任务，也不据此宣称闭环成功。

## 配对设计

复用v21完整union与zero/control/mixed，6个有效state/action统计必须精确相同。
原10条后退episode及全部5条新增纠偏episode129..133，每条取t0/20/60/100/140/180，
每个样本固定两个torch flow-noise种子；state51、实际previous_target、训练文本、黑图、
H10动作目标在三个checkpoint间相同。每模型额外重复首个输入一次并要求精确相等。
总15×6×2×3+3=543次推理，0训练更新、0物理rollout。

原始teacher NPZ与native dataset每个选中state/prior/text/H10逐项核验；t180仍有20帧，
H10不得padding。保存90个原始输入、每模型180个预测、原生processor消费的首样本张量、
三个重复smoke、全部源码/数据/权重哈希及完整日志。旧v21源码22文件先保存独立snapshot。

按original/correction、early(t0/20)/late(t60..180)分别报告first/E5/H10物理角度MSE，
包括全部15关节、明确名字映射的10腿关节与逐关节误差；persistence使用相同窗口和分母。
另外按episode保存均值。两次噪声不是独立训练，两条来源轨迹的重叠纠偏也不是5个独立身体；
所有状态都来自训练数据，没有新语言或heldout效果。

## 预期与判别

| 可能解释 | 本轮可见证据 | 不能据此声称 |
|---|---|---|
| 纠偏标签拟合不足 | mixed在纠偏状态不优于zero/control | 纠偏方法无效 |
| 样本上拟合但闭环离开分布 | mixed纠偏误差下降而v21任务仍失败 | 已证明唯一闭环失败机制 |
| 动作块后半段掩盖执行段问题 | H10改善而first/E5未改善 | 改E1必然有效 |
| 静止关节拉低平均误差 | 全关节改善而腿部/特定关节无改善 | 学会后退 |

不将原生flow-velocity训练loss与解码角度MSE直接比较。下一步是否测E1/E5取决于本轮结果，
本轮不附带追加训练或物理实验。独立审计重算原始输入、误差分母、噪声配对和processor契约。

## 运行

现有head_curriculum.py新增diagnose-correction-fit阶段，沿用SmolPolicy.predict与native dataset。
工件devbox /srv/duckmind-training/v22-owner/run，launcher run-correction-fit-curriculum.sh。
零更新诊断完成后记录结论并保持PR5未合并。首次本地lint发现过长日志字符串，修正后重新检查。
