# 大模型逻辑推理过程监督强化学习 step2

> 文档角色：`step2` 执行文档  
> 配套总览文档：`README.md`  
> 前置条件：`step1` 已完成

## 1. step2 要解决什么

`step2` 的目标不再是继续补环境，而是开始做第一版实验验证。

核心问题只有两个：

1. 当前规则版 `process reward v0` 是否足够合理，值得继续往下做
2. `final-only reward` 和 `final + process reward` 的第一版对照该如何建立

## 2. step2 的输入

进入 `step2` 时，默认已经具备这些输入：

- `GSM8K` 的 `raw / processed / debug` 数据
- 统一 schema
- `process reward v0` 配置和打分脚本
- `debug subset` 的首轮打分日志
- 基础测试全部通过

也就是说，`step2` 的起点不是“从零开始”，而是“在已有底座上做实验验证”。

## 3. step2 的主要任务

### 3.1 检查 reward 打分是否合理

先围绕 `debug subset` 做质量检查，重点看：

- 高分样本是不是确实更简洁、更有效
- 低分样本是不是存在重复、拆分过细、推进不足等问题
- reward 有没有明显偏向长输出
- flag 规则是否需要补充

### 3.2 建立 `final-only` 基线

第一版基线不追求复杂，只要求可比较。

最简单的做法是：

- 保持 `final_reward_weight: 1.0`
- 把 `process_reward_weight` 改为 `0.0`
- 用同一批输入样本重复打分

这样就能得到 `final-only` 的基线输出。

### 3.3 建立 `final + process` 对照

保留当前配置：

- `final_reward_weight: 1.0`
- `process_reward_weight: 0.3`

然后在同样的数据、同样的候选轨迹上输出对照结果。

### 3.4 明确对照的可比条件

为了让结论可用，`step2` 里所有比较都要尽量只改一个变量：

- 同一批输入题目
- 同一批候选轨迹
- 同一套最终答案标准化规则
- 只改变 `process_reward_weight`

### 3.5 输出第一版分析结论

`step2` 需要产出的是“第一版实验分析”，而不是大规模训练结果。

至少要回答：

- 加入过程奖励后，分数分布发生了什么变化
- 哪类样本受影响最大
- 当前 reward 是否值得进入下一阶段训练/筛选实验

## 4. step2 建议执行顺序

建议按这个顺序推进：

```text
先检查 debug scored 日志
-> 再跑 final-only 基线
-> 再对比 final-only vs final+process
-> 再整理结论与风险
```

## 5. step2 当前推荐命令

### 5.1 先看已有打分结果

```bash
cd ~/autodl-tmp/process_supervised_rl

head -n 5 logs/process_reward_v0/gsm8k_train_debug_scored.jsonl
```

### 5.2 跑 `final-only` 基线

把 `configs/reward/process_reward_v0.yaml` 中：

```yaml
process_reward_weight: 0.3
```

临时改成：

```yaml
process_reward_weight: 0.0
```

然后执行：

```bash
python scripts/score_samples.py \
  --input data/debug/gsm8k_train_debug.jsonl \
  --output logs/process_reward_v0/gsm8k_train_debug_final_only.jsonl \
  --reward-config configs/reward/process_reward_v0.yaml
```

### 5.3 恢复 `final + process`

把 `process_reward_weight` 改回 `0.3`，再执行：

```bash
python scripts/score_samples.py \
  --input data/debug/gsm8k_train_debug.jsonl \
  --output logs/process_reward_v0/gsm8k_train_debug_final_plus_process.jsonl \
  --reward-config configs/reward/process_reward_v0.yaml
```

## 6. step2 的验收标准

`step2` 完成的标准不是“训练已经做完”，而是下面几件事成立：

1. 有一份 `final-only` 输出
2. 有一份 `final + process` 输出
3. 两份结果在可比条件下生成
4. 对 reward 行为有第一版文字结论
5. 明确是否进入下一阶段训练或候选轨迹实验

## 7. step2 的产出物

`step2` 结束时，理想产出包括：

- `final-only` 打分结果文件
- `final + process` 打分结果文件
- 一份对照分析记录
- 下一阶段训练/筛选实验的执行建议

## 8. 当前判断

从现有 `debug subset` 的首轮检查来看，当前规则 reward 至少满足：

- 能稳定运行
- 输出分数分布正常
- 没有明显鼓励更长、更啰嗦的步骤

因此 `step2` 的主要工作不是再写 reward 骨架，而是把“是否值得继续做”这件事用第一版对照实验说清楚。

## 9. 候选轨迹 reranking 实验结果

在远端本地模型 `DeepSeekMath-7B-Instruct` 上完成了第一版真实候选轨迹实验。

模型路径：

```text
/root/autodl-tmp/models/deepseek-math-7b-instruct
```

实验设置：

- 输入：`data/debug/gsm8k_train_debug.jsonl`
- 规模：`100` 道题，每题 `4` 条候选，共 `400` 条候选
- 生成参数：`temperature=0.7`、`top_p=0.95`、`max_new_tokens=512`
- 对照方式：同一批候选上比较 `final-only` top1 与 `final+process` top1

正式产物：

- `logs/candidates/gsm8k_train_debug_candidates.jsonl`
- `logs/candidate_reward/gsm8k_train_debug_candidates_scored.jsonl`
- `logs/candidate_reward/final_only_vs_final_plus_process_selection_report.md`

核心结果：

```text
final_only_accuracy: 0.9300
final_plus_process_accuracy: 0.9300
final_only_selected_process_reward_mean: 0.6681
final_plus_selected_process_reward_mean: 0.6978
changed_selection_count: 50 / 100
changed_selection_breakdown: 1->1 = 46, 0->0 = 4, 1->0 = 0, 0->1 = 0
```

当前判断：

- `final+process` 没有降低最终答案准确率。
- 在准确率持平的情况下，`final+process` 选中了平均过程分更高的候选。
- 这说明 `process reward v0` 已经具备候选 reranking 信号价值。
- 但 `corr(num_steps, process_reward) = -0.8478`，说明当前 reward 对长步骤仍偏严格。进入训练前，需要先人工抽查 changed cases，并判断是否调整长推理惩罚。

## 10. step2 下一步

建议先不直接进入训练，而是做一次 reward 质量收尾：

1. 抽查 `final+process` 改变 top1 的样本。
2. 重点看长步骤但合理的推理是否被压低。
3. 必要时调整 `progress_contribution` 或过程分聚合方式。
4. 复跑同一批 `100×4` 候选。
5. 如果准确率仍不下降且过程质量更好，再进入小规模训练或 reward-weighted SFT。

