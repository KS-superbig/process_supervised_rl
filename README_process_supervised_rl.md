# 大模型逻辑推理过程监督强化学习 step1

> 文档角色：`step1` 执行与归档文档  
> 配套总览文档：`README.md`  
> 当前状态：`step1` 已完成，后续执行请看 `step2` 文档

## 1. 文档用途

这份文档记录 `step1` 为什么这样设计、实际完成了什么，以及为什么现在可以进入 `step2`。

`step1` 的核心任务不是做出完整论文系统，而是先把第一版工程底座和规则奖励闭环跑通。

## 2. step1 的目标

`step1` 的目标可以压缩成三句话：

1. 先把数学推理任务收敛到 `GSM8K`
2. 先把数据清洗、统一 schema、debug 子集这些基础设施跑通
3. 先把规则版 `process reward v0` 的打分链路跑通

它解决的是“有没有一条可以稳定复现的最小实验链路”，而不是“最终模型效果已经验证完毕”。

## 3. step1 的范围

`step1` 只覆盖下面这些内容：

- `GSM8K` 原始数据接入
- 原始数据到统一 schema 的预处理
- 最终答案标准化
- 规则版步骤切分
- 规则版 `process reward v0`
- debug 子集构建
- 样本级 reward 聚合与打分
- 远端环境核验与基础测试

`step1` 不要求完成：

- learned reward model
- PPO / GRPO 等在线 RL
- 大规模训练
- `final-only` 与 `final + process` 的正式效果对照结论

## 4. step1 实际完成情况

目前已经完成：

- 远端环境拉起并确认 `GPU / Python / PyTorch` 可用
- 仓库在远端成功拉取
- `PyYAML`、`pytest` 以及数据下载所需依赖补齐
- `GSM8K` 训练集预处理完成，共 `7473` 条
- `GSM8K` 测试集预处理完成，共 `1319` 条
- `debug` 子集生成完成，共 `100` 条
- 统一输出 schema 抽样核验通过
- `process reward v0` 已在 `debug subset` 上完成首轮打分
- `pytest tests -v` 已通过，当前共 `14` 个测试通过

远端当前关键产物包括：

- `data/raw/gsm8k_train.jsonl`
- `data/raw/gsm8k_test.jsonl`
- `data/processed/gsm8k_train.jsonl`
- `data/processed/gsm8k_test.jsonl`
- `data/debug/gsm8k_train_debug.jsonl`
- `logs/process_reward_v0/gsm8k_train_debug_scored.jsonl`

## 5. step1 的验收判断

判断 `step1` 是否完成，看下面 4 件事是否都成立：

1. 数据底座存在，且可以从原始数据稳定再生成
2. 统一 schema 可读、可复用、可被后续脚本消费
3. 规则版 reward 打分脚本可对 debug 子集稳定运行
4. 单元测试通过，说明基础代码结构没有明显损坏

现在这 4 条都已经满足，所以 `step1` 可以判定为完成。

## 6. step1 的主要结论

`step1` 最重要的价值不是“已经证明过程奖励有效”，而是：

- 证明了工程链路可以稳定复现
- 证明了规则奖励脚本可以产出可分析的样本级分数
- 证明了当前 reward 没有出现明显的“越长越高分”型 reward hacking

同时也要明确：

- 目前 `debug subset` 的首轮打分更像 sanity check
- 还不能仅凭这一步就得出 “`final + process` 优于 `final-only`” 的正式结论

## 7. 为什么进入 step2

既然数据、打分、测试都已经跑通，接下来最合理的事情就不是继续搭环境，而是进入真正的实验验证阶段。

`step2` 要回答的是：

- 规则版 `process reward v0` 到底合不合理
- `final-only` 与 `final + process` 的第一版对照该怎么做
- 需要补哪些候选轨迹、分析脚本和实验记录

后续执行请看：

- [README_process_supervised_rl_step2.md](/Users/bytedance/Documents/New%20project/README_process_supervised_rl_step2.md)
