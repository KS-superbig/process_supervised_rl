# 数学推理过程监督强化学习

这个仓库归档了一个围绕“数学推理过程监督”的研究工程项目。项目从 GSM8K 上的规则版过程奖励原型开始，随后尝试了 LLM judge 偏好标注、轻量 PRM 诊断、SFT warmup，最后收敛到 MATH Level 3/4 数据上的 gated PRM GRPO 路线。

这个仓库按公开展示和长期归档的方式整理：代码、测试、配置和精简实验记录保留在 Git 中；原始/处理后数据、完整训练日志、checkpoint 和 LoRA adapter 不直接提交到仓库。

## 项目概述

最终有效的训练链路是：

```text
deepseek-math-7b-instruct
  + SFT v3 / MATH L3-4 warmup adapter
  -> 将 warmup adapter merge 到模型权重
  + fresh r512 LoRA adapter
  -> 使用 Skywork PRM gated reward 做 GRPO
```

核心设计是把最终答案正确性作为硬门控：

```text
if final_answer_correct:
    reward = 1.0 + 0.2 * process_reward
else:
    reward = 0.0
```

这样可以避免“答案错了但过程看起来不错”的样本获得正奖励。对于数学推理任务，过程奖励主要用于区分答对样本中的推理质量，而不是替代最终答案验证。

## 仓库内容

```text
configs/       实验和 reward 配置示例
scripts/       数据处理、候选生成、LLM judge、PRM、SFT、GRPO 和评测脚本
src/psrl/      项目核心代码
tests/         数据、reward、PRM、训练入口等单元测试
docs/history/ 项目各阶段历史记录
docs/repro/   当前复现说明和 Git 归档策略
artifacts/    外部 adapter 的 checksum
```

历史文档记录了项目路线的变化和关键判断：

- [step1: 工程底座与规则 reward 原型](docs/history/README_process_supervised_rl_step1.md)
- [step2: 规则版过程 reward 与 reranking 实验](docs/history/README_process_supervised_rl_step2.md)
- [step3: LLM judge、PRM、SFT 和早期 GRPO 诊断](docs/history/README_process_supervised_rl_step3.md)
- [step4: MATH warmup 与 r512 gated PRM GRPO](docs/history/README_process_supervised_rl_step4.md)

## 主要结果

目前最值得保留的正向信号来自 step4 的 MATH 路线。

MATH500 前 40 题 quick check：

| 模型链路 | 准确率 |
| --- | ---: |
| SFT warmup | 16/40 = 40% |
| SFT warmup + r512 gated GRPO | 22/40 = 55% |

这个结果只能作为方向性 quick check，不作为完整 benchmark 结论。正式评测应使用 MATH500 first 100 或 full 500，`max_gen_toks=768`，并且必须复现训练时的加载链路：

```text
base + warmup SFT adapter merge + r512 GRPO adapter
```

如果只加载 `base + r512 GRPO adapter`，会漏掉 warmup SFT adapter 的能力，评测对象是错误的。

## 复现说明

短版复现流程见 [docs/repro/current_pipeline.md](docs/repro/current_pipeline.md)。

最终 MATH GRPO 路线依赖的主要资产如下：

| 资产 | 训练时默认位置 |
| --- | --- |
| Base model | `/root/autodl-tmp/models/deepseek-math-7b-instruct` |
| Warmup SFT adapter | `/root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final` |
| Skywork PRM | `/root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` |
| MATH L3/4 训练数据 | `data/processed/math_l34_train_3000_seed42.jsonl` |

warmup adapter 的 checksum 保存在：

```text
artifacts/sft_v3_math_l34_3000_e0p4_lr1e5_len2048.tar.gz.sha256
```

大体积 adapter 不提交到 Git。如果需要共享权重，建议通过 GitHub Release、Hugging Face model repo、网盘或其他外部 artifact 渠道发布，并在仓库中保留 checksum 用于校验。

## Git 归档策略

建议保留在 Git 中：

```text
configs/
scripts/
src/
tests/
docs/
小型 Markdown 结果摘要
adapter checksum
```

不建议保留在 Git 中：

```text
原始或处理后数据集
候选轨迹 JSONL
benchmark prediction JSONL
完整 reward-debug 日志
模型 checkpoint
LoRA adapter 二进制文件
远端机器专用输出或私密信息
```

## 工程说明

项目包含数据标准化、reward 聚合、候选选择、PRM 数据构造、PRM 训练和 GRPO reward 行为等测试。发布前建议运行：

```bash
pytest -q
```

其中大部分是轻量单元测试；完整模型训练和 MATH500 评测需要外部模型权重和 GPU 环境。
