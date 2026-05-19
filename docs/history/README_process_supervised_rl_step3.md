# 大模型逻辑推理过程监督强化学习 step3

> 文档角色：`step3` 执行文档  
> 配套总览文档：`README.md`  
> 前置条件：`step2` 已完成真实候选轨迹 reranking 实验

## 1. step3 要解决什么

`step3` 的目标是从“规则版过程 reward 原型”切换到“强 LLM judge 标注 -> PRM 训练数据 -> PRM”的路线。

核心判断：

- `process_reward_v0` 已经证明过程信号有用。
- 但规则 reward 存在短答案偏好，不适合作为 PRM 主训练标签。
- 下一阶段应该用更强的大模型 API 做过程裁判，生成更高质量的过程偏好数据。

一句话：

```text
reward0 proves the idea; LLM judge provides the teacher signal; PRM learns the teacher signal.
```

## 2. step3 总流程

推荐闭环：

```text
1. 本地 7B instruct 模型生成候选轨迹
2. 强 LLM API judge 对同题多个候选做过程质量评审
3. 将 judge 输出转成 ranking / score / preference pairs
4. 训练第一版 trajectory-level PRM
5. 用 PRM 复跑 candidate reranking
6. 对比 final-only vs final+PRM
7. 如果成立，再进入 reward-weighted SFT 或其它 post-training
```

第一版不要一上来做 step-level PRM。先做 trajectory-level / candidate-level preference，更稳、更容易验收。

## 3. 输入数据

沿用 `step2` 已有候选轨迹：

```text
logs/candidates/gsm8k_train_debug_candidates.jsonl
```

这个文件包含：

- `sample_id`
- `candidate_id`
- `question`
- `gold_final`
- `candidate_text`
- `candidate_final`
- `candidate_steps`

第一版规模：

```text
100 questions x 4 candidates = 400 candidates
```

## 4. LLM judge 的职责

LLM judge 不只是判断最终答案。它要判断候选过程质量。

每道题输入：

- 原题
- gold final answer
- 同一道题的 4 条候选解法

每道题输出：

```json
{
  "sample_id": "gsm8k-train-000001",
  "best_candidate_id": "gsm8k-train-000001-cand-03",
  "ranking": [
    "gsm8k-train-000001-cand-03",
    "gsm8k-train-000001-cand-01",
    "gsm8k-train-000001-cand-04",
    "gsm8k-train-000001-cand-02"
  ],
  "scores": {
    "gsm8k-train-000001-cand-01": 0.74,
    "gsm8k-train-000001-cand-02": 0.28,
    "gsm8k-train-000001-cand-03": 0.88,
    "gsm8k-train-000001-cand-04": 0.61
  },
  "pairwise_preferences": [
    {
      "chosen": "gsm8k-train-000001-cand-03",
      "rejected": "gsm8k-train-000001-cand-01",
      "reason": "Both reach the correct final answer, but cand-03 has fewer unsupported jumps and clearer arithmetic."
    }
  ],
  "notes": "Prefer candidates with correct final answer and faithful, complete reasoning. Penalize unrelated continuations, hidden contradictions, and thin explanations."
}
```

## 5. judge 评分原则

Judge prompt 应明确这几个优先级：

1. 最终答案正确性很重要，但不是唯一标准。
2. 如果多个候选最终答案都正确，优先选过程更清楚、更一致、更少跳步的。
3. 如果候选最终答案正确但过程跑题、乱码、代码污染、跳步严重，应降低过程分。
4. 不奖励“只是短”。短但完整可以高分；短但推理薄弱不能高分。
5. 不奖励“只是长”。长但重复、绕圈、污染也要扣分。
6. 如果所有候选都错，也要选过程相对最接近正确思路的一条。

第一版 judge 输出必须是 JSON，方便自动构造数据。

## 6. API 选择建议

截至 `2026-05-06`，推荐先用 DeepSeek API 小额跑通。

### 推荐主方案：DeepSeek

优点：

- 官方 API 支持 OpenAI 格式。
- 官方文档列出 `deepseek-v4-flash` 和 `deepseek-v4-pro`。
- 支持 JSON Output，适合结构化 judge。
- `deepseek-v4-flash` 官方价格适合批量标注：cache miss input `$0.14 / 1M tokens`，output `$0.28 / 1M tokens`。
- 先跑 `100x4` 这种规模，成本压力很低。

建议模型：

```text
deepseek-v4-flash
```

如发现 judge 不稳定，再抽样换：

```text
deepseek-v4-pro
```

### 备选复核：Kimi

Kimi 官方文档显示最新主力是 `Kimi K2.6`，上下文 `256K`，支持长思考、JSON Mode、ToolCalls 等能力。

但注意：

- 旧 `kimi-k2` 系列将在 `2026-05-25` 下线，不建议新流程依赖旧模型。
- 免费额度适合 smoke test 或边界样本复核，不建议把主流程押在免费额度上。
- 价格和限速以 Kimi 控制台为准；大规模标注前先用 `10` 道题估算实际 token 和费用。

建议用法：

```text
DeepSeek 负责主标注
Kimi K2.6 抽样复核 20 个边界样本
```

如果两者分歧很大，再考虑 ensemble judge。

参考文档：

- DeepSeek Models & Pricing: <https://api-docs.deepseek.com/quick_start/pricing>
- Kimi K2.6: <https://platform.kimi.com/docs/pricing/chat-k26>
- Kimi K2 legacy notice: <https://platform.kimi.com/docs/pricing/chat-k2>
- Kimi 充值与限速: <https://platform.kimi.com/docs/pricing/limits>

## 7. step3 需要新增的脚本

### 7.1 `scripts/judge_candidates_with_llm.py`

职责：

- 读取候选 JSONL
- 按 `sample_id` 聚合候选
- 构造 LLM judge prompt
- 调用 API
- 校验 JSON 输出
- 失败时重试
- 写出 judge JSONL

目标命令：

```bash
python scripts/judge_candidates_with_llm.py \
  --input logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --output logs/llm_judge/gsm8k_train_debug_candidates_judged.jsonl \
  --provider deepseek \
  --model deepseek-v4-flash \
  --limit 10
```

### 7.2 `scripts/build_prm_dataset.py`

职责：

- 读取候选 JSONL
- 读取 LLM judge JSONL
- 生成 PRM 训练数据

第一版输出 preference pairs：

```json
{
  "sample_id": "gsm8k-train-000001",
  "question": "...",
  "chosen_candidate_id": "...",
  "chosen_text": "...",
  "rejected_candidate_id": "...",
  "rejected_text": "...",
  "judge_reason": "..."
}
```

目标命令：

```bash
python scripts/build_prm_dataset.py \
  --candidates logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --judgements logs/llm_judge/gsm8k_train_debug_candidates_judged.jsonl \
  --output data/prm/gsm8k_train_debug_prm_preferences.jsonl
```

### 7.3 PRM 训练脚本

第一版先做最小 trajectory-level preference PRM：

```text
input: question + candidate_text
target: chosen/rejected preference
```

当前已新增轻量训练入口：

```bash
python scripts/train_prm.py \
  --preferences data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl \
  --output-dir logs/prm/gsm8k_debug_prm_smoke \
  --candidates logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --scored-output logs/prm/gsm8k_debug_prm_smoke/scored_candidates.jsonl \
  --report-output logs/prm/gsm8k_debug_prm_smoke/final_only_vs_final_plus_prm_selection_report.md
```

这个版本是纯 Python 线性词袋 preference PRM，用于 smoke test 数据闭环和 reranking 验收；不急着做复杂 step-level reward，也不急着上 transformer reward head。

## 8. step3 验收标准

`step3` 完成不等于大规模训练完成。第一版验收标准是：

1. 能稳定调用 LLM API judge，输出合法 JSON。
2. `10` 道题 smoke test 人工检查通过。
3. `100x4` candidates 完成 judge 标注。
4. 能生成 PRM preference dataset。
5. 第一版 PRM 能训练并输出分数。
6. `final-only` vs `final+PRM` reranking 不降低 final accuracy。
7. 人工抽查显示 `final+PRM` 选出的过程质量优于 `final-only` 或至少不更差。

## 9. 当前建议执行顺序

```text
1. 申请 / 购买少量 DeepSeek API 额度
2. 实现 LLM judge prompt 和调用脚本
3. 跑 10 道题 smoke test
4. 人工检查 JSON 输出和 ranking 是否合理
5. 跑完整 100x4 judge
6. 构造 preference dataset
7. 训练第一版 PRM
8. 用 PRM rerank 同一批 candidates
9. 再决定是否进入 post-training
```

## 10. 当前不做什么

暂时不做：

- 不继续把 `process_reward_v0` 调到复杂规则系统。
- 不直接用 `process_reward_v0` 生成 PRM 主训练标签。
- 不一上来做 step-level PRM。
- 不直接进入 RL。
- 不直接大规模购买 API 额度。

先跑小闭环，确认数据质量，再扩大规模。

## 11. 当前执行结果

已完成 DeepSeek API judge 的第一版 `100×4` 标注。

新增代码：

- `src/psrl/llm_judge.py`
- `src/psrl/prm_dataset.py`
- `src/psrl/prm.py`
- `scripts/judge_candidates_with_llm.py`
- `scripts/build_prm_dataset.py`
- `scripts/train_prm.py`
- `tests/test_llm_judge.py`
- `tests/test_prm_dataset.py`
- `tests/test_prm_training.py`
- `tests/test_train_prm_script.py`

远端验证：

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests
34 passed
```

DeepSeek judge 产物：

```text
logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl
data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl
```

核心统计：

```text
judge_rows: 100
preference_rows: 324
samples_with_explicit_pairwise_preferences: 98
samples_without_explicit_pairwise_preferences: 2

final_only_accuracy: 0.9300
llm_judge_top1_accuracy: 0.9300
changed_vs_final_only: 26
changed_breakdown: 1->1 = 22, 0->0 = 4
samples_with_at_least_one_correct_candidate: 93 / 100
```

Token 和费用：

```text
prompt_tokens: 131254
prompt_cache_miss_tokens: 113846
prompt_cache_hit_tokens: 17408
completion_tokens: 157518
reasoning_tokens: 112949
estimated_cost_usd: 0.060287
estimated_cost_rmb_at_7_2: 0.4341
```

当前判断：

- DeepSeek judge 能稳定输出结构化 JSON，但偶发会出现截断或越界分数，脚本已加入解析失败重试。
- LLM judge top1 没有降低 final accuracy。
- LLM judge 相比 final-only 改变了 `26/100` 道题，且没有 `1->0`。
- `324` 条 preference rows 已足够启动第一版 PRM 训练 smoke test。
- 第一版轻量 trajectory-level preference PRM 已完成 smoke training：`324` 个 preference pairs，训练集 pair accuracy `0.9722`，final loss `0.151557`。
- `final+PRM` top1 final accuracy 为 `0.9300`，与 final-only 持平；改变 `46/100` 道题，其中 `42` 个是 `1->1`、`4` 个是 `0->0`，没有 `1->0` 准确率伤害。
- 对 `46` 个 changed top1 样本完成 DeepSeek 二选一抽查：支持 PRM `17`、支持 final-only `29`（`0->0` 子集中 PRM `3`、final-only `1`）。
- 在轻量 PRM 上做小规模超参搜索后，较优配置 `epochs=120, lr=0.12, l2=0.0`：对原始 4-candidate judge 的一致率升至 `0.7100`（基线约 `0.6700`），但仍低于 final-only 的 `0.7400`；final accuracy 仍为 `0.9300`。

当前下一步：

```text
先做 PRM 建模改进（超出轻量线性词袋）
-> 目标：judge 一致率不低于 final-only，并保持 final accuracy 不下降
-> 满足后再扩展到 1k×4 candidates
-> 后续再进入 reward-weighted SFT / post-training
```
