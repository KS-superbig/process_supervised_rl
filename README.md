# Process-Supervised RL

面向“过程监督推理”实验的协作总览文档。

这个 `README.md` 的职责不是讲某一个 step 的全部细节，而是给合作者快速同步三件事：

- 这个仓库现在在做什么
- 当前进行到哪一个 step
- 如果要参与，应该先看哪些文档、先做哪些事

## 当前状态

- 当前总阶段：第一阶段
- 当前 step：`step3`
- 当前 step 文档：[README_process_supervised_rl_step3.md](README_process_supervised_rl_step3.md)
- 上一 step 归档文档：[README_process_supervised_rl_step2.md](README_process_supervised_rl_step2.md)
- 当前完成度：`step3` 已跑通 DeepSeek API judge 的 `100×4` 标注闭环，已生成 PRM preference 数据；下一步是训练第一版 trajectory-level preference PRM

## 当前目标

- 使用 `GSM8K` 跑通最小实验闭环
- 对比 `final-only`、`final+LLM-judge`、`final+PRM` 的候选轨迹选择效果
- 默认采用“本地开发，远端执行”的工作流
- 当前主线：用强 LLM API judge 生成过程偏好数据，再训练 PRM，并用 PRM 做 post-training 前的 reranking / 数据筛选
- 当前下一步：用 `324` 条 DeepSeek judge preference rows 做第一版 PRM smoke training

## 文档分工

### 1. `README.md`

这是协作者总览文档，会持续更新。

主要维护：

- 当前项目进度
- 当前正在做的 step
- 协作者如何上手
- 当前优先任务
- 后续文档入口

### 2. `README_process_supervised_rl.md`

这是 `step1` 的归档文档。

主要维护：

- `step1` 为什么这样设计
- `step1` 实际完成了什么
- `step1` 的验收判断
- 为什么现在进入 `step2`

### 3. `README_process_supervised_rl_step2.md`

这是 `step2` 的实验归档文档。

主要维护：

- `step2` 的实验目标
- `final-only` 与 `final + process` 的对照方式
- 当前推荐命令与产出物
- `step2` 的验收标准

### 4. `README_process_supervised_rl_step3.md`

这是当前下一阶段 `step3` 的执行文档。

主要维护：

- 为什么不再用 `process_reward_v0` 作为 PRM 主标注器
- LLM judge 如何输出结构化过程偏好
- 如何从 judge 结果构造 PRM 训练数据
- PRM 训练、reranking、post-training 的最小闭环

## 当前优先事项

当前默认优先顺序：

1. 训练第一版 trajectory-level preference PRM
2. 用 PRM 给同一批 `100×4` candidates 打分
3. 对比 `final-only` vs `final+PRM` top1
4. 人工抽查 `final+PRM` 改变 top1 的样本
5. 如果 PRM 不降低 final accuracy 且过程质量更好，再扩展到 `1k×4` candidates
6. 后续再进入 reward-weighted SFT 或其它 post-training

## 当前已完成

目前已经完成的内容：

- 远端完成仓库拉取与 Python 环境搭建
- 远端成功安装 `PyYAML`、`pytest` 与数据下载所需依赖
- `GSM8K` 训练集预处理完成，共生成 `7473` 条样本
- `GSM8K` 测试集预处理完成，共生成 `1319` 条样本
- `debug` 子集生成完成，共生成 `100` 条样本
- 统一输出 schema 已经验证可用
- `pytest tests -v` 已通过，当前共 `30` 个测试全部通过
- `process reward v0` 已在 `debug subset` 上完成首轮打分检查
- 远端已下载并验证 `DeepSeekMath-7B-Instruct` 本地模型：`/root/autodl-tmp/models/deepseek-math-7b-instruct`
- 已生成 `100` 道 debug 题、每题 `4` 条候选，共 `400` 条模型候选轨迹
- 已完成 `final-only` vs `final+process` 候选 reranking 分析：两者 top1 final accuracy 都是 `0.9300`，但 `final+process` 选中候选的平均 process reward 从 `0.6681` 提升到 `0.6978`
- `final+process` 改变了 `50/100` 道题的 top1 选择，其中 `46` 个是 `1->1`，`4` 个是 `0->0`，没有 `1->0` 的准确率伤害
- 已人工初审 changed cases：`process_reward_v0` 能过滤一部分跑题/污染候选，但也存在明显短答案偏好
- 已对 `process_reward_v0` 做一轮 sanity 修正，复跑后 final accuracy 仍为 `0.9300`，但 `corr(num_steps, process_reward)` 仍约为 `-0.7952`
- 已用 DeepSeek `deepseek-v4-flash` 对同一批 `100×4` candidates 完成 LLM judge 标注
- 已生成 `324` 条 PRM preference rows
- LLM judge top1 final accuracy 为 `0.9300`，相对 final-only 改变 `26/100` 道题，没有 `1->0` 准确率伤害
- DeepSeek judge 实际费用约 `$0.0603`，折合约 `0.43 RMB`

这说明当前工程已经具备：

- 远端可运行环境
- 可复现的数据预处理脚本
- 可用于后续 reward 计算和实验对照的数据底座
- 可用于 LLM judge 标注的候选轨迹数据
- 可用于 PRM smoke training 的第一版 preference 数据

## 合作者建议先看什么

如果你是第一次进入这个仓库，建议按这个顺序阅读：

1. 先看当前总览文档 `README.md`
2. 再看 `step1` 归档文档 `README_process_supervised_rl.md`
3. 然后看 `step2` 归档文档 `README_process_supervised_rl_step2.md`
4. 再看当前执行文档 `README_process_supervised_rl_step3.md`
5. 再看 `configs/`、`scripts/`、`src/` 的代码结构
6. 最后直接进入“当前下一步命令”执行区

## 工作流

### 本地负责

- 写代码
- 改配置
- 提交 Git
- 做少量无模型逻辑检查

### 远端负责

- 建 Python 环境
- 下载与清洗完整数据
- 跑推理
- 跑训练
- 跑评估

## 目录说明

- `configs/`: 数据、奖励、训练、评估配置
- `data/`: 原始数据、中间数据、处理后数据、调试子集
- `scripts/`: 命令行入口脚本
- `src/psrl/`: 核心 Python 包
- `tests/`: 单元测试
- `docs/`: 设计文档与实现计划

## 当前代码范围

第一版包含：

- `GSM8K` 数据准备
- 最终答案标准化
- 基于规则的步骤切分
- 基于规则的 `process reward v0`
- 样本级 reward 聚合与打分脚本
- 候选轨迹生成脚本
- 候选轨迹 reranking 分析脚本
- LLM judge prompt / JSON parsing / DeepSeek API 调用脚本
- PRM preference dataset 构造脚本

## 当前方法判断

`process_reward_v0` 的阶段性任务已经完成：

- 它证明了 final answer 之外的过程信号会改变候选选择。
- 它在 `100×4` 实验里没有造成 `1->0` 的 final accuracy 伤害。
- 它暴露了规则 reward 的短答案偏好和脆弱性。

因此，后续不再把 `process_reward_v0` 当作 PRM 训练数据的主标注器。新的主线是：

```text
local 7B model generates candidates
-> strong LLM API judges candidate process quality
-> build preference / score data
-> train PRM
-> final-only vs final+PRM reranking
-> post-training policy with PRM-filtered or PRM-weighted data
```

`process_reward_v0` 继续保留为：

- baseline
- sanity check
- LLM judge 输出的辅助对照
- 规则风险样例来源

## 当前下一步命令

下面这组命令对应的是当前 `step3` 继续推进 PRM 前的环境确认和数据检查。

### 1. 进入远端仓库

```bash
cd ~/autodl-tmp/process_supervised_rl
git checkout main
git pull --ff-only origin main
```

### 2. 确认模型、候选、judge 结果、PRM 数据和测试

```bash
ls /root/autodl-tmp/models/deepseek-math-7b-instruct
ls data/debug/gsm8k_train_debug.jsonl
ls logs/candidates/gsm8k_train_debug_candidates.jsonl
ls logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl
ls data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl
pytest tests -v
```

### 3. 如候选文件丢失，再复跑 `100×4` 候选生成

```bash
python scripts/generate_candidates.py \
  --input data/debug/gsm8k_train_debug.jsonl \
  --output logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --model-name /root/autodl-tmp/models/deepseek-math-7b-instruct \
  --limit 100 \
  --num-candidates 4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 512
```

### 4. 如 judge 文件丢失，再复跑 DeepSeek judge

```bash
export DEEPSEEK_API_KEY='<your-key>'

python scripts/judge_candidates_with_llm.py \
  --input logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --output logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl \
  --provider deepseek \
  --model deepseek-v4-flash \
  --limit 100 \
  --max-tokens 4096 \
  --parse-retries 1
```

### 5. 如 PRM preference 文件丢失，再重新构造

```bash
python scripts/build_prm_dataset.py \
  --candidates logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --judgements logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl \
  --output data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl
```

### 6. 当前下一步要新增的训练入口

PRM 训练脚本还没实现，目标命令形态如下：

```text
python scripts/train_prm.py \
  --preferences data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl \
  --output-dir logs/prm/gsm8k_debug_prm_smoke
```

当前正式产物：

- `logs/candidates/gsm8k_train_debug_candidates.jsonl`
- `logs/candidate_reward/gsm8k_train_debug_candidates_scored.jsonl`
- `logs/candidate_reward/final_only_vs_final_plus_process_selection_report.md`
- `logs/candidate_reward/changed_case_review_100x4.md`
- `logs/candidate_reward/changed_case_first_pass_labels.md`
- `logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl`
- `data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl`

## 数据清洗原则

这个项目的数据清洗脚本写在仓库里，不是“只在远端存在的一套脚本”。

实际使用方式是：

- 本地先用很小的样本验证逻辑
- 远端再用同一套脚本跑完整数据

第一版清洗主要做这些事情：

- 统一输出 schema
- 标准化最终答案
- 抽取原始解题过程
- 按规则切分步骤
- 生成可用于 debug 的小样本

## 下一步计划

下一步继续执行 `step3`：

1. 定义 LLM judge JSON schema。已完成
2. 实现 `scripts/judge_candidates_with_llm.py`。已完成
3. 先用 `10` 道题做 API smoke test。已完成
4. 再标注完整 `100×4` candidates。已完成
5. 生成 judge ranking / preference pairs。已完成
6. 训练第一版 trajectory-level PRM。
7. 用 `final-only` vs `final+PRM` 做候选 reranking。
8. 如果 PRM 不降低 final accuracy 且人工抽查过程质量更好，再进入 reward-weighted SFT / post-training。

## API 选择建议

截至 `2026-05-07`，建议继续优先使用 DeepSeek API 跑 judge 标注：

- DeepSeek 官方 API 文档明确支持 OpenAI 格式、JSON Output、1M context 的 `deepseek-v4-flash/pro`。
- `deepseek-v4-flash` 官方价格明显更适合批量标注：cache miss input `$0.14 / 1M tokens`，output `$0.28 / 1M tokens`。
- Kimi 官方文档显示最新主力是 `Kimi K2.6`，上下文 `256K`，能力强，但价格和限速需要在控制台确认；旧 `kimi-k2` 系列将在 `2026-05-25` 下线，不建议新实验依赖旧模型。
- 如果 Kimi 有免费额度，可以用来抽样复核 DeepSeek judge 的一致性，不建议第一版主流程押在免费额度上。

参考文档：

- DeepSeek Models & Pricing: <https://api-docs.deepseek.com/quick_start/pricing>
- Kimi K2.6: <https://platform.kimi.com/docs/pricing/chat-k26>
- Kimi K2 legacy notice: <https://platform.kimi.com/docs/pricing/chat-k2>
- Kimi 充值与限速: <https://platform.kimi.com/docs/pricing/limits>

推荐策略：

1. 先买少量 DeepSeek 额度，跑 `10` 道题 smoke test。
2. 确认 JSON 稳定、成本可控后，跑完整 `100×4`。
3. 用 Kimi K2.6 免费额度或小额充值复核 `20` 个边界样本。
4. 如果 DeepSeek 与 Kimi 在边界样本上分歧很大，再考虑 ensemble judge；否则先用 DeepSeek 继续推进 PRM 数据生产。

## 当前协作约定

- 总览信息优先维护在 `README.md`
- 每个 step 单独维护一份 step 文档
- 大更新可以积累一段后再统一 push
- 训练、完整数据、日志默认放远端，不提交到仓库；小规模 debug 实验报告和可复现实验产物可以按需提交

## 已完成但暂不重复执行的事项

下面这些已经完成，不需要当前协作者重复从头做一遍：

- 仓库拉取
- Python 环境核验
- `PyYAML`、`pytest` 与数据下载依赖安装
- `GSM8K` 预处理
- `debug` 子集生成
- `process reward v0` 首轮 debug 打分

除非远端环境丢失，否则优先直接进入“当前下一步命令”。
