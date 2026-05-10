# Process-Supervised RL

面向“过程监督推理”实验的协作总览文档。

这个 `README.md` 的职责不是讲某一个 step 的全部细节，而是给合作者快速同步三件事：

- 这个仓库现在在做什么
- 当前进行到哪一个 step
- 如果要参与，应该先看哪些文档、先做哪些事

## 当前状态

- 当前总阶段：第一阶段
- 当前 step：`step3` 收官，进入 `RL 对齐/benchmark`
- 当前 step 文档：[README_process_supervised_rl_step3.md](README_process_supervised_rl_step3.md)
- 上一 step 归档文档：[README_process_supervised_rl_step2.md](README_process_supervised_rl_step2.md)
- 当前完成度：`1k×4` 数据闭环、PRM v2 大规模调参、PRM 筛选后 LoRA-SFT、GRPO 小网格训练与 `64` 条快速 benchmark 已完成；下一步做中等规模/全量 benchmark 与 changed-case 抽查

## 当前目标

- 使用 `GSM8K` 跑通最小实验闭环
- 对比 `final-only`、`final+LLM-judge`、`final+PRM`、`PRM-filtered SFT` 的候选轨迹选择效果
- 默认采用“本地开发，远端执行”的工作流
- 当前主线：用强 LLM API judge 生成过程偏好数据，训练 PRM，再进入 RL（GRPO）对齐
- 当前下一步：交接 benchmark，对 `PRM-filtered SFT` 与最佳 GRPO adapter 做更大样本评测，并判断是否继续调 GRPO 或扩大 PRM 数据

## 当前数据集

- 主数据集：`GSM8K`
- 训练题：`data/processed/gsm8k_train.jsonl`（7473）
- 测试题：`data/processed/gsm8k_test.jsonl`（1319）
- 本阶段候选生成规模：`1000 questions × 4 candidates = 4000 trajectories`

说明：当前主线没有切到其它“高级 PMK 数据集”；你现在用的还是 `GSM8K` + 本地 7B 生成轨迹 + LLM judge 标注。

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

1. 固化当前 `PRM + SFT + GRPO grid` 产物版本
2. 由 benchmark 负责人按同一 decode config 评测 `base 7B` / `PRM-filtered-SFT` / `GRPO` adapters
3. 优先对 `cfg3` 做中等规模评测（建议先 `256` 条，不急着全量 `1319`）
4. 做 changed-case 抽查，确认没有 reward hacking
5. 如果 GRPO 仍无法超过 SFT，再决定是继续调 RL 超参，还是扩大 PRM 数据重训

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
- 已完成第一版轻量 trajectory-level preference PRM smoke training：`324` 个 preference pairs，训练集 pair accuracy `0.9722`
- `final+PRM` top1 final accuracy 为 `0.9300`，相对 final-only 改变 `46/100` 道题，其中 `42` 个是 `1->1`、`4` 个是 `0->0`，没有 `1->0` 准确率伤害
- 已完成 changed top1 的 DeepSeek 二选一抽查：`46` 个 changed 样本中，DeepSeek 支持 PRM `17` 个、支持 final-only `29` 个
- 已完成轻量 PRM 小规模调参（120/0.12/0.0）：对原始 4-candidate judge 的一致率提升到 `0.7100`（基线约 `0.6700`），但仍低于 final-only 的 `0.7400`
- 已完成 `1k×4` 候选生成（`4000` 条轨迹）与 DeepSeek judge 全量标注（`1000` 条）
- 已完成 `1k×4` PRM preference 构造，得到 `3205` 条 pairwise preference 数据
- 已完成 PRM v2 超参扫描（`108` 个 trial）
- PRM v2 最优配置：`max_features=12000, hidden_dim=384, epochs=16, lr=0.0015, wd=0.0001`
- PRM v2 最优指标：`judge_agree_prm_rate=0.795`，`final_plus_prm_accuracy=0.875`（与 final-only 持平，不降）
- 已完成 PRM 筛选后 LoRA-SFT（7B 增量微调）并产出 adapter
- SFT 训练摘要：`train_runtime=180.1s`，`train_loss=1.112`，`final eval_loss=1.068`
- SFT 产物路径：`logs/sft/prm_filtered_lora_1000x4/final/adapter_model.safetensors`
- 已修复并验证 GRPO LoRA 可训练性：训练前后多层 LoRA tensor hash 发生变化，保存后的 adapter 与训练后内存权重一致
- 已完成 `4` 组 GRPO 小网格训练，均从同一个 SFT adapter 初始化：
  - `cfg1_w0p1_lr5e6_b0p05`
  - `cfg2_w0p2_lr5e6_b0p05`
  - `cfg3_w0p2_lr2e6_b0p10`
  - `cfg4_w0p3_lr2e6_b0p10`
- 已完成 `64` 条 GSM8K test 快速 benchmark：
  - `base_7b`: `29/64 = 0.4531`
  - `prm_filtered_sft`: `36/64 = 0.5625`
  - `grpo_cfg1`: `34/64 = 0.5313`
  - `grpo_cfg2`: `31/64 = 0.4844`
  - `grpo_cfg3`: `36/64 = 0.5625`
  - `grpo_cfg4`: `35/64 = 0.5469`
- 当前快速结论：`cfg3` 追平 SFT，尚未证明超过 SFT；建议先做 `256` 条中等规模 benchmark，再决定是否跑全量 `1319`

这说明当前工程已经具备：

- 远端可运行环境
- 可复现的数据预处理脚本
- 可用于后续 reward 计算和实验对照的数据底座
- 可用于 LLM judge 标注的候选轨迹数据
- 可用于 PRM smoke training 的第一版 preference 数据
- 可执行的轻量 trajectory-level preference PRM 训练入口
- 第一版 `final-only` vs `final+PRM` reranking smoke report
- 可复现的 `1k×4` 级别 PRM 训练闭环（judge -> preference -> prm_v2 sweep）
- 可直接用于 RL 初始化的 SFT-LoRA policy adapter

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

## RL 阶段规范（下一会话直接执行）

下面是进入 GRPO/RL 前必须遵守的执行规范，避免 reward hacking 和无效训练。

### 1) 初始策略与数据

- policy init：使用已完成的 SFT adapter（`logs/sft/prm_filtered_lora_1000x4/final`）作为 RL 初始策略。
- 训练题集：继续用 `GSM8K train`（`data/processed/gsm8k_train.jsonl`），先做小规模 smoke run（例如前 `512` 题）。
- 评测题集：固定 `GSM8K test`（`data/processed/gsm8k_test.jsonl`），每次对比必须同解码参数。

### 2) 奖励定义（必须固定版本）

- 推荐总奖励：`R = w_final * final_reward + w_prm * prm_reward`。
- `final_reward`：答案正确性（当前二值或归一分）。
- `prm_reward`：PRM 分数标准化后使用（建议 z-score 或 min-max 后再 clip）。
- 建议初值：`w_final=1.0, w_prm=0.2~0.4`，先从保守权重起步，观察稳定性再调大。
- 必须记录奖励版本号与参数，避免“同名实验不同奖励函数”。

### 3) 训练安全阈值

- 强制监控：`reward_mean`、`KL`、`response_length`、`final_accuracy(proxy)`。
- 触发停训条件（任一命中即暂停）：
  - KL 持续暴涨（连续多个 eval window 超阈值）。
  - 平均输出长度异常缩短/拉长（疑似投机策略）。
  - final accuracy 明显下滑（相对 SFT baseline 下跌超预设阈值）。

### 4) 防 reward hacking 检查

- 每轮固定抽查 changed cases（建议 30~50 条）。
- 检查项：空洞模板化回答、无关展开、格式投机、只押最终答案不保过程质量。
- 必须保留抽查记录文件，作为是否继续扩训的 gate。

### 5) 实验对照与命名

- 统一四组对照：
  1. `base_7b`
  2. `prm_rerank`
  3. `prm_filtered_sft_lora`
  4. `grpo_policy`
- 命名建议：`exp_<date>_<policy>_<rewardver>_<seed>`。
- 每次实验至少固定 `seed`、`decode config`、`eval set` 三项。

### 6) 扩训条件（满足后才加预算）

- 与 `prm_filtered_sft_lora` 相比：
  - final accuracy 不下降；
  - judge 一致率或人工过程质量有明确提升；
  - 无明显 reward hacking。

## 阶段归档（截至 2026-05-08）

- `1k×4` teacher 标注闭环完成。
- PRM v2 达到高一致率并保持 final accuracy 不下降。
- PRM 数据筛选后的 LoRA-SFT 完成，policy init 已具备。
- 项目从 “PRM 构建阶段” 切换到 “RL 对齐主阶段”。

## 当前下一步命令

下面先给 RL 会话启动命令；其后的 `step3` 命令保留为历史参考。

### Benchmark 交接说明

远端当前在 AutoDL 上，主要执行目录：

```bash
cd ~/autodl-tmp/process_supervised_rl
```

关键产物路径：

- base 模型：`/root/autodl-tmp/models/deepseek-math-7b-instruct`（约 `13G`）
- GSM8K 数据：`data/processed/gsm8k_train.jsonl`、`data/processed/gsm8k_test.jsonl`
- SFT adapter：`logs/sft/prm_filtered_lora_1000x4/final`
- PRM v2：`logs/prm_v2/sweep_1000x4_v2/trial_056`
- GRPO 网格：`logs/rl/grpo_grid_20260510`
- 64 条快速 benchmark 结果：`logs/rl/grpo_grid_20260510/benchmark/results_64.jsonl`

注意：AutoDL 登录提示里明确说明 `/root/autodl-tmp` 是数据盘，通常不会随“保存镜像”一起保存。也就是说，只给组员一个系统镜像大概率不够；需要同时共享上面这些数据盘 artifact，或者让组员直接使用同一台实例/同一块数据盘。

已在系统盘打好精简交接包：

```bash
/root/psrl_benchmark_handoff_20260510.tar.gz
```

该包不包含 base 7B 模型；组员可自行下载 base 模型，再使用包内的 SFT/GRPO adapters 做 benchmark。

建议下一步 benchmark：

```bash
# 先做 256 条中等规模评测，不急着跑全量 1319
# 对照组：base_7b, prm_filtered_sft, grpo_cfg3
# 可选备选：grpo_cfg4
```

### RL 会话启动（推荐）

```bash
cd ~/autodl-tmp/process_supervised_rl
git checkout main
git pull --ff-only origin main

# 检查关键产物
ls logs/prm_v2/sweep_1000x4_v2/best.json
ls logs/sft/prm_filtered_lora_1000x4/final/adapter_model.safetensors
ls data/prm/gsm8k_train_1000_prm_preferences_deepseek_v4_flash.jsonl
```

### 历史命令（step3）

下面这组命令对应 `step3` 继续推进 PRM 前的环境确认和数据检查。

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

### 6. 训练第一版 trajectory-level preference PRM

第一版 PRM 是轻量 trajectory-level preference 模型，用 `question + candidate_text` 作为输入，用 chosen/rejected pairwise loss 学习 judge preference。它用于 smoke test 整个 PRM reranking 闭环，后续可以替换为 transformer reward head。

```bash
python scripts/train_prm.py \
  --preferences data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl \
  --output-dir logs/prm/gsm8k_debug_prm_smoke \
  --candidates logs/candidates/gsm8k_train_debug_candidates.jsonl \
  --scored-output logs/prm/gsm8k_debug_prm_smoke/scored_candidates.jsonl \
  --report-output logs/prm/gsm8k_debug_prm_smoke/final_only_vs_final_plus_prm_selection_report.md
```

当前正式产物：

- `logs/candidates/gsm8k_train_debug_candidates.jsonl`
- `logs/candidate_reward/gsm8k_train_debug_candidates_scored.jsonl`
- `logs/candidate_reward/final_only_vs_final_plus_process_selection_report.md`
- `logs/candidate_reward/changed_case_review_100x4.md`
- `logs/candidate_reward/changed_case_first_pass_labels.md`
- `logs/llm_judge/gsm8k_train_debug_candidates_judged_deepseek_v4_flash_100.jsonl`
- `data/prm/gsm8k_train_debug_prm_preferences_deepseek_v4_flash_100.jsonl`
- `logs/prm/gsm8k_debug_prm_smoke/model.json`
- `logs/prm/gsm8k_debug_prm_smoke/scored_candidates.jsonl`
- `logs/prm/gsm8k_debug_prm_smoke/final_only_vs_final_plus_prm_selection_report.md`

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
6. 训练第一版 trajectory-level PRM。已完成
7. 用 `final-only` vs `final+PRM` 做候选 reranking。已完成
8. 人工抽查 `final+PRM` 改变 top1 的样本。已完成
9. 基于抽查结果，先做 PRM 建模改进（超出轻量线性词袋），再决定是否扩展到 `1k×4`。
10. 新版 PRM 如果在 judge 一致率和人工抽查都不差于 final-only，再进入 `1k×4` 与后续 reward-weighted SFT / post-training。

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
