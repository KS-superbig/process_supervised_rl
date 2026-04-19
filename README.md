# Process-Supervised RL

面向“过程监督推理”实验的协作总览文档。

这个 `README.md` 的职责不是讲某一个 step 的全部细节，而是给合作者快速同步三件事：

- 这个仓库现在在做什么
- 当前进行到哪一个 step
- 如果要参与，应该先看哪些文档、先做哪些事

## 当前状态

- 当前总阶段：第一阶段
- 当前 step：`step1`
- 当前 step 文档：[README_process_supervised_rl.md](/Users/bytedance/Documents/New%20project/README_process_supervised_rl.md)
- 当前完成度：已跑通远端环境搭建、`GSM8K` 预处理链路、`process reward v0` 代码骨架

## 当前目标

- 使用 `GSM8K` 跑通最小实验闭环
- 对比 `final-only reward` 与 `final + process reward`
- 默认采用“本地开发，远端执行”的工作流

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

这是当前 `step1` 的执行文档。

主要维护：

- `step1` 为什么这样设计
- `step1` 具体要完成哪些事情
- `step1` 的方法、数据、实验范围
- `step1` 的验收标准

后续如果 `step1` 完成，会继续新增 `step2` 文档。到时候这个总览 README 会同步更新“当前 step”。

## 当前优先事项

当前默认优先顺序：

1. 实现过程奖励 `v0`
2. 对一批 `debug` 样本做逐题打分检查
3. 做 `final-only` 与 `final + process` 的第一版对照
4. 检查 reward 是否稳定、是否存在明显 reward hacking
5. 再进入第一版训练与评估

## 当前已完成

目前已经完成的内容：

- 远端完成仓库拉取与 Python 环境搭建
- 远端成功安装 `PyYAML` 与 `pytest`
- `GSM8K` 训练集预处理完成，共生成 `7473` 条样本
- `GSM8K` 测试集预处理完成，共生成 `1319` 条样本
- `debug` 子集生成完成，共生成 `100` 条样本
- 统一输出 schema 已经验证可用
- `pytest tests -v` 已通过，当前共 `8` 个测试全部通过

这说明当前工程已经具备：

- 远端可运行环境
- 可复现的数据预处理脚本
- 可用于后续 reward 计算和训练的数据底座

## 合作者建议先看什么

如果你是第一次进入这个仓库，建议按这个顺序阅读：

1. 先看当前总览文档 `README.md`
2. 再看当前 step 文档 `README_process_supervised_rl.md`
3. 然后看 `configs/`、`scripts/`、`src/` 的代码结构
4. 最后直接进入“当前下一步命令”执行区

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

第一版只包含：

- `GSM8K` 数据准备
- 最终答案标准化
- 基于规则的步骤切分
- 基于规则的 `process reward v0`
- 样本级 reward 聚合与打分脚本

## 当前下一步命令

下面这组命令不是环境初始化，而是当前阶段真正建议协作者继续执行的内容。

### 1. 进入仓库并激活环境

```bash
cd ~/autodl-tmp/process_supervised_rl
source .venv/bin/activate
git checkout main
```

### 2. 先确认当前数据底座存在

```bash
ls data/processed
ls data/debug
head -n 2 data/debug/gsm8k_train_debug.jsonl
pytest tests -v
```

### 3. 当前开发目标

当前不是重新跑环境配置，而是开始做过程奖励 `v0`。

建议先围绕这些目录开始：

```bash
ls src/psrl
ls src/psrl/data
cat configs/reward/process_reward_v0.yaml
```

### 4. 当前建议开发顺序

```text
先实现规则版 process reward
-> 先对 debug subset 做打分
-> 检查 reward 是否合理
-> 再做 final-only vs final+process 对照
```

### 5. 当前阶段建议的调试命令

```bash
python scripts/prepare_gsm8k.py --help
python scripts/build_debug_subset.py --help
python scripts/score_samples.py --help
```

### 6. 当前建议执行的 reward 打分命令

下面这组命令是给远端同学直接跑 `debug subset` 的。默认会输出每条样本的 `final_reward`、`process_reward`、`total_reward` 和组件均值。

```bash
cd ~/autodl-tmp/process_supervised_rl
source .venv/bin/activate

python scripts/score_samples.py \
  --input data/debug/gsm8k_train_debug.jsonl \
  --output logs/process_reward_v0/gsm8k_train_debug_scored.jsonl \
  --reward-config configs/reward/process_reward_v0.yaml
```

如果要先看 `final-only` 基线，把 `configs/reward/process_reward_v0.yaml` 里的 `process_reward_weight` 临时改成 `0.0`，再跑同一条命令即可。

如果要先看 `final + process`，就保留当前配置里的 `process_reward_weight: 0.3`。

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

下一步默认进入过程奖励 `v0` 阶段：

1. 先实现规则版过程奖励，不先训练 reward model
2. 先用固定权重把 `final reward` 和 `process reward` 合成总分
3. 先在 `debug` 子集和小规模样本上检查 reward 是否合理
4. 重点检查 reward 有没有崩、有没有明显偏向废话长输出
5. 再做第一版对照实验，判断 reward 有没有明显作用

## 当前协作约定

- 总览信息优先维护在 `README.md`
- 每个 step 单独维护一份 step 文档
- 大更新可以积累一段后再统一 push
- 训练、完整数据、日志默认放远端，不提交到仓库

## 已完成但暂不重复执行的事项

下面这些已经完成，不需要当前协作者重复从头做一遍：

- 仓库拉取
- Python 虚拟环境创建
- `PyYAML` 与 `pytest` 安装
- `GSM8K` 预处理
- `debug` 子集生成

除非远端环境丢失，否则优先直接进入“当前下一步命令”。
