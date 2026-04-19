# Process-Supervised RL

面向“过程监督推理”第一阶段实验的工程化仓库。

当前阶段目标：

- 使用 `GSM8K` 跑通最小实验闭环
- 对比 `final-only reward` 与 `final + process reward`
- 默认采用“本地开发，远端执行”的工作流

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

## 当前范围

第一版只包含：

- `GSM8K` 数据准备
- 最终答案标准化
- 基于规则的步骤切分
- 后续过程奖励模块的代码位置预留

## 首次启动流程

推荐工作方式：

- 本地：写代码、改配置、提交 Git
- 远端：建环境、放完整数据、跑预处理、跑训练、跑评估

### 第一步：拉代码

在远端服务器执行：

```bash
cd ~
git clone https://github.com/KS-superbig/process_supervised_rl.git
cd process_supervised_rl
git checkout main
```

### 第二步：创建 Python 环境

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pyyaml pytest
```

### 第三步：准备 GSM8K 原始数据

先创建目录：

```bash
mkdir -p data/raw/gsm8k
```

然后把原始数据放成下面两个文件：

```text
data/raw/gsm8k/train.jsonl
data/raw/gsm8k/test.jsonl
```

### 第四步：运行数据预处理

处理训练集：

```bash
python scripts/prepare_gsm8k.py \
  --input data/raw/gsm8k/train.jsonl \
  --output data/processed/gsm8k_train.jsonl \
  --split train
```

处理测试集：

```bash
python scripts/prepare_gsm8k.py \
  --input data/raw/gsm8k/test.jsonl \
  --output data/processed/gsm8k_test.jsonl \
  --split test
```

生成调试子集：

```bash
python scripts/build_debug_subset.py \
  --input data/processed/gsm8k_train.jsonl \
  --output data/debug/gsm8k_train_debug.jsonl \
  --limit 100
```

### 第五步：检查生成结果

```bash
ls data/processed
ls data/debug
head -n 2 data/processed/gsm8k_train.jsonl
head -n 2 data/debug/gsm8k_train_debug.jsonl
```

### 第六步：运行测试

```bash
pytest tests -v
```

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

## 远端建议

建议在远端安装至少这些依赖：

- Python 3.10+
- `pytest`
- `PyYAML`

后续如果进入正式训练，再补充 `transformers`、`datasets`、`peft` 等依赖。
