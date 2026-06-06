# 大模型数学推理 GRPO step4：MATH warmup 与 r512 gated PRM

> 文档角色：`step4` 历史归档  
> 时间范围：2026-06 初  
> 前置阶段：`step3` 的 GSM8K SFT/PRM/GRPO 诊断  
> 当前结论：旧 GSM8K 小步 GRPO 提升不明显；切到 MATH L3/4 后，`r512 + gated PRM + 1000 steps` 在 MATH500 quick check 上出现正向信号。

## 1. 为什么进入 step4

`step3` 后的主要问题是：在 GSM8K 上，SFT v3 spacefix 已经比较强，后续 GRPO 很难拉开差距。

我们最初尝试用 Skywork PRM 接入 GRPO，希望从 process reward 里拿到额外收益。但过程中发现，旧流程存在几类问题：

- baseline 一度混淆了 `base` 与 `instruct` 模型。
- 训练脚本早期默认接的是仓库内 MLP PRM v2，不是 Skywork PRM。
- Skywork PRM 必须作为 process reward 使用，不能当 final-only ORM。
- 旧 reward 是加法：`final_reward + 0.2 * prm_reward`，错误答案也可能靠 PRM 得正分。
- 旧 GRPO 只有约 `200` steps，LoRA rank 小，训练强度明显不够。
- MATH 场景中 completion 较长，`512` tokens 评测/训练上限可能截断部分长解。

因此 step4 的目标是重新搭一条更干净的数学 RL 路线：

```text
deepseek-math-7b-instruct
  + GSM8K SFT v3 spacefix adapter
  + MATH Level 3/4 light SFT warmup
  -> merge warmup into model weights
  + fresh r512 LoRA adapter
  -> GRPO with gated Skywork PRM reward
```

## 2. MATH L3/4 light SFT warmup

为了避免 GSM8K 已经被 SFT 吃满，我们切到 MATH 数据，选择 Level 3 和 Level 4 题目做轻量 SFT warmup。

核心目的不是让 SFT 直接把 MATH 做满，而是：

- 对齐 MATH 风格长推理。
- 强化 `\boxed{}` 最终答案格式。
- 给后续 GRPO 一个合理的冷启动点。
- 避免过拟合到训练题，保留 RL 探索空间。

产物：

```text
/root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final
```

本地归档校验文件：

```text
artifacts/sft_v3_math_l34_3000_e0p4_lr1e5_len2048.tar.gz.sha256
```

注意：这个 warmup adapter 是在此前 GSM8K SFT v3 spacefix 基础上继续轻量 SFT 得到的，因此它携带了 GSM8K SFT v3 和 MATH L3/4 warmup 两部分能力。

## 3. 旧 baseline 的主要失误

### 3.1 模型起点混乱

一开始需要确认到底是：

```text
deepseek-math-7b-base
```

还是：

```text
deepseek-math-7b-instruct
```

项目后续统一采用 `deepseek-math-7b-instruct` 作为 base model，再挂 SFT / RL adapter。

如果直接从 base model 跑 RL，或者忘记挂 SFT adapter，实验结论不可和当前主线比较。

### 3.2 Skywork PRM 接入方式错误风险

旧的 `train_grpo_smoke.py` 最初设计是加载仓库内 MLP PRM v2：

```text
model.pt
meta.json
```

而 Skywork PRM 是 Hugging Face 模型目录，包含：

```text
pytorch_model.bin
configuration_qwen2_rm.py
modeling_qwen2_rm.py
tokenizer.json
config.json
```

因此不能把 Skywork 目录直接塞给 `load_mlp_prm`。后续修正为 Skywork PRM process reward 路线，并验证：

```text
Qwen2ForRewardModel / manual Qwen2 v_head
has v_head: True
step_rewards=[...]
```

### 3.3 additive reward 不适合数学场景

旧 reward：

```text
total_reward = final_reward + 0.2 * prm_reward
```

问题是：最终答案错了，模型仍可能因为过程分拿到正 reward。

数学题有明确 ground truth，final correctness 应该是硬约束。因此 step4 改为 gated reward：

```text
if final_answer_correct:
    total_reward = 1.0 + 0.2 * prm_reward
else:
    total_reward = 0.0
```

这样 PRM 扮演的是“答对样本内部排序”的角色，而不是奖励错答案的漂亮过程。

### 3.4 训练步数和 LoRA 容量不足

旧实验大多是：

```text
max_steps = 200
small-rank LoRA
```

在 GRPO 场景下，200 steps 只能算 smoke / pilot，不足以说明 RL 是否有效。step4 改为：

```text
max_steps = 1000
new_lora_r = 512
new_lora_alpha = 1024
```

## 4. step4 最终采用的 GRPO 配置

训练链路：

```text
1. 加载 deepseek-math-7b-instruct
2. 加载 MATH L3/4 warmup SFT adapter
3. merge_and_unload()，把 warmup 写入当前模型权重
4. 挂 fresh r512 LoRA
5. 用 GRPO 训练这个新 r512 adapter
```

关键参数：

```text
train_jsonl = data/processed/math_l34_train_3000_seed42.jsonl
model_name = /root/autodl-tmp/models/deepseek-math-7b-instruct
sft_adapter = /root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final
prm_dir = /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
prm_backend = skywork

reward_mode = gated_prm
wrong_final_reward = 0.0
final_weight = 1.0
prm_weight = 0.2

num_generations = 4
generation_batch_size = 4
max_prompt_length = 512
max_completion_length = 768

learning_rate = 1e-6
beta = 0.04
loss_type = grpo
epsilon_low = 0.2
epsilon_high = 0.2

per_device_train_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = true
bf16 = true
max_steps = 1000

merge_sft_adapter = true
new_lora_r = 512
new_lora_alpha = 1024
new_lora_dropout = 0.05
new_lora_target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

输出目录：

```text
/root/autodl-tmp/psrl_outputs/grpo_math_l34_r512_gated_len768_1000_s42_20260604_202818
```

最终 adapter：

```text
/root/autodl-tmp/psrl_outputs/grpo_math_l34_r512_gated_len768_1000_s42_20260604_202818/final
```

## 5. 保存与磁盘策略

r512 LoRA 很大，单个 adapter 约 `4.8G`。TRL 保存 checkpoint 时还会额外保存 `ref/adapter_model.safetensors`，导致一个 checkpoint 接近 `9.6G`。

为避免数据盘爆掉，采用：

```text
save_steps = 500
save_total_limit = 1
save_only_model = true
delete_saved_ref_adapter = true
```

并加入保存后清理逻辑：

```text
checkpoint-*/ref
final/ref
```

训练完成后确认：

```text
checkpoint-1000/adapter_model.safetensors
final/adapter_model.safetensors
```

存在，且 `ref/` 已删除。

最终目录约 `9.0G`。

## 6. 训练结果与诊断

训练完成：

```text
1000 / 1000 steps
train_runtime ~= 54810 seconds
train_loss ~= 0.0002661
epoch ~= 0.6667
```

adapter 诊断显示 LoRA 参数确实更新并保存：

```text
pre-train lora_B norm = 0
post-train lora_B norm > 0
saved adapter sha256 == post-train memory sha256
```

这说明训练不是空跑，最终保存的 adapter 和内存中训练后的参数一致。

## 7. 正确评测链路

一个关键点：因为训练时是先 merge warmup SFT，再挂 r512 LoRA，所以评测不能简单写成：

```text
base + r512_grpo_adapter
```

这会漏掉 warmup SFT 能力。

正确链路是：

```text
base
  + warmup SFT adapter
  -> merge
  + r512 GRPO adapter
```

也就是：

```python
base = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base, warmup_adapter)
model = model.merge_and_unload()
model = PeftModel.from_pretrained(model, r512_grpo_adapter)
```

后续正式评测和组员复现实验必须遵守这个链路。

## 8. MATH500 quick check

为了快速判断 GRPO 是否有方向性效果，先在 MATH500 上做了 quick check。

评测设置：

```text
dataset = HuggingFaceH4/MATH-500
samples = first 40 examples
max_new_tokens = 512
decode = greedy
verifier = math_verify
```

注意：这是 quick check，不是最终正式分数。训练时 completion 上限是 `768`，正式评测建议也使用 `max_gen_toks=768`。

结果：

| 样本范围 | SFT warmup | r512 gated GRPO |
| --- | ---: | ---: |
| idx 0-19 | 9/20 = 45% | 11/20 = 55% |
| idx 20-39 | 7/20 = 35% | 11/20 = 55% |
| 合计 | 16/40 = 40% | 22/40 = 55% |

格式率：

```text
SFT 约 87.5%
GRPO 约 87.5%
```

平均生成长度：

```text
SFT 约 421 tokens
GRPO 约 385 tokens
```

临时结论：

```text
r512 + gated PRM + 1000 steps 在 quick check 上出现正向信号。
```

但 40 条样本太少，不能作为最终论文/汇报结论。下一步应跑：

```text
MATH500 subset100 或 full 500
max_gen_toks = 768
正确链路：base + warmup merge + r512 adapter
```

## 9. 本阶段经验总结

这段实验的核心教训：

1. 不要只看 adapter 是否存在，要确认它挂在哪个 base 上。
2. 不要把 Skywork PRM 当普通 final reward model。
3. 数学场景中 final correctness 应作为硬门控。
4. GRPO 的 200 steps 只能算 smoke，不能轻易据此否定 RL。
5. 小 rank adapter 容量可能限制 RL 更新。
6. MATH 题更长，`512` token 上限可能不够，应优先使用 `768` 或更高做正式评测。
7. merge warmup + fresh RL LoRA 后，评测链路必须复现 merge，否则评估对象错误。
8. r512 adapter 占用大，必须提前设计 checkpoint/ref 清理策略。

## 10. 下一步建议

step4 后的直接下一步：

1. 用正确链路跑 MATH500 subset100，`max_gen_toks=768`。
2. 如果 subset100 仍有明显提升，再跑 full MATH500。
3. 固化评测脚本，避免每次手写 merge/load 逻辑。
4. 再决定是否继续做消融：
   - r256 vs r512
   - `prm_weight=0.1/0.2/0.3`
   - `max_completion_length=768/1024`
   - `gated_prm` vs additive
   - 1000 steps vs 1500/2000 steps

根 README 后续应作为“给组员的当前执行指南”重写，而不是继续堆历史细节。
