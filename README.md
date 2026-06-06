# MATH GRPO 消融实验任务书

这个文档是当前给组员执行实验用的主 README。历史探索和弯路已经归档到 `docs/history/`，这里不再复述早期 GSM8K 旧 baseline 和旧消融。

当前结论很明确：之前围绕旧 baseline 做的消融实验不再作为有效对比，因为那版 RL 基本没有拉开效果。现在统一以 step4 的新 baseline 为锚点，重新跑消融。

## 1. 当前实验主线

当前模型链路：

```text
deepseek-math-7b-instruct
  + SFT v3 / MATH L3-4 warmup adapter
  -> merge 到模型权重
  + fresh r512 LoRA adapter
  -> Skywork PRM gated reward GRPO
```

必须注意：GRPO 训练和评测都不是简单的 `base + r512 adapter`。正确链路是：

```text
base + warmup SFT adapter merge + r512 GRPO adapter
```

如果评测时只挂 `r512 GRPO adapter`，会漏掉 warmup SFT 能力，结果无效。

消融实验还必须注意另一点：每个实验都要重新训练一个自己的 fresh r512 adapter。不要在已经训好的 `baseline_r512_gated_1000/final` 上继续训练。

正确做法：

```text
每个实验都从同一个 warmup SFT adapter 出发
  -> merge warmup
  -> 新建 fresh r512 LoRA
  -> 只改该实验变量重新 GRPO
```

错误做法：

```text
baseline_r512_gated_1000/final
  -> 继续训练 Exp2/Exp3/...
```

后者会把 baseline 已经学到的更新混进消融变量里，实验不可解释。

## 2. 当前 baseline

baseline 已完成训练：

```text
exp_name = baseline_r512_gated_1000
output = /root/autodl-tmp/psrl_outputs/grpo_math_l34_r512_gated_len768_1000_s42_20260604_202818
final_adapter = /root/autodl-tmp/psrl_outputs/grpo_math_l34_r512_gated_len768_1000_s42_20260604_202818/final
```

quick check 结果：

```text
MATH500 first 40 examples
SFT warmup: 16/40 = 40%
GRPO r512 gated: 22/40 = 55%
delta = +15 points
```

这说明新 GRPO baseline 有正向信号。但这只是 quick check，不是最终正式结论。正式对比需要跑 `MATH500 subset100`，建议 `max_gen_toks=768`。

## 3. 固定资产路径

远端默认路径：

| 资产 | 路径 |
| --- | --- |
| 代码 | `/root/autodl-tmp/process_supervised_rl` |
| base model | `/root/autodl-tmp/models/deepseek-math-7b-instruct` |
| warmup SFT adapter | `/root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final` |
| Skywork PRM | `/root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` |
| MATH L3/4 train | `data/processed/math_l34_train_3000_seed42.jsonl` |
| 输出根目录 | `/root/autodl-tmp/psrl_outputs` |

## 4. Baseline 固定配置

所有消融实验默认继承以下配置。每组只改自己负责的变量。

每组都必须重新创建 fresh r512 LoRA。`baseline_r512_gated_1000/final` 只作为对照组评测对象，不作为任何消融实验的初始化 adapter。

```yaml
base_model: /root/autodl-tmp/models/deepseek-math-7b-instruct
sft_adapter: /root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final
merge_sft_adapter: true

new_lora_r: 512
new_lora_alpha: 1024
new_lora_dropout: 0.05
new_lora_target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

train_data: data/processed/math_l34_train_3000_seed42.jsonl
limit: 3000

num_generations: 4
generation_batch_size: 4
max_prompt_length: 512
max_completion_length: 768
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-6
beta: 0.04
max_steps: 1000
seed: 42
gradient_checkpointing: true
bf16: true

loss_type: grpo
epsilon_low: 0.2
epsilon_high: 0.2

reward_mode: gated_prm
prm_backend: skywork
prm_dir: /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
final_weight: 1.0
prm_weight: 0.2
wrong_final_reward: 0.0

dynamic_sampling: false
length_penalty_weight: 0.0
length_penalty_start: 0
length_penalty_max: 0

save_steps: 500
save_total_limit: 1
save_only_model: true
delete_saved_ref_adapter: true
```

## 5. 五组消融实验

每组只改表中变量，其余全部保持 baseline 一致。

所有消融都从同一个起点开始：

```text
base_model + warmup SFT adapter merge + fresh r512 LoRA
```

不要从 baseline 的 r512 adapter 继续训练。

| 实验 | 变量 | 改动 | 备注 |
| --- | --- | --- | --- |
| Exp1 | rollout 数 | `num_generations: 8`, `generation_batch_size: 8`, `gradient_accumulation_steps: 4` | 保持每次更新约 32 条 rollout，做更纯的 G 消融 |
| Exp2 | Clip-Higher | `epsilon_low: 0.2`, `epsilon_high: 0.28` | 最便宜，优先跑 |
| Exp3 | DAPO loss | `loss_type: dapo` | 最便宜，优先跑 |
| Exp4 | Dynamic Sampling | `--dynamic-sampling` | 只保留 `0 < correct_count < G` 的 group |
| Exp5 | Soft length penalty | `length_penalty_weight: 0.2`, `length_penalty_start: 512`, `length_penalty_max: 768` | 注意不是超过 768 扣分 |

Exp5 解释：当前训练 `max_completion_length=768`，所以如果把惩罚起点也设为 `768`，惩罚永远不会生效。这里采用 `512 -> 768` 线性扣分，超过 512 开始逐渐扣，到 768 达到最大惩罚。

## 6. 推荐分工

三个人分工时，建议按成本平衡：

```text
同学 A：Exp2 + Exp3
同学 B：Exp4 + Exp5
同学 C：Exp1 + baseline/subset100 正式评测补锚点
```

如果只分五组消融，Exp1 最慢，可以单独给显存和时间最宽裕的人。

## 7. 训练命令模板

正式训练前先 smoke：

```bash
cd /root/autodl-tmp/process_supervised_rl

EXP=exp2_clip_higher_smoke
OUT=/root/autodl-tmp/psrl_outputs/$EXP
mkdir -p "$OUT"

python scripts/train_grpo_smoke.py \
  --train-jsonl data/processed/math_l34_train_3000_seed42.jsonl \
  --model-name /root/autodl-tmp/models/deepseek-math-7b-instruct \
  --sft-adapter /root/autodl-tmp/psrl_outputs/sft_v3_math_l34_3000_e0p4_lr1e5_len2048/final \
  --prm-dir /root/autodl-tmp/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
  --prm-backend skywork \
  --output-dir "$OUT" \
  --reward-debug-jsonl "$OUT/reward_debug.jsonl" \
  --limit 8 \
  --num-generations 4 \
  --generation-batch-size 4 \
  --reward-mode gated_prm \
  --wrong-final-reward 0.0 \
  --final-weight 1.0 \
  --prm-weight 0.2 \
  --max-prompt-length 512 \
  --max-completion-length 768 \
  --learning-rate 1e-6 \
  --beta 0.04 \
  --loss-type grpo \
  --epsilon-low 0.2 \
  --epsilon-high 0.28 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --max-steps 1 \
  --save-steps 1 \
  --save-total-limit 1 \
  --save-only-model \
  --delete-saved-ref-adapter \
  --merge-sft-adapter \
  --new-lora-r 512 \
  --new-lora-alpha 1024 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed 42 \
  --device cuda \
  2>&1 | tee "$OUT/train.log"
```

正式训练只改：

```text
--limit 3000
--max-steps 1000
--save-steps 500
--output-dir /root/autodl-tmp/psrl_outputs/<exp_name>
```

并替换对应实验参数。

## 8. 各实验参数差异

### Exp1: G=8

```bash
--num-generations 8 \
--generation-batch-size 8 \
--gradient-accumulation-steps 4
```

baseline 是 `G=4` 且 `gradient_accumulation_steps=8`，每次 optimizer update 约看到 `4 x 8 = 32` 条 rollout。Exp1 如果保持 accumulation 为 8，会变成 `8 x 8 = 64` 条 rollout，相当于同时改了 G 和 effective batch。

因此 Exp1 固定采用 `gradient_accumulation_steps=4`，保持 `8 x 4 = 32`，更接近“只验证每题多采样是否有用”的纯消融。

如果显存不够，可以先把 `generation_batch_size` 降到 `4`，但 `num_generations` 仍保持 `8`，`gradient_accumulation_steps` 仍保持 `4`。

### Exp2: Clip-Higher

```bash
--epsilon-low 0.2 \
--epsilon-high 0.28
```

### Exp3: DAPO loss

```bash
--loss-type dapo
```

注意：`loss_type=dapo` 依赖当前 TRL/项目适配层支持。smoke 时必须检查 `run_manifest.json` 里确实记录：

```json
"loss_type": "dapo"
```

并确认训练没有 fallback 到默认 `grpo`。

### Exp4: Dynamic Sampling

```bash
--dynamic-sampling
```

检查日志里是否出现：

```text
dynamic_sampling/keep_rate
dynamic_sampling/all_wrong_rate
dynamic_sampling/all_correct_rate
```

如果 `dynamic_sampling/keep_rate < 0.3` 持续约 100 step，说明有效 group 太少。不要直接中断正式实验，但需要在结果记录里标记：

```text
early stop candidate / low keep_rate
```

后续分析时可以考虑补跑 `max_steps=1500` 版本，让有效训练量和其他组更可比。

### Exp5: Soft length penalty

```bash
--length-penalty-weight 0.2 \
--length-penalty-start 512 \
--length-penalty-max 768
```

这个设置的含义：

```text
length <= 512: penalty = 0
length = 768: penalty = -1.0
weighted penalty = 0.2 * penalty
```

也就是最长处最多扣 `0.2`，大约是答对 reward 的 15%-20%。力度不算极端，主要用于减少无效长输出，不应该压到模型不敢写必要推理。

## 9. Smoke 必查项

每组正式跑前，smoke 必须确认：

1. 加载 `deepseek-math-7b-instruct`。
2. 加载 warmup SFT adapter。
3. 打印 merge/fresh LoRA 后的 trainable stats，r512 应约 `1.199B` trainable params。
4. 加载 Skywork PRM，并看到 `has v_head: True` 或 manual v_head backend。
5. 打印 `step_rewards=[...]`。
6. `reward_debug.jsonl` 存在，且 wrong final 不应拿到正 PRM 总 reward。
7. `adapter_diagnostics.json` 存在，训练后 LoRA 参数发生变化。
8. 保存目录里不要保留大号 `ref/` 目录。

正式训练跑到 500 step checkpoint 时，再做一次中期检查：

1. 看 `train.log` 中 reward mean / reward std / KL / completion length 是否异常。
2. 如果 reward mean 明显低于 baseline 同期超过约 20%，在实验记录里标记 `early stop candidate`。
3. 除非 OOM、磁盘爆掉或代码报错，不要因为 500 step 看起来差就提前停。仍跑满 1000 step，保证消融数据完整。

## 10. 评测规范

统一使用 MATH500。

```yaml
dataset: HuggingFaceH4/MATH-500
task: minerva_math500
subset: first 100 examples
max_gen_toks: 768
decode: greedy
verifier: math_verify
```

评测链路必须是：

```text
base + warmup SFT merge + experiment adapter
```

不要直接用：

```text
base + experiment adapter
```

后者会漏掉 warmup SFT，结果无效。

## 11. 输出目录规范

统一输出到：

```text
/root/autodl-tmp/psrl_outputs/<exp_name>
```

命名：

```text
baseline_r512_gated_1000
exp1_g8
exp2_clip_higher
exp3_dapo_loss
exp4_dynamic_sampling
exp5_length_penalty
```

每组至少保留：

```text
train.log
run_manifest.json
adapter_diagnostics.json
reward_debug.jsonl
final/adapter_config.json
final/adapter_model.safetensors
eval_results.json
```

## 12. 结果记录模板

```markdown
### Exp X: <name>

训练：
- max_steps:
- train_runtime:
- train_loss:
- reward mean 趋势:
- reward std 趋势:
- KL final:
- completion mean length:
- clipped_ratio:
- GPU peak memory:

评测 MATH500 subset100：
- accuracy: __/100 = __%
- format_rate:
- avg_gen_tokens:

对比：
- baseline subset100: __%
- this exp subset100: __%
- delta: +/- __ points

异常：
- 是否 OOM:
- 是否保存 ref 目录:
- 是否有截断偏高:
- 其他备注:
```

## 13. 当前优先级

建议执行顺序：

```text
0. 先补 baseline_r512_gated_1000 的 MATH500 subset100 正式评测
1. Exp2 clip_higher
2. Exp3 dapo_loss
3. Exp4 dynamic_sampling
4. Exp5 length_penalty
5. Exp1 g8
```

Exp2/Exp3 成本最低，最适合作为组员先跑通流程的任务。Exp1 最贵，放最后。

## 14. 历史归档

历史文档：

- [step1 bootstrap](docs/history/README_process_supervised_rl_step1.md)
- [step2 process reward baseline](docs/history/README_process_supervised_rl_step2.md)
- [step3 PRM/SFT/GRPO diagnostics](docs/history/README_process_supervised_rl_step3.md)
- [step4 MATH warmup 与 r512 gated PRM](docs/history/README_process_supervised_rl_step4.md)

step4 记录了这段时间的主要弯路和最终有效配置。组员执行实验时优先看当前 README，不要从历史长文开始。
