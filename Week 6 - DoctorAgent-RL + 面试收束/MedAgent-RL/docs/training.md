# Training

## 环境

推荐单机 8 张 24GB NVIDIA GPU、CUDA 12.1 和 Conda：

```bash
git submodule update --init --recursive
bash scripts/setup.sh
conda activate medagent-rl
```

训练前确认四个固定数据文件位于 `data/`。可以先按 README 使用脱敏样例生成同构的小文件，验证安装和数据读取。

## SFT 冷启动

```bash
bash scripts/train_sft_8gpu.sh 8 checkpoints/medagent-rl-sft
```

默认配置：Qwen2.5-7B-Instruct、LoRA Rank 32、LoRA Alpha 16、最大长度 4096、3 Epoch、Gradient Checkpointing。模型路径和其他 Hydra 参数可追加到命令末尾覆盖。

合并 LoRA：

```bash
bash scripts/merge_lora.sh \
  /path/to/Qwen2.5-7B-Instruct \
  checkpoints/medagent-rl-sft/global_step_129 \
  checkpoints/medagent-rl-sft-merged
```

## GRPO Smoke

```bash
bash scripts/smoke_grpo_8gpu.sh \
  checkpoints/medagent-rl-sft-merged \
  /path/to/Qwen2.5-7B-Instruct
```

Smoke 配置读取 8 个病例、每个病例采样 4 条轨迹、最多问询 2 轮，并完成一次参数更新。它用于检查模型加载、Ray Worker、vLLM Rollout、奖励、State Masking 和 FSDP 更新链路，不用于汇报泛化指标。

## 完整 GRPO

```bash
bash scripts/train_grpo_8gpu.sh \
  checkpoints/medagent-rl-sft-merged \
  /path/to/Qwen2.5-7B-Instruct
```

关键配置：

| 配置 | 值 |
| --- | ---: |
| GPU | 8×24GB |
| 每病例 Rollout | 8 |
| 策略 Tensor Parallel | 4 |
| 患者模型 Tensor Parallel | 4 |
| 最大问询轮数 | 10 |
| Advantage Estimator | GRPO |
| State Masking | 开启 |
| Reference Policy / KL | 关闭 |
| 策略参数卸载 | CPU |

`config/base.yaml` 与 `config/env/medical_consultation.yaml` 提供同一套配置化入口：

```bash
python -m ragen.train medical_consultation \
  training.total_training_steps=1 \
  trainer.val_before_train=false
```

## 评测

```bash
bash scripts/evaluate.sh /path/to/consultation-policy
```

评测先生成多轮问诊结果，再输出诊断与建议质量、信息收集、轮数、重复率、格式错误率和完成率。输出目录为 `outputs/evaluation/<model-name>/`。
