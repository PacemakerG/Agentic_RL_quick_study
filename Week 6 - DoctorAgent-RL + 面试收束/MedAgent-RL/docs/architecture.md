# Architecture

## 训练闭环

MedAgent-RL 将每个病例视为一个多轮环境。初始状态由患者首轮自述和病例元数据组成；Consultation Policy 生成一个动作；Patient Simulator 返回观察；Consultation Evaluator 计算奖励并判断对话是否结束。

```text
state_t
  ├─ patient self-report
  ├─ dialogue history
  └─ remaining turns
          │
          ▼
Consultation Policy ── action_t ──► Medical Consultation Env
                                          │
                           Patient Simulator / Evaluator
                                          │
                                          ▼
                               observation_t+1, reward_t
```

策略动作只有两类：

1. `Question`：提出一个有针对性的追问，进入下一轮。
2. `Diagnosis + Recommendation`：结束对话并输出诊断和建议。

## GRPO 与 State Masking

同一病例并行采样多条轨迹，GRPO 在组内比较回报并计算优势。轨迹同时包含策略输出和患者模拟器返回的状态文本；State Masking 仅保留 Consultation Policy 动作对应的 Token Loss，避免模型拟合环境观察。

## 分布式执行

- Ray 管理 Actor、Rollout 与环境模型 Worker。
- vLLM 批量生成策略动作和患者回答。
- FSDP 分片策略模型参数，并可将参数卸载到 CPU。
- 单机 8×24GB 配置将策略和患者模型的 Tensor Parallel 均设为 4，以适配 Qwen2.5-7B 的 Attention Head 数量。

## 模块映射

| 模块 | 路径 |
| --- | --- |
| 医疗环境 | `ragen/env/medical_consultation/` |
| GRPO 入口 | `ragen/trainer/main_ppo.py` |
| 多轮 Rollout 与 State Masking | `ragen/trainer/ppo/ray_trainer.py` |
| Actor / 环境 Worker | `ragen/workers/` |
| SFT Trainer | `ragen/trainer/fsdp_sft_trainer.py` |
| 数据构建 | `scripts/data/` |
| 离线评测 | `ragen/env/medical_consultation/evaluation/` |
