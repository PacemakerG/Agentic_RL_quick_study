# MedAgent-RL

MedAgent-RL 面向多轮医疗问询场景，训练一个能够主动追问、收集关键信息并在信息充分后给出诊断与建议的 **Consultation Policy**。项目完成了医疗数据构建、SFT 冷启动、GRPO 强化学习和离线评测链路；训练后的策略接入 MedAgent 属于下一阶段工作。

## 项目目标

通用医疗问答模型容易在患者首轮描述信息不足时直接作答。MedAgent-RL 将问诊建模为多轮决策过程：策略模型每轮只能选择继续追问，或结束问询并输出诊断与建议；患者模拟器根据病例信息回答问题；评价器对有效追问、重复提问、输出格式、问询轮数和最终诊疗结果计算奖励。

## 架构

```text
病例与患者自述
      │
      ▼
Consultation Policy ──提问──► Patient Simulator
      ▲                           │
      └────────患者回答───────────┘
      │
      ├─ 信息充分：输出诊断与建议
      ▼
Consultation Evaluator ──奖励──► GRPO / State Masking / FSDP
```

- **Consultation Policy**：Qwen2.5-7B-Instruct 经 SFT 冷启动和 GRPO 训练得到的问诊策略。
- **Patient Simulator**：基于病例画像回答策略模型的多轮问题，不主动泄露诊断标签。
- **Consultation Evaluator**：组合有效提问、重复与格式惩罚、轮数约束以及诊断/建议质量信号。
- **训练基础设施**：Ray 调度双模型多智能体 Rollout，vLLM 提供批量生成，FSDP 负责参数分片与卸载，State Masking 只对策略动作计算损失。

更完整的数据流与模块边界见 [docs/architecture.md](docs/architecture.md)。

## 个人改造

- 将通用 Agent 强化学习框架收缩为医疗问诊主线，移除 Toy 环境、奖励分支和无关实验入口。
- 建立病例级数据处理与质量检查流程，固定 SFT、GRPO 和测试数据接口，并提供脱敏样例。
- 完成 Consultation Policy 的 LoRA SFT 冷启动、LoRA 合并与多轮 GRPO 训练流程。
- 面向单机 8×24GB GPU 重配 Ray、vLLM、FSDP、Tensor Parallel=4、参数卸载和 KV Cache 占用。
- 整理离线评测入口，输出诊断与建议、问询轮数、重复率、格式错误率、完成率及语义质量指标。

## 数据

完整实验使用以下固定产物：

| 产物 | 规模 | 用途 |
| --- | ---: | --- |
| `MTMedDialog_sft_train.parquet` | 5,516 | SFT 训练 |
| `MTMedDialog_sft_val.parquet` | 549 | SFT 验证 |
| `MTMedDialog_RL.parquet` | 7,068 | GRPO 训练 |
| `MTMedDialog_test.json` | 2,082 | 离线评测 |

完整数据不进入 Git。仓库仅保留脱敏样例、构建脚本和病例级无重叠检查，详见 [DATASET.md](DATASET.md)。

## SFT

SFT 使用 Qwen2.5-7B-Instruct、LoRA Rank 32、最大长度 4096，在单机 8×24GB GPU 上训练：

```bash
bash scripts/train_sft_8gpu.sh 8 checkpoints/medagent-rl-sft
```

选择训练产生的 `global_step_*` 后合并 LoRA：

```bash
bash scripts/merge_lora.sh \
  /path/to/Qwen2.5-7B-Instruct \
  checkpoints/medagent-rl-sft/global_step_129 \
  checkpoints/medagent-rl-sft-merged
```

## GRPO

完整配置采用单机 8×24GB GPU、每个病例 8 条 Rollout、Tensor Parallel=4、Ray、vLLM、FSDP 和 State Masking。先运行单步 Smoke，再启动完整训练：

```bash
bash scripts/smoke_grpo_8gpu.sh \
  checkpoints/medagent-rl-sft-merged \
  /path/to/Qwen2.5-7B-Instruct

bash scripts/train_grpo_8gpu.sh \
  checkpoints/medagent-rl-sft-merged \
  /path/to/Qwen2.5-7B-Instruct
```

配置化入口为：

```bash
python -m ragen.train medical_consultation
```

具体显存配置与可覆盖参数见 [docs/training.md](docs/training.md)。

## 评测

```bash
bash scripts/evaluate.sh /path/to/consultation-policy
```

评测结果写入 `outputs/evaluation/`，包含病例级诊断与建议，以及诊断/建议质量、信息收集、问询轮数、重复率、格式错误率和完成率汇总。

## Quick Start

```bash
git clone git@github.com:PacemakerG/MedAgent-RL.git
cd MedAgent-RL
git submodule update --init --recursive
bash scripts/setup.sh
```

先用仓库样例生成与正式训练一致的文件接口：

```bash
python scripts/data/materialize_artifacts.py \
  --sft-train-json data/samples/MTMedDialog_sft_train_first_10.json \
  --sft-val-json data/samples/MTMedDialog_sft_val_first_10.json \
  --rl-json data/samples/MTMedDialog_RL_first_10.json \
  --test-json data/samples/MTMedDialog_test_first_10.json \
  --output-dir data
```

该命令会校验四个病例集合无重叠，并生成 SFT、GRPO、Test 四个目标文件；生成物已被 `.gitignore` 排除。

## 当前状态

| 模块 | 状态 |
| --- | --- |
| 数据构建与固定接口 | 已完成 |
| SFT 冷启动与模型合并 | 已完成 |
| 多轮 GRPO 与 State Masking | 已完成 |
| 离线评测 | 已完成 |
| MedAgent 在线接入 | Roadmap |

仓库不发布缺少训练日志支撑的提升百分比；当前版本重点保证代码主线、训练配置和复现入口一致。

## Roadmap

- 将训练后的 Consultation Policy 接入 MedAgent 的问询节点。
- 定义 MedAgent 会话状态与策略模型输入/输出协议。
- 增加端到端回归测试、线上安全边界和人工评测。

接入边界见 [docs/medagent-integration-roadmap.md](docs/medagent-integration-roadmap.md)。

## Acknowledgements

本项目基于 RAGEN、veRL 及 DoctorAgent-RL 的基础研究实现继续开发；来源、许可和改造边界见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。
