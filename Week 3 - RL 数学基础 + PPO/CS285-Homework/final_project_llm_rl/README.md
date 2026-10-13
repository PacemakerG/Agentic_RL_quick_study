# CS 185/285 LLM RL 默认期末大作业指南

本仓库包含 LLM RL（大语言模型强化学习）默认期末大作业代码库。

本项目研究针对开放式指令遵循的 RLHF（人类反馈强化学习），实验配置如下：
- **基座模型**：`Qwen/Qwen2.5-1.5B-Instruct`
- **基准数据集**：精选自过滤后的 WildChat 提示词与大模型打分评级的 `5k` 条样本数据集
- **主要评估指标**：相对冻结基座模型的 GPT-5.4 双盲对抗胜率（Head-to-head Win Rate）

核心 Python 包为：
- `llm_rl_final_proj`

## 仓库布局

- `llm_rl_final_proj/`：训练、评估与数据加载核心代码
- `scripts/modal_train.py`：用于训练与评估的 Modal 云端入口脚本
- `dataset/`：用于训练的 RLHF 本地数据集
- `public_eval/`：用于本地评估和 Gradescope 评测的公开评估资源

## 环境配置

初始设置：

```bash
uv run modal token new
uvx wandb login
```

若要在本地对仓库评估集运行大模型对抗胜率评测，还需导出你自己的 OpenAI API 密钥：

```bash
export OPENAI_API_KEY=...
```

## 基准数据集

RLHF 数据集保存在本地：
- `dataset/wildchat_min4_judged_5k_v1`

划分规模：
- `train_prefs = 4744`（训练偏好对）
- `test_prefs = 256`（测试偏好对）
- `train_gen = 4744`（训练生成 Prompt）
- `test_gen = 256`（测试生成 Prompt）

说明：
- `train_prefs`：包含 chosen/rejected 偏好对，用于离线偏好优化（如 DPO/IPO）和奖励模型（RM）训练
- `test_prefs`：留出偏好对，用于奖励模型评估和离线诊断
- `train_gen`：纯 Prompt 数据切分，用于在线 RLHF 的 Rollout 采样
- `test_gen`：留出 Prompt 切分，用于策略模型对比基座模型的胜率评估

`train_prefs` 和 `train_gen` 使用相同的 Prompt 分布，区别在于前者包含配对回答和偏好标签，后者是在线交互 Rollout 的 Prompt 源。

## 基准测试规约（Benchmark Rules）

无论是运行 Part 1 必做算法还是进行 Part 2 深入探索，都必须固定基准设置：

- 严禁在 `dataset/wildchat_min4_judged_5k_v1` 之外引入额外训练 Prompt 或偏好数据
- 严禁更换基座模型，必须固定使用 `Qwen/Qwen2.5-1.5B-Instruct`
- 固定 `max_prompt_tokens = 700`
- 固定策略生成预算：在线 Rollout 和所有提交的策略 JSONL 必须使用 `max_new_tokens = 256`
- 固定偏好/奖励打分的响应 Token 预算：`max_response_tokens = 512`（适用于离线训练、奖励模型训练及在线 RLHF 打分）
- 生成提交 JSONL 时，必须使用确定性解码：`temperature = 0.0`，`top_p = 1.0`

Part 2 的核心是在此固定基准下改进算法本身。改动评估 Token 预算、在额外数据上训练或堆叠大量检查点均不算作有效的算法扩展。

## 上传基准资源到 Modal

训练前，请创建 Modal 存储卷并上传数据集（若已创建可跳过创建命令）：

```bash
uv run modal volume create llm-rl-final-project-volume
```

### 上传训练数据集
```bash
uv run modal volume put llm-rl-final-project-volume dataset/wildchat_min4_judged_5k_v1 /synthetic_datasets/
```

数据集将在 Modal 容器内挂载至：
- `/vol/synthetic_datasets/wildchat_min4_judged_5k_v1`

验证上传：
```bash
uv run modal volume ls llm-rl-final-project-volume /synthetic_datasets
```

### 评估文件说明
评估文件已存在于本仓库的 `public_eval/` 目录下。从本仓库启动 Modal 入口时，代码库会自动挂载到容器内的 `/root/project/`，因此无需额外上传 `public_eval/`。

## 项目 Part 1 必做算法清单

**离线算法（Offline）**：
- `dpo`（直接偏好优化）
- `ipo`（恒等偏好优化）
- `aot`

**在线算法（Online）**：
- Bradley-Terry 奖励模型（Reward Model）
- `grpo`（分组相对策略优化）
- `drgrpo`
- `gspo`

## 典型训练命令

### 1. 奖励模型（Reward Model）训练
```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_v1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_interval 25 \
  --save_interval 50 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_reward_model_v1
```

### 2. 离线偏好优化训练（以 DPO 为例）
```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1 \
  --beta 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 100 \
  --eval_interval 100 \
  --save_interval 100 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_dpo_beta01_v1
```

针对必选的离线 Part 1 提交，推荐使用以下参考检查点：
- DPO: `step_000100`
- IPO: `step_000300`
- AOT: `step_000550`

### 3. 在线 RLHF 训练（以 GRPO 为例）
```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_rm445_v1 \
  --steps 25 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_limit 32 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_grpo_rm445_v1
```

对于 `dr_grpo` 和 `gspo`，使用相同命令，仅修改 `--algo`、`--output_dir` 与 `--wandb_name`。Part 1 在线提交推荐使用检查点 `step_000025`。

## 构建 Gradescope 提交文件

共有三类提交要求：
- Part 1 策略方法：在 `128` 条评测集上生成回答
- Part 1 奖励模型：为 `256` 对留出偏好打分
- Part 2 最佳方法：在相同的 `128` 条评测集上生成回答

推荐 Part 1 检查点路径映射：

| 文件名称 | 适配器检查点路径 |
| --- | --- |
| `policy_generations/dpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_dpo_beta01_v1/checkpoints/step_000100/adapter` |
| `policy_generations/ipo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_ipo_v1/checkpoints/step_000300/adapter` |
| `policy_generations/aot.jsonl` | `/vol/runs/wildchat_min4_judged_5k_aot_beta02_v1/checkpoints/step_000550/adapter` |
| `policy_generations/grpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_grpo_rm445_v1/checkpoints/step_000025/adapter` |
| `policy_generations/drgrpo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_drgrpo_rm445_v1/checkpoints/step_000025/adapter` |
| `policy_generations/gspo.jsonl` | `/vol/runs/wildchat_min4_judged_5k_gspo_rm445_v1/checkpoints/step_000025/adapter` |
| `reward_model/public_test_pref_scores.jsonl` | `/vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000445/adapter` |

### 生成策略提交文件示例（128 Prompt）：
```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/<run>/checkpoints/<step>/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/dpo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

### 生成奖励模型打分提交文件示例：
```bash
uv run modal run scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/<reward_model_run>/checkpoints/<step>/adapter \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl /vol/submissions/public_test_pref_scores.jsonl \
  --max_prompt_tokens 700 \
  --max_response_tokens 512
```

## 从 Modal 下载提交产物并打包

本地创建目录：
```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations
mkdir -p llm_rl_final_proj_public_submission/reward_model
mkdir -p llm_rl_final_proj_public_submission/part2
```

下载所有产物文件：
```bash
uv run modal volume get llm-rl-final-project-volume /submissions/dpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/ipo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/aot.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/grpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/drgrpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/gspo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/public_test_pref_scores.jsonl llm_rl_final_proj_public_submission/reward_model/
uv run modal volume get llm-rl-final-project-volume /submissions/offline_best.jsonl llm_rl_final_proj_public_submission/part2/
uv run modal volume get llm-rl-final-project-volume /submissions/online_best.jsonl llm_rl_final_proj_public_submission/part2/
```

压缩为提交 Zip 包：
```bash
zip -r llm_rl_final_proj_public_submission.zip llm_rl_final_proj_public_submission -x "*.DS_Store"
```

## 本地自动评分器（Local Autograder）

在下载好的 `llm_rl_final_proj_public_submission/` 上执行本地评测：
```bash
uv run python student_autograder/run_local_autograder.py \
  --submission_dir llm_rl_final_proj_public_submission \
  --output_json student_autograder_results.json
```

## 满分胜率阈值标准
- 奖励模型对配对准确率：`0.74`
- Part 1 算法（DPO / IPO / AOT / GRPO / DrGRPO / GSPO）：`0.65 / 0.60 / 0.65 / 0.60 / 0.60 / 0.60`
- Part 2 最佳离线方法胜率：`0.82`
- Part 2 最佳在线方法胜率：`0.70`（Part 2 离线或在线任一达标即可得满分）
