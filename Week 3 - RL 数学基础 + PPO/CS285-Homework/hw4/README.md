# CS285 作业 4：大语言模型强化学习（LLM RL）

本仓库默认通过 `scripts/modal_train.py` 配置为在 Modal 云端运行。默认分配的 Modal GPU 为 **NVIDIA H100**。

## 快速上手

```bash
uv sync
uv run modal token new
uvx wandb login
```

以下所有训练命令均假定在仓库根目录下执行。

## 必跑实验（Required Runs）

### 1. Format Copy（格式复刻任务）+ GRPO

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task format_copy \
  --algo grpo \
  --output_dir /vol/runs/modal_format_copy_grpo \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --clip_eps 0.2 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name format_copy_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### 2. Format Copy + REINFORCE

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task format_copy \
  --algo reinforce \
  --output_dir /vol/runs/modal_format_copy_reinforce \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name format_copy_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
```

### 3. Math Hard（困难数学推理）+ REINFORCE

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task math_hard \
  --algo reinforce \
  --output_dir /vol/runs/modal_math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```

### 4. Math Hard + GRPO

```bash
uv run modal run --detach scripts/modal_train.py -- \
  --task math_hard \
  --algo grpo \
  --output_dir /vol/runs/modal_math_hard_grpo \
  --steps 501 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --clip_eps 0.2 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
```

## 构建 Gradescope 提交包

为每个想要包含的已完成实验传递一次 `--run_dir`。在四个实验全部跑完之前，也可以构建并提交部分结果；未完成的实验在 Gradescope 上只会显示为缺失/0分，稍后补充即可。

```bash
uv run modal run scripts/modal_train.py::bundle_submission_remote -- \
  --run_dir /vol/runs/modal_format_copy_grpo \
  --run_dir /vol/runs/modal_format_copy_reinforce \
  --run_dir /vol/runs/modal_math_hard_grpo \
  --run_dir /vol/runs/modal_math_hard_reinforce \
  --output_dir /vol/submissions/hw4_gradescope_submission \
  --overwrite
```

从 Modal 下载打包结果：

```bash
uv run modal volume get hw4-llm-rl-volume /submissions/hw4_gradescope_submission.zip .
```

## 打包并提交到 Gradescope

1. 解压 `hw4_gradescope_submission.zip`。
2. 将你的作业代码目录 `hw4/` 和解压后的 `hw4_gradescope_submission/` 文件夹并排放置在同一个父目录下。
3. 直接将这两个文件夹一起打包为 zip，外层不要嵌套其他包装目录。

顶层目录结构应如下所示：

```text
hw4_submission.zip
  hw4/
    ...
  hw4_gradescope_submission/
    ...
```

如果你的当前工作目录是包含这两个文件夹的父目录，可以执行便捷打包命令：

```bash
zip -r hw4_submission.zip hw4 hw4_gradescope_submission \
  -x "hw4/.venv/*" "hw4/.git/*" "hw4/**/__pycache__/*" "*.pyc" ".DS_Store"
```
