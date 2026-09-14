#!/usr/bin/env bash

set -euo pipefail
set -x

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "用法: bash $0 <已合并的SFT完整模型目录> [Patient模型目录]"
    exit 1
fi

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "$CONDA_PREFIX/bin/python" ]; then
    echo "错误：请先执行 conda activate medagent-rl。"
    exit 1
fi

policy_model_path=$(cd "$1" && pwd)
patient_model_path=${2:-Qwen2.5-7B-Instruct}
patient_model_path=$(cd "$patient_model_path" && pwd)

if [ -f "$policy_model_path/adapter_config.json" ]; then
    echo "错误：传入的是 LoRA adapter 目录，不是已合并的完整问诊策略模型。"
    exit 1
fi

if [ ! -f "$policy_model_path/config.json" ]; then
    echo "错误：问诊策略模型目录缺少 config.json：$policy_model_path"
    exit 1
fi

python_bin="$CONDA_PREFIX/bin/python"
run_stamp=$(date +%Y%m%d-%H%M%S)
project_name=MedAgent-RL
exp_name=medagent-rl-grpo-8x24gb-${run_stamp}
log_dir="outputs/rl_full_8gpu/${exp_name}"
checkpoint_dir="checkpoints/${project_name}/${exp_name}"

mkdir -p "$log_dir"

# 单节点 8 张 24 GB RTX 4090 D 的保守正式训练配置。
# 每批 8 个病例，每个病例生成 8 条 GRPO 轨迹；策略模型参数卸载到 CPU，Patient Simulator 和 Adam optimizer 留在 GPU。
# env.max_turns=-1 使每条轨迹随机使用 2–10 轮，外层 max_turns=10 提供上限。
# GRPO 不启动 Critic；关闭 KL 后也不启动 Reference Policy。
# Qwen2.5-7B 有 28 个 attention heads，tensor parallel 使用可整除 28 的 4。
# Patient/Reward 只需要生成文本，关闭无用的 prompt/token logprobs 并降低 vLLM KV cache 比例。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

"$python_bin" -m ragen.trainer.main_ppo \
  hydra.run.dir="outputs/exp_configs/logs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)" \
  data.train_files=data/MTMedDialog_RL.parquet \
  data.val_files=data/MTMedDialog_RL.parquet \
  data.train_data_num=null \
  data.val_data_num=32 \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  data.max_prompt_length=6528 \
  data.max_response_length=256 \
  data.max_start_length=512 \
  data.max_obs_length=256 \
  data.shuffle=True \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +algorithm.no_ref_policy=True \
  +algorithm.no_think_rl=False \
  +algorithm.reward_norm_type=grpo \
  actor_rollout_ref.model.path="$policy_model_path" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  +actor_rollout_ref.actor.optim.betas=[0.9,0.95] \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.state_masking=True \
  actor_rollout_ref.rollout.n_agent=8 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  reward_model.enable=False \
  "trainer.logger=['console','wandb']" \
  +trainer.val_only=false \
  trainer.val_before_train=true \
  trainer.resume_mode=disable \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$checkpoint_dir" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$exp_name" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  +trainer.ref_update_steps=null \
  env.name=medical_consultation_patient_llm_rm \
  env.use_env_llm=True \
  +env.max_turns=-1 \
  env.env_llm.fsdp_config.fsdp_size=-1 \
  env.env_llm.fsdp_config.param_offload=False \
  env.env_llm.fsdp_config.model_dtype=bf16 \
  env.env_llm.vllm_config.tensor_parallel_size=4 \
  env.env_llm.vllm_config.gpu_memory_utilization=0.15 \
  env.env_llm.vllm_config.max_num_batched_tokens=4096 \
  env.env_llm.vllm_config.max_num_seqs=64 \
  env.env_llm.model.path="$patient_model_path" \
  env.env_llm.model.trust_remote_code=True \
  env.env_llm.model.use_liger=False \
  env.env_llm.model.override_config.max_position_embeddings=4000 \
  env.env_llm.generation.prompt_length=2272 \
  env.env_llm.generation.response_length=128 \
  env.env_llm.generation.max_model_len=2400 \
  env.env_llm.generation.temperature=0.01 \
  env.env_llm.generation.top_p=1.0 \
  env.env_llm.generation.top_k=-1 \
  env.env_llm.generation.repetition_penalty=1.0 \
  env.env_llm.generation.do_sample=False \
  env.env_llm.generation.num_beams=1 \
  env.env_llm.generation.best_of=1 \
  env.env_llm.generation.min_p=0.0 \
  env.env_llm.generation.n=1 \
  env.env_llm.generation.use_cache=True \
  env.env_llm.generation.use_beam_search=False \
  env.env_llm.generation.detokenize=False \
  env.env_llm.generation.ignore_eos=False \
  env.env_llm.generation.free_cache_engine=True \
  env.env_llm.generation.prompt_logprobs=null \
  env.env_llm.generation.generation_logprobs=0 \
  env.env_llm.generation.disable_log_stats=True \
  env.env_llm.generation.dtype=bfloat16 \
  env.env_llm.generation.enforce_eager=True \
  env.env_llm.generation.enable_chunked_prefill=True \
  env.env_llm.generation.tensor_model_parallel_size=4 \
  env.env_llm.generation.gpu_memory_utilization=0.15 \
  env.env_llm.generation.max_tokens_per_batch=4096 \
  env.env_llm.generation.load_format=dummy_dtensor \
  env.env_llm.ulysses_sequence_parallel_size=1 \
  max_turns=10 \
  logging.log_images=true \
  logging.log_image_dir=log/trajectory \
  logging.log_image_step_size=10 \
  logging.log_n_image_per_batch=8 \
  2>&1 | tee "$log_dir/train.log"

echo "RL training finished."
echo "Log: $log_dir/train.log"
echo "Checkpoint: $checkpoint_dir"
