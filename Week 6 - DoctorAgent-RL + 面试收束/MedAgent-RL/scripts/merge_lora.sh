#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "用法: bash $0 <基座模型目录> <LoRA Checkpoint目录> <合并后策略模型目录>"
    exit 1
fi

base_model=$1
ckpt_path=$2
target_dir=$3

python verl/scripts/model_merger.py \
    --backend fsdp \
    --tie-word-embedding \
    --hf_model_path "$base_model" \
    --local_dir "$ckpt_path" \
    --target_dir "$target_dir"

for tokenizer_file in vocab.json tokenizer.json tokenizer_config.json; do
    if [ -f "$base_model/$tokenizer_file" ]; then
        cp "$base_model/$tokenizer_file" "$target_dir/"
    fi
done
