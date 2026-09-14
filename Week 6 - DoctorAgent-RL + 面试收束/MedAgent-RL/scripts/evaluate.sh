#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "用法: bash $0 <策略模型目录> [更多策略模型目录...]"
    exit 1
fi

for model_path in "$@"; do
    bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh "$model_path"
done
