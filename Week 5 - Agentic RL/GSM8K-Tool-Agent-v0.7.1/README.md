# GSM8K Tool Agent：官方 v0.7.1 示例

用于学习计划 Week 5 Day 3。这里保存官方预处理、训练脚本及两份配置的原样副本；完整训练框架使用同版本官方仓库。

- 来源：[verl v0.7.1](https://github.com/verl-project/verl/tree/bec9ef74768dd201881cd4e54cd0385e87caae27)。
- 固定 commit：`bec9ef74768dd201881cd4e54cd0385e87caae27`。
- [数据预处理](examples/data_preprocess/gsm8k_tool_agent_loop.py)、[训练脚本](examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh)、[训练配置](examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml)、[工具配置](examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml)。

在 Linux/GPU 训练环境中，使用独立工作目录获取匹配版本：

```bash
git clone --branch v0.7.1 --depth 1 https://github.com/verl-project/verl.git verl-tool-agent-v0.7.1
cd verl-tool-agent-v0.7.1
git rev-parse HEAD
```

确认 commit 与上方一致，按该版本的 [安装文档](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/docs/start/install.rst) 和 [rollout trace 文档](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/docs/advance/rollout_trace.rst) 安装 SGLang、配置 MLflow 与模型，再在完整仓库根目录运行：

```bash
python examples/data_preprocess/gsm8k_tool_agent_loop.py
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh
```

官方脚本以 8×H100 为参考，默认 2 个 training steps。按实际设备调整时记录差异。此处仅核对文件来源与路径，未运行 GPU 训练。旁边的 `verl` 是另一版本的课程源码快照，不将这个旧版示例直接混入其配置环境。
