# Week 5：Agentic RL

[学习计划](../%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92_%E6%A0%B8%E5%AF%B9%E4%BF%AE%E8%AE%A2%E7%89%88_%E5%85%88SFT%E5%90%8ERL.md) · [官方仓库](https://github.com/verl-project/verl)

按 verl 官方 Agentic RL、Agent Loop 和 Multi-turn Rollout 文档，使用本目录已有源码与官方示例。

| Day | 官方课程 / 作业内容 | 本地资料 | 本地代码 / 测试 | 当天实践 |
| --- | --- | --- | --- | --- |
| Day 1 | Agentic RL 架构与异步 rollout | [Agentic RL 文档](verl/docs/start/agentic_rl.rst) | [Agent Loop 源码](verl/verl/experimental/agent_loop) | 核对当前版本配置，区分异步请求与训练/采样并发。 |
| Day 2 | Multi-turn Rollout；Tool Calling | [Multi-turn 文档](verl/docs/sglang_multiturn/multiturn.rst) | [Agent Loop 源码](verl/verl/experimental/agent_loop) | 对照多轮 messages、tool observation 和 train mask。 |
| Day 3 | GSM8K Tool Agent 官方示例 | [预处理脚本](GSM8K-Tool-Agent-v0.7.1/examples/data_preprocess/gsm8k_tool_agent_loop.py) | [训练脚本](GSM8K-Tool-Agent-v0.7.1/examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh) | 完成预处理、rollout、trace 和至少一次参数更新。 |
| Day 4 | Tool / Reward 扩展与失败分析 | [rollout 配置](verl/verl/trainer/config/rollout/rollout.yaml) | [工具实现](verl/verl/tools) | 定位工具注册、reward、mask，分析至少两条失败轨迹。 |


Day 3 使用 [官方 v0.7.1 示例与运行说明](GSM8K-Tool-Agent-v0.7.1/README.md)，固定版本运行；Day 1、2、4 的源码阅读使用已有 verl 快照。
