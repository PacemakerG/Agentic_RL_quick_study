# 作业 2：策略梯度（Policy Gradients）

## 环境安装与配置

有关通用的环境安装与 Modal 云端使用说明，请参阅 [作业 1 的 README](../hw1/README.md)。

## 运行示例

以下是常用的运行命令示例。请在 `hw2` 目录下运行：

* **在本地计算机上运行（推荐）：**
  ```bash
  uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```

* **在 Modal 云端运行：**
  ```bash
  uv run modal run src/scripts/modal_run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
  ```
  * **注意**：本次作业大概率**不需要**使用 Modal。在测试中，在本地笔记本电脑 CPU 上训练比在 Modal 上快得多。当然，如果你愿意，依然可以使用 Modal。
  * 你可以通过修改 `src/scripts/modal_run.py` 中的变量来请求不同的 GPU 类型、CPU 核心数与内存大小。
  * 使用 `modal run --detach` 可以让训练任务在后台持续运行。

## 故障排除（Troubleshooting）

* 如果在安装 `box2d-py` 时遇到关于 `swig` 的报错，你需要在你的机器上安装 `swig` 和 `cmake`。
  * **macOS（使用 Homebrew）**：运行 `brew install swig cmake`
  * **Linux (Ubuntu/Debian)**：运行 `sudo apt-get install swig cmake`
  * 在 Modal 云端环境中，这些依赖已经默认预装好。

## 作业核心文件结构

```
src/
├── agents/
│   └── pg_agent.py          # 策略梯度核心智能体（需实现：回报计算、GAE优势估计、Actor/Critic更新）
├── networks/
│   ├── policies.py          # 策略网络 MLPPolicyPG（需实现：离散/连续动作分布、策略损失计算）
│   └── critics.py           # 价值网络 ValueCritic（需实现：前向传播、MSE 损失更新）
├── infrastructure/
│   ├── pytorch_util.py      # PyTorch 张量与设备辅助函数
│   └── utils.py             # 采样轨迹、日志记录工具
└── scripts/
    └── run.py               # 本地训练启动入口脚本
```
