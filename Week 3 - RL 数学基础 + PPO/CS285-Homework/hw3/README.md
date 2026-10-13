# CS285 作业 3：Q-Learning 与 Actor-Critic 算法

本次作业涵盖用于离散动作空间的 **深度 Q 网络（DQN）**，以及用于连续动作空间的 **软行动者-评论家算法（Soft Actor-Critic, SAC）**。

## 前置环境要求

本次作业需要 **Python 3.10** 和系统级依赖包 `swig`（`gym[box2d]` 所必需）。

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install swig
```

**macOS (Homebrew):**
```bash
brew install swig
```

如果你无法在本地安装 `swig` 或遇到其他环境问题，可以使用 Modal 直接在云端 GPU 运行所有实验（参见下文 Modal 章节）。

## 环境安装

使用 `uv` 一键安装依赖并同步环境：

```bash
uv sync
```

## 运行实验

### 1. DQN 实验（离散动作空间）

```bash
# CartPole（调试与冒烟测试）
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml

# LunarLander（Double-DQN）
uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1

# MsPacman（雅达利游戏环境）
uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml
```

### 2. SAC 实验（连续动作空间）

```bash
# InvertedPendulum（完整性检查）
uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml

# HalfCheetah（标准连续控制基准）
uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml

# Hopper（测试不同 Q-backup 备份策略）
uv run src/scripts/run_sac.py -cfg experiments/sac/hopper_clipq.yaml
```

### 3. Modal（云端计算，可选）

首先完成 Modal 身份认证：
```bash
uv run modal token new
```

远程提交实验任务：
```bash
# 在 Modal 上运行 DQN
uv run modal run src/scripts/modal_run_dqn.py -- -cfg experiments/dqn/mspacman.yaml

# 在 Modal 上运行 SAC
uv run modal run src/scripts/modal_run_sac.py -- -cfg experiments/sac/halfcheetah.yaml
```

从持久卷下载实验结果到本地：
```bash
uv run modal volume get hw3-ql-volume exp/
```

## 目录结构

```
src/
├── agents/           # DQN 与 SAC 智能体实现
├── networks/         # 神经网络模块（策略网络 Policies、评价网络 Critics）
├── infrastructure/   # 基础设施（经验回放池 Replay Buffer、日志记录等）
├── configs/          # 配置工厂
└── scripts/          # 训练启动脚本
experiments/          # YAML 配置文件目录
```
