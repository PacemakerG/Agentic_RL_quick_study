# 作业 1：模仿学习（Imitation Learning）与通用环境指南

## 环境配置

本项目使用 `uv` 进行依赖和虚拟环境管理。`uv` 是由 [Astral](https://astral.sh) 开发的超高速 Python 包管理工具，它通过统一简明的命令行界面替代了 `pip`、`pipx`、`conda` 和 `virtualenv`。

### 安装 `uv`

在终端中执行以下命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，请重新打开终端或重新加载配置，确保 `uv` 已加入系统 `PATH`。

### 统一使用 `uv run` 执行命令

**请勿直接运行 `python` 或 `pip`**。始终通过 `uv run` 来执行脚本，这样依赖和虚拟环境将被自动隔离和加载。
如果你想添加新依赖，可以使用 `uv add <package>`，这会自动将其添加到 `pyproject.toml`，更新 `uv.lock` 并安装到虚拟环境中。

示例：

```bash
uv run src/hw1_imitation/train.py --help
```

在提供的官方骨架代码下，此命令即可直接运行。

## Weights & Biases (WandB) 登录配置

本课程各作业均使用 [Weights & Biases (WandB)](https://wandb.ai) 记录与可视化实验数据（学术用途完全免费）。在启动正式训练脚本前，你需要通过 API 密钥登录 WandB：

```bash
uv run wandb login
```

根据终端提示粘贴你的 WandB API Key 即可。

## 使用 Modal（云端算力，可选）

**注意：本次作业及大部分作业在本地 CPU 笔记本上运行通常比在 Modal 上更快。但后续某些大作业可能需要使用 GPU 算力，配置方法如下：**

首先注册 Modal 账户（新用户通常有免费额度）。然后可以通过以下命令在 Modal 上启动训练：

```bash
uv run modal run src/hw1_imitation/modal_train.py
```

这会自动构建远程容器并在云端启动训练。传参方式与本地脚本完全一致。如果你已经在本地登录了 WandB，你的 API 密钥会自动传递到 Modal 容器中。

训练日志和模型检查点将保存在名为 `hw1-imitation-volume` 的持久化云存储卷（Modal Volume）中。查看日志命令：

```bash
uv run modal volume ls hw1-imitation-volume exp
```

将远程日志和模型下载到本地：

```bash
uv run modal volume get hw1-imitation-volume exp/<experiment_name>
```