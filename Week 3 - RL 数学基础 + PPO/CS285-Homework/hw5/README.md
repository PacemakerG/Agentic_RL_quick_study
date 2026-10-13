# 作业 5：离线强化学习（Offline RL）

## 环境配置

有关通用的环境安装与 Modal 云端使用说明，请参阅 [作业 1 的 README](../hw1/README.md)。

## 运行示例

以下是常用运行命令示例。请在 `hw5` 目录下运行：

* **在本地计算机上运行：**
  ```bash
  uv run src/scripts/run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```

* **在 Modal 云端运行：**
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * 你可以通过修改 `src/scripts/modal_run.py` 中的变量来请求不同的 GPU 类型、CPU 核心数与内存大小。
  * 使用 `modal run --detach` 可以让训练任务在后台持续运行。
  * 如果任务被抢占中断，重新运行相同命令将从 `exp/` 中的最新检查点自动恢复。
  * 如果想使用相同参数强制启动全新运行，请使用不同的随机种子（seed）或显式指定 `exp_name`。

* **在 Modal 上单卡并行运行 4 个任务：**
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=q1 --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```

* **从 Modal 下载日志与检查点：**
  ```bash
  mkdir -p exp
  uv run modal volume get hw5-offline-rl-volume / exp
  ```
