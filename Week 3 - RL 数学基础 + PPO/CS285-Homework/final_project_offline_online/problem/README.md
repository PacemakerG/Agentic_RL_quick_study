# 离线到在线强化学习（Offline to Online RL）期末大作业

## 环境配置

有关通用的环境安装与 Modal 云端使用说明，请参阅 [作业 1 的 README](../../hw1/README.md)。

## 运行示例

以下是常用运行命令示例。请在 `final_project_offline_online` 目录下运行：

* **在本地计算机上运行：**
  ```bash
  uv run src/scripts/run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```

* **在 Modal 云端运行：**
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```
  * 你可以通过修改 `src/scripts/modal_run.py` 中的变量来请求不同的 GPU 类型、CPU 核心数与内存大小。
  * 使用 `modal run --detach` 可以让作业在后台持续运行。

* **在 Modal 上单卡并行运行 4 个任务：**
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000"
  ```
