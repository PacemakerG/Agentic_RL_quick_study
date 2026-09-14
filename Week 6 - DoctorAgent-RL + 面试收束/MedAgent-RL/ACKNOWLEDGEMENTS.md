# Acknowledgements

MedAgent-RL 使用并改造了以下开源研究与训练基础设施：

- [RAGEN](https://github.com/ZihanWang314/ragen)：提供 Agent 强化学习训练框架与多轮环境抽象。
- [veRL](https://github.com/volcengine/verl)：以 Git Submodule 形式保留，提供 Ray、vLLM、FSDP 等分布式训练能力。
- [DoctorAgent-RL](https://github.com/JarvisUSTC/DoctorAgent-RL)：提供多智能体医疗问诊研究实现与基础代码参考。

本仓库在上述基础上完成了医疗主线精选、数据处理与固定接口、8×24GB 训练适配、资源配置、评测指标整理、公开命名统一和可复现文档。仓库保留原 Apache-2.0 `LICENSE`；使用或再分发时请同时遵守上游项目及数据集各自的许可条款。

为避免误解：MedAgent 在线接入当前尚未包含在代码中，相关工作仅记录在路线图。
