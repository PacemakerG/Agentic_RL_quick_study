# Week 6：DoctorAgent-RL + 面试收束

[学习计划](../%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92_%E6%A0%B8%E5%AF%B9%E4%BF%AE%E8%AE%A2%E7%89%88_%E5%85%88SFT%E5%90%8ERL.md) · [官方仓库](https://github.com/PacemakerG/MedAgent-RL)

以本目录的 MedAgent-RL 项目源码、训练说明与评测入口为主线，对照前五周课程完成项目复盘。

| Day | 官方课程 / 作业内容 | 本地资料 | 本地代码 / 测试 | 当天实践 |
| --- | --- | --- | --- | --- |
| Day 1 | 项目架构与 rollout/update 时序 | [项目架构](MedAgent-RL/docs/architecture.md) | [项目源码](MedAgent-RL/ragen) | 沿实际入口追踪样本，标注各模型角色。 |
| Day 2 | Reward Ablation | [训练说明](MedAgent-RL/docs/training.md) | [训练与评测脚本](MedAgent-RL/scripts) | 固定 baseline，只改变一个 reward component，比较独立指标与行为。 |
| Day 3 | GRPO/rollout 参数消融与失败分析 | [训练配置说明](MedAgent-RL/docs/training.md) | [环境与评测源码](MedAgent-RL/ragen/env/medical_consultation) | 选参数做对照，检查至少 10 条失败轨迹，整理 Top-3 failure modes。 |
| Day 4 | Transformer / SFT / RL / Agentic RL 复盘 | [完整课程计划](../%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92_%E6%A0%B8%E5%AF%B9%E4%BF%AE%E8%AE%A2%E7%89%88_%E5%85%88SFT%E5%90%8ERL.md) | [项目入口](MedAgent-RL/README.md) | 结合真实代码与日志，完成 30–45 分钟模拟面试和 30 道自测。 |
