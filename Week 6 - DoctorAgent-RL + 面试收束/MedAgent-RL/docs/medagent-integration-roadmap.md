# MedAgent Integration Roadmap

## 当前边界

当前仓库已经完成 Consultation Policy 的数据、SFT、GRPO 和离线评测，但**尚未包含 MedAgent 接入代码**。现阶段训练产物是一个独立的多轮问询策略模型。

## 目标接口

MedAgent 侧需要向策略模型提供：

- 患者首轮输入与结构化上下文；
- 已发生的问答历史；
- 剩余问询轮数与会话终止状态。

策略模型返回统一的二选一结果：

- `Question`：交还 MedAgent 获取患者回答，并追加到会话状态；
- `Diagnosis + Recommendation`：结束问询，将收集到的信息和策略结论交给 MedAgent 后续节点。

## 实施阶段

1. 定义 MedAgent 会话状态到 Consultation Policy Prompt 的适配层。
2. 增加格式校验、超轮终止、重复提问拦截和失败回退。
3. 接入训练后 Checkpoint，完成端到端多轮问询测试。
4. 对离线评测、人工安全评测和线上观测指标建立回归基线。
5. 接入完成并验证后，再更新 README 状态和对外简历表述。
