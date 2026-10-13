# Agentic RL 6 周学习计划 v0.3

> 周期：6 周 × 每周 4 天  
> 单日投入：2–4 小时  
> 目标：Agent / Agentic RL / LLM Post-training 算法实习  
> 作业优先级：官方课程 Homework > 官方框架 Tutorial / Example > 官方论文复现 > 自定义 Lab  
> 说明：CS336 A3 Scaling、A4 Data 与当前 Agentic RL 求职目标相关性较低，因此不纳入主线；CS336 A2 只选与 attention / memory / inference 相关的官方题。自定义任务只保留在官方课程没有覆盖的 Agentic RL 多轮工具训练和个人项目验收部分。

---

## Week 1｜Transformer 基础恢复：CS336 A1 为主，A2 精选

| Day | 学习内容 | 学习资料 | 学习方式 | 官方作业 / 实践 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | Decoder-only Transformer 整体结构；Linear、Embedding、RMSNorm、SwiGLU、Residual、Pre-Norm | Stanford CS336 2026 Lecture 3: Architectures；CS336 Assignment 1: Basics | 先看 Lecture 3 中 Transformer architecture 部分；随后直接读 A1 handout 对应 Model Architecture 题，不额外找二手教程。遇到不懂的公式再和 GPT 交互。 | **CS336 A1 官方题**：完成 Linear、Embedding、RMSNorm、SwiGLU / FFN 相关实现与官方单测。 | 不看资料画出 Decoder-only LM 和单层 Transformer Block；解释 RMSNorm、Pre-Norm、SwiGLU 各自作用；对应测试通过。 |
| Day 2 | Self-Attention、Q/K/V、Causal Mask、Multi-Head Attention、RoPE | CS336 Lecture 3；CS336 Assignment 1 | 继续 Lecture 3 Attention / RoPE 部分；公式自己推一遍后再写作业。 | **CS336 A1 官方题**：完成 RoPE、Scaled Dot-Product Attention、Causal Self-Attention、Multi-Head Self-Attention、Transformer Block 相关实现和测试。 | 能从 `[B,T,D]` 推到 `[B,H,T,D_head]`；讲清 Q/K/V、`1/sqrt(d)`、causal mask、MHA 的真实作用；官方测试通过。 |
| Day 3 | LM Head、Softmax、Cross Entropy、AdamW、Training Loop、自回归语言模型训练 | CS336 Assignment 1 handout / repo | 这一天以作业为主：先读 handout 对应题，再实现；不再看长视频。 | **CS336 A1 官方题**：完成 Cross Entropy、AdamW、Gradient Clipping、Training Loop 等与 LM 训练直接相关的核心题；Tokenizer/BPE 若时间不足可跳过。跑通最小 LM 训练。 | 能解释 `[B,T,D]→[B,T,V]→logprob→CE`；能回答“softmax 分母是什么、为什么整个词表都有梯度”；核心官方测试通过并成功训练。 |
| Day 4 | Prefill / Decode、KV Cache、Attention 内存与吞吐 | CS336 Lecture 10: Inference；CS336 Assignment 2: Systems | 学 Lecture 10 的 inference / KV-cache 核心部分；A2 不完整做，只抽与本路线直接相关的官方 Systems 题。 | **CS336 A2 精选官方题**：做 resource accounting / model profiling / attention benchmarking 中与 FLOPs、memory、attention 性能有关的题；不要求完成 Triton FlashAttention2 和完整 distributed training。 | 能解释 Prefill vs Decode、为什么缓存 K/V 而非 Q、KV Cache 如何随 batch/seq/head 变化；能把这些问题连接到 Agent rollout 成本。 |

### Week 1 官方资料

- CS336 2026 Course: https://cs336.stanford.edu/
- CS336 Lecture 3: https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf
- CS336 Lecture 10: https://cs336.stanford.edu/lectures/?trace=lecture_10
- CS336 Assignment 1: https://github.com/stanford-cs336/assignment1-basics
- CS336 Assignment 1 Handout: https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf
- CS336 Assignment 2: https://github.com/stanford-cs336/assignment2-systems
- CS336 Assignment 2 Handout: https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf

---

## Week 2｜强化学习基础：CS285 Lecture 4–10 + HW2

| Day | 学习内容 | 学习资料 | 学习方式 | 官方作业 / 实践 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | MDP、Trajectory、Return、Value、Q、Advantage | Berkeley CS185/285 2026 Lecture 4: RL Basics | 以官方 slides 为主；公式逐页推。看完后让 GPT 用 LLM / Tool Agent 场景重新口试一遍。 | 暂不做自定义代码作业；完成 Lecture 4 公式与例题，并开始阅读 **HW2 官方说明**。 | 能把普通 RL、单轮 LLM、Tool Agent 都写成 MDP；能解释 trajectory、return、terminal reward。 |
| Day 2 | REINFORCE、Policy Gradient、Reward-to-Go、Variance Reduction | CS285 Lecture 5: Policy Gradients；CS285 HW2 | Slides → 白纸推导 → 做官方 HW2，不先看答案。 | **CS285 HW2 官方题**：完成 vanilla Policy Gradient / REINFORCE / reward-to-go 相关实现和指定实验。 | 闭卷推导 Policy Gradient；回答“reward 不可导为什么还能训练”；对应实验能跑通。 |
| Day 3 | Baseline、Value Baseline、GAE、Actor-Critic 关系 | CS285 Lecture 6: Actor Critic；CS285 HW2 | 看 Lecture 6 后继续完成 HW2；重点理解 baseline 与 GAE，而不是扩展到所有 actor-critic 算法。 | **继续完成 CS285 HW2 官方题**：baseline、advantage estimation / GAE 及要求的核心实验、报告。 | 解释 baseline 为什么降低 variance 而不改变期望；能手算简单 GAE；HW2 核心结果完成。 |
| Day 4 | Importance Sampling、Policy Ratio、Trust Region、PPO | CS285 Lecture 9–10: Advanced Policy Gradients | 只学和 PPO / LLM RL 直接相关的部分；其它高级 policy optimization 内容按需跳过。 | **不自编额外算法作业**：完成 HW2 收尾；同时按官方 lecture 公式自行推 PPO，作为 Week 3 的前置知识。 | 能讲清 old/current/reference policy 的区别；解释 importance ratio、PPO clip、为什么同一 rollout 只能有限复用。 |

### Week 2 官方资料

- CS185/285 Spring 2026 Course: https://rail.eecs.berkeley.edu/deeprlcourse/
- CS285 Spring 2026 Homework Repo: https://github.com/berkeleydeeprlcourse/homework_spring2026
- CS285 HW2: https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw2

---

## Week 3｜LLM Post-training 与 GRPO：CS336 A5 为主

| Day | 学习内容 | 学习资料 | 学习方式 | 官方作业 / 实践 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | SFT、RLHF、Preference、Reward、Post-training 全景 | CS336 Lecture 15: Mid/Post-training (SFT/RLHF)；CS336 Assignment 5 | 先完整学习 Lecture 15；随后阅读 A5 handout 的整体任务和 SFT / tokenization / metrics 部分。 | **CS336 A5 官方题**：开始完成 metrics、tokenization、response mask / log-prob、SFT 相关核心实现。 | 能解释 SFT loss、completion mask、sequence/token logprob；能画出 SFT → reward / RL 的 post-training 流程。 |
| Day 2 | RLVR、GRPO、Group Relative Advantage、KL、Policy Gradient Loss | CS336 Lecture 16: Post-training - RLVR；CS336 A5 | Lecture 16 认真学，不跳 GRPO 公式；随后直接做 A5 对应实现。 | **CS336 A5 官方题**：reward calculation、group-normalized rewards / advantage、policy-gradient loss、KL / clipping 等 GRPO 核心部分。 | 能手算 group advantage；回答 GRPO 为什么不需要 critic、它节省什么又增加什么；能把公式映射到代码。 |
| Day 3 | 完整 GRPO Train Step、Microbatch、Gradient Accumulation、Loss Normalization | CS336 A5 | 纯官方作业日；不学新课。 | **继续 CS336 A5 官方题**：完成 GRPO train step、microbatch loss aggregation、梯度累积、gradient clipping 等；运行 `pytest tests/test_grpo.py`。 | `test_grpo.py` 核心测试通过；任意给一行 GRPO 代码都能解释输入 tensor、shape、公式含义。 |
| Day 4 | SFT + GRPO 实验；DPO 作为次要扩展 | CS336 A5；A5 optional safety/RLHF supplement | 完成 A5 要求的训练 / evaluation 结果；若当周时间充足，再做 supplemental DPO，不强制。 | **优先完成 A5 Required 部分**；DPO 只做官方 Optional，不单独自编 DPO 作业。 | 能从 prompt → rollout → reward → advantage → logprob → loss → optimizer step 完整讲一遍；A5 Required 核心任务完成。 |

### Week 3 官方资料

- CS336 Lecture 15: https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf
- CS336 Lecture 16: https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf
- CS336 Assignment 5: https://github.com/stanford-cs336/assignment5-alignment
- CS336 A5 Handout: https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf
- CS336 A5 Optional Safety/RLHF Supplement: https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf

---

## Week 4｜LLM RL 实战：CS285 Lecture 14 + HW4

| Day | 学习内容 | 学习资料 | 学习方式 | 官方作业 / 实践 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | LLM RL 总体框架；REINFORCE 与 GRPO 在语言模型中的实现 | CS285 2026 Lecture 14: LLM RL；CS285 HW4 | 先学 Lecture 14，再完整读 HW4 README / handout 和代码骨架。 | **CS285 HW4 官方题 / Required Runs**：完成简单任务上的 REINFORCE 与 GRPO 训练，如官方 Format Copy 配置。 | 能解释 HW4 的 prompt、rollout、reward、group、old/ref logprob、update 数据流；至少两组 required run 正常。 |
| Day 2 | Reasoning RL / Math Task 上的 REINFORCE vs GRPO | CS285 HW4 | 不看新课，直接跑官方要求的 reasoning / Math Hard 等 required runs。先写预测，再运行。 | **继续 CS285 HW4 官方 Required Runs**：完成 REINFORCE 与 GRPO 对比。 | 能依据 reward、response length、sample generation 解释两种方法差异，不只看 loss。 |
| Day 3 | LLM RL 训练代码：group size、PPO epochs、minibatch、clip、KL | CS285 HW4 源码 | 沿官方 HW4 一次 train step 追源码；不另外造 toy trainer。 | **完成 HW4 官方实现 / 报告要求**；把要求记录的关键 metric、sample 和曲线整理出来。 | 能解释 `group_size / ppo_epochs / minibatch_size / clip_eps / kl_coef` 在算法和代码中的作用。 |
| Day 4 | 官方 HW4 实验分析与报告 | CS285 HW4 | 只做作业收尾、结果分析、报告；如果官方题要求超参或算法比较，严格按官方要求完成。 | **完成 HW4 官方 submission / report 范围**。如果算力不足，允许缩小训练规模学习，但必须保留同一比较逻辑。 | 能对训练曲线、KL、reward、长度、sample 做完整诊断；Week 4 结束后应达到 GRPO L3。 |

### Week 4 官方资料

- CS285 2026 Course / Lecture 14: https://rail.eecs.berkeley.edu/deeprlcourse/
- CS285 HW4 LLM RL: https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw4

---

## Week 5｜Agentic RL：verl 官方 Agentic RL / Agent Loop

> 说明：这一段目前没有找到与 CS336 / CS285 同等级、同时包含公开 Lecture + Auto-graded Homework 的大学课程。因此不再虚构“课程作业”，而是直接使用 verl 官方 Agentic RL 示例作为实验主线；只有官方示例没有覆盖的验证项才做最小自定义 Lab。

| Day | 学习内容 | 学习资料 | 学习方式 | 官方作业 / 实践 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | Agentic RL 系统架构；Server-based Async Rollout；Trainer / Agent / Inference Server 分离 | verl Agentic RL Training 官方文档 | 直接读官方文档；边读边追官方示例配置，不看二手课程。 | **verl 官方 Example**：跑 / 读 Agentic RL 官方启动示例，确认 async rollout 配置、inference engine、agent client 数据流。 | 能解释 Agent RL 为什么容易 GPU idle；为什么 tool latency 和 long-tail rollout 会成为系统瓶颈。 |
| Day 2 | Multi-turn Conversation、Tool Call、Agent Loop、Trajectory | verl Multi-turn Rollout + Agent Loop 官方文档 | 文档 + 源码对照，重点看 `AgentLoopBase.run()`、tool agent loop、message/trajectory 组织。 | **verl 官方 Multi-turn / Tool Agent Example**：跑官方多轮工具调用示例，检查 assistant/tool message 和最终 trajectory。 | 能指出哪些内容是 policy action、哪些是 environment observation；讲清 multi-turn rollout 与 single-turn GRPO 的结构差异。 |
| Day 3 | GSM8K Tool Agent RL：真正跑一条工具 Agent 训练链路 | verl Agentic RL GSM8K Tool Agent 官方示例 | 按官方数据预处理、tool agent、training、trace 流程跑；规模可缩小。 | **verl 官方 GSM8K Tool Agent Example**：产生真实 tool-call rollout / trace，并完成至少一个小规模训练运行。 | 任取一条 trace，能解释 LLM → Tool → Observation → LLM → Reward → Update；能区分 policy failure、tool failure、environment failure。 |
| Day 4 | 自定义 Tool / Reward / Agent Loop 扩展 | verl How to Extend / Agent Loop 官方文档 | 先按官方 extension interface 实现；只有这里保留自定义任务，因为官方没有给与你个人方向完全对应的 autograded homework。 | **最小自定义 Lab（官方课程缺口）**：基于 verl 官方接口增加 1 个 tool 或修改 1 个 reward，并设计正常 / invalid / reward-hacking case 验证。 | 至少找到 2 个 reward / tool design 的失败模式；能说明环境、reward、policy 三者边界。 |

### Week 5 官方资料

- verl Agentic RL Training: https://verl.readthedocs.io/en/latest/start/agentic_rl.html
- verl Agent Loop: https://verl.readthedocs.io/en/latest/advance/agent_loop.html
- verl Documentation: https://verl.readthedocs.io/en/latest/

---

## Week 6｜DoctorAgent-RL 项目收束 + 面试复盘

> 说明：这一周不再冒充“课程作业”。这是把前五周官方课程知识迁移到个人项目的项目验收周。实验设计参考 Berkeley CS285 2026 LLM RL Final Project 的固定 benchmark、对照算法、held-out evaluation 和在线 RLHF 实验方式。

| Day | 学习内容 | 学习资料 | 学习方式 | 项目任务 | 当天验收 |
|---|---|---|---|---|---|
| Day 1 | DoctorAgent-RL 完整代码地图；current / old / reference / patient / reward / rollout | 自己的 DoctorAgent-RL；对照 CS336 A5、CS285 HW4、verl Agent Loop | 从真实代码入口沿一条样本追到底；课程资料只作为 reference implementation。 | 画项目架构图和单样本 rollout → reward → GRPO update 时序图；标清所有模型角色和 tensor / trajectory 数据。 | 不看代码 10 分钟讲清完整训练流程；任何一个 policy / reward / trajectory 角色不能混淆。 |
| Day 2 | Reward Evaluation / Reward Ablation | Berkeley CS285 LLM RL Final Project 的 benchmark / reward-model / online RLHF 设计作为参考 | 先固定 eval set、metric、baseline，再做实验；禁止边跑边改评价规则。 | 对 DoctorAgent-RL 做 1 个有明确假设的 reward ablation：删除或调整一个 reward component；保存代表性 trajectory。 | 不只汇报 reward；必须说明该组件具体改变了什么行为、是否影响真实 success / quality。 |
| Day 3 | GRPO / Rollout Ablation + Failure Analysis | CS285 HW4 参数；CS285 LLM RL Final Project；verl Agentic RL | 先写 hypothesis，再选 1–2 个最有理论意义的变量跑对照。 | 从 group size / num generations、KL、clip、max response length 等选 1–2 个；同时标注至少 10 条失败 trajectory。 | 输出 Top-3 failure mode，每类都有“现象 → 证据 → 假设 → 下一实验”；结果与预测冲突时能提出新假设。 |
| Day 4 | 最终面试复盘：Transformer + PPO + GRPO + Agentic RL + 项目 | 快速回看 CS336 Lecture 3 / 10 / 15 / 16、CS285 Lecture 4 / 5 / 9 / 10 / 14、自己的 Week 1–6 产出 | 不重新系统听课，只按薄弱点跳转复习；最后和 GPT 做 30–45 分钟压力面试。 | 完成 10 分钟 DoctorAgent-RL 项目讲解；30 道核心问答；白板重画 Transformer、PPO/GRPO、Agent trajectory。 | Transformer / PPO / GRPO 达 L3；Agent rollout / reward / failure analysis 达 L3–L4；30 题正确率 ≥ 80%。 |

### Week 6 官方参考

- CS285 2026 LLM RL Final Project: https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/final_project_llm_rl
- CS336 Assignment 5: https://github.com/stanford-cs336/assignment5-alignment
- CS285 HW4: https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw4
- verl Agentic RL: https://verl.readthedocs.io/en/latest/start/agentic_rl.html

---

## 最终要求

| 模块 | 最终必须达到 |
|---|---|
| Transformer | 能从输入 token 推到 logits / logprob；能解释 MHA、RoPE、RMSNorm、SwiGLU、KV Cache、Prefill / Decode |
| RL 基础 | 能推 Policy Gradient；理解 Value / Advantage / GAE / Importance Sampling |
| PPO | 能解释 old policy、ratio、clip、KL、为什么有限复用 rollout |
| GRPO | 能手推 group advantage；能读懂 token-level GRPO loss；能解释 critic-free 的收益与代价 |
| LLM RL 工程 | 完成 CS336 A5 Required 核心题和 CS285 HW4 核心实验；能追一次完整 train step |
| Agentic RL | 能解释 multi-turn trajectory、tool observation、agent loop、async rollout、reward / environment 边界 |
| 项目 | DoctorAgent-RL 至少有 baseline、1 个 reward ablation、1 个 rollout/GRPO ablation、failure taxonomy |
| 面试 | 10 分钟项目陈述 + 20–30 分钟追问；核心问题正确率 ≥80% |
