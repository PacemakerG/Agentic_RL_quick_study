# Week 4：LLM RL + GRPO 实现与实验

[学习计划](../%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92_%E6%A0%B8%E5%AF%B9%E4%BF%AE%E8%AE%A2%E7%89%88_%E5%85%88SFT%E5%90%8ERL.md) · [官方仓库](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw4)

CS285 HW4 在本目录；CS336 A5 复用 Week 2 的同一份仓库，避免 SFT 与 GRPO 各维护一套副本。A5 命令在 Week 2/CS336-A5-Alignment 中执行，HW4 命令在本目录 CS285-Homework/hw4 中执行。

| Day | 官方课程 / 作业内容 | 本地资料 | 本地代码 / 测试 | 当天实践 |
| --- | --- | --- | --- | --- |
| Day 1 | A5 prompting baseline 与 on-policy GRPO | [A5 主 handout](../Week%202%20-%20SFT%20+%20%E5%90%8E%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80/CS336-A5-Alignment/cs336_spring2026_assignment5_alignment.pdf) | [A5 GRPO 测试](../Week%202%20-%20SFT%20+%20%E5%90%8E%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80/CS336-A5-Alignment/tests/test_grpo.py) | 实现 reward / logprob / advantage / loss / train step，启动 on-policy baseline。 |
| Day 2 | A5 off-policy；HW4 学生 TODO | [HW4 README](CS285-Homework/hw4/README.md) | [HW4 实现目录](CS285-Homework/hw4/hw4) | 完成 logprob、mask、advantage、minibatch、REINFORCE/GRPO update 和 KL proxy；最小训练验证。 |
| Day 3 | Format Copy 两组 Required Runs | [官方训练命令](CS285-Homework/hw4/README.md) | [算法代码](CS285-Homework/hw4/hw4/rl) | 完成 REINFORCE 与 GRPO 实验，保存配置和结果。 |
| Day 4 | Math Hard 两组 Required Runs；A5 后续变体 | [官方训练命令](CS285-Homework/hw4/README.md) | [共用 A5 仓库](../Week%202%20-%20SFT%20+%20%E5%90%8E%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80/CS336-A5-Alignment) | 汇总四组 HW4 实验；随后按 A5 handout 补做消融与变体。 |
