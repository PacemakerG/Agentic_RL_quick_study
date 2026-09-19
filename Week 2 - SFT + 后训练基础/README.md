# Week 2：SFT + 后训练基础

[学习计划](../%E5%AD%A6%E4%B9%A0%E8%AE%A1%E5%88%92_%E6%A0%B8%E5%AF%B9%E4%BF%AE%E8%AE%A2%E7%89%88_%E5%85%88SFT%E5%90%8ERL.md) · [官方仓库](https://github.com/stanford-cs336/assignment5-alignment)

使用 CS336 A5 官方 Safety/RLHF Supplement 第 3–5 节。A5 完整仓库保存在本目录；Week 4 继续使用同一份代码。

| Day | 官方课程 / 作业内容 | 本地资料 | 本地代码 / 测试 | 当天实践 |
| --- | --- | --- | --- | --- |
| Day 1 | Baseline Evaluation；look_at_sft | [官方 handout，第 3、4.1 节](CS336-A5-Alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | [评测与训练数据](CS336-A5-Alignment/data) | 完成四项 base-model baseline，查看 10 条 instruction-tuning 样本。 |
| Day 2 | data_loading | [官方 handout，第 4.2.1 节](CS336-A5-Alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | [测试适配入口](CS336-A5-Alignment/tests/adapters.py) | 实现 packed dataset / batching，运行 test_packed_sft_dataset 与 test_iterate_batches。 |
| Day 3 | sft_script；sft | [官方 handout，第 4.2.2 节](CS336-A5-Alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | [实现目录](CS336-A5-Alignment/cs336_alignment) | 按官方题目完成训练脚本、SFT，保存模型、tokenizer、配置和曲线。 |
| Day 4 | mmlu_sft；gsm8k_sft；alpaca_eval_sft；sst_sft | [官方 handout，第 5.1–5.4 节](CS336-A5-Alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | [评测脚本资源](CS336-A5-Alignment/scripts) | 按官方要求对照 base 与 SFT 结果，完成书面回答。 |
