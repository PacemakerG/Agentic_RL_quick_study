# 学生本地自动评分器（Local Autograder）

本目录包含用于本地开发调试的 GPT-5.4 自动评估脚本。

## 使用方法

在项目根目录下运行：

```bash
export OPENAI_API_KEY=your_key_here
uv run python student_autograder/run_local_autograder.py \
  --submission_dir llm_rl_final_proj_public_submission \
  --output_json student_autograder_results.json
```

本地评分器使用与 Gradescope 完全相同的评估文件和胜率阈值。

## 评分项目
- Part 1 奖励模型评估：在 `reward_model/public_test_pref_scores.jsonl` 上打分
- Part 1 策略算法评估：在公开的 `128` 条 Prompt 评估集上对比胜率
- Part 2 最佳离线方法（Offline Best）：在相同的 `128` 条 Prompt 评估集上评测
- Part 2 最佳在线方法（Online Best）：在相同的 `128` 条 Prompt 评估集上评测

## 依赖要求
安装所需 Python 依赖：

```bash
uv pip install -r student_autograder/requirements.txt
```
