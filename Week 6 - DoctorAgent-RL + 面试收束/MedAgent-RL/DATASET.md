# Dataset

## 固定数据接口

训练和评测入口只依赖四个文件：

| 文件 | 完整实验规模 | 核心字段 |
| --- | ---: | --- |
| `data/MTMedDialog_sft_train.parquet` | 5,516 | `data_source`、`prompt`、`response`、`reward_model`、`extra_info` |
| `data/MTMedDialog_sft_val.parquet` | 549 | 与 SFT Train 相同 |
| `data/MTMedDialog_RL.parquet` | 7,068 | `data_source`、`prompt`、`reward_model`、`extra_info` |
| `data/MTMedDialog_test.json` | 2,082 | 患者自述、参考诊断/建议、患者画像与类别信息 |

完整数据、原始数据缓存和生成结果不进入 Git；`data/samples/` 仅包含每类 10 条脱敏样例。

## 数据流程

```text
IMCS21 / CHIP-MDCFNPC / MedDG
              │
              ▼
格式归一化与质量门控
              │
              ▼
病例级划分 ──► SFT 冷启动记录
      │
      ├──────► GRPO 病例与奖励字段
      └──────► Test 病例与患者画像
```

`scripts/data/build_dataset.py` 负责可检查的原始数据归一化、去重和病例级划分；`generate_cold_start_sft.py` 生成逐轮策略推理；`generate_patient_profiles.py` 构建患者模拟所需的增强画像；`materialize_artifacts.py` 是最终接口门禁，写文件前会拒绝跨集合病例泄漏。

## 构建步骤

1. 下载可公开获取的原始数据：

```bash
bash scripts/data/download_raw_datasets.sh
```

CHIP-MDCFNPC 需要按脚本提示从数据平台手动下载并放入指定目录。

2. 构建确定性病例级中间产物：

```bash
python scripts/data/build_dataset.py \
  --sample-per-source 50 \
  --seed 42 \
  --output-dir data/processed_samples
```

3. 如需生成 SFT 推理，使用环境变量提供 API 配置：

```bash
export API_KEY='...'
export API_BASE_URL='https://api.deepseek.com'
export API_MODEL_NAME='deepseek-v3'
python scripts/data/generate_cold_start_sft.py
```

4. 将准备好的四个 JSON 集合物化为固定接口：

```bash
python scripts/data/materialize_artifacts.py \
  --sft-train-json /path/to/sft_train.json \
  --sft-val-json /path/to/sft_val.json \
  --rl-json /path/to/rl.json \
  --test-json /path/to/test.json \
  --output-dir data
```

## 数据质量约束

- 以归一化后的患者首轮自述作为病例键，SFT Train、SFT Val、GRPO 和 Test 之间不得重叠。
- 同一 SFT 病例可展开为多条逐轮训练记录，但只能属于一个集合。
- 原始文本清理、病例筛选和划分使用固定随机种子。
- 完整数据的发布与使用必须遵守对应源数据的许可和隐私要求。
