# Data Directory

该目录只提交说明和 `samples/` 下的少量脱敏样例。以下生成物被 `.gitignore` 排除：

- `MTMedDialog_sft_train.parquet`
- `MTMedDialog_sft_val.parquet`
- `MTMedDialog_RL.parquet`
- `MTMedDialog_test.json`
- `raw_sources/`、`processed/`、`processed_samples/`

使用样例生成可运行的数据接口：

```bash
python scripts/data/materialize_artifacts.py \
  --sft-train-json data/samples/MTMedDialog_sft_train_first_10.json \
  --sft-val-json data/samples/MTMedDialog_sft_val_first_10.json \
  --rl-json data/samples/MTMedDialog_RL_first_10.json \
  --test-json data/samples/MTMedDialog_test_first_10.json \
  --output-dir data
```

完整流程与字段说明见仓库根目录的 [DATASET.md](../DATASET.md)。
