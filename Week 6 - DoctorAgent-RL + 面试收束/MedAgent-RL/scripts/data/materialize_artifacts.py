#!/usr/bin/env python3
"""Validate prepared records and materialize MedAgent-RL training artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DESCRIPTION_PATTERN = re.compile(
    r"Patient's description:\s*(.*?)(?:\n+Decide next action:|$)",
    re.DOTALL,
)


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every row must be a JSON object: {path}")
    return records


def case_key(record: dict) -> str:
    self_report = record.get("self_report")
    if isinstance(self_report, str) and self_report.strip():
        return " ".join(self_report.split()).casefold()

    for message in record.get("prompt", []):
        content = message.get("content", "")
        match = DESCRIPTION_PATTERN.search(content)
        if match:
            return " ".join(match.group(1).split()).casefold()

    raise ValueError("Unable to derive a case key from self_report or prompt")


def validate_case_splits(splits: dict[str, list[dict]]) -> dict[str, int]:
    case_sets = {
        split: {case_key(record) for record in records}
        for split, records in splits.items()
    }
    names = list(case_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = case_sets[left] & case_sets[right]
            if overlap:
                example = next(iter(overlap))
                raise ValueError(
                    f"Case leakage between {left} and {right}: "
                    f"{len(overlap)} overlapping case(s), for example {example!r}"
                )
    return {name: len(values) for name, values in case_sets.items()}


def write_parquet(records: list[dict], output_path: Path) -> None:
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path, compression="zstd", version="2.6")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the fixed SFT, GRPO, and evaluation data interfaces."
    )
    parser.add_argument("--sft-train-json", type=Path, required=True)
    parser.add_argument("--sft-val-json", type=Path, required=True)
    parser.add_argument("--rl-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    splits = {
        "sft_train": load_records(args.sft_train_json),
        "sft_val": load_records(args.sft_val_json),
        "rl": load_records(args.rl_json),
        "test": load_records(args.test_json),
    }
    unique_cases = validate_case_splits(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sft_train": args.output_dir / "MTMedDialog_sft_train.parquet",
        "sft_val": args.output_dir / "MTMedDialog_sft_val.parquet",
        "rl": args.output_dir / "MTMedDialog_RL.parquet",
        "test": args.output_dir / "MTMedDialog_test.json",
    }
    write_parquet(splits["sft_train"], outputs["sft_train"])
    write_parquet(splits["sft_val"], outputs["sft_val"])
    write_parquet(splits["rl"], outputs["rl"])
    outputs["test"].write_text(
        json.dumps(splits["test"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "rows": {name: len(records) for name, records in splits.items()},
        "unique_cases": unique_cases,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
