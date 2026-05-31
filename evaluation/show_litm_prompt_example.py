#!/usr/bin/env python3
"""重建一个 LITM-KV prompt，展示实际输入格式。"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


DATA_DIR = Path("/home/public/bjh/dym/NLP/evaluation/benchmarks/lost_middle/data")
PRED_PATH = Path("/home/public/bjh/dym/NLP/evaluation/outputs/qwen3_0_6b_litm_kv_official5_merged/predictions.jsonl")

PROMPT_PREFIX = (
    "Extract the value corresponding to the specified key from the records below. "
    "Return only the value.\n\n"
    "Records:\n"
)


def record_line(key: str, value: str) -> str:
    return f"Key: {key} | Value: {value}\n"


def move_target_record(records, target_key: str, position_ratio: float):
    target = None
    rest = []
    for record in records:
        if record[0] == target_key:
            target = record
        else:
            rest.append(record)
    if target is None:
        raise RuntimeError("target key not found")
    target_index = round(position_ratio * (len(records) - 1))
    return rest[:target_index] + [target] + rest[target_index:], target_index


def main() -> int:
    pred = None
    for line in PRED_PATH.open(encoding="utf-8"):
        row = json.loads(line)
        if row["example_id"] == "kv75_pos0.50_sample2":
            pred = row
            break
    if pred is None:
        raise RuntimeError("prediction not found")

    rows = []
    with gzip.open(DATA_DIR / "kv-retrieval-75_keys.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    source = rows[pred["source_index"]]
    records, target_index = move_target_record(source["ordered_kv_records"], pred["key"], pred["position_ratio"])

    print("## 实际 prompt 结构示例")
    print(f"example_id: {pred['example_id']}")
    print(f"source_index: {pred['source_index']}")
    print(f"target_index: {target_index}")
    print(f"query key: {pred['key']}")
    print(f"gold value: {pred['value']}")
    print()
    print("```text")
    print(PROMPT_PREFIX, end="")
    for i, (key, value) in enumerate(records[:3]):
        print(f"[{i}] " + record_line(key, value), end="")
    print("...")
    for i in range(target_index - 1, target_index + 2):
        key, value = records[i]
        marker = "  <-- target record" if i == target_index else ""
        print(f"[{i}] " + record_line(key, value).rstrip() + marker)
    print("...")
    for i in range(len(records) - 2, len(records)):
        key, value = records[i]
        print(f"[{i}] " + record_line(key, value), end="")
    print()
    print(f"Question: What is the value associated with key {pred['key']}?")
    print("Answer:")
    print("```")
    print()
    print("模型输出：")
    print(f"```text\n{pred['generated_text']}\n```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
