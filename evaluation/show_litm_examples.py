#!/usr/bin/env python3
"""导出 LITM-KV 真实评测样例，说明输入、输出与打分。"""

from __future__ import annotations

import json
from pathlib import Path


PRED_PATH = Path("/home/public/bjh/dym/NLP/evaluation/outputs/qwen3_0_6b_litm_kv_official5_merged/predictions.jsonl")


def load_rows():
    return [json.loads(line) for line in PRED_PATH.open(encoding="utf-8")]


def pick(rows, condition):
    for row in rows:
        if condition(row):
            return row
    raise RuntimeError("没有找到符合条件的样本")


def format_record(row):
    return [
        f"example_id: {row['example_id']}",
        f"num_keys / position: {row['num_keys']} / {int(row['position_ratio'] * 100)}%",
        f"query key: {row['key']}",
        f"gold value: {row['value']}",
        f"model output: {row['generated_text']!r}",
        f"first_extracted_value: {row.get('first_extracted_value')}",
        f"first_value_match: {row['first_value_match']}",
        f"wrong_context_value: {row['wrong_context_value']}",
        f"hallucinated_value: {row['hallucinated_value']}",
        f"format_error: {row['format_error']}",
        f"target_index: {row['target_index']}",
        f"predicted_index: {row.get('predicted_index')}",
        f"abs_index_delta: {row.get('abs_index_delta')}",
    ]


def main() -> int:
    rows = load_rows()
    examples = [
        ("正确样例", pick(rows, lambda r: r["num_keys"] == 75 and r["position_ratio"] == 0.5 and r["first_value_match"] == 1)),
        ("上下文内错误 value 样例", pick(rows, lambda r: r["num_keys"] == 300 and r["position_ratio"] == 0.5 and r["wrong_context_value"] == 1)),
        ("幻觉 UUID 样例", pick(rows, lambda r: r["num_keys"] == 300 and r["position_ratio"] == 0.5 and r["hallucinated_value"] == 1)),
    ]
    for title, row in examples:
        print(f"\n## {title}")
        print("\n".join(format_record(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
