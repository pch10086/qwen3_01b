#!/usr/bin/env python3
"""打印本次 LITM-KV 评测真实输入和输出样例。"""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path("/home/public/bjh/dym/NLP_longcontext/evaluation")
RUN = BASE / "outputs/ours_stage1r_v3_hard_retrieval_litm_kv_short_2k_7pos_s20"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact_prompt(prompt: str, head: int = 8, tail: int = 8) -> str:
    lines = prompt.splitlines()
    if len(lines) <= head + tail:
        return prompt
    return "\n".join(lines[:head] + [f"... ({len(lines) - head - tail} lines omitted) ..."] + lines[-tail:])


def main() -> int:
    examples = {row["example_id"]: row for row in read_jsonl(RUN / "examples.jsonl")}
    predictions = {row["example_id"]: row for row in read_jsonl(RUN / "predictions.jsonl")}
    sample_ids = [
        "kv32_pos0.00_sample0",
        "kv32_pos0.25_sample7",
        "kv32_pos1.00_sample1",
        "kv50_pos1.00_sample13",
    ]
    for example_id in sample_ids:
        e = examples[example_id]
        p = predictions[example_id]
        print("=" * 100)
        print("example_id:", example_id)
        print("num_keys:", e["num_keys"])
        print("position_ratio:", e["position_ratio"])
        print("target_index:", e["target_index"])
        print("actual_prompt_tokens:", e["actual_prompt_tokens"])
        print("query_key:", e["key"])
        print("gold_value:", e["value"])
        print("contains_question_template:", "What is the value associated with key" in e["prompt"])
        print("\n[PROMPT]")
        print(compact_prompt(e["prompt"]))
        print("\n[GENERATED_TEXT]")
        print(repr(p["generated_text"]))
        print("\n[SCORE]")
        print("first_extracted_value:", p["first_extracted_value"])
        print("first_value_match:", p["first_value_match"])
        print("format_error:", p["format_error"])
        print("wrong_context_value:", p["wrong_context_value"])
        print("hallucinated_value:", p["hallucinated_value"])
        print("predicted_index:", p["predicted_index"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
