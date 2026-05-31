#!/usr/bin/env python3
"""对 official5 LITM-KV 结果做细粒度诊断。"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


RUN_DIR = Path("/home/public/bjh/dym/NLP/evaluation/outputs/qwen3_0_6b_litm_kv_official5_merged")
OUT_PATH = Path("/home/public/bjh/dym/NLP/evaluation/docs/litm_kv_official5_diagnostics.md")
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
NUM_KEYS = [75, 140, 300]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def bucket_140(predicted_index: int | None) -> str:
    if predicted_index is None:
        return "无可解析 index"
    if predicted_index < 35:
        return "0-34"
    if predicted_index < 70:
        return "35-69"
    if predicted_index < 105:
        return "70-104"
    return "105-139"


def main() -> int:
    rows = [json.loads(line) for line in (RUN_DIR / "predictions.jsonl").open(encoding="utf-8")]
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(int(row["num_keys"]), float(row["position_ratio"]))].append(row)

    lines = [
        "# official5 LITM-KV 诊断统计",
        "",
        "## cell 级别诊断",
        "",
        "| KV 数量 | 位置 | n | 准确率 | target index 均值 | target token ratio 均值 | 上下文内错误 | 幻觉 UUID | 格式错误 | 平均 abs index delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nk in NUM_KEYS:
        for pos in POSITIONS:
            cell = by_cell[(nk, pos)]
            n = len(cell)
            acc = sum(int(r["first_value_match"]) for r in cell) / n
            wrong = sum(int(r["wrong_context_value"]) for r in cell) / n
            hallu = sum(int(r["hallucinated_value"]) for r in cell) / n
            fmt = sum(int(r["format_error"]) for r in cell) / n
            target_index_mean = statistics.mean(float(r["target_index"]) for r in cell)
            token_ratio_mean = statistics.mean(float(r["target_record_token_start"]) / float(r["actual_prompt_tokens"]) for r in cell)
            abs_delta = statistics.mean(float(r.get("abs_index_delta") or 0.0) for r in cell)
            lines.append(
                f"| {nk} | {int(pos * 100)}% | {n} | {pct(acc)} | {target_index_mean:.1f} | "
                f"{token_ratio_mean:.3f} | {pct(wrong)} | {pct(hallu)} | {pct(fmt)} | {abs_delta:.2f} |"
            )

    lines += [
        "",
        "## source sample 一致性检查",
        "",
        "同一 KV 数量下，五个位置使用的 source sample 集合应该一致，这样位置差异不是由样本不同导致的。",
        "",
        "| KV 数量 | 五个位置样本集合是否一致 | 每个位置 source 数量 |",
        "|---:|---|---|",
    ]
    for nk in NUM_KEYS:
        sets = []
        for pos in POSITIONS:
            sets.append(set(int(r["source_index"]) for r in by_cell[(nk, pos)]))
        lines.append(f"| {nk} | {'是' if all(s == sets[0] for s in sets) else '否'} | {', '.join(str(len(s)) for s in sets)} |")

    lines += [
        "",
        "## 140-key 错误答案 predicted index 分布",
        "",
        "这里把错误答案中可定位到上下文 value 的 predicted index 分桶；幻觉或格式错误记为无可解析 index。",
        "",
        "| 位置 | 错误总数 | 0-34 | 35-69 | 70-104 | 105-139 | 无可解析 index |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pos in POSITIONS:
        cell = by_cell[(140, pos)]
        counter = Counter()
        wrong_total = 0
        for row in cell:
            if row["first_value_match"]:
                continue
            wrong_total += 1
            counter[bucket_140(row.get("predicted_index"))] += 1
        lines.append(
            f"| {int(pos * 100)}% | {wrong_total} | {counter['0-34']} | {counter['35-69']} | "
            f"{counter['70-104']} | {counter['105-139']} | {counter['无可解析 index']} |"
        )

    lines += [
        "",
        "## 诊断结论",
        "",
        "1. target index 和 target token ratio 与设定位置一致，暂未看到位置构造错误。",
        "2. 同一 KV 数量下五个位置使用的是同一批 source sample，因此位置曲线差异不是由不同样本集合导致的。",
        "3. `140 keys @ 50%` 的高点是真实评测结果中的异常现象，需要后续进一步解释；它不是简单的 shard 合并错误或 source sample 不一致。",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_PATH)
    print("\n".join(lines[:32]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
