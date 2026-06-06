#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path("/home/public/bjh/dym/NLP_longcontext/evaluation")
RUNS = {
    "rope_none": BASE / "outputs/stage2_kv4k_rope_none_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100",
    "rope_ntk": BASE / "outputs/stage2_kv4k_rope_ntk_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100",
    "rope_yarn_fixed": BASE / "outputs/stage2_kv4k_rope_yarn_fixed_minikv_digit8_frontpos_0_1_2_5_10_len2k4k_s100",
}


def bucket_predicted_index(predicted_index: int, num_records: int) -> str:
    if predicted_index < 0:
        return "not_in_context"
    if predicted_index == 0:
        return "target_0"
    ratio = predicted_index / max(1, num_records - 1)
    if ratio < 0.10:
        return "front_0_10"
    if ratio < 0.25:
        return "early_10_25"
    if ratio < 0.50:
        return "mid_25_50"
    if ratio < 0.75:
        return "late_50_75"
    if ratio < 0.90:
        return "tail_75_90"
    return "end_90_100"


def load_summary(model: str, run_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (run_dir / "summary.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "model": model,
                    "target_tokens": row["target_tokens"],
                    "position_ratio": row["position_ratio"],
                    "accuracy": row["exact_match"],
                    "format_error": row["format_error_rate"],
                    "wrong_context": row["wrong_context_value_rate"],
                    "hallucinated": row["hallucinated_value_rate"],
                }
            )
    return rows


def load_zero_error_buckets(model: str, run_dir: Path) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[int, float], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[int, float]] = Counter()
    corrects: Counter[tuple[int, float]] = Counter()
    with (run_dir / "predictions.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row["position_ratio"]) != 0.0:
                continue
            key = (int(row["target_tokens"]), float(row["position_ratio"]))
            totals[key] += 1
            if int(row["exact_match"]) == 1:
                corrects[key] += 1
                continue
            bucket = bucket_predicted_index(int(row["predicted_index"]), int(row["num_records"]))
            grouped[key][bucket] += 1

    bucket_order = [
        "not_in_context",
        "target_0",
        "front_0_10",
        "early_10_25",
        "mid_25_50",
        "late_50_75",
        "tail_75_90",
        "end_90_100",
    ]
    rows: list[dict[str, str | int | float]] = []
    for key in sorted(totals):
        target_tokens, position = key
        error_n = totals[key] - corrects[key]
        out: dict[str, str | int | float] = {
            "model": model,
            "target_tokens": target_tokens,
            "position_ratio": position,
            "n": totals[key],
            "correct": corrects[key],
            "error_n": error_n,
        }
        for bucket in bucket_order:
            count = grouped[key][bucket]
            out[f"{bucket}_count"] = count
            out[f"{bucket}_rate"] = count / error_n if error_n else 0.0
        rows.append(out)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summary_rows: list[dict[str, str]] = []
    bucket_rows: list[dict[str, str | int | float]] = []
    for model, run_dir in RUNS.items():
        summary_rows.extend(load_summary(model, run_dir))
        bucket_rows.extend(load_zero_error_buckets(model, run_dir))

    summary_out = BASE / "outputs/stage2_kv4k_frontpos_key_metrics.csv"
    bucket_out = BASE / "outputs/stage2_kv4k_frontpos_pos0_predicted_index_buckets.csv"
    write_rows(summary_out, summary_rows)
    write_rows(bucket_out, bucket_rows)
    print(summary_out)
    print(bucket_out)


if __name__ == "__main__":
    main()
