#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


BASE = Path("/home/public/bjh/dym/NLP_longcontext/evaluation")
RUNS = {
    "rope_none": BASE / "outputs/stage2_kv4k_rope_none_minikv_digit8_len2k4k_pos5_s100/summary.csv",
    "rope_linear": BASE / "outputs/stage2_kv4k_rope_linear_minikv_digit8_len2k4k_pos5_s100/summary.csv",
    "rope_ntk": BASE / "outputs/stage2_kv4k_rope_ntk_minikv_digit8_len2k4k_pos5_s100/summary.csv",
    "rope_yarn": BASE / "outputs/stage2_kv4k_rope_yarn_minikv_digit8_len2k4k_pos5_s100/summary.csv",
}


def main() -> None:
    rows = []
    for model_name, path in RUNS.items():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "model": model_name,
                        "target_tokens": row["target_tokens"],
                        "position_ratio": row["position_ratio"],
                        "accuracy": row["exact_match"],
                        "format_error": row["format_error_rate"],
                        "wrong_context": row["wrong_context_value_rate"],
                        "hallucinated": row["hallucinated_value_rate"],
                    }
                )

    out = BASE / "outputs/stage2_kv4k_rope_compare_key_metrics.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "target_tokens",
                "position_ratio",
                "accuracy",
                "format_error",
                "wrong_context",
                "hallucinated",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
