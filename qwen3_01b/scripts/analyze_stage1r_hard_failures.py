#!/usr/bin/env python3
"""Analyze Stage1R hard-negative failures for V3 data design."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--output-dir", default="analysis/stage1r_v3_hard_retrieval_20260602/failure_analysis")
    p.add_argument(
        "--eval-dir",
        action="append",
        default=[
            "analysis/stage1r_v2_mixed_replay_20260601/eval_stage1r_v2b_step1000_v1eval",
            "analysis/stage1r_v2_mixed_replay_20260601/eval_stage1r_v2b_step1000_v2eval",
        ],
        help="Eval output directory containing answer_rows.csv and candidate_scores.csv. Can be repeated.",
    )
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return sum(vals) / max(1, len(vals))


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    out_dir = resolve(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_answer_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    for eval_dir_arg in args.eval_dir:
        eval_dir = resolve(root, eval_dir_arg)
        eval_name = eval_dir.name
        answer_rows = read_csv(eval_dir / "answer_rows.csv")
        candidate_rows = read_csv(eval_dir / "candidate_scores.csv")
        by_ex: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidate_rows:
            by_ex[row["example_id"]].append(row)
        for row in answer_rows:
            row = dict(row)
            row["eval_name"] = eval_name
            all_answer_rows.append(row)
            candidates = by_ex[row["example_id"]]
            wrong = [c for c in candidates if c["candidate_kind"] != "gold"]
            wrong_by_sum = sorted(wrong, key=lambda c: float(c["candidate_sum_logprob"]), reverse=True)
            wrong_by_mean = sorted(wrong, key=lambda c: float(c["candidate_mean_nll"]))
            top_sum = wrong_by_sum[0] if wrong_by_sum else None
            top_mean = wrong_by_mean[0] if wrong_by_mean else None
            is_fail = int(row["rank_correct_sum_logprob"]) == 0
            hard_rows.append(
                {
                    "eval_name": eval_name,
                    "example_id": row["example_id"],
                    "failed_sum_rank": int(is_fail),
                    "target_length": int(row["target_length"]),
                    "evidence_position": row["evidence_position"],
                    "task_type": row["task_type"],
                    "answer_type": row["answer_type"],
                    "gold_value": row["gold_value"],
                    "gold_rank_sum_logprob": int(float(row["gold_rank_sum_logprob"])),
                    "gold_sum_logprob": float(row["answer_sum_logprob"]),
                    "best_wrong_sum_logprob": float(row["best_wrong_sum_logprob"]),
                    "sum_logprob_margin": float(row["sum_logprob_margin"]),
                    "best_wrong_value_sum": top_sum["candidate_value"] if top_sum else "",
                    "best_wrong_mean_nll": float(top_mean["candidate_mean_nll"]) if top_mean else None,
                    "best_wrong_value_mean": top_mean["candidate_value"] if top_mean else "",
                    "answer_token_count": int(float(row["answer_token_count"])),
                    "evidence_to_answer_distance": float(row["evidence_to_answer_distance"]),
                }
            )

    write_csv(out_dir / "hard_failure_rows.csv", hard_rows)
    failed = [row for row in hard_rows if row["failed_sum_rank"]]
    total = len(hard_rows)
    summary: dict[str, Any] = {
        "total_examples": total,
        "failed_examples": len(failed),
        "failure_rate": len(failed) / max(1, total),
        "failure_counts_by_task": dict(Counter(row["task_type"] for row in failed)),
        "failure_counts_by_length": dict(Counter(str(row["target_length"]) for row in failed)),
        "failure_counts_by_answer_type": dict(Counter(row["answer_type"] for row in failed)),
        "failure_counts_by_position": dict(Counter(row["evidence_position"] for row in failed)),
    }

    group_rows: list[dict[str, Any]] = []
    dimensions = [
        ("task_type",),
        ("target_length",),
        ("answer_type",),
        ("evidence_position",),
        ("task_type", "target_length"),
        ("task_type", "answer_type"),
        ("task_type", "target_length", "answer_type"),
    ]
    for dims in dimensions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in hard_rows:
            key = tuple(row[d] for d in dims)
            grouped[key].append(row)
        for key, rows in grouped.items():
            fails = [row for row in rows if row["failed_sum_rank"]]
            group_rows.append(
                {
                    "group_by": "+".join(dims),
                    **{dim: value for dim, value in zip(dims, key)},
                    "n": len(rows),
                    "failures": len(fails),
                    "failure_rate": len(fails) / max(1, len(rows)),
                    "mean_margin": group_mean(rows, "sum_logprob_margin"),
                    "mean_gold_rank": group_mean(rows, "gold_rank_sum_logprob"),
                    "mean_distance": group_mean(rows, "evidence_to_answer_distance"),
                }
            )
    group_rows.sort(key=lambda row: (-float(row["failure_rate"]), -int(row["failures"]), str(row["group_by"])))
    write_csv(out_dir / "failure_groups.csv", group_rows)
    summary["top_failure_groups"] = group_rows[:30]
    write_json(out_dir / "summary.json", summary)

    lines = [
        "# Stage1R V3 Hard Failure Analysis",
        "",
        f"- Total examples: `{total}`",
        f"- Failed examples: `{len(failed)}`",
        f"- Failure rate: `{summary['failure_rate']:.4f}`",
        "",
        "## Failure Counts",
        "",
        f"- By task: `{summary['failure_counts_by_task']}`",
        f"- By length: `{summary['failure_counts_by_length']}`",
        f"- By answer type: `{summary['failure_counts_by_answer_type']}`",
        f"- By position: `{summary['failure_counts_by_position']}`",
        "",
        "## Top Failure Groups",
        "",
        "| Group | n | failures | failure rate | mean margin | mean gold rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows[:20]:
        labels = []
        for key, value in row.items():
            if key in {"group_by", "n", "failures", "failure_rate", "mean_margin", "mean_gold_rank", "mean_distance"}:
                continue
            labels.append(f"{key}={value}")
        lines.append(
            f"| {'; '.join(labels)} | {row['n']} | {row['failures']} | {row['failure_rate']:.4f} | "
            f"{row['mean_margin']:.4f} | {row['mean_gold_rank']:.4f} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
