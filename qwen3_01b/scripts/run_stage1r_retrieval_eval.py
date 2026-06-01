#!/usr/bin/env python3
"""Evaluate Stage1R retrieval/copy examples by answer likelihood and ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer-json", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--candidate-limit", type=int, default=None)
    p.add_argument("--label", default=None)
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


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


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def score_candidate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    candidate_text: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt)
    cand_ids = tokenizer.encode(candidate_text)
    if not cand_ids:
        raise ValueError("empty candidate tokenization")
    input_ids = torch.tensor([prompt_ids + cand_ids], dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids)
    selected = logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(cand_ids), :].float()
    targets = torch.tensor(cand_ids, dtype=torch.long, device=device)
    log_probs = F.log_softmax(selected, dim=-1)
    token_logprobs = log_probs.gather(1, targets[:, None]).squeeze(1)
    vals = [float(x) for x in token_logprobs.detach().cpu()]
    sum_logprob = float(sum(vals))
    mean_nll = float(-sum_logprob / max(1, len(vals)))
    return {
        "candidate_token_count": int(len(cand_ids)),
        "candidate_sum_logprob": sum_logprob,
        "candidate_mean_nll": mean_nll,
        "candidate_ppl": math.exp(min(80.0, mean_nll)),
        "candidate_token_logprobs": vals,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return sum(vals) / max(1, len(vals))


def aggregate(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        keys = [
            ("ALL", "ALL", "ALL", "ALL"),
            (row["target_length"], "ALL", "ALL", "ALL"),
            (row["target_length"], row["evidence_position"], "ALL", "ALL"),
            (row["target_length"], row["evidence_position"], row["task_type"], "ALL"),
            (row["target_length"], row["evidence_position"], row["task_type"], row["answer_type"]),
        ]
        for key in keys:
            groups[key].append(row)
    out: list[dict[str, Any]] = []
    for (length, position, task, answer_type), group in sorted(
        groups.items(),
        key=lambda item: (
            10**9 if item[0][0] == "ALL" else int(item[0][0]),
            str(item[0][1]),
            str(item[0][2]),
            str(item[0][3]),
        ),
    ):
        out.append(
            {
                "label": label,
                "target_length": length,
                "evidence_position": position,
                "task_type": task,
                "answer_type": answer_type,
                "n": len(group),
                "answer_mean_nll": mean(group, "answer_mean_nll"),
                "answer_ppl": math.exp(min(80.0, mean(group, "answer_mean_nll"))),
                "mean_sum_logprob_margin": mean(group, "sum_logprob_margin"),
                "mean_mean_nll_margin": mean(group, "mean_nll_margin"),
                "rank_acc_sum_logprob": mean(group, "rank_correct_sum_logprob"),
                "rank_acc_mean_nll": mean(group, "rank_correct_mean_nll"),
                "mean_gold_rank_sum_logprob": mean(group, "gold_rank_sum_logprob"),
                "mean_gold_rank_mean_nll": mean(group, "gold_rank_mean_nll"),
                "mean_evidence_to_answer_distance": mean(group, "evidence_to_answer_distance"),
            }
        )
    return out


@torch.inference_mode()
def evaluate_examples(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    candidate_limit: int | None,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    answer_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for idx, ex in enumerate(examples):
        if idx and idx % 25 == 0:
            print(f"[{label}] scored {idx}/{len(examples)} examples in {time.perf_counter() - start:.1f}s", flush=True)
        candidates = [str(ex["gold_value"])] + [str(x) for x in ex.get("distractor_values", [])]
        if candidate_limit is not None:
            candidates = candidates[: max(1, int(candidate_limit))]
        scores: list[tuple[str, str, dict[str, Any]]] = []
        for ci, value in enumerate(candidates):
            kind = "gold" if ci == 0 else "distractor"
            score = score_candidate(
                model=model,
                tokenizer=tokenizer,
                prompt=ex["prompt"],
                candidate_text=" " + value,
                device=device,
            )
            scores.append((kind, value, score))
            row = {
                "label": label,
                "example_id": ex["example_id"],
                "candidate_kind": kind,
                "candidate_value": value,
                "candidate_index": ci,
                "target_length": ex["target_length"],
                "evidence_position": ex["evidence_position"],
                "task_type": ex["task_type"],
                "answer_type": ex["answer_type"],
                "distractor_count": ex["distractor_count"],
                "prompt_tokens": ex["prompt_tokens"],
                "answer_tokens": ex["answer_tokens"],
                "evidence_to_answer_distance": ex["evidence_to_answer_distance"],
                **{k: v for k, v in score.items() if k != "candidate_token_logprobs"},
            }
            candidate_rows.append(row)
        gold_score = scores[0][2]
        wrong_scores = [score for kind, _value, score in scores if kind != "gold"]
        sorted_by_sum = sorted(scores, key=lambda item: item[2]["candidate_sum_logprob"], reverse=True)
        sorted_by_mean = sorted(scores, key=lambda item: item[2]["candidate_mean_nll"])
        gold_rank_sum = next(i + 1 for i, (kind, _value, _score) in enumerate(sorted_by_sum) if kind == "gold")
        gold_rank_mean = next(i + 1 for i, (kind, _value, _score) in enumerate(sorted_by_mean) if kind == "gold")
        best_wrong_sum = max((s["candidate_sum_logprob"] for s in wrong_scores), default=float("-inf"))
        best_wrong_mean = min((s["candidate_mean_nll"] for s in wrong_scores), default=float("inf"))
        answer_rows.append(
            {
                "label": label,
                "example_id": ex["example_id"],
                "target_length": ex["target_length"],
                "evidence_position": ex["evidence_position"],
                "position_ratio": ex.get("position_ratio"),
                "task_type": ex["task_type"],
                "answer_type": ex["answer_type"],
                "distractor_count": ex["distractor_count"],
                "gold_value": ex["gold_value"],
                "prompt_tokens": ex["prompt_tokens"],
                "answer_token_count": gold_score["candidate_token_count"],
                "answer_mean_nll": gold_score["candidate_mean_nll"],
                "answer_ppl": gold_score["candidate_ppl"],
                "answer_sum_logprob": gold_score["candidate_sum_logprob"],
                "best_wrong_sum_logprob": best_wrong_sum,
                "best_wrong_mean_nll": best_wrong_mean,
                "sum_logprob_margin": gold_score["candidate_sum_logprob"] - best_wrong_sum,
                "mean_nll_margin": best_wrong_mean - gold_score["candidate_mean_nll"],
                "gold_rank_sum_logprob": gold_rank_sum,
                "gold_rank_mean_nll": gold_rank_mean,
                "rank_correct_sum_logprob": int(gold_rank_sum == 1),
                "rank_correct_mean_nll": int(gold_rank_mean == 1),
                "evidence_token_start": ex["evidence_token_start"],
                "evidence_token_end": ex["evidence_token_end"],
                "evidence_to_answer_distance": ex["evidence_to_answer_distance"],
                "num_candidates": len(scores),
            }
        )
    return answer_rows, candidate_rows, aggregate(answer_rows, label)


def write_report(
    *,
    output_dir: Path,
    label: str,
    checkpoint: Path,
    eval_jsonl: Path,
    examples: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    elapsed_sec: float,
) -> None:
    overall = next(row for row in summary_rows if row["target_length"] == "ALL")
    length_rows = [row for row in summary_rows if row["evidence_position"] == "ALL" and row["task_type"] == "ALL" and row["target_length"] != "ALL"]
    lines = [
        f"# Stage1R Retrieval Eval: {label}",
        "",
        "## Scope",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Eval data: `{eval_jsonl}`",
        f"- Examples: `{len(examples)}`",
        f"- Elapsed seconds: `{elapsed_sec:.1f}`",
        "",
        "## Overall",
        "",
        f"- Answer mean NLL: `{overall['answer_mean_nll']:.4f}`",
        f"- Answer PPL: `{overall['answer_ppl']:.4f}`",
        f"- Hard-negative rank acc by summed logprob: `{overall['rank_acc_sum_logprob']:.4f}`",
        f"- Hard-negative rank acc by mean NLL: `{overall['rank_acc_mean_nll']:.4f}`",
        f"- Mean summed-logprob margin: `{overall['mean_sum_logprob_margin']:.4f}`",
        f"- Mean mean-NLL margin: `{overall['mean_mean_nll_margin']:.4f}`",
        "",
        "## By Length",
        "",
        "| Length | n | Answer NLL | Answer PPL | Sum rank acc | Mean-NLL rank acc | Sum margin | Mean-NLL margin |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in length_rows:
        lines.append(
            f"| {row['target_length']} | {row['n']} | {row['answer_mean_nll']:.4f} | {row['answer_ppl']:.4f} | "
            f"{row['rank_acc_sum_logprob']:.4f} | {row['rank_acc_mean_nll']:.4f} | "
            f"{row['mean_sum_logprob_margin']:.4f} | {row['mean_mean_nll_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Negatives are same-context distractor values generated in the eval example.",
            "- This eval scores answer likelihood only; it does not require instruction following or free generation.",
            "- A positive margin means the gold value is preferred over the strongest distractor under that ranking rule.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
    from qwen3_01b.training import build_model_from_checkpoint

    checkpoint = resolve(root, args.checkpoint)
    eval_jsonl = resolve(root, args.eval_jsonl)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    label = args.label or checkpoint.parent.name

    tokenizer_path = resolve(root, args.tokenizer_json) if args.tokenizer_json else None
    if tokenizer_path is None:
        # Prefer sibling manifest metadata when eval_jsonl lives under a generated dataset.
        candidate = eval_jsonl.parent / "manifest.json"
        if candidate.is_file():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            tokenizer_path = Path(manifest["tokenizer_json"])
        else:
            tokenizer_path = root / "qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json"

    write_json(output_dir / "run_config.json", vars(args) | {"resolved_checkpoint": str(checkpoint), "resolved_eval_jsonl": str(eval_jsonl), "tokenizer_json": str(tokenizer_path)})
    print(f"Loading model {checkpoint} on {device}", flush=True)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    model = build_model_from_checkpoint(checkpoint, device)
    examples = read_jsonl(eval_jsonl, args.limit)
    start = time.perf_counter()
    answer_rows, candidate_rows, summary_rows = evaluate_examples(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        candidate_limit=args.candidate_limit,
        label=label,
    )
    elapsed = time.perf_counter() - start
    write_csv(output_dir / "answer_rows.csv", answer_rows)
    write_csv(output_dir / "candidate_scores.csv", candidate_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_report(
        output_dir=output_dir,
        label=label,
        checkpoint=checkpoint,
        eval_jsonl=eval_jsonl,
        examples=examples,
        summary_rows=summary_rows,
        elapsed_sec=elapsed,
    )
    overall = next(row for row in summary_rows if row["target_length"] == "ALL")
    print(json.dumps({"output_dir": str(output_dir), "label": label, "overall": overall, "elapsed_sec": elapsed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
