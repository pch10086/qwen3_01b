#!/usr/bin/env python3
"""Teacher-forced answer scoring plus answer-token attention summaries.

This analysis is meant for base/pretraining checkpoints that may not follow an
instruction prompt during free generation. Instead of requiring the model to
freely emit the whole numeric answer, it scores the gold answer continuation
under teacher forcing and records attention from the answer-token positions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
from qwen3_01b.training import build_model_from_checkpoint, get_device


def load_generation_module() -> Any:
    """Load the V2 generation-aware analysis module for shared probe/hook code."""
    path = ROOT / "qwen3_01b" / "scripts" / "generation_attention_analysis.py"
    spec = importlib.util.spec_from_file_location("generation_attention_analysis_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared analysis module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gen = load_generation_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-json",
        default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json",
        help="Path to tokenizer.json relative to repo root or absolute.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="qwen3_01b/runs/stage2_16k_seq16384_rope_*/checkpoint_last.pt",
        help="Glob for checkpoints, relative to repo root.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Explicit checkpoint in name=path form. Can be repeated. Overrides --checkpoint-glob when set.",
    )
    parser.add_argument("--output-dir", default="analysis/attention_collapse/forced_answer_probe_v3")
    parser.add_argument(
        "--examples-jsonl",
        default="analysis/attention_collapse/position_probe_v1/examples.jsonl",
        help="Existing examples.jsonl to reuse. If absent, examples are rebuilt from context/position args.",
    )
    parser.add_argument("--context-lengths", nargs="*", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--positions", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method name filter.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--answer-prefix",
        default=" ",
        help="Prefix prepended before the gold answer for teacher-forced scoring. Default is one space after 'Answer:'.",
    )
    parser.add_argument(
        "--attention-scope",
        choices=["first", "last", "all"],
        default="all",
        help="Which forced answer token positions to average attention over.",
    )
    parser.add_argument(
        "--write-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write lightweight SVG summary figures.",
    )
    parser.add_argument(
        "--dtype-autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 autocast on CUDA for model forward.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def maybe_autocast(device: torch.device, enabled: bool):
    return gen.maybe_autocast(device, enabled)


def load_or_build_examples(tokenizer: Any, args: argparse.Namespace) -> list[Any]:
    if args.examples_jsonl:
        path = resolve_path(args.examples_jsonl)
        if path.is_file():
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(gen.ProbeExample(**json.loads(line)))
            return rows[: args.max_examples] if args.max_examples is not None else rows

    examples = []
    for context_length in args.context_lengths:
        for position in args.positions:
            if not 0.0 <= position <= 1.0:
                raise ValueError(f"Position must be in [0, 1], got {position}")
            for sample_index in range(args.samples_per_cell):
                examples.append(
                    gen.build_probe_example(
                        tokenizer,
                        seed=args.seed,
                        context_length=context_length,
                        position=position,
                        sample_index=sample_index,
                    )
                )
    return examples[: args.max_examples] if args.max_examples is not None else examples


def teacher_forced_score_and_attention(
    *,
    model: Any,
    tokenizer: Any,
    collector: Any,
    example: Any,
    answer_text: str,
    attention_scope: str,
    device: torch.device,
    dtype_autocast: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt_ids = tokenizer.encode(example.prompt)
    answer_ids = tokenizer.encode(answer_text)
    if not answer_ids:
        raise ValueError(f"Gold answer tokenized to no tokens for {example.example_id!r}")

    input_ids_list = prompt_ids + answer_ids
    context_length = int(model.cfg["context_length"])
    if len(input_ids_list) > context_length:
        raise ValueError(
            f"{example.example_id} prompt+answer has {len(input_ids_list)} tokens, "
            f"exceeding context_length={context_length}"
        )

    if attention_scope == "first":
        query_position = len(prompt_ids)
        query_window = 1
    elif attention_scope == "last":
        query_position = len(input_ids_list) - 1
        query_window = 1
    else:
        query_position = len(input_ids_list) - 1
        query_window = len(answer_ids)

    collector.query_window = query_window
    collector.begin(
        evidence_start=example.evidence_token_start,
        evidence_end=example.evidence_token_end,
        query_position=query_position,
    )

    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    start = time.perf_counter()
    try:
        with torch.inference_mode(), maybe_autocast(device, dtype_autocast):
            logits = model(input_ids)
        latency = time.perf_counter() - start
        selected_logits = logits[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(answer_ids), :].float()
        target_ids = torch.tensor(answer_ids, dtype=torch.long, device=selected_logits.device)
        log_probs = F.log_softmax(selected_logits, dim=-1)
        token_logprobs_tensor = log_probs.gather(1, target_ids[:, None]).squeeze(1)
        token_logprobs = [float(x) for x in token_logprobs_tensor.detach().cpu()]
        token_nlls = [-x for x in token_logprobs]
    finally:
        metric_rows = [dict(row) for row in collector.rows]
        collector.end()

    sum_logprob = float(sum(token_logprobs))
    mean_nll = float(sum(token_nlls) / len(token_nlls))
    # Clamp only for numeric display; mean_nll should be modest for sane logits.
    ppl = float(math.exp(min(80.0, mean_nll)))
    score_row = {
        "example_id": example.example_id,
        "context_length_target": example.context_length_target,
        "position_ratio": example.position_ratio,
        "sample_index": example.sample_index,
        "answer": example.answer,
        "forced_answer_text": answer_text,
        "answer_prefix": answer_text[: max(0, len(answer_text) - len(example.answer))],
        "prompt_tokens": len(prompt_ids),
        "answer_token_count": len(answer_ids),
        "analysis_seq_len": len(input_ids_list),
        "attention_scope": attention_scope,
        "attention_query_start": min(row["query_start"] for row in metric_rows) if metric_rows else query_position,
        "attention_query_end": max(row["query_end"] for row in metric_rows) if metric_rows else query_position + 1,
        "answer_token_ids": answer_ids,
        "answer_token_texts": [tokenizer.decode([token_id]) for token_id in answer_ids],
        "answer_token_logprobs": token_logprobs,
        "answer_token_nlls": token_nlls,
        "answer_sum_logprob": sum_logprob,
        "answer_mean_nll": mean_nll,
        "answer_ppl": ppl,
        "latency_sec": latency,
    }
    for row in metric_rows:
        row.update(
            {
                "example_id": example.example_id,
                "context_length_target": example.context_length_target,
                "position_ratio": example.position_ratio,
                "sample_index": example.sample_index,
                "answer": example.answer,
                "forced_answer_text": answer_text,
                "answer_token_count": len(answer_ids),
                "answer_mean_nll": mean_nll,
                "answer_sum_logprob": sum_logprob,
                "answer_ppl": ppl,
                "attention_scope": attention_scope,
                "prompt_tokens": len(prompt_ids),
                "analysis_seq_len": len(input_ids_list),
                "evidence_token_start": example.evidence_token_start,
                "evidence_token_end": example.evidence_token_end,
            }
        )
    return score_row, metric_rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_keys() -> list[str]:
    return [
        "entropy",
        "entropy_norm",
        "top1_mass",
        "top5_mass",
        "sink_mass_first16",
        "local_mass_128",
        "local_mass_512",
        "local_mass_2048",
        "far_mass_4096_plus",
        "far_mass_8192_plus",
        "evidence_mass",
        "effective_span_80",
    ]


def add_collapse_score(record: dict[str, Any]) -> None:
    record["collapse_score"] = (
        (1.0 - float(record["entropy_norm"]))
        + float(record["top5_mass"])
        + float(record["sink_mass_first16"])
    ) / 3.0


def aggregate_attention_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(group_keys, key_values)}
        record["n"] = len(group)
        for numeric_key in metric_keys() + ["answer_mean_nll", "answer_sum_logprob", "answer_ppl"]:
            vals = [
                float(row[numeric_key])
                for row in group
                if numeric_key in row and not math.isnan(float(row[numeric_key]))
            ]
            record[numeric_key] = sum(vals) / len(vals) if vals else float("nan")
        add_collapse_score(record)
        out.append(record)
    return out


def aggregate_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["context_length_target"], row["position_ratio"])].append(row)

    out: list[dict[str, Any]] = []
    for (method, length, position), group in sorted(groups.items(), key=lambda item: item[0]):
        n = len(group)
        mean_nll = sum(float(row["answer_mean_nll"]) for row in group) / n
        mean_sum_logprob = sum(float(row["answer_sum_logprob"]) for row in group) / n
        record = {
            "method": method,
            "context_length_target": length,
            "position_ratio": position,
            "n": n,
            "mean_answer_mean_nll": mean_nll,
            "mean_answer_sum_logprob": mean_sum_logprob,
            "mean_answer_ppl": math.exp(min(80.0, mean_nll)),
            "mean_answer_token_count": sum(int(row["answer_token_count"]) for row in group) / n,
            "mean_latency_sec": sum(float(row["latency_sec"]) for row in group) / n,
        }
        out.append(record)
    return out


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    escaped = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{escaped}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def write_line_plot(
    path: Path,
    *,
    title: str,
    subtitle: str,
    y_label: str,
    methods: list[str],
    positions: list[float],
    values: dict[tuple[str, float], float],
    lower_is_better: bool = False,
) -> None:
    width, height = 920, 540
    margin = {"left": 82, "right": 190, "top": 74, "bottom": 74}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#f97316"]
    vals = [float(v) for v in values.values() if not math.isnan(float(v))]
    y_min = min(vals + [0.0])
    y_max = max(vals + [1.0])
    if lower_is_better:
        y_min = min(vals) if vals else 0.0
        pad = max(0.01, (y_max - y_min) * 0.08)
        y_min = max(0.0, y_min - pad)
        y_max = y_max + pad

    def x_scale(pos: float) -> float:
        return margin["left"] + pos * plot_w

    def y_scale(value: float) -> float:
        if y_max <= y_min:
            return margin["top"] + plot_h / 2
        return margin["top"] + (1.0 - (value - y_min) / (y_max - y_min)) * plot_h

    body = [
        svg_text(width / 2, 32, title, size=18, anchor="middle", weight="700"),
        svg_text(width / 2, 54, subtitle, size=11, anchor="middle"),
    ]
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        y = y_scale(value)
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{value:.3g}", size=10, anchor="end"))
    for pos in positions:
        x = x_scale(pos)
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 24, f"{int(pos * 100)}%", size=10, anchor="middle"))
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    body.append(svg_text(margin["left"] + plot_w / 2, height - 22, "Evidence position", size=12, anchor="middle", weight="700"))
    body.append(svg_text(18, margin["top"] + plot_h / 2, y_label, size=12, anchor="middle", weight="700"))

    for index, method in enumerate(methods):
        color = colors[index % len(colors)]
        points = [(x_scale(pos), y_scale(values[(method, pos)])) for pos in positions if (method, pos) in values]
        if points:
            body.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + f'" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
            )
        for x, y in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = margin["top"] + 8 + index * 24
        legend_x = width - margin["right"] + 28
        body.append(f'<line x1="{legend_x}" y1="{legend_y:.1f}" x2="{legend_x + 26}" y2="{legend_y:.1f}" stroke="{color}" stroke-width="2.5"/>')
        body.append(svg_text(legend_x + 36, legend_y + 4, method, size=11))

    write_svg(path, width, height, body)


def write_figures(score_summary: list[dict[str, Any]], attention_summary: list[dict[str, Any]], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    by_length_scores: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in score_summary:
        by_length_scores[int(row["context_length_target"])].append(row)
    for length, rows in sorted(by_length_scores.items()):
        methods = sorted({str(row["method"]) for row in rows})
        positions = sorted({float(row["position_ratio"]) for row in rows})
        nll_values = {(str(row["method"]), float(row["position_ratio"])): float(row["mean_answer_mean_nll"]) for row in rows}
        write_line_plot(
            figures_dir / f"forced_answer_nll_by_position_len{length}.svg",
            title=f"Forced Gold Answer NLL by Position ({length} tokens)",
            subtitle="Lower is better; teacher-forced score of the gold answer continuation",
            y_label="Mean NLL",
            methods=methods,
            positions=positions,
            values=nll_values,
            lower_is_better=True,
        )

    by_length_attention: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attention_summary:
        by_length_attention[int(row["context_length_target"])].append(row)
    for length, rows in sorted(by_length_attention.items()):
        methods = sorted({str(row["method"]) for row in rows})
        positions = sorted({float(row["position_ratio"]) for row in rows})
        evidence_values = {(str(row["method"]), float(row["position_ratio"])): float(row["evidence_mass"]) for row in rows}
        write_line_plot(
            figures_dir / f"forced_answer_evidence_mass_by_position_len{length}.svg",
            title=f"Forced Answer Evidence Mass by Position ({length} tokens)",
            subtitle="Attention from teacher-forced gold answer tokens to evidence span",
            y_label="Evidence mass",
            methods=methods,
            positions=positions,
            values=evidence_values,
        )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "forced_answer_scores.jsonl"
    metrics_path = output_dir / "forced_answer_attention_metrics.jsonl"
    for path in [scores_path, metrics_path]:
        if path.exists():
            path.unlink()

    tokenizer_path = resolve_path(args.tokenizer_json)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    examples = load_or_build_examples(tokenizer, args)
    checkpoints = gen.discover_checkpoints(args)
    device = get_device(None if args.device == "auto" else args.device)

    write_json(
        output_dir / "run_config.json",
        vars(args)
        | {
            "resolved_tokenizer_json": str(tokenizer_path),
            "device_resolved": str(device),
            "analysis_query": f"teacher_forced_answer_tokens:{args.attention_scope}",
        },
    )
    write_jsonl(output_dir / "examples.jsonl", [gen.asdict(example) for example in examples])
    print(f"Built/loaded {len(examples)} probe examples.")
    print(f"Selected checkpoints: {', '.join(name for name, _ in checkpoints)}")
    print(f"Writing scores to {scores_path}")
    print(f"Writing attention metrics to {metrics_path}")

    all_score_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []

    for method, checkpoint_path in checkpoints:
        print(f"\n=== Loading {method}: {checkpoint_path} ===", flush=True)
        model = build_model_from_checkpoint(checkpoint_path, device)
        model.eval()
        collector = gen.AttentionSummaryCollector(evidence_start=None, evidence_end=None, query_window=1)
        gen.attach_attention_summary_hooks(model, collector)

        for index, example in enumerate(examples, start=1):
            answer_text = f"{args.answer_prefix}{example.answer}"
            with torch.inference_mode():
                score_row, metric_rows = teacher_forced_score_and_attention(
                    model=model,
                    tokenizer=tokenizer,
                    collector=collector,
                    example=example,
                    answer_text=answer_text,
                    attention_scope=args.attention_scope,
                    device=device,
                    dtype_autocast=args.dtype_autocast,
                )

            score_row.update({"method": method, "checkpoint": str(checkpoint_path)})
            append_jsonl(scores_path, score_row)
            all_score_rows.append(score_row)

            for row in metric_rows:
                row.update({"method": method, "checkpoint": str(checkpoint_path)})
                append_jsonl(metrics_path, row)
            all_metric_rows.extend(metric_rows)

            print(
                f"[{method}] {index}/{len(examples)} {example.example_id} "
                f"tokens={score_row['prompt_tokens']} answer={example.answer} "
                f"answer_tokens={score_row['answer_token_count']} mean_nll={score_row['answer_mean_nll']:.4f} "
                f"ppl={score_row['answer_ppl']:.2f} rows={len(metric_rows)} time={score_row['latency_sec']:.2f}s",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    score_summary = aggregate_score_rows(all_score_rows)
    layer_head_summary = aggregate_attention_rows(
        all_metric_rows,
        group_keys=("method", "context_length_target", "position_ratio", "layer", "head"),
    )
    method_position_summary = aggregate_attention_rows(
        all_metric_rows,
        group_keys=("method", "context_length_target", "position_ratio"),
    )
    write_csv(output_dir / "forced_answer_score_summary.csv", score_summary)
    write_csv(output_dir / "forced_answer_layer_head_summary.csv", layer_head_summary)
    write_csv(output_dir / "forced_answer_method_position_summary.csv", method_position_summary)
    if args.write_figures:
        write_figures(score_summary, method_position_summary, output_dir)

    print(f"\nWrote examples: {output_dir / 'examples.jsonl'}")
    print(f"Wrote forced answer scores: {scores_path}")
    print(f"Wrote forced answer attention metrics: {metrics_path}")
    print(f"Wrote score summary: {output_dir / 'forced_answer_score_summary.csv'}")
    print(f"Wrote layer/head summary: {output_dir / 'forced_answer_layer_head_summary.csv'}")
    print(f"Wrote method/position summary: {output_dir / 'forced_answer_method_position_summary.csv'}")
    if args.write_figures:
        print(f"Wrote figures under: {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
