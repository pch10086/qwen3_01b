#!/usr/bin/env python3
"""Analysis-only attention collapse probes for self-trained qwen3_01b checkpoints.

This script intentionally avoids changing the model implementation used for
training. It loads checkpoints normally, wraps each attention module at runtime,
and computes small attention summaries for selected query positions while the
model still uses its configured attention implementation for the actual forward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen3_01b.rope import apply_rope
from qwen3_01b.tokenizer_utils import load_tokenizer_from_json
from qwen3_01b.training import build_model_from_checkpoint, get_device


FILLER_SENTENCES = [
    "The archive entry describes routine maintenance notes and ordinary status updates.",
    "Several unrelated project summaries are listed here to create a realistic document.",
    "This paragraph contains background information that is not useful for the question.",
    "The committee reviewed timelines, budgets, and staffing plans in a neutral report.",
    "A short memo records meeting logistics, room assignments, and minor schedule changes.",
    "The document includes generic operational details that should be ignored by the reader.",
]

PROJECT_NAMES = [
    "Aurora",
    "Beacon",
    "Cobalt",
    "Delta",
    "Ember",
    "Falcon",
    "Granite",
    "Harbor",
    "Ion",
    "Juniper",
    "Keystone",
    "Lumen",
]


@dataclass
class ProbeExample:
    example_id: str
    context_length_target: int
    position_ratio: float
    sample_index: int
    project: str
    answer: str
    prompt: str
    evidence: str
    actual_prompt_tokens: int
    evidence_token_start: int
    evidence_token_end: int


class AttentionSummaryCollector:
    def __init__(
        self,
        *,
        evidence_start: int | None,
        evidence_end: int | None,
        query_window: int,
        local_windows: tuple[int, ...] = (128, 512, 2048),
        far_distances: tuple[int, ...] = (4096, 8192),
        sink_tokens: int = 16,
        topk: tuple[int, ...] = (1, 5),
    ) -> None:
        self.evidence_start = evidence_start
        self.evidence_end = evidence_end
        self.query_window = query_window
        self.local_windows = local_windows
        self.far_distances = far_distances
        self.sink_tokens = sink_tokens
        self.topk = topk
        self.rows: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.rows.clear()

    def add_layer_summary(
        self,
        *,
        layer: int,
        attn: torch.Tensor,
        query_positions: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Record metrics from attention weights shaped [1, heads, queries, seq]."""
        weights = attn.detach().float().cpu()[0]
        qpos = query_positions.detach().cpu().tolist()
        num_heads = weights.shape[0]

        for head in range(num_heads):
            head_weights = weights[head]
            query_metrics = [
                compute_query_metrics(
                    probs=head_weights[q_index],
                    query_position=int(query_position),
                    seq_len=seq_len,
                    evidence_start=self.evidence_start,
                    evidence_end=self.evidence_end,
                    local_windows=self.local_windows,
                    far_distances=self.far_distances,
                    sink_tokens=self.sink_tokens,
                    topk=self.topk,
                )
                for q_index, query_position in enumerate(qpos)
            ]
            row = average_metric_dicts(query_metrics)
            row.update(
                {
                    "layer": layer,
                    "head": head,
                    "num_queries": len(query_metrics),
                    "query_start": int(min(qpos)),
                    "query_end": int(max(qpos)) + 1,
                    "seq_len": seq_len,
                }
            )
            self.rows.append(row)


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
    parser.add_argument("--output-dir", default="analysis/attention_collapse/position_probe_v1")
    parser.add_argument("--context-lengths", nargs="*", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--positions", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--samples-per-cell", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-window", type=int, default=1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit examples after construction; useful for smoke tests.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method name filter, e.g. rope_none rope_linear.",
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


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def make_answer(seed: int, context_length: int, position: float, sample_index: int) -> str:
    rng = random.Random((seed * 1000003) + (context_length * 101) + int(position * 1000) + sample_index)
    return f"{rng.randint(10000, 99999)}"


def build_prompt(prefix: str, evidence: str, suffix: str, project: str) -> str:
    return (
        "You are given a document. Answer the question using only the document. "
        "Return only the access code digits.\n\n"
        "Document:\n"
        f"{prefix}"
        f"{evidence}\n"
        f"{suffix}"
        "\nQuestion: "
        f"What is the access code for Project {project}?\n"
        "Answer:"
    )


def build_prompt_before_evidence(prefix: str) -> str:
    return (
        "You are given a document. Answer the question using only the document. "
        "Return only the access code digits.\n\n"
        "Document:\n"
        f"{prefix}"
    )


def grow_filler_to_tokens(tokenizer: Any, target_tokens: int, start_offset: int = 0) -> str:
    if target_tokens <= 0:
        return ""
    chunks: list[str] = []
    index = start_offset
    total_tokens = 0
    while total_tokens < target_tokens:
        sentence = FILLER_SENTENCES[index % len(FILLER_SENTENCES)]
        chunk = f"{sentence} Reference item {index:05d}.\n"
        chunks.append(chunk)
        total_tokens += token_count(tokenizer, chunk)
        index += 1
    return "".join(chunks)


def trim_text_to_token_budget(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_probe_example(
    tokenizer: Any,
    *,
    seed: int,
    context_length: int,
    position: float,
    sample_index: int,
) -> ProbeExample:
    project_index = (sample_index + int(position * 10) + context_length) % len(PROJECT_NAMES)
    project = PROJECT_NAMES[project_index]
    answer = make_answer(seed, context_length, position, sample_index)
    evidence = f"The access code for Project {project} is {answer}."

    empty_prompt = build_prompt("", evidence, "", project)
    fixed_tokens = token_count(tokenizer, empty_prompt)
    filler_budget = max(0, context_length - fixed_tokens)
    prefix_budget = int(math.floor(filler_budget * position))
    suffix_budget = max(0, filler_budget - prefix_budget)

    prefix = grow_filler_to_tokens(tokenizer, prefix_budget, start_offset=sample_index * 1000)
    prefix = trim_text_to_token_budget(tokenizer, prefix, prefix_budget)
    suffix = grow_filler_to_tokens(tokenizer, suffix_budget, start_offset=sample_index * 1000 + 500000)
    suffix = trim_text_to_token_budget(tokenizer, suffix, suffix_budget)

    prompt = build_prompt(prefix, evidence, suffix, project)
    evidence_token_start = token_count(tokenizer, build_prompt_before_evidence(prefix))
    evidence_token_end = evidence_token_start + token_count(tokenizer, evidence)

    return ProbeExample(
        example_id=f"len{context_length}_pos{position:.2f}_sample{sample_index}",
        context_length_target=context_length,
        position_ratio=position,
        sample_index=sample_index,
        project=project,
        answer=answer,
        prompt=prompt,
        evidence=evidence,
        actual_prompt_tokens=token_count(tokenizer, prompt),
        evidence_token_start=evidence_token_start,
        evidence_token_end=evidence_token_end,
    )


def build_examples(tokenizer: Any, args: argparse.Namespace) -> list[ProbeExample]:
    examples: list[ProbeExample] = []
    for context_length in args.context_lengths:
        for position in args.positions:
            if not 0.0 <= position <= 1.0:
                raise ValueError(f"Position must be in [0, 1], got {position}")
            for sample_index in range(args.samples_per_cell):
                examples.append(
                    build_probe_example(
                        tokenizer,
                        seed=args.seed,
                        context_length=context_length,
                        position=position,
                        sample_index=sample_index,
                    )
                )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    return examples


def discover_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.checkpoint:
        pairs: list[tuple[str, Path]] = []
        for value in args.checkpoint:
            if "=" not in value:
                raise ValueError("--checkpoint must use name=path form")
            name, raw_path = value.split("=", 1)
            pairs.append((name, resolve_path(raw_path)))
    else:
        pairs = []
        for path in sorted(ROOT.glob(args.checkpoint_glob)):
            method = path.parent.name
            if method.startswith("stage2_16k_seq16384_"):
                method = method.removeprefix("stage2_16k_seq16384_")
            pairs.append((method, path))

    if args.methods:
        wanted = set(args.methods)
        pairs = [(name, path) for name, path in pairs if name in wanted]
    if not pairs:
        raise FileNotFoundError("No checkpoints selected.")
    for _, path in pairs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    return pairs


def compute_query_metrics(
    *,
    probs: torch.Tensor,
    query_position: int,
    seq_len: int,
    evidence_start: int | None,
    evidence_end: int | None,
    local_windows: tuple[int, ...],
    far_distances: tuple[int, ...],
    sink_tokens: int,
    topk: tuple[int, ...],
) -> dict[str, float]:
    valid_len = query_position + 1
    valid = probs[:valid_len]
    valid = valid / valid.sum().clamp_min(1e-20)
    entropy = -(valid * valid.clamp_min(1e-20).log()).sum().item()
    entropy_norm = entropy / math.log(max(valid_len, 2))

    metrics: dict[str, float] = {
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "sink_mass_first16": valid[: min(sink_tokens, valid_len)].sum().item(),
    }

    for k in topk:
        kk = min(k, valid_len)
        metrics[f"top{k}_mass"] = torch.topk(valid, kk).values.sum().item()

    for window in local_windows:
        start = max(0, query_position - window + 1)
        metrics[f"local_mass_{window}"] = probs[start : query_position + 1].sum().item()

    for distance in far_distances:
        end = query_position - distance + 1
        metrics[f"far_mass_{distance}_plus"] = probs[: max(0, end)].sum().item()

    if evidence_start is None or evidence_end is None:
        metrics["evidence_mass"] = float("nan")
    else:
        start = max(0, evidence_start)
        end = min(valid_len, evidence_end)
        metrics["evidence_mass"] = probs[start:end].sum().item() if end > start else 0.0

    reversed_valid = torch.flip(valid, dims=[0])
    cumulative = torch.cumsum(reversed_valid, dim=0)
    span_hits = torch.nonzero(cumulative >= 0.8, as_tuple=False)
    metrics["effective_span_80"] = float(span_hits[0].item() + 1 if len(span_hits) else valid_len)
    return metrics


def average_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in rows]
        vals = [v for v in vals if not math.isnan(float(v))]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def make_attention_wrapper(original_forward: Any, layer_idx: int, collector: AttentionSummaryCollector):
    def wrapped(self: Any, x: torch.Tensor, mask: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor, past_kv=None, use_cache=False, position_offset=0):
        if past_kv is None and not use_cache and x.shape[0] == 1:
            with torch.no_grad():
                summarize_attention_layer(
                    module=self,
                    x=x,
                    mask=mask,
                    cos=cos,
                    sin=sin,
                    layer_idx=layer_idx,
                    collector=collector,
                    position_offset=position_offset,
                )
        return original_forward(x, mask, cos, sin, past_kv=past_kv, use_cache=use_cache, position_offset=position_offset)

    return wrapped


def summarize_attention_layer(
    *,
    module: Any,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_idx: int,
    collector: AttentionSummaryCollector,
    position_offset: int,
) -> None:
    bsz, num_tokens, _ = x.shape
    if bsz != 1:
        raise ValueError("Attention summary currently expects batch size 1.")

    queries = module.W_query(x)
    keys = module.W_key(x)

    queries = queries.view(bsz, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
    keys = keys.view(bsz, num_tokens, module.num_kv_groups, module.head_dim).transpose(1, 2)

    if module.q_norm:
        queries = module.q_norm(queries)
    if module.k_norm:
        keys = module.k_norm(keys)

    queries = apply_rope(queries, cos, sin, position_offset=position_offset)
    keys = apply_rope(keys, cos, sin, position_offset=position_offset)
    keys = keys.repeat_interleave(module.group_size, dim=1)

    query_window = max(1, min(collector.query_window, num_tokens))
    query_positions = torch.arange(num_tokens - query_window, num_tokens, device=x.device)
    q = queries.index_select(dim=2, index=query_positions)
    scores = q @ keys.transpose(2, 3)
    scores = scores / math.sqrt(module.head_dim)

    key_positions = torch.arange(num_tokens, device=x.device)
    causal_mask = key_positions.view(1, 1, 1, num_tokens) > query_positions.view(1, 1, query_window, 1)
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    attn = torch.softmax(scores.float(), dim=-1)
    collector.add_layer_summary(
        layer=layer_idx,
        attn=attn,
        query_positions=query_positions,
        seq_len=num_tokens,
    )


def attach_attention_summary_hooks(model: Any, collector: AttentionSummaryCollector) -> None:
    for layer_idx, block in enumerate(model.trf_blocks):
        att = block.att
        original = att.forward
        att.forward = MethodType(make_attention_wrapper(original, layer_idx, collector), att)


def run_model_example(
    *,
    model: Any,
    tokenizer: Any,
    collector: AttentionSummaryCollector,
    example: ProbeExample,
    device: torch.device,
    dtype_autocast: bool,
) -> list[dict[str, Any]]:
    ids = tokenizer.encode(example.prompt)
    context_length = int(model.cfg["context_length"])
    if len(ids) > context_length:
        raise ValueError(
            f"{example.example_id} has {len(ids)} tokens, exceeding checkpoint context_length={context_length}"
        )
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    collector.evidence_start = example.evidence_token_start
    collector.evidence_end = example.evidence_token_end
    collector.clear()

    if dtype_autocast and device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(input_ids)
    else:
        _ = model(input_ids)

    return [dict(row) for row in collector.rows]


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


def aggregate_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    numeric_keys = [
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
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(group_keys, key_values)}
        record["n"] = len(group)
        for numeric_key in numeric_keys:
            vals = [float(row[numeric_key]) for row in group if numeric_key in row and not math.isnan(float(row[numeric_key]))]
            record[numeric_key] = sum(vals) / len(vals) if vals else float("nan")
        record["collapse_score"] = (
            (1.0 - float(record["entropy_norm"]))
            + float(record["top5_mass"])
            + float(record["sink_mass_first16"])
        ) / 3.0
        out.append(record)
    return out


def fmt_num(value: float) -> str:
    if math.isnan(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def write_text_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{escaped}</text>'
    )


def heat_color(value: float, lo: float, hi: float) -> str:
    if math.isnan(float(value)):
        return "#e5e7eb"
    value = max(lo, min(hi, float(value)))
    t = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    # White -> orange -> red for higher collapse-like values.
    r1, g1, b1 = 255, 247, 237
    r2, g2, b2 = 220, 38, 38
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def write_layer_head_heatmaps(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = {
        "entropy_collapse": ("entropy_norm", lambda v: 1.0 - v, 0.0, 1.0, "1 - normalized entropy"),
        "sink_mass": ("sink_mass_first16", lambda v: v, 0.0, 1.0, "Attention mass on first 16 tokens"),
        "evidence_mass": ("evidence_mass", lambda v: v, 0.0, 1.0, "Attention mass on evidence span"),
    }
    by_panel: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_panel[(str(row["method"]), int(row["context_length_target"]))].append(row)

    for (method, length), rows in by_panel.items():
        layers = sorted({int(row["layer"]) for row in rows})
        heads = sorted({int(row["head"]) for row in rows})
        if not layers or not heads:
            continue
        values_by_lh = {(int(row["layer"]), int(row["head"])): row for row in rows}
        cell_w, cell_h = 42, 30
        left, top = 80, 80
        width = left + len(heads) * cell_w + 36
        height = top + len(layers) * cell_h + 60
        for metric_name, (source_key, transform, lo, hi, title) in metrics.items():
            body: list[str] = [
                svg_text(width / 2, 30, f"{method} {length}: {title}", size=18, anchor="middle", weight="700"),
                svg_text(width / 2, 52, "rows = layer, columns = head", size=11, anchor="middle"),
            ]
            for j, head in enumerate(heads):
                body.append(svg_text(left + j * cell_w + cell_w / 2, top - 14, str(head), size=10, anchor="middle", weight="700"))
            for i, layer in enumerate(layers):
                y = top + i * cell_h
                body.append(svg_text(left - 12, y + cell_h / 2 + 4, str(layer), size=10, anchor="end", weight="700"))
                for j, head in enumerate(heads):
                    x = left + j * cell_w
                    row = values_by_lh.get((layer, head))
                    raw_value = float(row[source_key]) if row else float("nan")
                    value = transform(raw_value) if not math.isnan(raw_value) else raw_value
                    body.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                        f'fill="{heat_color(value, lo, hi)}" stroke="white" stroke-width="1"/>'
                    )
                    body.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, f"{value:.2f}" if not math.isnan(value) else "na", size=9, anchor="middle"))
            write_text_svg(output_dir / f"{metric_name}_{method}_len{length}.svg", width, height, body)


def write_method_position_plot(method_rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        by_length[int(row["context_length_target"])].append(row)

    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#f97316"]
    for length, rows in sorted(by_length.items()):
        methods = sorted({str(row["method"]) for row in rows})
        positions = sorted({float(row["position_ratio"]) for row in rows})
        values = {(str(row["method"]), float(row["position_ratio"])): float(row["evidence_mass"]) for row in rows}

        width, height = 900, 520
        margin = {"left": 74, "right": 180, "top": 70, "bottom": 72}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        def x_scale(pos: float) -> float:
            return margin["left"] + pos * plot_w

        def y_scale(value: float) -> float:
            return margin["top"] + (1.0 - max(0.0, min(1.0, value))) * plot_h

        body = [
            svg_text(width / 2, 32, f"Evidence Attention Mass by Position ({length} tokens)", size=18, anchor="middle", weight="700"),
            svg_text(width / 2, 54, "Mean across layers, heads, and samples; higher means answer query attends to evidence span", size=11, anchor="middle"),
        ]
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = y_scale(tick)
            body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            body.append(svg_text(margin["left"] - 10, y + 4, f"{tick:.2f}", size=10, anchor="end"))
        for pos in positions:
            x = x_scale(pos)
            body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="#f3f4f6" stroke-width="1"/>')
            body.append(svg_text(x, margin["top"] + plot_h + 24, f"{int(pos * 100)}%", size=10, anchor="middle"))
        body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.2"/>')
        body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.2"/>')
        body.append(svg_text(margin["left"] + plot_w / 2, height - 22, "Evidence position", size=12, anchor="middle", weight="700"))
        body.append(svg_text(18, margin["top"] + plot_h / 2, "Evidence mass", size=12, anchor="middle", weight="700"))

        for index, method in enumerate(methods):
            color = colors[index % len(colors)]
            points = [(x_scale(pos), y_scale(values.get((method, pos), float("nan")))) for pos in positions if (method, pos) in values]
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

        write_text_svg(output_dir / f"evidence_mass_by_position_len{length}.svg", width, height, body)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    tokenizer_path = resolve_path(args.tokenizer_json)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    examples = build_examples(tokenizer, args)
    checkpoints = discover_checkpoints(args)
    device = get_device(None if args.device == "auto" else args.device)

    write_json(output_dir / "run_config.json", vars(args) | {"resolved_tokenizer_json": str(tokenizer_path), "device_resolved": str(device)})
    write_jsonl(output_dir / "examples.jsonl", [asdict(example) for example in examples])
    print(f"Built {len(examples)} probe examples.")
    print(f"Selected checkpoints: {', '.join(name for name, _ in checkpoints)}")
    print(f"Writing metrics to {metrics_path}")

    all_rows: list[dict[str, Any]] = []
    for method, checkpoint_path in checkpoints:
        print(f"\n=== Loading {method}: {checkpoint_path} ===", flush=True)
        model = build_model_from_checkpoint(checkpoint_path, device)
        model.eval()
        collector = AttentionSummaryCollector(
            evidence_start=None,
            evidence_end=None,
            query_window=args.query_window,
        )
        attach_attention_summary_hooks(model, collector)

        with torch.inference_mode():
            for index, example in enumerate(examples, start=1):
                start = time.perf_counter()
                rows = run_model_example(
                    model=model,
                    tokenizer=tokenizer,
                    collector=collector,
                    example=example,
                    device=device,
                    dtype_autocast=args.dtype_autocast,
                )
                elapsed = time.perf_counter() - start
                for row in rows:
                    row.update(
                        {
                            "method": method,
                            "checkpoint": str(checkpoint_path),
                            "example_id": example.example_id,
                            "context_length_target": example.context_length_target,
                            "position_ratio": example.position_ratio,
                            "sample_index": example.sample_index,
                            "answer": example.answer,
                            "actual_prompt_tokens": example.actual_prompt_tokens,
                            "evidence_token_start": example.evidence_token_start,
                            "evidence_token_end": example.evidence_token_end,
                        }
                    )
                    append_jsonl(metrics_path, row)
                all_rows.extend(rows)
                print(
                    f"[{method}] {index}/{len(examples)} {example.example_id} "
                    f"tokens={example.actual_prompt_tokens} rows={len(rows)} time={elapsed:.2f}s",
                    flush=True,
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    layer_head_summary = aggregate_rows(
        all_rows,
        group_keys=("method", "context_length_target", "position_ratio", "layer", "head"),
    )
    method_summary = aggregate_rows(
        all_rows,
        group_keys=("method", "context_length_target", "position_ratio"),
    )
    write_csv(output_dir / "layer_head_summary.csv", layer_head_summary)
    write_csv(output_dir / "method_position_summary.csv", method_summary)
    if args.write_figures:
        figures_dir = output_dir / "figures"
        write_layer_head_heatmaps(layer_head_summary, figures_dir)
        write_method_position_plot(method_summary, figures_dir)

    print(f"\nWrote examples: {output_dir / 'examples.jsonl'}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote layer/head summary: {output_dir / 'layer_head_summary.csv'}")
    print(f"Wrote method/position summary: {output_dir / 'method_position_summary.csv'}")
    if args.write_figures:
        print(f"Wrote figures under: {output_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
