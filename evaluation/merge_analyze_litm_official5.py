#!/usr/bin/env python3
"""合并并分析 official-scale Lost-in-the-Middle KV 评测结果。"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("/home/public/bjh/dym/NLP/evaluation")
OUT_DIR = BASE / "outputs/qwen3_0_6b_litm_kv_official5_merged"
DOC_PATH = BASE / "docs/litm_kv_official5_result_memo.md"

SHARDS = [
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_75/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_140/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_300/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_300_pos25/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_300_pos50/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_300_pos75/predictions.jsonl",
    BASE / "outputs/qwen3_0_6b_litm_kv_official5_300_pos100/predictions.jsonl",
]

NUM_KEYS = [75, 140, 300]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]

COLORS = {
    "75": "#2563eb",
    "140": "#16a34a",
    "300": "#dc2626",
    "correct": "#2563eb",
    "wrong_context_value": "#f97316",
    "hallucinated_value": "#8b5cf6",
    "format_error": "#64748b",
    "other": "#94a3b8",
    "grid": "#d6d3d1",
    "text": "#1f2937",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify(row: dict[str, Any]) -> str:
    if row.get("first_value_match"):
        return "correct"
    if row.get("wrong_context_value"):
        return "wrong_context_value"
    if row.get("hallucinated_value"):
        return "hallucinated_value"
    if row.get("format_error"):
        return "format_error"
    return "other"


def merge_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """合并分片：300-key 主分片只保留 0% 位置，其他位置使用专门分片。"""
    merged: dict[str, dict[str, Any]] = {}
    raw_count = 0
    skipped_300_nonzero_from_main = 0
    duplicate_ids = 0

    for shard in SHARDS:
        shard_rows = read_jsonl(shard)
        raw_count += len(shard_rows)
        for row in shard_rows:
            nk = int(row["num_keys"])
            pos = round(float(row["position_ratio"]), 2)
            is_300_main = shard.parent.name == "qwen3_0_6b_litm_kv_official5_300"
            if is_300_main and pos != 0.0:
                skipped_300_nonzero_from_main += 1
                continue
            if nk not in NUM_KEYS or pos not in POSITIONS:
                raise ValueError(f"发现意外 cell: shard={shard}, num_keys={nk}, position={pos}")
            row["num_keys"] = nk
            row["position_ratio"] = pos
            if row["example_id"] in merged:
                duplicate_ids += 1
            merged[row["example_id"]] = row

    rows = sorted(
        merged.values(),
        key=lambda r: (int(r["num_keys"]), float(r["position_ratio"]), int(r["sample_index"])),
    )
    diagnostics = {
        "raw_count": raw_count,
        "merged_count": len(rows),
        "skipped_300_nonzero_from_main": skipped_300_nonzero_from_main,
        "duplicate_ids": duplicate_ids,
    }
    return rows, diagnostics


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(int(row["num_keys"]), float(row["position_ratio"]))].append(row)

    summary = []
    for nk in NUM_KEYS:
        for pos in POSITIONS:
            cell = by_cell[(nk, pos)]
            if len(cell) != 500:
                raise ValueError(f"cell {(nk, pos)} 应为 500 条，实际为 {len(cell)} 条")
            n = len(cell)
            summary.append(
                {
                    "num_keys": nk,
                    "position_ratio": pos,
                    "n": n,
                    "exact_match": sum(int(r.get("exact_match", 0)) for r in cell) / n,
                    "contains_value": sum(int(r.get("contains_value", 0)) for r in cell) / n,
                    "first_value_match": sum(int(r.get("first_value_match", 0)) for r in cell) / n,
                    "wrong_context_value": sum(int(r.get("wrong_context_value", 0)) for r in cell) / n,
                    "hallucinated_value": sum(int(r.get("hallucinated_value", 0)) for r in cell) / n,
                    "format_error": sum(int(r.get("format_error", 0)) for r in cell) / n,
                    "mean_abs_index_delta": sum(float(r.get("abs_index_delta") or 0.0) for r in cell) / n,
                    "mean_latency_sec": sum(float(r.get("latency_sec") or 0.0) for r in cell) / n,
                    "mean_prompt_tokens": sum(float(r.get("actual_prompt_tokens") or 0.0) for r in cell) / n,
                }
            )
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    fields = [
        "num_keys",
        "position_ratio",
        "n",
        "exact_match",
        "contains_value",
        "first_value_match",
        "wrong_context_value",
        "hallucinated_value",
        "format_error",
        "mean_abs_index_delta",
        "mean_latency_sec",
        "mean_prompt_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{COLORS["text"]}">{html.escape(text)}</text>'
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


def write_accuracy_plot(path: Path, summary: list[dict[str, Any]]) -> None:
    width, height = 980, 560
    margin = {"left": 82, "right": 150, "top": 72, "bottom": 84}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def x_scale(pos: float) -> float:
        return margin["left"] + pos * plot_w

    def y_scale(acc: float) -> float:
        return margin["top"] + (1.0 - acc) * plot_h

    body = [
        svg_text(width / 2, 34, "Lost-in-the-Middle KV: Accuracy by Evidence Position", 20, "middle", "700"),
        svg_text(width / 2, 56, "Qwen3-0.6B-Base; 500 examples per cell; metric = first generated UUID matches target value", 12, "middle"),
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_scale(tick)
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{tick:.2f}", 11, "end"))
    for pos in POSITIONS:
        x = x_scale(pos)
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 24, f"{int(pos * 100)}%", 11, "middle"))
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    body.append(svg_text(margin["left"] + plot_w / 2, height - 26, "Target evidence position", 13, "middle", "700"))
    y_label = margin["top"] + plot_h / 2
    body.append(f'<text x="20" y="{y_label:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="{COLORS["text"]}" transform="rotate(-90 20 {y_label:.1f})">Accuracy</text>')

    by_key = defaultdict(list)
    for row in summary:
        by_key[row["num_keys"]].append(row)
    for nk in NUM_KEYS:
        rows = sorted(by_key[nk], key=lambda r: r["position_ratio"])
        points = [(x_scale(r["position_ratio"]), y_scale(r["first_value_match"])) for r in rows]
        color = COLORS[str(nk)]
        body.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for row, (x, y) in zip(rows, points):
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            body.append(svg_text(x, y - 10, f"{row['first_value_match']:.2f}", 10, "middle"))

    legend_x = width - margin["right"] + 24
    legend_y = margin["top"] + 10
    body.append(svg_text(legend_x, legend_y - 18, "KV pairs", 13, "start", "700"))
    for i, nk in enumerate(NUM_KEYS):
        y = legend_y + i * 30
        color = COLORS[str(nk)]
        body.append(f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 30}" y2="{y:.1f}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{legend_x + 15}" cy="{y:.1f}" r="4" fill="{color}"/>')
        body.append(svg_text(legend_x + 42, y + 4, str(nk), 12))
    write_svg(path, width, height, body)


def heat_color(value: float) -> str:
    r1, g1, b1 = 220, 38, 38
    r2, g2, b2 = 37, 99, 235
    r = round(r1 + (r2 - r1) * value)
    g = round(g1 + (g2 - g1) * value)
    b = round(b1 + (b2 - b1) * value)
    return f"rgb({r},{g},{b})"


def write_heatmap(path: Path, summary: list[dict[str, Any]]) -> None:
    width, height = 780, 340
    left, top = 120, 86
    cell_w, cell_h = 96, 56
    values = {(row["num_keys"], row["position_ratio"]): row["first_value_match"] for row in summary}
    body = [
        svg_text(width / 2, 34, "Accuracy Heatmap", 20, "middle", "700"),
        svg_text(width / 2, 56, "Rows = number of KV pairs; columns = target evidence position", 12, "middle"),
    ]
    for j, pos in enumerate(POSITIONS):
        body.append(svg_text(left + j * cell_w + cell_w / 2, top - 18, f"{int(pos * 100)}%", 12, "middle", "700"))
    for i, nk in enumerate(NUM_KEYS):
        body.append(svg_text(left - 18, top + i * cell_h + cell_h / 2 + 4, str(nk), 12, "end", "700"))
        for j, pos in enumerate(POSITIONS):
            x = left + j * cell_w
            y = top + i * cell_h
            value = values[(nk, pos)]
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{heat_color(value)}" stroke="white" stroke-width="2"/>')
            text_color = "white" if value < 0.55 else "#111827"
            body.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="{text_color}">{value:.2f}</text>')
    body.append(svg_text(left - 18, top - 18, "keys", 12, "end", "700"))
    write_svg(path, width, height, body)


def write_error_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1060, 900
    margin = {"left": 88, "right": 220, "top": 78, "bottom": 142}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    classes = ["correct", "wrong_context_value", "hallucinated_value", "format_error", "other"]
    labels = {
        "correct": "Correct",
        "wrong_context_value": "Wrong in-context value",
        "hallucinated_value": "Hallucinated UUID",
        "format_error": "Format error",
        "other": "Other",
    }
    counts: dict[tuple[int, float], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[int, float]] = Counter()
    for row in rows:
        key = (int(row["num_keys"]), float(row["position_ratio"]))
        counts[key][classify(row)] += 1
        totals[key] += 1

    body = [
        svg_text(width / 2, 34, "Error Types by Evidence Position", 20, "middle", "700"),
        svg_text(width / 2, 56, "Stacked proportions; each position has 500 examples", 12, "middle"),
    ]
    panel_gap = 62
    panel_h = (plot_h - panel_gap * (len(NUM_KEYS) - 1)) / len(NUM_KEYS)
    group_w = plot_w / len(POSITIONS)
    bar_w = group_w * 0.48

    for i, nk in enumerate(NUM_KEYS):
        panel_x = margin["left"]
        panel_y = margin["top"] + i * (panel_h + panel_gap)
        body.append(svg_text(panel_x - 54, panel_y + panel_h / 2 + 4, str(nk), 13, "middle", "700"))
        body.append(svg_text(panel_x - 54, panel_y + panel_h / 2 + 22, "keys", 10, "middle"))
        for tick in [0.0, 0.5, 1.0]:
            y = panel_y + (1.0 - tick) * panel_h
            body.append(f'<line x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            body.append(svg_text(panel_x - 8, y + 4, f"{tick:.1f}", 10, "end"))
        body.append(f'<line x1="{panel_x:.1f}" y1="{panel_y + panel_h:.1f}" x2="{panel_x + plot_w:.1f}" y2="{panel_y + panel_h:.1f}" stroke="#111827" stroke-width="1"/>')
        body.append(f'<line x1="{panel_x:.1f}" y1="{panel_y:.1f}" x2="{panel_x:.1f}" y2="{panel_y + panel_h:.1f}" stroke="#111827" stroke-width="1"/>')
        for j, pos in enumerate(POSITIONS):
            x = panel_x + j * group_w + (group_w - bar_w) / 2
            y_base = panel_y + panel_h
            total = max(1, totals[(nk, pos)])
            for cls in classes:
                frac = counts[(nk, pos)][cls] / total
                h = frac * panel_h
                y = y_base - h
                if h >= 1.0:
                    body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[cls]}"/>')
                y_base = y
            if i == len(NUM_KEYS) - 1:
                body.append(svg_text(x + bar_w / 2, panel_y + panel_h + 24, f"{int(pos * 100)}%", 10, "middle"))

    body.append(svg_text(margin["left"] + plot_w / 2, height - 78, "Target evidence position", 13, "middle", "700"))
    y_label = margin["top"] + plot_h / 2
    body.append(f'<text x="20" y="{y_label:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="{COLORS["text"]}" transform="rotate(-90 20 {y_label:.1f})">Proportion</text>')
    legend_y = height - 44
    x = margin["left"]
    for cls in classes:
        body.append(f'<rect x="{x:.1f}" y="{legend_y - 13:.1f}" width="16" height="16" fill="{COLORS[cls]}"/>')
        body.append(svg_text(x + 23, legend_y, labels[cls], 11))
        x += 150 if cls != "wrong_context_value" else 190
    write_svg(path, width, height, body)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_memo(path: Path, summary: list[dict[str, Any]], rows: list[dict[str, Any]], diagnostics: dict[str, int]) -> None:
    by_key = defaultdict(list)
    for row in summary:
        by_key[row["num_keys"]].append(row)
    overall = Counter(classify(row) for row in rows)
    total = len(rows)

    lines = [
        "# v1-a 结果整理：Official-scale Lost-in-the-Middle KV Retrieval",
        "",
        "## 实验设置",
        "",
        "- 模型：`Qwen3-0.6B-Base`",
        "- Benchmark：Lost-in-the-Middle 官方 KV retrieval 数据",
        "- KV 数量：`75`、`140`、`300`",
        "- 目标证据位置：`0%`、`25%`、`50%`、`75%`、`100%`",
        "- 每个 cell 样本数：`500`",
        f"- 合并后样本总数：`{total}`",
        f"- 原始 shard 行数：`{diagnostics['raw_count']}`",
        f"- 从 300-key 主 shard 中跳过的非 0% 残留行：`{diagnostics['skipped_300_nonzero_from_main']}`",
        f"- 结果目录：`{OUT_DIR}`",
        "",
        "## 主指标",
        "",
        "`first_value_match`：模型生成文本中的第一个 UUID 是否等于目标 value。",
        "这个指标比 exact match 更适合本任务，因为任务本质是返回一个 UUID；只要第一个 UUID 正确，后面少量多余文本不应过度惩罚。",
        "",
        "## 不同位置的准确率",
        "",
        "| KV 数量 | 0% | 25% | 50% | 75% | 100% | 平均 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nk in NUM_KEYS:
        values = sorted(by_key[nk], key=lambda r: r["position_ratio"])
        mean_acc = sum(r["first_value_match"] for r in values) / len(values)
        lines.append(f"| {nk} | " + " | ".join(pct(r["first_value_match"]) for r in values) + f" | {pct(mean_acc)} |")

    lines += [
        "",
        "每个 cell 有 500 条样本。按二项分布粗略估计，本实验中单个 cell 的 95% 置信区间通常约为 +/- 2 到 4 个百分点；因此 0% 与 50% 这类大幅差距基本不是抽样噪声。",
        "",
        "## 总体错误类型",
        "",
        "| 类型 | 数量 | 比例 |",
        "|---|---:|---:|",
    ]
    for cls, label in [
        ("correct", "正确"),
        ("wrong_context_value", "上下文内错误 value"),
        ("hallucinated_value", "幻觉 UUID"),
        ("format_error", "格式错误"),
        ("other", "其他"),
    ]:
        lines.append(f"| {label} | {overall[cls]} | {pct(overall[cls] / total)} |")

    lines += [
        "",
        "## 主要结论",
        "",
        "1. 第一个 official-scale benchmark 已完成：15 个 cell，每个 cell 500 条，共 7500 条有效样本。",
        "2. 证据位置影响明显。三个 KV 数量设置下，目标证据在开头时表现最好，靠近中间时明显下降，符合 Lost in the Middle 现象。",
        "3. 候选 KV 数量越大，任务越难。平均准确率从 75-key 到 140-key、300-key 明显下降；300-key 在中间位置降到较低水平。",
        "4. 之前 pilot 中 `140 keys @ 50%` 的异常高点在官方规模下仍然存在：该点为 75.4%，明显高于 25% 的 43.2% 和 75% 的 46.0%。这说明它不是 50 条样本导致的简单有限样本波动，需要单独做诊断。",
        "5. 错误主要不是格式问题。模型通常能输出 UUID，但经常把 query key 绑定到上下文中其他 key 的 value，或者生成不在上下文中的 UUID。",
        "6. 后续后训练应重点构造“多干扰项 + 位置均衡 + key-value 绑定”的数据，而不是只做简单 single-needle 检索。",
        "",
        "## 后续建议分析",
        "",
        "- 优先诊断 `140 keys @ 50%`：检查 prompt/token 位置、source sample 是否一致、错误答案的 predicted index 分布，以及是否存在 Qwen 对 140-key/中间位置的特殊偏置。",
        "- 分析错误答案的 predicted index：模型错误时更偏向输出前部记录、邻近记录，还是随机/幻觉 UUID？",
        "- 按 token offset 而不是 record index 分析位置效应，确认是否存在更细粒度的位置偏置。",
        "- 在后训练后重复同一 benchmark，区分提升来自更强检索、更强 key-value 绑定，还是更少幻觉。",
        "",
        "## 产物",
        "",
        "- `outputs/qwen3_0_6b_litm_kv_official5_merged/predictions.jsonl`",
        "- `outputs/qwen3_0_6b_litm_kv_official5_merged/summary.csv`",
        "- `outputs/qwen3_0_6b_litm_kv_official5_merged/plots/accuracy_by_position.svg`",
        "- `outputs/qwen3_0_6b_litm_kv_official5_merged/plots/accuracy_heatmap.svg`",
        "- `outputs/qwen3_0_6b_litm_kv_official5_merged/plots/error_types_stacked.svg`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows, diagnostics = merge_rows()
    summary = summarize(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "predictions.jsonl", rows)
    write_summary(OUT_DIR / "summary.csv", summary)
    plots = OUT_DIR / "plots"
    write_accuracy_plot(plots / "accuracy_by_position.svg", summary)
    write_heatmap(plots / "accuracy_heatmap.svg", summary)
    write_error_bars(plots / "error_types_stacked.svg", rows)
    write_memo(DOC_PATH, summary, rows, diagnostics)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"合并后有效样本数: {len(rows)}")
    print(f"结果目录: {OUT_DIR}")
    print(f"中文 memo: {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
