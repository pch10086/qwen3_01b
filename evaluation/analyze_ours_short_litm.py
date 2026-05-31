#!/usr/bin/env python3
"""分析自研模型短上下文 LITM-KV 诊断实验。"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path("/home/public/bjh/dym/NLP/evaluation")
RUNS = [
    {
        "id": "stage1",
        "label": "Stage1",
        "dir": BASE / "outputs/ours_stage1_litm_kv_short_2k_7pos_s20",
    },
    {
        "id": "stage2_16k",
        "label": "Stage2-16K",
        "dir": BASE / "outputs/ours_stage2_16k_litm_kv_short_2k_7pos_s20",
    },
]
OUT_DIR = BASE / "outputs/ours_litm_kv_short_2k_stage1_vs_stage2"
DOC_PATH = BASE / "docs/ours_litm_kv_short_2k_stage1_vs_stage2_memo.md"

COLORS = {
    "32": "#2563eb",
    "40": "#16a34a",
    "50": "#dc2626",
    "stage1": "#475569",
    "stage2_16k": "#0f766e",
    "correct": "#2563eb",
    "wrong_context": "#f97316",
    "hallucinated": "#8b5cf6",
    "format_error": "#64748b",
    "grid": "#d6d3d1",
    "text": "#1f2937",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"num_keys", "n"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify(row: dict[str, Any]) -> str:
    if row.get("first_value_match"):
        return "correct"
    if row.get("wrong_context_value"):
        return "wrong_context"
    if row.get("hallucinated_value"):
        return "hallucinated"
    if row.get("format_error"):
        return "format_error"
    return "other"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pct1(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def heat_color(value: float, high_good: bool = True) -> str:
    low = (229, 231, 235)
    high = (37, 99, 235) if high_good else (220, 38, 38)
    r = round(low[0] + (high[0] - low[0]) * value)
    g = round(low[1] + (high[1] - low[1]) * value)
    b = round(low[2] + (high[2] - low[2]) * value)
    return f"rgb({r},{g},{b})"


def write_accuracy_heatmap(path: Path, all_summary: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 1180, 440
    left, top = 118, 104
    cell_w, cell_h = 84, 52
    panel_gap = 92
    keys = sorted({row["num_keys"] for rows in all_summary.values() for row in rows})
    positions = sorted({row["position_ratio"] for rows in all_summary.values() for row in rows})
    panel_w = len(positions) * cell_w

    body: list[str] = [
        svg_text(width / 2, 34, "Short-Context LITM-KV Accuracy", 20, "middle", "700"),
        svg_text(width / 2, 58, "Metric: first generated UUID matches the target value; 20 examples per cell", 12, "middle"),
    ]

    for ridx, run in enumerate(RUNS):
        panel_x = left + ridx * (panel_w + panel_gap)
        rows = all_summary[run["id"]]
        values = {(row["num_keys"], row["position_ratio"]): row["first_value_match"] for row in rows}
        body.append(svg_text(panel_x + panel_w / 2, 86, run["label"], 15, "middle", "700"))
        for j, pos in enumerate(positions):
            body.append(svg_text(panel_x + j * cell_w + cell_w / 2, top - 16, f"{int(pos * 100)}%", 11, "middle", "700"))
        for i, nk in enumerate(keys):
            y = top + i * cell_h
            body.append(svg_text(panel_x - 16, y + cell_h / 2 + 4, str(nk), 12, "end", "700"))
            for j, pos in enumerate(positions):
                x = panel_x + j * cell_w
                value = values[(nk, pos)]
                color = heat_color(min(value / 0.05, 1.0), high_good=True)
                body.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                    f'fill="{color}" stroke="white" stroke-width="2"/>'
                )
                body.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 5, f"{value:.2f}", 12, "middle", "700"))
        body.append(svg_text(panel_x - 16, top - 16, "keys", 11, "end", "700"))
    body.append(svg_text(width / 2, height - 28, "Target evidence position", 13, "middle", "700"))
    write_svg(path, width, height, body)


def write_error_stacked(path: Path, all_predictions: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 980, 560
    margin = {"left": 82, "right": 230, "top": 82, "bottom": 94}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    classes = ["correct", "wrong_context", "hallucinated", "format_error"]
    labels = {
        "correct": "Correct",
        "wrong_context": "Wrong in-context value",
        "hallucinated": "Hallucinated UUID",
        "format_error": "Format error",
    }
    bars: list[tuple[str, str, int, Counter[str]]] = []
    for run in RUNS:
        by_key: dict[int, Counter[str]] = defaultdict(Counter)
        for row in all_predictions[run["id"]]:
            by_key[int(row["num_keys"])][classify(row)] += 1
        for nk in sorted(by_key):
            bars.append((run["id"], run["label"], nk, by_key[nk]))

    body: list[str] = [
        svg_text(width / 2, 34, "Short-Context LITM-KV Error Composition", 20, "middle", "700"),
        svg_text(width / 2, 56, "Stacked proportions by checkpoint and number of KV pairs", 12, "middle"),
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin["top"] + (1.0 - tick) * plot_h
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{tick:.2f}", 10, "end"))
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.4"/>')

    group_w = plot_w / len(bars)
    bar_w = group_w * 0.56
    for i, (run_id, run_label, nk, counts) in enumerate(bars):
        x = margin["left"] + i * group_w + (group_w - bar_w) / 2
        total = max(1, sum(counts.values()))
        y_base = margin["top"] + plot_h
        for cls in classes:
            frac = counts[cls] / total
            h = frac * plot_h
            if h > 0:
                body.append(f'<rect x="{x:.1f}" y="{y_base - h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[cls]}"/>')
            y_base -= h
        body.append(svg_text(x + bar_w / 2, margin["top"] + plot_h + 22, str(nk), 10, "middle"))
        if i in {1, 4}:
            body.append(svg_text(x + bar_w / 2, margin["top"] + plot_h + 44, run_label, 11, "middle", "700"))

    body.append(svg_text(margin["left"] + plot_w / 2, height - 18, "KV pairs grouped by checkpoint", 13, "middle", "700"))
    y_label = margin["top"] + plot_h / 2
    body.append(
        f'<text x="20" y="{y_label:.1f}" font-family="Arial, sans-serif" font-size="13" font-weight="700" '
        f'text-anchor="middle" fill="{COLORS["text"]}" transform="rotate(-90 20 {y_label:.1f})">Proportion</text>'
    )

    legend_x = margin["left"] + plot_w + 30
    legend_y = margin["top"] + 8
    body.append(svg_text(legend_x, legend_y - 18, "Class", 13, "start", "700"))
    for i, cls in enumerate(classes):
        y = legend_y + i * 30
        body.append(f'<rect x="{legend_x}" y="{y - 12:.1f}" width="16" height="16" fill="{COLORS[cls]}"/>')
        body.append(svg_text(legend_x + 26, y + 1, labels[cls], 12))
    write_svg(path, width, height, body)


def write_accuracy_line(path: Path, all_summary: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 1040, 520
    margin = {"left": 74, "right": 170, "top": 76, "bottom": 76}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    positions = sorted({row["position_ratio"] for rows in all_summary.values() for row in rows})
    keys = sorted({row["num_keys"] for rows in all_summary.values() for row in rows})
    max_y = 0.10

    def x_scale(pos: float) -> float:
        return margin["left"] + pos * plot_w

    def y_scale(value: float) -> float:
        return margin["top"] + (1.0 - min(value / max_y, 1.0)) * plot_h

    body: list[str] = [
        svg_text(width / 2, 34, "Short-Context LITM-KV Accuracy by Position", 20, "middle", "700"),
        svg_text(width / 2, 56, "Y-axis capped at 0.10 because all accuracies are near zero", 12, "middle"),
    ]
    for tick in [0.0, 0.025, 0.05, 0.075, 0.10]:
        y = y_scale(tick)
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{tick:.3f}", 10, "end"))
    for pos in positions:
        x = x_scale(pos)
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 22, f"{int(pos * 100)}%", 10, "middle"))
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.4"/>')
    body.append(svg_text(margin["left"] + plot_w / 2, height - 20, "Target evidence position", 13, "middle", "700"))

    dash_by_run = {"stage1": "6 5", "stage2_16k": "0"}
    for run in RUNS:
        rows = all_summary[run["id"]]
        for nk in keys:
            series = sorted([row for row in rows if row["num_keys"] == nk], key=lambda r: r["position_ratio"])
            points = [(x_scale(row["position_ratio"]), y_scale(row["first_value_match"])) for row in series]
            dash = dash_by_run[run["id"]]
            color = COLORS[str(nk)]
            body.append(
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" '
                f'stroke="{color}" stroke-width="2.6" stroke-dasharray="{dash}" stroke-linejoin="round"/>'
            )
            for x, y in points:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" stroke="white" stroke-width="1"/>')

    legend_x = margin["left"] + plot_w + 26
    legend_y = margin["top"] + 4
    body.append(svg_text(legend_x, legend_y - 16, "KV pairs", 13, "start", "700"))
    for i, nk in enumerate(keys):
        y = legend_y + i * 26
        body.append(f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 30}" y2="{y:.1f}" stroke="{COLORS[str(nk)]}" stroke-width="3"/>')
        body.append(svg_text(legend_x + 40, y + 4, str(nk), 12))
    style_y = legend_y + 100
    body.append(svg_text(legend_x, style_y, "Checkpoint", 13, "start", "700"))
    body.append(f'<line x1="{legend_x}" y1="{style_y + 26}" x2="{legend_x + 30}" y2="{style_y + 26}" stroke="#111827" stroke-width="2.6" stroke-dasharray="6 5"/>')
    body.append(svg_text(legend_x + 40, style_y + 30, "Stage1", 12))
    body.append(f'<line x1="{legend_x}" y1="{style_y + 54}" x2="{legend_x + 30}" y2="{style_y + 54}" stroke="#111827" stroke-width="2.6"/>')
    body.append(svg_text(legend_x + 40, style_y + 58, "Stage2-16K", 12))
    write_svg(path, width, height, body)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(classify(row) for row in rows)
    by_key: dict[int, Counter[str]] = defaultdict(Counter)
    by_pos: dict[float, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cls = classify(row)
        by_key[int(row["num_keys"])][cls] += 1
        by_pos[float(row["position_ratio"])][cls] += 1
    return {
        "n": len(rows),
        "counts": counts,
        "by_key": by_key,
        "by_pos": by_pos,
    }


def write_comparison_csv(path: Path, all_predictions: dict[str, list[dict[str, Any]]]) -> None:
    fields = [
        "checkpoint",
        "num_keys",
        "n",
        "accuracy",
        "format_error_rate",
        "wrong_context_rate",
        "hallucinated_rate",
    ]
    rows = []
    for run in RUNS:
        by_key: dict[int, Counter[str]] = defaultdict(Counter)
        for pred in all_predictions[run["id"]]:
            by_key[int(pred["num_keys"])][classify(pred)] += 1
        for nk in sorted(by_key):
            counts = by_key[nk]
            n = sum(counts.values())
            rows.append(
                {
                    "checkpoint": run["label"],
                    "num_keys": nk,
                    "n": n,
                    "accuracy": counts["correct"] / n,
                    "format_error_rate": counts["format_error"] / n,
                    "wrong_context_rate": counts["wrong_context"] / n,
                    "hallucinated_rate": counts["hallucinated"] / n,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_memo(path: Path, all_predictions: dict[str, list[dict[str, Any]]]) -> None:
    stats = {run["id"]: summarize_predictions(all_predictions[run["id"]]) for run in RUNS}
    lines: list[str] = [
        "# 自研模型短上下文 LITM-KV 诊断结果",
        "",
        "## 实验目的",
        "",
        "这次实验不是正式长上下文 benchmark，而是为了回答一个更基础的问题：当前模型在 2k 左右上下文内是否已经具备稳定的 KV 检索和高熵 UUID 复制能力，以及 stage2 长上下文继续训练是否相对 stage1 引入明显退化。",
        "",
        "## 设置",
        "",
        "- 数据来源：Lost-in-the-Middle 官方 KV retrieval 数据，使用官方样本裁剪出更短的 KV 列表，目标 key-value 始终保留。",
        "- Key 数量：32、40、50，平均 prompt token 数约为 1.8k、2.24k、2.79k。",
        "- 证据位置：0%、10%、25%、50%、75%、90%、100%。",
        "- 每格样本数：20；每个 checkpoint 共 420 条。",
        "- 推理方式：自研模型 `generate_with_cache`，temperature=0，max_new_tokens=64。",
        "- 评分：抽取生成文本中的第一个 UUID；若它等于 gold value，则记为正确；若没有 UUID，则记为 format error；若 UUID 来自上下文但不是 gold，则记为 wrong in-context value；否则记为 hallucinated UUID。",
        "",
        "## 总体结果",
        "",
        "| Checkpoint | N | Correct | Accuracy | Format error | Wrong in-context | Hallucinated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        s = stats[run["id"]]
        n = s["n"]
        c = s["counts"]
        lines.append(
            f"| {run['label']} | {n} | {c['correct']} | {pct(c['correct'] / n)} | "
            f"{pct(c['format_error'] / n)} | {pct(c['wrong_context'] / n)} | {pct(c['hallucinated'] / n)} |"
        )

    lines += [
        "",
        "## 按 key 数量划分",
        "",
        "| Checkpoint | Keys | N | Accuracy | Format error | Wrong in-context | Hallucinated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        by_key = stats[run["id"]]["by_key"]
        for nk in sorted(by_key):
            c = by_key[nk]
            n = sum(c.values())
            lines.append(
                f"| {run['label']} | {nk} | {n} | {pct(c['correct'] / n)} | "
                f"{pct(c['format_error'] / n)} | {pct(c['wrong_context'] / n)} | {pct(c['hallucinated'] / n)} |"
            )

    lines += [
        "",
        "## 关键观察",
        "",
        "1. stage1 在 2k 左右上下文内几乎不能稳定输出合法 UUID：420 条中 410 条是 format error，整体正确率只有 0.48%。这说明当前问题不能主要归因于 16k 位置编码或长上下文外推，因为在 stage1 可覆盖的短上下文范围内基础复制行为已经不稳定。",
        "2. stage2_16k 的 format error 从 97.62% 降到 74.05%，说明继续训练后模型更容易生成 UUID 形态的字符串；但正确率仍只有 0.71%，大部分非 format-error 输出要么是上下文里的错误 value，要么是幻觉 UUID。",
        "3. 正确样本几乎只出现在 100% 位置，符合一种更弱的 recency/copying 行为：模型在 query 紧邻最近记录或尾部记录时偶尔能续写对，但尚未形成真正的按 key 检索能力。",
        "4. 因为 32 keys 平均只有约 1.8k tokens 仍然失败，当前 full 75/140/280 keys 长上下文评测的低分不能解释为单纯的 KV cache bug 或 16k 数据不足；更根本的问题是模型还没有学会稳定的结构化复制和键值绑定。",
        "",
        "## 对研究推进的含义",
        "",
        "当前不建议直接扩大正式长上下文评测规模。更合理的路线是先把模型的短上下文 KV 检索能力训练/诊断到可用水平，再进入证据位置效应分析。否则 long-context 曲线主要反映 format failure，而不是 lost-in-the-middle 或位置编码问题。",
        "",
        "建议下一步做两个更有解释力的实验：",
        "",
        "1. 做一个极短 KV sanity check：4/8/16 keys，每格 50 条。如果 4 或 8 keys 仍失败，说明需要先做复制/格式 SFT 或继续预训练中的结构化数据增强。",
        "2. 做 gold-value log-likelihood/ranking 评测：给定同一个 prompt，比较 gold value 与若干上下文 distractor value 的条件概率。这样可以绕开自由生成 UUID 的格式失败，判断模型内部是否至少能区分正确 value。",
        "",
        "## 产物路径",
        "",
        f"- 结果目录：`{OUT_DIR}`",
        f"- stage1 predictions：`{RUNS[0]['dir'] / 'predictions.jsonl'}`",
        f"- stage2_16k predictions：`{RUNS[1]['dir'] / 'predictions.jsonl'}`",
        f"- 本 memo：`{DOC_PATH}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: dict[str, list[dict[str, Any]]] = {}
    all_predictions: dict[str, list[dict[str, Any]]] = {}
    for run in RUNS:
        all_summary[run["id"]] = read_csv(run["dir"] / "summary.csv")
        all_predictions[run["id"]] = read_jsonl(run["dir"] / "predictions.jsonl")

    write_accuracy_heatmap(OUT_DIR / "accuracy_heatmap.svg", all_summary)
    write_accuracy_line(OUT_DIR / "accuracy_by_position.svg", all_summary)
    write_error_stacked(OUT_DIR / "error_composition_by_checkpoint.svg", all_predictions)
    write_comparison_csv(OUT_DIR / "comparison_by_key_count.csv", all_predictions)
    write_memo(DOC_PATH, all_predictions)

    print(f"wrote {OUT_DIR}")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
