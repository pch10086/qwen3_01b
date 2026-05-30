#!/usr/bin/env python3
"""Analyze Lost-in-the-Middle KV benchmark outputs and write report artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COLORS = {
    "75": "#2563eb",
    "140": "#16a34a",
    "300": "#dc2626",
    "correct": "#2563eb",
    "wrong_context_value": "#f97316",
    "hallucinated_value": "#8b5cf6",
    "format_error": "#64748b",
    "grid": "#d6d3d1",
    "text": "#1f2937",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="/home/public/bjh/dym/NLP/evaluation/outputs/qwen3_0_6b_litm_kv_v1",
    )
    parser.add_argument(
        "--memo-path",
        default="/home/public/bjh/dym/NLP/evaluation/docs/litm_kv_v1_result_memo.md",
    )
    return parser.parse_args()


def read_summary(path: Path) -> list[dict[str, Any]]:
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


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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


def write_accuracy_line_plot(path: Path, summary: list[dict[str, Any]]) -> None:
    width, height = 900, 560
    margin = {"left": 78, "right": 150, "top": 70, "bottom": 78}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    positions = sorted({row["position_ratio"] for row in summary})
    num_keys = sorted({row["num_keys"] for row in summary})

    def x_scale(pos: float) -> float:
        return margin["left"] + pos * plot_w

    def y_scale(acc: float) -> float:
        return margin["top"] + (1.0 - acc) * plot_h

    body: list[str] = []
    body.append(svg_text(width / 2, 34, "Lost-in-the-Middle KV: Accuracy by Target Position", 20, "middle", "700"))
    body.append(svg_text(width / 2, 55, "Metric: first generated UUID equals target value", 12, "middle"))

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_scale(tick)
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(margin["left"] - 10, y + 4, f"{tick:.2f}", 11, "end"))

    for pos in positions:
        x = x_scale(pos)
        body.append(f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{margin["top"] + plot_h}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        body.append(svg_text(x, margin["top"] + plot_h + 24, f"{int(pos * 100)}%", 11, "middle"))

    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#111827" stroke-width="1.5"/>')
    body.append(svg_text(margin["left"] + plot_w / 2, height - 24, "Target key-value position", 13, "middle", "700"))
    y_label = margin["top"] + plot_h / 2
    body.append(
        f'<text x="18" y="{y_label:.1f}" font-family="Arial, sans-serif" font-size="13" '
        f'font-weight="700" text-anchor="middle" fill="{COLORS["text"]}" '
        f'transform="rotate(-90 18 {y_label:.1f})">Accuracy</text>'
    )

    by_key = defaultdict(list)
    for row in summary:
        by_key[row["num_keys"]].append(row)
    for nk in num_keys:
        rows = sorted(by_key[nk], key=lambda r: r["position_ratio"])
        points = [(x_scale(r["position_ratio"]), y_scale(r["first_value_match"])) for r in rows]
        point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        color = COLORS[str(nk)]
        body.append(f'<polyline points="{point_attr}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for row, (x, y) in zip(rows, points):
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            body.append(svg_text(x, y - 10, f"{row['first_value_match']:.2f}", 10, "middle"))

    legend_x = width - margin["right"] + 28
    legend_y = margin["top"] + 8
    body.append(svg_text(legend_x, legend_y - 16, "KV pairs", 13, "start", "700"))
    for i, nk in enumerate(num_keys):
        y = legend_y + i * 28
        color = COLORS[str(nk)]
        body.append(f'<line x1="{legend_x}" y1="{y:.1f}" x2="{legend_x + 28}" y2="{y:.1f}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<circle cx="{legend_x + 14}" cy="{y:.1f}" r="4" fill="{color}"/>')
        body.append(svg_text(legend_x + 40, y + 4, str(nk), 12))

    write_svg(path, width, height, body)


def heat_color(value: float) -> str:
    # Blue for high accuracy, red for low accuracy.
    r1, g1, b1 = 220, 38, 38
    r2, g2, b2 = 37, 99, 235
    r = round(r1 + (r2 - r1) * value)
    g = round(g1 + (g2 - g1) * value)
    b = round(b1 + (b2 - b1) * value)
    return f"rgb({r},{g},{b})"


def write_heatmap(path: Path, summary: list[dict[str, Any]]) -> None:
    width, height = 880, 360
    left, top = 120, 86
    cell_w, cell_h = 92, 54
    positions = sorted({row["position_ratio"] for row in summary})
    num_keys = sorted({row["num_keys"] for row in summary})
    values = {(row["num_keys"], row["position_ratio"]): row["first_value_match"] for row in summary}

    body: list[str] = []
    body.append(svg_text(width / 2, 34, "Accuracy Heatmap", 20, "middle", "700"))
    body.append(svg_text(width / 2, 55, "Rows: number of key-value pairs; columns: target position", 12, "middle"))

    for j, pos in enumerate(positions):
        x = left + j * cell_w + cell_w / 2
        body.append(svg_text(x, top - 18, f"{int(pos * 100)}%", 12, "middle", "700"))
    for i, nk in enumerate(num_keys):
        y = top + i * cell_h + cell_h / 2
        body.append(svg_text(left - 18, y + 4, str(nk), 12, "end", "700"))
        for j, pos in enumerate(positions):
            x = left + j * cell_w
            y0 = top + i * cell_h
            value = values[(nk, pos)]
            body.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{heat_color(value)}" stroke="white" stroke-width="2"/>')
            text_color = "white" if value < 0.58 else "#111827"
            body.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" '
                f'font-family="Arial, sans-serif" font-size="13" font-weight="700" '
                f'text-anchor="middle" fill="{text_color}">{value:.2f}</text>'
            )

    body.append(svg_text(left - 18, top - 18, "keys", 12, "end", "700"))
    write_svg(path, width, height, body)


def classify_prediction(row: dict[str, Any]) -> str:
    if row.get("first_value_match"):
        return "correct"
    if row.get("wrong_context_value"):
        return "wrong_context_value"
    if row.get("hallucinated_value"):
        return "hallucinated_value"
    if row.get("format_error"):
        return "format_error"
    return "other"


def write_error_stacked_bars(path: Path, predictions: list[dict[str, Any]]) -> None:
    width, height = 980, 860
    margin = {"left": 86, "right": 210, "top": 76, "bottom": 132}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    positions = sorted({float(row["position_ratio"]) for row in predictions})
    num_keys = sorted({int(row["num_keys"]) for row in predictions})
    classes = ["correct", "wrong_context_value", "hallucinated_value", "format_error"]
    labels = {
        "correct": "Correct",
        "wrong_context_value": "Wrong in-context value",
        "hallucinated_value": "Hallucinated UUID",
        "format_error": "Format error",
    }

    counts: dict[tuple[int, float], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[int, float]] = Counter()
    for row in predictions:
        key = (int(row["num_keys"]), float(row["position_ratio"]))
        counts[key][classify_prediction(row)] += 1
        totals[key] += 1

    body: list[str] = []
    body.append(svg_text(width / 2, 34, "Error Types by Target Position", 20, "middle", "700"))
    body.append(svg_text(width / 2, 55, "Stacked proportions; each row is one key-value size", 12, "middle"))

    panel_gap = 54
    panel_h = (plot_h - panel_gap * (len(num_keys) - 1)) / len(num_keys)
    panel_w = plot_w
    bar_w = panel_w / len(positions) * 0.52

    for pidx, nk in enumerate(num_keys):
        panel_x = margin["left"]
        panel_y = margin["top"] + pidx * (panel_h + panel_gap)
        body.append(svg_text(panel_x - 54, panel_y + panel_h / 2 + 4, f"{nk}", 13, "middle", "700"))
        body.append(svg_text(panel_x - 54, panel_y + panel_h / 2 + 22, "keys", 10, "middle"))
        for tick in [0.0, 0.5, 1.0]:
            y = panel_y + (1.0 - tick) * panel_h
            body.append(f'<line x1="{panel_x}" y1="{y:.1f}" x2="{panel_x + panel_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
            body.append(svg_text(panel_x - 8, y + 4, f"{tick:.1f}", 10, "end"))
        body.append(f'<line x1="{panel_x:.1f}" y1="{panel_y + panel_h:.1f}" x2="{panel_x + panel_w:.1f}" y2="{panel_y + panel_h:.1f}" stroke="#111827" stroke-width="1"/>')
        body.append(f'<line x1="{panel_x:.1f}" y1="{panel_y:.1f}" x2="{panel_x:.1f}" y2="{panel_y + panel_h:.1f}" stroke="#111827" stroke-width="1"/>')
        for j, pos in enumerate(positions):
            x = panel_x + (j + 0.5) * (panel_w / len(positions)) - bar_w / 2
            y_base = panel_y + panel_h
            total = max(1, totals[(nk, pos)])
            for cls in classes:
                frac = counts[(nk, pos)][cls] / total
                h = frac * panel_h
                y = y_base - h
                if h > 0:
                    body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[cls]}"/>')
                y_base = y
            if pidx == len(num_keys) - 1:
                body.append(svg_text(x + bar_w / 2, panel_y + panel_h + 24, f"{int(pos * 100)}", 10, "middle"))

    body.append(svg_text(margin["left"] + plot_w / 2, height - 74, "Target position (%)", 13, "middle", "700"))
    y_label = margin["top"] + plot_h / 2
    body.append(
        f'<text x="18" y="{y_label:.1f}" font-family="Arial, sans-serif" font-size="13" '
        f'font-weight="700" text-anchor="middle" fill="{COLORS["text"]}" '
        f'transform="rotate(-90 18 {y_label:.1f})">Proportion</text>'
    )

    legend_x = margin["left"]
    legend_y = height - 42
    body.append(svg_text(legend_x, legend_y - 20, "Class", 13, "start", "700"))
    x = legend_x + 56
    for cls in classes:
        body.append(f'<rect x="{x:.1f}" y="{legend_y - 13:.1f}" width="16" height="16" fill="{COLORS[cls]}"/>')
        body.append(svg_text(x + 24, legend_y, labels[cls], 11))
        x += 178 if cls != "wrong_context_value" else 224

    write_svg(path, width, height, body)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_memo(path: Path, run_dir: Path, summary: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> None:
    overall = Counter(classify_prediction(row) for row in predictions)
    total = len(predictions)

    by_num = defaultdict(list)
    for row in summary:
        by_num[row["num_keys"]].append(row)

    lines: list[str] = []
    lines.append("# v1-a Result Memo: Lost-in-the-Middle KV Retrieval")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append("- model: `Qwen3-0.6B-Base`")
    lines.append("- benchmark: Lost-in-the-Middle official KV retrieval data")
    lines.append("- key-value sizes: `75`, `140`, `300`")
    lines.append("- target positions: `0%`, `10%`, `25%`, `50%`, `75%`, `90%`, `100%`")
    lines.append("- examples per cell: `50`")
    lines.append(f"- total examples: `{total}`")
    lines.append(f"- run directory: `{run_dir}`")
    lines.append("")
    lines.append("## Main Metric")
    lines.append("")
    lines.append("The main metric is `first_value_match`: whether the first UUID generated by the model equals the target value.")
    lines.append("This avoids over-penalizing harmless extra text while still requiring the model to output the correct value first.")
    lines.append("")
    lines.append("## Accuracy by Position")
    lines.append("")
    header = "| keys | 0% | 10% | 25% | 50% | 75% | 90% | 100% |"
    lines.append(header)
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nk in sorted(by_num):
        rows = sorted(by_num[nk], key=lambda r: r["position_ratio"])
        values = " | ".join(pct(row["first_value_match"]) for row in rows)
        lines.append(f"| {nk} | {values} |")
    lines.append("")
    lines.append("## Overall Error Types")
    lines.append("")
    lines.append("| error type | count | rate |")
    lines.append("|---|---:|---:|")
    for cls, label in [
        ("correct", "correct"),
        ("wrong_context_value", "wrong context value"),
        ("hallucinated_value", "hallucinated UUID"),
        ("format_error", "format error"),
    ]:
        lines.append(f"| {label} | {overall[cls]} | {pct(overall[cls] / total)} |")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    lines.append("1. The official KV retrieval task is much more diagnostic than the earlier single-needle sanity check. The model solved simple single-evidence numeric retrieval at 4K/8K/16K, but drops sharply on this benchmark.")
    lines.append("2. There is a clear position effect. Accuracy is strongest when the target record is at the beginning, especially for 140 and 300 key-value pairs.")
    lines.append("3. Candidate scale amplifies the failure. With 300 key-value pairs, accuracy is only 8.0% at the 50% and 90% positions.")
    lines.append("4. Most failures are not format failures. The model usually emits a UUID, but often binds the query key to a wrong in-context value or generates a UUID not present in the context.")
    lines.append("5. The result supports using harder retrieval and hard-negative data in later post-training rather than relying on simple single-needle examples.")
    lines.append("")
    lines.append("## Non-Monotonic Point: 140 Keys at 50%")
    lines.append("")
    lines.append("The 140-key curve has a local high point at the 50% position: 72.0%, compared with 46.0% at 25% and 44.0% at 75%.")
    lines.append("A paired check over the same 50 source examples shows that this is not caused by a target-position indexing bug: the target record is placed at index 70 and the target token position is near 49.9% of the prompt.")
    lines.append("The most likely explanation is finite-sample and model-specific non-monotonicity: at 50 examples per cell, a 10-15 example swing can visibly bend the curve. This point should be treated as a caveat and rechecked with a larger 140-key-only run before making strong claims about that exact position.")
    lines.append("The broader conclusion is still stable because the 300-key setting shows a much stronger degradation at non-edge positions, and error types are dominated by wrong value binding rather than formatting.")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- `plots/accuracy_by_position.svg`")
    lines.append("- `plots/accuracy_heatmap.svg`")
    lines.append("- `plots/error_types_stacked.svg`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = read_summary(run_dir / "summary.csv")
    predictions = read_predictions(run_dir / "predictions.jsonl")
    plots_dir = run_dir / "plots"

    write_accuracy_line_plot(plots_dir / "accuracy_by_position.svg", summary)
    write_heatmap(plots_dir / "accuracy_heatmap.svg", summary)
    write_error_stacked_bars(plots_dir / "error_types_stacked.svg", predictions)
    write_memo(Path(args.memo_path), run_dir, summary, predictions)

    print(f"Wrote plots to {plots_dir}")
    print(f"Wrote memo to {args.memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
