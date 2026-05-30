#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pretrain_data_utils import (  # noqa: E402
    DEFAULT_SOURCE_DIRS,
    iter_clean_documents,
    resolve_source_paths,
    stable_source_seed,
    text_fingerprint,
)


LABELS = {
    "fineweb_edu": "FineWeb-Edu",
    "finemath_or_openwebmath": "FineMath/OpenWebMath",
    "pes2o": "peS2o",
    "pg19": "PG-19",
    "wikipedia_en": "Wikipedia-en",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Simple corpus sampling check for tokenizer quality.")
    p.add_argument("--raw_dir", type=Path, default=project_dir / "data/raw")
    p.add_argument(
        "--tokenizer_json",
        type=Path,
        default=project_dir / "tokenizers/bpe_64k_clean/tokenizer.json",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=project_dir / "reports/tokenizer_eval_64k_simple",
    )
    p.add_argument("--docs_per_source", type=int, default=500)
    p.add_argument("--max_chars_per_doc", type=int, default=8000)
    p.add_argument("--min_chars", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260527)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(args.tokenizer_json))
    vocab_size = tok.get_vocab_size()
    unk_id = tok.token_to_id("<|unk|>")
    source_paths = resolve_source_paths(args.raw_dir)
    sources = list(DEFAULT_SOURCE_DIRS.keys())

    doc_rows: list[dict[str, float | int | str]] = []
    source_stats: dict[str, dict[str, float | int | str]] = {}
    global_freq: Counter[int] = Counter()
    piece_len_freq: Counter[int] = Counter()
    sample_html: dict[str, str] = {}

    @lru_cache(maxsize=None)
    def decoded_piece(tid: int) -> str:
        try:
            return tok.decode([tid], skip_special_tokens=False)
        except TypeError:
            return tok.decode([tid])

    @lru_cache(maxsize=None)
    def piece_len(tid: int) -> int:
        return len(decoded_piece(tid))

    for source in sources:
        seen: set[bytes] = set()
        docs = iter_clean_documents(
            source,
            source_paths[source],
            parquet_batch_size=1024,
            shuffle_files=True,
            seed=stable_source_seed(source, args.seed),
            min_chars=args.min_chars,
            min_alpha_ratio=0.25,
        )
        rows: list[dict[str, float | int | str]] = []
        source_freq: Counter[int] = Counter()
        token_count = 0
        char_count = 0
        byte_count = 0
        unk_count = 0
        single_piece_count = 0
        duplicates = 0

        for doc in docs:
            fp = text_fingerprint(doc.text)
            if fp in seen:
                duplicates += 1
                continue
            seen.add(fp)
            text = doc.text[: args.max_chars_per_doc]
            ids = tok.encode(text).ids
            if not ids:
                continue

            chars = len(text)
            bts = len(text.encode("utf-8", errors="replace"))
            toks = len(ids)
            unks = sum(1 for i in ids if i == unk_id)
            singles = sum(1 for i in ids if piece_len(i) <= 1)

            for i in ids:
                piece_len_freq[min(piece_len(i), 16)] += 1
            source_freq.update(ids)
            global_freq.update(ids)

            token_count += toks
            char_count += chars
            byte_count += bts
            unk_count += unks
            single_piece_count += singles

            row = {
                "source": source,
                "chars": chars,
                "bytes": bts,
                "tokens": toks,
                "chars_per_token": chars / toks,
                "bytes_per_token": bts / toks,
                "tokens_per_1k_chars": toks * 1000 / chars,
                "unk_rate": unks / toks,
                "single_piece_rate": singles / toks,
            }
            rows.append(row)
            doc_rows.append(row)

            if source not in sample_html:
                chips = []
                for tid in ids[:180]:
                    piece = decoded_piece(tid)
                    shown = piece.replace("\n", "\\n").replace("\t", "\\t")
                    if shown == " ":
                        shown = "[space]"
                    chips.append(
                        f'<span class="tok"><b>{tid}</b><em>{html.escape(shown)}</em></span>'
                    )
                sample_html[source] = "".join(chips)

            if len(rows) >= args.docs_per_source:
                break

        cpt_values = [float(r["chars_per_token"]) for r in rows]
        tpk_values = [float(r["tokens_per_1k_chars"]) for r in rows]
        source_stats[source] = {
            "label": LABELS[source],
            "docs": len(rows),
            "duplicates_skipped": duplicates,
            "chars": char_count,
            "bytes": byte_count,
            "tokens": token_count,
            "unique_token_ids": len(source_freq),
            "vocab_utilization_pct": len(source_freq) / vocab_size * 100,
            "chars_per_token_mean": statistics.mean(cpt_values),
            "chars_per_token_median": statistics.median(cpt_values),
            "chars_per_token_p10": float(np.percentile(cpt_values, 10)),
            "chars_per_token_p90": float(np.percentile(cpt_values, 90)),
            "tokens_per_1k_chars_mean": statistics.mean(tpk_values),
            "unk_rate": unk_count / token_count if token_count else 0,
            "single_piece_rate": single_piece_count / token_count if token_count else 0,
        }
        print(source, source_stats[source], flush=True)

    with open(args.out_dir / "doc_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(doc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(doc_rows)

    with open(args.out_dir / "source_metrics.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["source", *next(iter(source_stats.values())).keys()]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for source, stats in source_stats.items():
            writer.writerow({"source": source, **stats})

    freqs = np.array(sorted(global_freq.values(), reverse=True), dtype=np.int64)
    cum = np.cumsum(freqs) / freqs.sum()
    coverage_at: dict[str, float] = {}
    for k in [1000, 4000, 8000, 16000, 32000, 48000, 64000]:
        idx = min(k, len(cum)) - 1
        coverage_at[str(k)] = float(cum[idx]) if idx >= 0 else 0.0

    total_chars = int(sum(int(r["chars"]) for r in doc_rows))
    total_tokens = int(sum(int(r["tokens"]) for r in doc_rows))
    global_summary = {
        "tokenizer_json": str(args.tokenizer_json),
        "vocab_size": vocab_size,
        "docs_sampled": len(doc_rows),
        "chars_sampled": total_chars,
        "tokens_sampled": total_tokens,
        "chars_per_token_global": total_chars / total_tokens,
        "unk_rate_global": sum(float(r["unk_rate"]) * int(r["tokens"]) for r in doc_rows) / total_tokens,
        "single_piece_rate_global": sum(float(r["single_piece_rate"]) * int(r["tokens"]) for r in doc_rows)
        / total_tokens,
        "unique_token_ids_sampled": len(global_freq),
        "vocab_utilization_pct_sampled": len(global_freq) / vocab_size * 100,
        "top_k_token_coverage": coverage_at,
        "source_stats": source_stats,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    labels = [LABELS[s] for s in sources]
    box_data = [[float(r["chars_per_token"]) for r in doc_rows if r["source"] == s] for s in sources]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.boxplot(box_data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("chars per token")
    ax.set_title("Tokenizer compression by corpus source")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.out_dir / "chars_per_token_boxplot.png", dpi=180)
    plt.close(fig)

    means = [float(source_stats[s]["chars_per_token_mean"]) for s in sources]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(labels, means, color=["#4464ad", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    ax.set_ylabel("mean chars per token")
    ax.set_title("Average compression: 64K BPE across sources")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "chars_per_token_bar.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, cum * 100, color="#2a9d8f", linewidth=2)
    for k in [8000, 16000, 32000, 64000]:
        if k <= len(cum):
            ax.axvline(k, color="#888888", linestyle="--", linewidth=0.8)
            ax.text(k, min(99, cum[k - 1] * 100 + 1.2), f"top {k // 1000}K\n{cum[k - 1] * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("top-K token ids by sample frequency")
    ax.set_ylabel("cumulative token coverage (%)")
    ax.set_title("Vocabulary frequency coverage in sampled corpus")
    ax.set_ylim(0, 101)
    fig.tight_layout()
    fig.savefig(args.out_dir / "vocab_coverage_curve.png", dpi=180)
    plt.close(fig)

    length_keys = sorted(piece_len_freq)
    length_vals = [piece_len_freq[k] for k in length_keys]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(
        [str(k) if k < 16 else "16+" for k in length_keys],
        np.array(length_vals) / sum(length_vals) * 100,
        color="#4464ad",
    )
    ax.set_xlabel("decoded characters represented by one token")
    ax.set_ylabel("token occurrence share (%)")
    ax.set_title("Token piece length distribution")
    fig.tight_layout()
    fig.savefig(args.out_dir / "token_piece_length_hist.png", dpi=180)
    plt.close(fig)

    rows_html = "\n".join(
        f"<tr><td>{LABELS[s]}</td><td>{source_stats[s]['docs']}</td><td>{source_stats[s]['tokens']:,}</td>"
        f"<td>{source_stats[s]['chars_per_token_mean']:.2f}</td><td>{source_stats[s]['chars_per_token_median']:.2f}</td>"
        f"<td>{source_stats[s]['unk_rate'] * 100:.4f}%</td><td>{source_stats[s]['single_piece_rate'] * 100:.2f}%</td>"
        f"<td>{source_stats[s]['unique_token_ids']:,}</td><td>{source_stats[s]['vocab_utilization_pct']:.1f}%</td></tr>"
        for s in sources
    )
    coverage_rows = "\n".join(f"<tr><td>top {k}</td><td>{v * 100:.2f}%</td></tr>" for k, v in coverage_at.items())
    sample_sections = "\n".join(
        f"<h3>{LABELS[s]}</h3><div class='chips'>{sample_html.get(s, '')}</div>" for s in sources
    )
    html_text = f"""
<!doctype html><html><head><meta charset="utf-8"><title>Qwen3 0.1B Tokenizer Eval</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; color: #1f2933; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; background: #f8fafc; }}
.card b {{ display:block; font-size: 24px; margin-top: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #d8dee8; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; border: 1px solid #d8dee8; border-radius: 8px; margin: 12px 0 24px; }}
.tok {{ display:inline-flex; flex-direction:column; border:1px solid #cbd5e1; background:#f8fafc; border-radius:6px; padding:3px 5px; margin:2px; font-size:11px; }}
.tok b {{ color:#334155; }} .tok em {{ color:#0f766e; font-style:normal; max-width:130px; overflow:hidden; text-overflow:ellipsis; }}
</style></head><body>
<h1>Qwen3 0.1B 64K BPE Tokenizer Sampling Check</h1>
<p>Sample: {global_summary['docs_sampled']} docs, {global_summary['chars_sampled']:,} chars, {global_summary['tokens_sampled']:,} tokens from 5 corpus families.</p>
<div class="grid">
<div class="card">Vocab size<b>{vocab_size:,}</b></div>
<div class="card">Sample vocab used<b>{global_summary['unique_token_ids_sampled']:,} ({global_summary['vocab_utilization_pct_sampled']:.1f}%)</b></div>
<div class="card">Global chars/token<b>{global_summary['chars_per_token_global']:.2f}</b></div>
<div class="card">UNK rate<b>{global_summary['unk_rate_global'] * 100:.4f}%</b></div>
</div>
<h2>Source Metrics</h2>
<table><thead><tr><th>source</th><th>docs</th><th>tokens</th><th>mean chars/token</th><th>median chars/token</th><th>UNK</th><th>single-char token</th><th>unique ids</th><th>vocab used</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Plots</h2>
<img src="chars_per_token_boxplot.png"><img src="chars_per_token_bar.png"><img src="vocab_coverage_curve.png"><img src="token_piece_length_hist.png">
<h2>Top-K Coverage</h2><table><tbody>{coverage_rows}</tbody></table>
<h2>Tokenization Samples</h2>{sample_sections}
</body></html>
"""
    (args.out_dir / "index.html").write_text(html_text, encoding="utf-8")

    summary_txt = (
        f"Tokenizer: {args.tokenizer_json}\n"
        f"Vocab size: {vocab_size}\n"
        f"Sample docs: {global_summary['docs_sampled']}\n"
        f"Sample chars: {global_summary['chars_sampled']:,}\n"
        f"Sample tokens: {global_summary['tokens_sampled']:,}\n"
        f"Global chars/token: {global_summary['chars_per_token_global']:.3f}\n"
        f"UNK rate: {global_summary['unk_rate_global'] * 100:.6f}%\n"
        f"Single-char token rate: {global_summary['single_piece_rate_global'] * 100:.3f}%\n"
        f"Unique token ids observed: {global_summary['unique_token_ids_sampled']:,} "
        f"({global_summary['vocab_utilization_pct_sampled']:.2f}% of vocab)\n"
        f"Top-K coverage: {coverage_at}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary_txt, encoding="utf-8")
    print(summary_txt)
    print("REPORT_DIR", args.out_dir.resolve())


if __name__ == "__main__":
    main()
