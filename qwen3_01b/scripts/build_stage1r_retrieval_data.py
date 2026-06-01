#!/usr/bin/env python3
"""Build Stage1R short-context retrieval/copy warmup data.

The training shard is intentionally a normal raw-token manifest so it can be
used by the existing cli_pretrain.py path without changing the trainer. Each
training window is exactly seq_len + 1 tokens and the recommended training
stride is also seq_len + 1, so windows do not cross synthetic block boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


FILLER_SENTENCES = [
    "The file also includes routine scheduling notes that do not determine the stored value.",
    "A reviewer added neutral background text about staffing, rooms, dates, and ordinary updates.",
    "This unrelated paragraph describes maintenance logs, checklist items, and archived comments.",
    "Several entries in the document are distractors and should not be reused for the target record.",
    "The surrounding report contains generic prose about planning, budgets, and meeting summaries.",
    "A separate memo records historical notes and labels that are unrelated to the requested field.",
    "The archive repeats harmless status descriptions to make the document longer and less direct.",
    "Additional lines mention unrelated teams, reference numbers, and administrative observations.",
]

COLORS = [
    "blue",
    "green",
    "silver",
    "amber",
    "violet",
    "white",
    "black",
    "crimson",
    "teal",
    "gold",
]

NOUNS = [
    "river",
    "orbit",
    "harbor",
    "lantern",
    "forest",
    "summit",
    "copper",
    "signal",
    "meadow",
    "matrix",
]

CITIES = [
    "Denver",
    "Austin",
    "Boston",
    "Seattle",
    "Raleigh",
    "Phoenix",
    "Madison",
    "Columbus",
    "Portland",
    "Albany",
]

TASK_TYPES = [
    "kv_lookup",
    "passkey",
    "copy_span",
    "multi_field",
    "ledger_lookup",
]

POSITION_RATIOS = {
    "front": 0.12,
    "middle": 0.50,
    "near_end": 0.82,
}

TRAIN_LENGTH_WEIGHTS = [
    (256, 0.20),
    (512, 0.30),
    (1024, 0.30),
    (1800, 0.20),
]


@dataclass
class ExampleRecord:
    example_id: str
    split: str
    task_type: str
    template_family: str
    target_length: int
    evidence_position: str
    position_ratio: float
    distractor_count: int
    answer_type: str
    key: str
    gold_value: str
    distractor_values: list[str]
    prompt: str
    answer_text: str
    full_text: str
    prompt_tokens: int
    answer_tokens: int
    full_tokens: int
    evidence_token_start: int
    evidence_token_end: int
    evidence_to_answer_distance: int
    block_index: int | None = None
    token_start_in_block: int | None = None
    token_end_in_block: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--tokenizer-json", default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json")
    p.add_argument("--output-dir", default="data/processed/stage1r_retrieval_2k_bpe64k_v1")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--train-blocks", type=int, default=10000)
    p.add_argument("--eval-samples-per-cell", type=int, default=4)
    p.add_argument("--eval-lengths", nargs="*", type=int, default=[512, 1024, 2048])
    p.add_argument("--eval-positions", nargs="*", default=["front", "middle", "near_end"])
    p.add_argument("--eval-distractors", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--eod-id", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


def decode_prefix(tokenizer: Any, ids: list[int], max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    return tokenizer.decode(ids[:max_tokens])


def make_filler(tokenizer: Any, budget_tokens: int, rng: random.Random, label: str) -> str:
    if budget_tokens <= 0:
        return ""
    chunks: list[str] = []
    while token_count(tokenizer, "".join(chunks)) < budget_tokens + 24:
        sent = rng.choice(FILLER_SENTENCES)
        ref = rng.randint(100000, 999999)
        chunks.append(f"{sent} Reference {label}-{ref}.\n")
    ids = tokenizer.encode("".join(chunks))
    return decode_prefix(tokenizer, ids, budget_tokens)


def weighted_choice(rng: random.Random, weighted: list[tuple[int, float]]) -> int:
    total = sum(w for _, w in weighted)
    x = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += weight
        if x <= acc:
            return value
    return weighted[-1][0]


def numeric_value(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)}"


def alpha_code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(1000, 9999)}"


def phrase_value(rng: random.Random) -> str:
    # Equal word count makes length-normalized and summed-logprob rankings easier to interpret.
    return f"{rng.choice(COLORS)} {rng.choice(NOUNS)} {rng.randint(100, 999)} {rng.choice(NOUNS)}"


def unique_values(rng: random.Random, count: int, answer_type: str) -> list[str]:
    gen = {
        "number": numeric_value,
        "alphanumeric": alpha_code,
        "phrase": phrase_value,
    }[answer_type]
    values: list[str] = []
    seen: set[str] = set()
    while len(values) < count:
        v = gen(rng)
        if v not in seen:
            seen.add(v)
            values.append(v)
    return values


def make_key(rng: random.Random, task_type: str) -> str:
    if task_type == "multi_field":
        return f"{rng.randint(1000, 9999)}"
    if task_type == "ledger_lookup":
        return f"LEDGER-{rng.randint(10000, 99999)}"
    prefixes = {
        "kv_lookup": "ITEM",
        "passkey": "ARCHIVE",
        "copy_span": "CHANNEL",
    }
    return f"{prefixes.get(task_type, 'KEY')}_{rng.randint(10000, 99999)}"


def choose_answer_type(rng: random.Random, task_type: str) -> str:
    if task_type == "copy_span":
        return "phrase"
    if task_type == "passkey":
        return rng.choice(["number", "alphanumeric"])
    if task_type == "multi_field":
        return "alphanumeric"
    if task_type == "ledger_lookup":
        return rng.choice(["number", "alphanumeric"])
    return rng.choice(["number", "alphanumeric", "phrase"])


def split_template_variants(split: str, task_type: str) -> dict[str, str]:
    train = split == "train"
    if task_type == "kv_lookup":
        return {
            "header": "Record list:\n" if train else "Catalog extract:\n",
            "target_line": "{key} = {value}\n" if train else "entry {key} stores value {value}\n",
            "distractor_line": "{key} = {value}\n" if train else "entry {key} stores value {value}\n",
            "final": "The stored value for {key} is" if train else "The catalog value assigned to {key} is",
        }
    if task_type == "passkey":
        return {
            "header": "Archive notes:\n" if train else "Recovery file:\n",
            "target_line": "The recovery code assigned to {key} is {value}.\n" if train else "For {key}, the active recovery marker is {value}.\n",
            "distractor_line": "A decoy recovery code for {key} is {value}.\n" if train else "Inactive marker for {key}: {value}.\n",
            "final": "When {key} is summarized, the recovery code is" if train else "The active marker repeated for {key} is",
        }
    if task_type == "copy_span":
        return {
            "header": "Verification passages:\n" if train else "Copied phrase archive:\n",
            "target_line": "The verification phrase for {key} is \"{value}\".\n" if train else "Channel {key} carries the exact phrase \"{value}\".\n",
            "distractor_line": "The verification phrase for {key} is \"{value}\".\n" if train else "Channel {key} carries the exact phrase \"{value}\".\n",
            "final": "The verification phrase for {key} is repeated as" if train else "The exact phrase copied for {key} is",
        }
    if task_type == "multi_field":
        return {
            "header": "User records:\n" if train else "Account table:\n",
            "target_line": "ID {key} | city: {city} | token: {value} | color: {color}\n" if train else "account {key}; token={value}; city={city}; color={color}\n",
            "distractor_line": "ID {key} | city: {city} | token: {value} | color: {color}\n" if train else "account {key}; token={value}; city={city}; color={color}\n",
            "final": "The token for ID {key} is" if train else "The account token for {key} is",
        }
    return {
        "header": "Ledger records:\n" if train else "Audit ledger excerpt:\n",
        "target_line": "ledger {key} -> confirmation {value}\n" if train else "confirmation attached to {key}: {value}\n",
        "distractor_line": "ledger {key} -> confirmation {value}\n" if train else "confirmation attached to {key}: {value}\n",
        "final": "The confirmation attached to ledger {key} is" if train else "The audit confirmation for {key} is",
    }


def render_line(template: str, *, key: str, value: str, rng: random.Random) -> str:
    return template.format(
        key=key,
        value=value,
        city=rng.choice(CITIES),
        color=rng.choice(COLORS),
    )


def build_core_parts(
    *,
    split: str,
    task_type: str,
    distractor_count: int,
    rng: random.Random,
) -> tuple[str, str, str, str, str, list[str], str, str]:
    answer_type = choose_answer_type(rng, task_type)
    values = unique_values(rng, distractor_count + 1, answer_type)
    gold = values[0]
    distractors = values[1:]
    key = make_key(rng, task_type)
    distractor_keys = [make_key(rng, task_type) for _ in range(distractor_count)]
    templates = split_template_variants(split, task_type)
    records = [(key, gold, True)] + [(k, v, False) for k, v in zip(distractor_keys, distractors)]
    rng.shuffle(records)

    lines = [templates["header"]]
    evidence_text = ""
    for rec_key, rec_value, is_target in records:
        line_tmpl = templates["target_line"] if is_target else templates["distractor_line"]
        line = render_line(line_tmpl, key=rec_key, value=rec_value, rng=rng)
        lines.append(line)
        if is_target:
            evidence_text = line
    record_block = "".join(lines)
    final_prefix = templates["final"].format(key=key)
    prompt_tail = f"\nSummary line:\n{final_prefix}"
    answer_text = f" {gold}"
    full_suffix = ".\n"
    return record_block, prompt_tail, answer_text, full_suffix, key, distractors, gold, answer_type, evidence_text


def assemble_example(
    tokenizer: Any,
    *,
    split: str,
    example_id: str,
    task_type: str,
    target_length: int,
    evidence_position: str,
    distractor_count: int,
    rng: random.Random,
) -> ExampleRecord:
    ratio = POSITION_RATIOS[evidence_position]
    (
        record_block,
        prompt_tail,
        answer_text,
        full_suffix,
        key,
        distractors,
        gold,
        answer_type,
        evidence_text,
    ) = build_core_parts(
        split=split,
        task_type=task_type,
        distractor_count=distractor_count,
        rng=rng,
    )
    prefix_label = f"{split}-{example_id}-pre"
    suffix_label = f"{split}-{example_id}-post"

    fixed_prompt = record_block + prompt_tail
    fixed_full = fixed_prompt + answer_text + full_suffix
    fixed_tokens = token_count(tokenizer, fixed_full)
    budget = max(0, target_length - fixed_tokens)
    prefix_budget = int(math.floor(budget * ratio))
    suffix_budget = max(0, budget - prefix_budget)

    # Iteratively trim filler if token-boundary effects push the example over budget.
    for _ in range(6):
        prefix = make_filler(tokenizer, prefix_budget, rng, prefix_label)
        suffix = make_filler(tokenizer, suffix_budget, rng, suffix_label)
        prompt = prefix + record_block + suffix + prompt_tail
        full_text = prompt + answer_text + full_suffix
        full_tokens = token_count(tokenizer, full_text)
        if full_tokens <= target_length:
            break
        overflow = full_tokens - target_length
        if suffix_budget >= overflow:
            suffix_budget -= overflow
        elif prefix_budget > 0:
            prefix_budget = max(0, prefix_budget - (overflow - suffix_budget))
            suffix_budget = 0
        else:
            break

    before_evidence = prefix + record_block.split(evidence_text, 1)[0]
    evidence_start = token_count(tokenizer, before_evidence)
    evidence_end = evidence_start + token_count(tokenizer, evidence_text)
    prompt_tokens = token_count(tokenizer, prompt)
    answer_tokens = token_count(tokenizer, answer_text)
    full_tokens = token_count(tokenizer, full_text)
    return ExampleRecord(
        example_id=example_id,
        split=split,
        task_type=task_type,
        template_family="heldin" if split == "train" else "heldout",
        target_length=target_length,
        evidence_position=evidence_position,
        position_ratio=ratio,
        distractor_count=distractor_count,
        answer_type=answer_type,
        key=key,
        gold_value=gold,
        distractor_values=distractors,
        prompt=prompt,
        answer_text=answer_text,
        full_text=full_text,
        prompt_tokens=prompt_tokens,
        answer_tokens=answer_tokens,
        full_tokens=full_tokens,
        evidence_token_start=evidence_start,
        evidence_token_end=evidence_end,
        evidence_to_answer_distance=max(0, prompt_tokens - evidence_end),
    )


def build_train_data(args: argparse.Namespace, tokenizer: Any, out_dir: Path) -> dict[str, Any]:
    block_tokens = args.seq_len + 1
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    shard_path = train_dir / "shard_00000.bin"
    meta_path = out_dir / "train_examples.jsonl"
    if shard_path.exists() and not args.overwrite:
        raise SystemExit(f"{shard_path} exists; pass --overwrite to rebuild")

    rng = random.Random(args.seed)
    task_counts: dict[str, int] = {k: 0 for k in TASK_TYPES}
    length_counts: dict[str, int] = {}
    position_counts: dict[str, int] = {k: 0 for k in POSITION_RATIOS}
    answer_type_counts: dict[str, int] = {}
    example_count = 0
    block: list[int] = []
    block_index = 0

    with shard_path.open("wb") as bin_f, meta_path.open("w", encoding="utf-8") as meta_f:
        while block_index < args.train_blocks:
            task_type = rng.choice(TASK_TYPES)
            target_length = weighted_choice(rng, TRAIN_LENGTH_WEIGHTS)
            evidence_position = rng.choice(list(POSITION_RATIOS))
            distractor_count = rng.choice([4, 8, 16, 32])
            ex = assemble_example(
                tokenizer,
                split="train",
                example_id=f"train_{example_count:08d}",
                task_type=task_type,
                target_length=target_length,
                evidence_position=evidence_position,
                distractor_count=distractor_count,
                rng=rng,
            )
            ids = tokenizer.encode(ex.full_text) + [int(args.eod_id)]
            if len(ids) > block_tokens:
                # The generator should avoid this, but skip defensively.
                continue
            if block and len(block) + len(ids) > block_tokens:
                block.extend([int(args.eod_id)] * (block_tokens - len(block)))
                np.asarray(block, dtype=np.uint16).tofile(bin_f)
                block_index += 1
                block = []
                if block_index >= args.train_blocks:
                    break
            token_start = len(block)
            block.extend(ids)
            ex.block_index = block_index
            ex.token_start_in_block = token_start
            ex.token_end_in_block = token_start + len(ids)
            write_jsonl_line(meta_f, asdict(ex))
            example_count += 1
            task_counts[task_type] += 1
            length_counts[str(target_length)] = length_counts.get(str(target_length), 0) + 1
            position_counts[evidence_position] += 1
            answer_type_counts[ex.answer_type] = answer_type_counts.get(ex.answer_type, 0) + 1

        if block_index < args.train_blocks and block:
            block.extend([int(args.eod_id)] * (block_tokens - len(block)))
            np.asarray(block, dtype=np.uint16).tofile(bin_f)
            block_index += 1

    tokens = int(block_index * block_tokens)
    return {
        "path": str(shard_path.relative_to(out_dir)),
        "tokens": tokens,
        "dtype": "uint16",
        "blocks": block_index,
        "block_tokens": block_tokens,
        "examples": example_count,
        "task_counts": task_counts,
        "target_length_counts": length_counts,
        "evidence_position_counts": position_counts,
        "answer_type_counts": answer_type_counts,
    }


def build_eval_data(args: argparse.Namespace, tokenizer: Any, out_dir: Path) -> dict[str, Any]:
    eval_path = out_dir / "eval_examples.jsonl"
    rng = random.Random(args.seed + 97_531)
    rows = 0
    task_counts: dict[str, int] = {k: 0 for k in TASK_TYPES}
    with eval_path.open("w", encoding="utf-8") as f:
        for length in args.eval_lengths:
            for position in args.eval_positions:
                if position not in POSITION_RATIOS:
                    raise SystemExit(f"unknown eval position: {position}")
                for task_type in TASK_TYPES:
                    for sample in range(args.eval_samples_per_cell):
                        ex = assemble_example(
                            tokenizer,
                            split="eval",
                            example_id=f"eval_len{length}_{position}_{task_type}_{sample:03d}",
                            task_type=task_type,
                            target_length=int(length),
                            evidence_position=position,
                            distractor_count=int(args.eval_distractors),
                            rng=rng,
                        )
                        write_jsonl_line(f, asdict(ex))
                        rows += 1
                        task_counts[task_type] += 1
    return {
        "path": str(eval_path.relative_to(out_dir)),
        "examples": rows,
        "task_counts": task_counts,
        "lengths": [int(x) for x in args.eval_lengths],
        "positions": list(args.eval_positions),
        "distractors": int(args.eval_distractors),
        "samples_per_cell": int(args.eval_samples_per_cell),
    }


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    train = manifest["stage1r_train"]
    eval_meta = manifest["stage1r_eval"]
    lines = [
        "# Stage1R Retrieval Warmup Dataset V1",
        "",
        "This dataset is designed for a short Stage1R experiment: teach a base LM",
        "to prefer evidence-conditioned continuations in 2K context before long-context",
        "RoPE extrapolation experiments.",
        "",
        "## Training Shard",
        "",
        f"- Blocks: `{train['blocks']}`",
        f"- Block tokens: `{train['block_tokens']}`",
        f"- Total raw tokens: `{train['tokens']}`",
        f"- Examples packed: `{train['examples']}`",
        "- Recommended training seq_len: `2048`",
        "- Recommended training stride: `2049`",
        "",
        "The shard is packed so each training window is exactly one synthetic block.",
        "Use `--stride 2049` with the existing token-manifest dataset.",
        "",
        "## Task Mix",
        "",
        "| Task | Train examples | Eval examples |",
        "|---|---:|---:|",
    ]
    for task in TASK_TYPES:
        lines.append(f"| {task} | {train['task_counts'].get(task, 0)} | {eval_meta['task_counts'].get(task, 0)} |")
    lines.extend(
        [
            "",
            "## Design Notes",
            "",
            "- Training templates and eval templates are intentionally different.",
            "- Eval negatives are same-context distractor values, not random unrelated values.",
            "- Evidence positions cover front, middle, and near-end within the 2K window.",
            "- Answers cover numbers, alphanumeric codes, and fixed-word-count phrase spans.",
            "- This V1 shard uses ordinary causal LM loss; answer-only loss is a later improvement.",
            "",
        ]
    )
    (out_dir / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json

    out_dir = resolve(root, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = resolve(root, args.tokenizer_json)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else 64000

    train_meta = build_train_data(args, tokenizer, out_dir)
    eval_meta = build_eval_data(args, tokenizer, out_dir)
    manifest = {
        "format": "raw_token_ids",
        "dtype": "uint16",
        "vocab_size": int(vocab_size),
        "tokenizer_json": str(tokenizer_path),
        "eod_id": int(args.eod_id),
        "seq_len": int(args.seq_len),
        "recommended_stride": int(args.seq_len + 1),
        "stage": "stage1r_retrieval_warmup_v1",
        "selection_method": "synthetic retrieval/copy continuation blocks with held-out eval templates",
        "total_tokens": int(train_meta["tokens"]),
        "stage1r_train": train_meta,
        "stage1r_eval": eval_meta,
        "shards": [
            {
                "path": train_meta["path"],
                "tokens": int(train_meta["tokens"]),
                "dtype": "uint16",
                "blocks": int(train_meta["blocks"]),
                "block_tokens": int(train_meta["block_tokens"]),
            }
        ],
        "run_config": vars(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run_config.json", vars(args))
    write_data_card(out_dir, manifest)
    print(json.dumps({"output_dir": str(out_dir), "manifest": str(out_dir / "manifest.json"), "train": train_meta, "eval": eval_meta}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
