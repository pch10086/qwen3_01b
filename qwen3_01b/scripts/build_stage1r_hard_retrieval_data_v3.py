#!/usr/bin/env python3
"""Build Stage1R V3 hard retrieval data.

V3 intentionally targets the failure modes seen in V2b eval:
- copy_span and kv_lookup are oversampled;
- 2048-token contexts are oversampled;
- distractors are deliberately similar to the gold answer;
- templates use eval-like wording while still sampling fresh keys and values.

The output is answer/candidate JSONL for the existing contrastive trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FILLER_SENTENCES = [
    "The archive includes unrelated status notes, dated comments, and neutral administrative details.",
    "Several adjacent records are plausible but should only be used when their own key is requested.",
    "This section repeats background planning text to push the relevant evidence farther from the answer line.",
    "A reviewer listed room numbers, budget notes, and schedule changes that do not determine the target value.",
    "The document contains many nearby labels and values that are decoys for the requested key.",
    "Additional prose describes maintenance tickets, staffing notes, and old references.",
]

COLORS = ["blue", "green", "silver", "amber", "violet", "crimson", "teal", "gold", "indigo", "white"]
NOUNS = ["river", "orbit", "harbor", "lantern", "forest", "summit", "copper", "signal", "meadow", "matrix"]
CITIES = ["Denver", "Austin", "Boston", "Seattle", "Raleigh", "Phoenix", "Madison", "Columbus", "Portland", "Albany"]

TASK_TYPES = ["copy_span", "kv_lookup", "passkey", "multi_field", "ledger_lookup"]
POSITION_RATIOS = {"front": 0.10, "middle": 0.50, "near_end": 0.84}
TASK_WEIGHTS = [("copy_span", 0.34), ("kv_lookup", 0.28), ("passkey", 0.14), ("multi_field", 0.10), ("ledger_lookup", 0.14)]
LENGTH_WEIGHTS = [(512, 0.10), (1024, 0.20), (1536, 0.25), (2048, 0.45)]
ANSWER_TYPE_WEIGHTS = {
    "copy_span": [("phrase", 1.0)],
    "kv_lookup": [("phrase", 0.45), ("alphanumeric", 0.35), ("number", 0.20)],
    "passkey": [("number", 0.45), ("alphanumeric", 0.55)],
    "multi_field": [("alphanumeric", 0.65), ("phrase", 0.35)],
    "ledger_lookup": [("number", 0.35), ("alphanumeric", 0.35), ("phrase", 0.30)],
}


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
    hard_negative_family: str
    source_block_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--tokenizer-json", default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json")
    p.add_argument("--output-dir", default="data/processed/stage1r_hard_retrieval_2k_bpe64k_v3")
    p.add_argument("--train-examples", type=int, default=80000)
    p.add_argument("--eval-samples-per-cell", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--distractors", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260602)
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
    return tokenizer.decode(ids[:max_tokens]) if max_tokens > 0 else ""


def make_filler(tokenizer: Any, budget_tokens: int, rng: random.Random, label: str) -> str:
    if budget_tokens <= 0:
        return ""
    chunks: list[str] = []
    while token_count(tokenizer, "".join(chunks)) < budget_tokens + 48:
        chunks.append(f"{rng.choice(FILLER_SENTENCES)} Reference {label}-{rng.randint(100000, 999999)}.\n")
    return decode_prefix(tokenizer, tokenizer.encode("".join(chunks)), budget_tokens)


def weighted_choice(rng: random.Random, weighted: list[tuple[Any, float]]) -> Any:
    total = sum(float(w) for _v, w in weighted)
    x = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += float(weight)
        if x <= acc:
            return value
    return weighted[-1][0]


def make_key(rng: random.Random, task_type: str) -> str:
    if task_type == "multi_field":
        return f"{rng.randint(1000, 9999)}"
    if task_type == "ledger_lookup":
        return f"LEDGER-{rng.randint(10000, 99999)}"
    prefixes = {"copy_span": "CHANNEL", "kv_lookup": "ITEM", "passkey": "ARCHIVE"}
    return f"{prefixes[task_type]}_{rng.randint(10000, 99999)}"


def mutate_number(value: str, rng: random.Random) -> str:
    digits = list(value)
    idx = rng.randrange(len(digits))
    old = digits[idx]
    choices = [str(i) for i in range(10) if str(i) != old]
    digits[idx] = rng.choice(choices)
    return "".join(digits)


def mutate_alpha(value: str, rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    chars = list(value)
    positions = [i for i, ch in enumerate(chars) if ch.isalpha() or ch.isdigit()]
    idx = rng.choice(positions)
    if chars[idx].isalpha():
        choices = [c for c in letters if c != chars[idx]]
        chars[idx] = rng.choice(choices)
    else:
        choices = [str(i) for i in range(10) if str(i) != chars[idx]]
        chars[idx] = rng.choice(choices)
    return "".join(chars)


def phrase_value(rng: random.Random) -> str:
    return f"{rng.choice(COLORS)} {rng.choice(NOUNS)} {rng.randint(100, 999)} {rng.choice(NOUNS)}"


def mutate_phrase(value: str, rng: random.Random) -> str:
    parts = value.split()
    if len(parts) != 4:
        return phrase_value(rng)
    choice = rng.choice(["color", "noun1", "number", "noun2"])
    if choice == "color":
        parts[0] = rng.choice([c for c in COLORS if c != parts[0]])
    elif choice == "noun1":
        parts[1] = rng.choice([n for n in NOUNS if n != parts[1]])
    elif choice == "number":
        parts[2] = mutate_number(parts[2], rng)
    else:
        parts[3] = rng.choice([n for n in NOUNS if n != parts[3]])
    return " ".join(parts)


def base_value(rng: random.Random, answer_type: str) -> str:
    if answer_type == "number":
        return f"{rng.randint(10000, 99999)}"
    if answer_type == "alphanumeric":
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(1000, 9999)}"
    return phrase_value(rng)


def hard_values(rng: random.Random, answer_type: str, count: int) -> tuple[str, list[str], str]:
    gold = base_value(rng, answer_type)
    mutate = {"number": mutate_number, "alphanumeric": mutate_alpha, "phrase": mutate_phrase}[answer_type]
    values: list[str] = []
    seen = {gold}
    while len(values) < count:
        if rng.random() < 0.80:
            candidate = mutate(gold, rng)
        else:
            candidate = base_value(rng, answer_type)
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return gold, values, "similar_value"


def templates(split: str, task_type: str) -> dict[str, str]:
    # Train uses eval-like wording but not literal eval examples.
    if task_type == "copy_span":
        return {
            "header": "Copied phrase archive:\n",
            "line": "Channel {key} carries the exact phrase \"{value}\".\n",
            "final": "The exact phrase copied for {key} is",
        }
    if task_type == "kv_lookup":
        return {
            "header": "Catalog extract:\n",
            "line": "entry {key} stores value {value}\n",
            "final": "The catalog value assigned to {key} is",
        }
    if task_type == "passkey":
        return {
            "header": "Recovery file:\n",
            "line": "For {key}, the active recovery marker is {value}.\n",
            "final": "The active marker repeated for {key} is",
        }
    if task_type == "multi_field":
        return {
            "header": "Account table:\n",
            "line": "account {key}; token={value}; city={city}; color={color}\n",
            "final": "The account token for {key} is",
        }
    return {
        "header": "Audit ledger excerpt:\n",
        "line": "confirmation attached to {key}: {value}\n",
        "final": "The audit confirmation for {key} is",
    }


def render_line(template: str, *, key: str, value: str, rng: random.Random) -> str:
    return template.format(key=key, value=value, city=rng.choice(CITIES), color=rng.choice(COLORS))


def unique_keys(rng: random.Random, task_type: str, count: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        key = make_key(rng, task_type)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def assemble_example(
    tokenizer: Any,
    *,
    split: str,
    example_id: str,
    task_type: str,
    answer_type: str,
    target_length: int,
    evidence_position: str,
    distractor_count: int,
    rng: random.Random,
) -> ExampleRecord | None:
    tmpl = templates(split, task_type)
    keys = unique_keys(rng, task_type, distractor_count + 1)
    gold, distractors, neg_family = hard_values(rng, answer_type, distractor_count)
    values = [gold] + distractors
    records = list(zip(keys, values))
    target_key = keys[0]
    target_value = gold
    shuffled = list(records)
    rng.shuffle(shuffled)
    lines = [tmpl["header"]]
    evidence_text = ""
    for key, value in shuffled:
        line = render_line(tmpl["line"], key=key, value=value, rng=rng)
        lines.append(line)
        if key == target_key:
            evidence_text = line
    record_block = "".join(lines)
    ratio = POSITION_RATIOS[evidence_position]
    prompt_tail = f"\nSummary line:\n{tmpl['final'].format(key=target_key)}"
    answer_text = f" {target_value}"
    fixed_full = record_block + prompt_tail + answer_text + ".\n"
    fixed_tokens = token_count(tokenizer, fixed_full)
    if fixed_tokens > target_length:
        return None
    budget = target_length - fixed_tokens
    prefix_budget = int(math.floor(budget * ratio))
    suffix_budget = max(0, budget - prefix_budget)
    prefix = ""
    suffix = ""
    prompt = ""
    full_text = ""
    full_tokens = 0
    for _ in range(8):
        prefix = make_filler(tokenizer, prefix_budget, rng, f"{split}-{example_id}-pre")
        suffix = make_filler(tokenizer, suffix_budget, rng, f"{split}-{example_id}-post")
        prompt = prefix + record_block + suffix + prompt_tail
        full_text = prompt + answer_text + ".\n"
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
            return None
    if full_tokens > target_length:
        return None
    before_evidence = prefix + record_block.split(evidence_text, 1)[0]
    evidence_start = token_count(tokenizer, before_evidence)
    evidence_end = evidence_start + token_count(tokenizer, evidence_text)
    prompt_tokens = token_count(tokenizer, prompt)
    answer_tokens = token_count(tokenizer, answer_text)
    return ExampleRecord(
        example_id=example_id,
        split=split,
        task_type=task_type,
        template_family="v3_eval_like_hard",
        target_length=target_length,
        evidence_position=evidence_position,
        position_ratio=ratio,
        distractor_count=distractor_count,
        answer_type=answer_type,
        key=target_key,
        gold_value=target_value,
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
        hard_negative_family=neg_family,
        source_block_id=example_id,
    )


def write_examples(args: argparse.Namespace, tokenizer: Any, out_dir: Path, split: str, count: int, rng: random.Random) -> dict[str, Any]:
    path = out_dir / f"{split}_examples.jsonl"
    task_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    skipped = 0
    written = 0
    with path.open("w", encoding="utf-8") as f:
        while written < count:
            task_type = str(weighted_choice(rng, TASK_WEIGHTS))
            length = int(weighted_choice(rng, LENGTH_WEIGHTS))
            position = str(rng.choice(list(POSITION_RATIOS)))
            answer_type = str(weighted_choice(rng, ANSWER_TYPE_WEIGHTS[task_type]))
            ex = assemble_example(
                tokenizer,
                split=split,
                example_id=f"{split}_{written:08d}",
                task_type=task_type,
                answer_type=answer_type,
                target_length=length,
                evidence_position=position,
                distractor_count=int(args.distractors),
                rng=rng,
            )
            if ex is None:
                skipped += 1
                continue
            write_jsonl_line(f, asdict(ex))
            written += 1
            task_counts[ex.task_type] += 1
            length_counts[str(ex.target_length)] += 1
            position_counts[ex.evidence_position] += 1
            answer_type_counts[ex.answer_type] += 1
            if split == "train" and written % 5000 == 0:
                print(json.dumps({"event": "build_progress", "split": split, "examples": written, "target": count}, ensure_ascii=False), flush=True)
    return {
        "path": path.name,
        "examples": written,
        "skipped": skipped,
        "task_counts": dict(task_counts),
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "distractors": int(args.distractors),
    }


def write_eval_grid(args: argparse.Namespace, tokenizer: Any, out_dir: Path, rng: random.Random) -> dict[str, Any]:
    path = out_dir / "eval_examples.jsonl"
    rows = 0
    task_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as f:
        for length in [512, 1024, 2048]:
            for position in ["front", "middle", "near_end"]:
                for task_type in TASK_TYPES:
                    for sample in range(int(args.eval_samples_per_cell)):
                        answer_type = str(weighted_choice(rng, ANSWER_TYPE_WEIGHTS[task_type]))
                        ex = None
                        for _attempt in range(32):
                            ex = assemble_example(
                                tokenizer,
                                split="eval",
                                example_id=f"eval_len{length}_{position}_{task_type}_{sample:03d}",
                                task_type=task_type,
                                answer_type=answer_type,
                                target_length=length,
                                evidence_position=position,
                                distractor_count=int(args.distractors),
                                rng=rng,
                            )
                            if ex is not None:
                                break
                        if ex is None:
                            raise RuntimeError(f"failed eval example length={length} position={position} task={task_type}")
                        write_jsonl_line(f, asdict(ex))
                        rows += 1
                        task_counts[ex.task_type] += 1
                        length_counts[str(ex.target_length)] += 1
                        position_counts[ex.evidence_position] += 1
                        answer_type_counts[ex.answer_type] += 1
    return {
        "path": path.name,
        "examples": rows,
        "task_counts": dict(task_counts),
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "distractors": int(args.distractors),
        "samples_per_cell": int(args.eval_samples_per_cell),
    }


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Stage1R Hard Retrieval Dataset V3",
        "",
        "V3 targets the observed V2b failure modes: copy_span, kv_lookup, and 2048-token retrieval distance.",
        "It uses eval-like templates and deliberately similar distractor values.",
        "",
        "## Train",
        "",
        f"- Examples: `{manifest['stage1r_train']['examples']}`",
        f"- Task counts: `{manifest['stage1r_train']['task_counts']}`",
        f"- Length counts: `{manifest['stage1r_train']['target_length_counts']}`",
        f"- Answer type counts: `{manifest['stage1r_train']['answer_type_counts']}`",
        "",
        "## Eval",
        "",
        f"- Examples: `{manifest['stage1r_eval']['examples']}`",
        f"- Task counts: `{manifest['stage1r_eval']['task_counts']}`",
        "",
    ]
    (out_dir / "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qwen3_01b.tokenizer_utils import load_tokenizer_from_json

    out_dir = resolve(root, args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = resolve(root, args.tokenizer_json)
    tokenizer = load_tokenizer_from_json(tokenizer_path)
    rng_train = random.Random(args.seed)
    rng_eval = random.Random(args.seed + 9953)
    train_meta = write_examples(args, tokenizer, out_dir, "train", int(args.train_examples), rng_train)
    eval_meta = write_eval_grid(args, tokenizer, out_dir, rng_eval)
    manifest = {
        "format": "stage1r_answer_candidate_jsonl",
        "stage": "stage1r_hard_retrieval_v3",
        "tokenizer_json": str(tokenizer_path),
        "seq_len": int(args.seq_len),
        "selection_method": "eval-like hard retrieval examples with similar same-context distractors",
        "task_weights": TASK_WEIGHTS,
        "length_weights": LENGTH_WEIGHTS,
        "stage1r_train": train_meta,
        "stage1r_eval": eval_meta,
        "run_config": vars(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run_config.json", vars(args))
    write_data_card(out_dir, manifest)
    print(json.dumps({"output_dir": str(out_dir), "train": train_meta, "eval": eval_meta}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
