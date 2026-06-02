#!/usr/bin/env python3
"""Build Stage1R V2 retrieval warmup data.

V2 is designed for answer-only retrieval training, not ordinary full-token LM
training.  The important difference from V1 is that many training examples are
multi-query records: every record in a small table becomes the gold answer for
one query, while the other records are same-context hard negatives.  This makes
the supervision target key-value binding instead of memorizing answer formats or
globally common values.
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
    "The surrounding note lists routine scheduling changes and archived status details.",
    "A neutral paragraph describes maintenance tickets, calendar updates, and unrelated comments.",
    "This background line records ordinary planning information that does not determine the requested value.",
    "Several adjacent entries are decoys and should only be used when their own key is requested.",
    "The memo repeats administrative observations about rooms, teams, budgets, and dates.",
    "A separate section includes historical labels and reference numbers unrelated to the target key.",
    "The archive contains harmless filler text to make the evidence farther from the final continuation.",
    "Additional prose mentions staffing, checkpoints, summaries, and generic operational remarks.",
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
    "orange",
    "indigo",
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
    "anchor",
    "quartz",
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
    "Chicago",
    "Dallas",
]

TASK_TYPES = [
    "kv_lookup",
    "passkey",
    "copy_span",
    "multi_field",
    "ledger_lookup",
]

POSITION_RATIOS = {
    "front": 0.10,
    "middle": 0.50,
    "near_end": 0.84,
}

# V2 intentionally oversamples long effective distances inside the same 2K
# context window because V1 was weakest at 2048.
TRAIN_LENGTH_WEIGHTS = [
    (512, 0.15),
    (1024, 0.25),
    (1536, 0.25),
    (2048, 0.35),
]

TASK_WEIGHTS = [
    ("kv_lookup", 0.18),
    ("passkey", 0.22),
    ("copy_span", 0.24),
    ("multi_field", 0.16),
    ("ledger_lookup", 0.20),
]

ANSWER_TYPE_WEIGHTS = {
    "kv_lookup": [("number", 0.35), ("alphanumeric", 0.35), ("phrase", 0.30)],
    "passkey": [("number", 0.55), ("alphanumeric", 0.45)],
    "copy_span": [("phrase", 1.0)],
    "multi_field": [("alphanumeric", 0.60), ("phrase", 0.40)],
    "ledger_lookup": [("number", 0.40), ("alphanumeric", 0.35), ("phrase", 0.25)],
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
    query_group_id: str
    query_index: int
    records_per_group: int
    source_block_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="/home/public/bjh/dym/NLP_longcontext")
    p.add_argument("--tokenizer-json", default="qwen3_01b/tokenizers/bpe_64k_clean/tokenizer.json")
    p.add_argument("--output-dir", default="data/processed/stage1r_retrieval_2k_bpe64k_v2")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--train-examples", type=int, default=120000)
    p.add_argument("--eval-samples-per-cell", type=int, default=8)
    p.add_argument("--eval-lengths", nargs="*", type=int, default=[512, 1024, 2048])
    p.add_argument("--eval-positions", nargs="*", default=["front", "middle", "near_end"])
    p.add_argument("--eval-distractors", type=int, default=16)
    p.add_argument("--min-records-per-group", type=int, default=4)
    p.add_argument("--max-records-per-group", type=int, default=8)
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
    if max_tokens <= 0:
        return ""
    return tokenizer.decode(ids[:max_tokens])


def make_filler(tokenizer: Any, budget_tokens: int, rng: random.Random, label: str) -> str:
    if budget_tokens <= 0:
        return ""
    chunks: list[str] = []
    while token_count(tokenizer, "".join(chunks)) < budget_tokens + 48:
        sent = rng.choice(FILLER_SENTENCES)
        ref = rng.randint(100000, 999999)
        chunks.append(f"{sent} Reference {label}-{ref}.\n")
    ids = tokenizer.encode("".join(chunks))
    return decode_prefix(tokenizer, ids, budget_tokens)


def weighted_choice(rng: random.Random, weighted: list[tuple[Any, float]]) -> Any:
    total = sum(float(w) for _value, w in weighted)
    x = rng.random() * total
    acc = 0.0
    for value, weight in weighted:
        acc += float(weight)
        if x <= acc:
            return value
    return weighted[-1][0]


def numeric_value(rng: random.Random) -> str:
    return f"{rng.randint(10000, 99999)}"


def alpha_code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(1000, 9999)}"


def phrase_value(rng: random.Random) -> str:
    # Four content-like fields keep summed-logprob and mean-NLL ranking less
    # sensitive to answer length.
    return f"{rng.choice(COLORS)} {rng.choice(NOUNS)} {rng.randint(100, 999)} {rng.choice(NOUNS)}"


def make_value(rng: random.Random, answer_type: str) -> str:
    if answer_type == "number":
        return numeric_value(rng)
    if answer_type == "alphanumeric":
        return alpha_code(rng)
    if answer_type == "phrase":
        return phrase_value(rng)
    raise ValueError(answer_type)


def unique_values(rng: random.Random, count: int, answer_type: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        value = make_value(rng, answer_type)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


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


def unique_keys(rng: random.Random, task_type: str, count: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < count:
        key = make_key(rng, task_type)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def choose_answer_type(rng: random.Random, task_type: str) -> str:
    return str(weighted_choice(rng, ANSWER_TYPE_WEIGHTS[task_type]))


def template_variants(split: str, task_type: str) -> dict[str, str]:
    train = split == "train"
    if task_type == "kv_lookup":
        return {
            "header": "Record list:\n" if train else "Catalog extract:\n",
            "line": "{key} = {value}\n" if train else "entry {key} stores value {value}\n",
            "final": "The stored value for {key} is" if train else "The catalog value assigned to {key} is",
        }
    if task_type == "passkey":
        return {
            "header": "Archive notes:\n" if train else "Recovery file:\n",
            "line": "The active recovery code assigned to {key} is {value}.\n" if train else "For {key}, the active recovery marker is {value}.\n",
            "final": "When {key} is summarized, the active recovery code is" if train else "The active marker repeated for {key} is",
        }
    if task_type == "copy_span":
        return {
            "header": "Verification passages:\n" if train else "Copied phrase archive:\n",
            "line": "The verification phrase for {key} is \"{value}\".\n" if train else "Channel {key} carries the exact phrase \"{value}\".\n",
            "final": "The verification phrase for {key} is repeated as" if train else "The exact phrase copied for {key} is",
        }
    if task_type == "multi_field":
        return {
            "header": "User records:\n" if train else "Account table:\n",
            "line": "ID {key} | city: {city} | token: {value} | color: {color}\n" if train else "account {key}; token={value}; city={city}; color={color}\n",
            "final": "The token for ID {key} is" if train else "The account token for {key} is",
        }
    return {
        "header": "Ledger records:\n" if train else "Audit ledger excerpt:\n",
        "line": "ledger {key} -> confirmation {value}\n" if train else "confirmation attached to {key}: {value}\n",
        "final": "The confirmation attached to ledger {key} is" if train else "The audit confirmation for {key} is",
    }


def render_line(template: str, *, key: str, value: str, rng: random.Random) -> str:
    return template.format(
        key=key,
        value=value,
        city=rng.choice(CITIES),
        color=rng.choice(COLORS),
    )


def build_record_group(
    *,
    split: str,
    task_type: str,
    answer_type: str,
    records_per_group: int,
    rng: random.Random,
) -> tuple[str, list[dict[str, str]], dict[str, str]]:
    templates = template_variants(split, task_type)
    keys = unique_keys(rng, task_type, records_per_group)
    values = unique_values(rng, records_per_group, answer_type)
    rows = [{"key": key, "value": value} for key, value in zip(keys, values)]
    shuffled = list(rows)
    rng.shuffle(shuffled)
    lines = [templates["header"]]
    rendered_by_key: dict[str, str] = {}
    for row in shuffled:
        line = render_line(templates["line"], key=row["key"], value=row["value"], rng=rng)
        lines.append(line)
        rendered_by_key[row["key"]] = line
    return "".join(lines), rows, rendered_by_key


def assemble_from_group(
    tokenizer: Any,
    *,
    split: str,
    example_id: str,
    query_group_id: str,
    query_index: int,
    task_type: str,
    answer_type: str,
    target_length: int,
    evidence_position: str,
    record_block: str,
    rows: list[dict[str, str]],
    rendered_by_key: dict[str, str],
    rng: random.Random,
) -> ExampleRecord | None:
    target = rows[query_index]
    key = target["key"]
    gold = target["value"]
    distractors = [row["value"] for row in rows if row["key"] != key]
    templates = template_variants(split, task_type)
    ratio = POSITION_RATIOS[evidence_position]
    prompt_tail = f"\nSummary line:\n{templates['final'].format(key=key)}"
    answer_text = f" {gold}"
    full_suffix = ".\n"

    fixed_prompt = record_block + prompt_tail
    fixed_full = fixed_prompt + answer_text + full_suffix
    fixed_tokens = token_count(tokenizer, fixed_full)
    if fixed_tokens > target_length:
        return None
    budget = max(0, target_length - fixed_tokens)
    prefix_budget = int(math.floor(budget * ratio))
    suffix_budget = max(0, budget - prefix_budget)

    prefix = ""
    suffix = ""
    prompt = fixed_prompt
    full_text = fixed_full
    full_tokens = fixed_tokens
    for _ in range(8):
        prefix = make_filler(tokenizer, prefix_budget, rng, f"{split}-{example_id}-pre")
        suffix = make_filler(tokenizer, suffix_budget, rng, f"{split}-{example_id}-post")
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
            return None
    if full_tokens > target_length:
        return None

    evidence_text = rendered_by_key[key]
    before_evidence = prefix + record_block.split(evidence_text, 1)[0]
    evidence_start = token_count(tokenizer, before_evidence)
    evidence_end = evidence_start + token_count(tokenizer, evidence_text)
    prompt_tokens = token_count(tokenizer, prompt)
    answer_tokens = token_count(tokenizer, answer_text)
    return ExampleRecord(
        example_id=example_id,
        split=split,
        task_type=task_type,
        template_family="v2_multiquery_heldin" if split == "train" else "v2_multiquery_heldout",
        target_length=target_length,
        evidence_position=evidence_position,
        position_ratio=ratio,
        distractor_count=len(distractors),
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
        query_group_id=query_group_id,
        query_index=query_index,
        records_per_group=len(rows),
        source_block_id=query_group_id,
    )


def assemble_many_from_group(
    tokenizer: Any,
    *,
    split: str,
    example_id_start: int,
    query_group_id: str,
    task_type: str,
    answer_type: str,
    target_length: int,
    evidence_position: str,
    record_block: str,
    rows: list[dict[str, str]],
    rendered_by_key: dict[str, str],
    query_order: list[int],
    rng: random.Random,
    max_examples: int | None = None,
) -> list[ExampleRecord]:
    """Assemble several queries from one record group with shared filler.

    This is the fast path for the train split. Reusing the same filler within a
    group keeps the intended hard-negative structure and avoids repeatedly
    tokenizing near-identical 2K contexts for every query.
    """

    if max_examples is not None:
        query_order = query_order[:max(0, int(max_examples))]
    if not query_order:
        return []

    templates = template_variants(split, task_type)
    ratio = POSITION_RATIOS[evidence_position]
    fixed_token_max = 0
    per_query: list[tuple[int, str, str, str, list[str]]] = []
    for query_index in query_order:
        target = rows[query_index]
        key = target["key"]
        gold = target["value"]
        distractors = [row["value"] for row in rows if row["key"] != key]
        prompt_tail = f"\nSummary line:\n{templates['final'].format(key=key)}"
        answer_text = f" {gold}"
        fixed_full = record_block + prompt_tail + answer_text + ".\n"
        fixed_token_max = max(fixed_token_max, token_count(tokenizer, fixed_full))
        per_query.append((query_index, key, gold, answer_text, distractors))
    if fixed_token_max > target_length:
        return []

    budget = max(0, target_length - fixed_token_max)
    prefix_budget = int(math.floor(budget * ratio))
    suffix_budget = max(0, budget - prefix_budget)
    prefix = ""
    suffix = ""
    built: list[tuple[str, str, str, int]] = []
    for _ in range(8):
        prefix = make_filler(tokenizer, prefix_budget, rng, f"{split}-{query_group_id}-pre")
        suffix = make_filler(tokenizer, suffix_budget, rng, f"{split}-{query_group_id}-post")
        built = []
        max_full_tokens = 0
        for _query_index, key, gold, answer_text, _distractors in per_query:
            prompt_tail = f"\nSummary line:\n{templates['final'].format(key=key)}"
            prompt = prefix + record_block + suffix + prompt_tail
            full_text = prompt + answer_text + ".\n"
            full_tokens = token_count(tokenizer, full_text)
            max_full_tokens = max(max_full_tokens, full_tokens)
            built.append((prompt, answer_text, full_text, full_tokens))
        if max_full_tokens <= target_length:
            break
        overflow = max_full_tokens - target_length
        if suffix_budget >= overflow:
            suffix_budget -= overflow
        elif prefix_budget > 0:
            prefix_budget = max(0, prefix_budget - (overflow - suffix_budget))
            suffix_budget = 0
        else:
            return []
    if any(full_tokens > target_length for *_rest, full_tokens in built):
        return []

    examples: list[ExampleRecord] = []
    for local_idx, ((query_index, key, gold, answer_text, distractors), (prompt, _answer_text, full_text, full_tokens)) in enumerate(zip(per_query, built)):
        evidence_text = rendered_by_key[key]
        before_evidence = prefix + record_block.split(evidence_text, 1)[0]
        evidence_start = token_count(tokenizer, before_evidence)
        evidence_end = evidence_start + token_count(tokenizer, evidence_text)
        prompt_tokens = token_count(tokenizer, prompt)
        answer_tokens = token_count(tokenizer, answer_text)
        examples.append(
            ExampleRecord(
                example_id=f"{split}_{example_id_start + local_idx:08d}",
                split=split,
                task_type=task_type,
                template_family="v2_multiquery_heldin" if split == "train" else "v2_multiquery_heldout",
                target_length=target_length,
                evidence_position=evidence_position,
                position_ratio=ratio,
                distractor_count=len(distractors),
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
                query_group_id=query_group_id,
                query_index=query_index,
                records_per_group=len(rows),
                source_block_id=query_group_id,
            )
        )
    return examples


def build_examples(
    *,
    tokenizer: Any,
    split: str,
    requested_examples: int,
    rng: random.Random,
    eval_grid: list[tuple[int, str, str, int]] | None,
    min_records: int,
    max_records: int,
) -> tuple[list[ExampleRecord], dict[str, Any]]:
    examples: list[ExampleRecord] = []
    group_index = 0
    task_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    group_sizes: Counter[str] = Counter()
    skipped_too_long = 0

    def one_group(length: int, position: str, task_type: str, records_per_group: int) -> None:
        nonlocal group_index, skipped_too_long
        answer_type = choose_answer_type(rng, task_type)
        group_id = f"{split}_group_{group_index:08d}"
        group_index += 1
        record_block, rows, rendered_by_key = build_record_group(
            split=split,
            task_type=task_type,
            answer_type=answer_type,
            records_per_group=records_per_group,
            rng=rng,
        )
        order = list(range(len(rows)))
        rng.shuffle(order)
        for query_index in order:
            if len(examples) >= requested_examples:
                break
            example_id = f"{split}_{len(examples):08d}"
            ex = assemble_from_group(
                tokenizer,
                split=split,
                example_id=example_id,
                query_group_id=group_id,
                query_index=query_index,
                task_type=task_type,
                answer_type=answer_type,
                target_length=length,
                evidence_position=position,
                record_block=record_block,
                rows=rows,
                rendered_by_key=rendered_by_key,
                rng=rng,
            )
            if ex is None:
                skipped_too_long += 1
                continue
            examples.append(ex)
            task_counts[ex.task_type] += 1
            length_counts[str(ex.target_length)] += 1
            position_counts[ex.evidence_position] += 1
            answer_type_counts[ex.answer_type] += 1
            group_sizes[str(ex.records_per_group)] += 1

    if eval_grid is not None:
        for length, position, task_type, sample in eval_grid:
            if len(examples) >= requested_examples:
                break
            # Keep eval distractor count controlled by records_per_group - 1,
            # but add exactly one query per grid cell so coverage is not skewed
            # by the multi-query group size.
            ex = None
            for _attempt in range(32):
                answer_type = choose_answer_type(rng, task_type)
                group_id = f"{split}_group_{group_index:08d}"
                group_index += 1
                record_block, rows, rendered_by_key = build_record_group(
                    split=split,
                    task_type=task_type,
                    answer_type=answer_type,
                    records_per_group=max_records,
                    rng=rng,
                )
                query_index = rng.randrange(len(rows))
                example_id = f"{split}_{len(examples):08d}"
                ex = assemble_from_group(
                    tokenizer,
                    split=split,
                    example_id=example_id,
                    query_group_id=group_id,
                    query_index=query_index,
                    task_type=task_type,
                    answer_type=answer_type,
                    target_length=length,
                    evidence_position=position,
                    record_block=record_block,
                    rows=rows,
                    rendered_by_key=rendered_by_key,
                    rng=rng,
                )
                if ex is not None:
                    break
                skipped_too_long += 1
            if ex is None:
                raise RuntimeError(f"could not build eval example for length={length} position={position} task={task_type}")
            # Add the sample id to keep eval rows stable and easier to inspect.
            ex.example_id = f"eval_len{length}_{position}_{task_type}_{sample:03d}"
            examples.append(ex)
            task_counts[ex.task_type] += 1
            length_counts[str(ex.target_length)] += 1
            position_counts[ex.evidence_position] += 1
            answer_type_counts[ex.answer_type] += 1
            group_sizes[str(ex.records_per_group)] += 1
        requested_examples = len(examples)

    while len(examples) < requested_examples:
        length = int(weighted_choice(rng, TRAIN_LENGTH_WEIGHTS))
        position = str(rng.choice(list(POSITION_RATIOS)))
        task_type = str(weighted_choice(rng, TASK_WEIGHTS))
        records_per_group = rng.randint(min_records, max_records)
        one_group(length, position, task_type, records_per_group)

    meta = {
        "examples": len(examples),
        "groups": group_index,
        "task_counts": dict(task_counts),
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "records_per_group_counts": dict(group_sizes),
        "skipped_too_long": skipped_too_long,
    }
    return examples, meta


def build_train_data(args: argparse.Namespace, tokenizer: Any, out_dir: Path) -> dict[str, Any]:
    train_path = out_dir / "train_examples.jsonl"
    if train_path.exists() and not args.overwrite:
        raise SystemExit(f"{train_path} exists; pass --overwrite to rebuild")
    rng = random.Random(args.seed)
    task_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    position_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    group_sizes: Counter[str] = Counter()
    skipped_too_long = 0
    example_count = 0
    group_index = 0
    with train_path.open("w", encoding="utf-8") as f:
        while example_count < int(args.train_examples):
            length = int(weighted_choice(rng, TRAIN_LENGTH_WEIGHTS))
            position = str(rng.choice(list(POSITION_RATIOS)))
            task_type = str(weighted_choice(rng, TASK_WEIGHTS))
            answer_type = choose_answer_type(rng, task_type)
            records_per_group = rng.randint(int(args.min_records_per_group), int(args.max_records_per_group))
            group_id = f"train_group_{group_index:08d}"
            group_index += 1
            record_block, rows, rendered_by_key = build_record_group(
                split="train",
                task_type=task_type,
                answer_type=answer_type,
                records_per_group=records_per_group,
                rng=rng,
            )
            order = list(range(len(rows)))
            rng.shuffle(order)
            remaining = int(args.train_examples) - example_count
            group_examples = assemble_many_from_group(
                tokenizer,
                split="train",
                example_id_start=example_count,
                query_group_id=group_id,
                task_type=task_type,
                answer_type=answer_type,
                target_length=length,
                evidence_position=position,
                record_block=record_block,
                rows=rows,
                rendered_by_key=rendered_by_key,
                query_order=order,
                rng=rng,
                max_examples=remaining,
            )
            if not group_examples:
                skipped_too_long += 1
                continue
            for ex in group_examples:
                write_jsonl_line(f, asdict(ex))
                example_count += 1
                task_counts[ex.task_type] += 1
                length_counts[str(ex.target_length)] += 1
                position_counts[ex.evidence_position] += 1
                answer_type_counts[ex.answer_type] += 1
                group_sizes[str(ex.records_per_group)] += 1
            if example_count % 5000 < len(group_examples):
                print(
                    json.dumps(
                        {
                            "event": "build_train_progress",
                            "examples": example_count,
                            "groups": group_index,
                            "target": int(args.train_examples),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    meta = {
        "examples": example_count,
        "groups": group_index,
        "task_counts": dict(task_counts),
        "target_length_counts": dict(length_counts),
        "evidence_position_counts": dict(position_counts),
        "answer_type_counts": dict(answer_type_counts),
        "records_per_group_counts": dict(group_sizes),
        "skipped_too_long": skipped_too_long,
    }
    meta |= {"path": str(train_path.relative_to(out_dir))}
    return meta


def build_eval_data(args: argparse.Namespace, tokenizer: Any, out_dir: Path) -> dict[str, Any]:
    eval_path = out_dir / "eval_examples.jsonl"
    rng = random.Random(args.seed + 97_531)
    grid: list[tuple[int, str, str, int]] = []
    for length in args.eval_lengths:
        for position in args.eval_positions:
            if position not in POSITION_RATIOS:
                raise SystemExit(f"unknown eval position: {position}")
            for task_type in TASK_TYPES:
                for sample in range(int(args.eval_samples_per_cell)):
                    grid.append((int(length), str(position), task_type, int(sample)))
    records_per_group = int(args.eval_distractors) + 1
    examples, meta = build_examples(
        tokenizer=tokenizer,
        split="eval",
        requested_examples=len(grid),
        rng=rng,
        eval_grid=grid,
        min_records=records_per_group,
        max_records=records_per_group,
    )
    with eval_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            write_jsonl_line(f, asdict(ex))
    meta |= {
        "path": str(eval_path.relative_to(out_dir)),
        "lengths": [int(x) for x in args.eval_lengths],
        "positions": list(args.eval_positions),
        "distractors": int(args.eval_distractors),
        "samples_per_cell": int(args.eval_samples_per_cell),
    }
    return meta


def write_data_card(out_dir: Path, manifest: dict[str, Any]) -> None:
    train = manifest["stage1r_train"]
    eval_meta = manifest["stage1r_eval"]
    lines = [
        "# Stage1R Retrieval Warmup Dataset V2",
        "",
        "V2 is built for answer-only retrieval warmup with ordinary LM replay.",
        "It should not be trained with full-token LM loss over the synthetic text.",
        "",
        "## Why V2",
        "",
        "- V1 answer-only loss improved hard-negative ranking, but 2K distance and copy/passkey tasks remained weak.",
        "- V1 full-token synthetic LM loss improved answer NLL while hurting hard-negative ranking.",
        "- V2 therefore makes retrieval supervision explicit: only the final answer tokens are trained.",
        "",
        "## Main Data Change",
        "",
        "Most training groups are multi-query tables. A group contains several key-value records; each key is",
        "queried in a separate example, so values that are distractors in one example become gold in another.",
        "This attacks the core failure mode: choosing plausible values instead of binding the requested key to",
        "its evidence line.",
        "",
        "## Training Split",
        "",
        f"- Examples: `{train['examples']}`",
        f"- Groups: `{train['groups']}`",
        f"- Target lengths: `{train['target_length_counts']}`",
        f"- Evidence positions: `{train['evidence_position_counts']}`",
        f"- Answer types: `{train['answer_type_counts']}`",
        "",
        "## Eval Split",
        "",
        f"- Examples: `{eval_meta['examples']}`",
        f"- Lengths: `{eval_meta['lengths']}`",
        f"- Positions: `{eval_meta['positions']}`",
        f"- Same-context distractors per example: `{eval_meta['distractors']}`",
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
            "## Recommended Use",
            "",
            "Use `qwen3_01b/scripts/train_stage1r_mixed_replay.py` with roughly 80%-90% ordinary LM replay",
            "and 10%-20% retrieval answer-only loss. Keep the V1 eval set as the fixed regression set,",
            "then evaluate this V2 eval set as an additional harder check.",
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
        "format": "stage1r_answer_only_jsonl",
        "vocab_size": int(vocab_size),
        "tokenizer_json": str(tokenizer_path),
        "seq_len": int(args.seq_len),
        "stage": "stage1r_retrieval_warmup_v2",
        "selection_method": "multi-query synthetic retrieval/copy continuation examples for answer-only loss",
        "stage1r_train": train_meta,
        "stage1r_eval": eval_meta,
        "task_weights": TASK_WEIGHTS,
        "length_weights": TRAIN_LENGTH_WEIGHTS,
        "position_ratios": POSITION_RATIOS,
        "run_config": vars(args),
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run_config.json", vars(args))
    write_data_card(out_dir, manifest)
    print(json.dumps({"output_dir": str(out_dir), "train": train_meta, "eval": eval_meta}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
