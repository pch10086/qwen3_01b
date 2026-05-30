# Benchmark v1/v2 Design: Evidence Position in Long Contexts

## Objective

This benchmark suite studies how evidence position affects long-context behavior
for a small pretrained language model. The immediate model is
`Qwen3-0.6B-Base`. The self-trained checkpoint is intentionally excluded from
this round because its long-context window is still being trained; it can later
be evaluated with the same suite.

The suite is deliberately narrow. It does not aim to report a broad leaderboard
score. It aims to produce interpretable evidence for three questions:

1. What happens when the relevant evidence is moved across the context?
2. Does task difficulty or candidate confusion amplify the position effect?
3. Which failures should later post-training target?

## Related Work Rationale

The suite uses existing benchmark protocols rather than a fully custom task.

- **Lost in the Middle** studies how models use relevant information at
  different positions in long inputs, using multi-document QA and key-value
  retrieval. It directly motivates Benchmark v1.
- **RULER** shows that simple needle-in-a-haystack tasks can overestimate
  long-context ability, and provides configurable synthetic retrieval tasks.
  It motivates Benchmark v2.

This gives the project external credibility while preserving the position-
controlled analysis needed for the course project.

## Model Scope

Initial benchmark model:

- `Qwen3-0.6B-Base`
- local path: `/home/public/bjh/dym/NLP/models/Qwen3-0.6B-Base`

Excluded for now:

- `/home/public/bjh/dym/NLP/evaluation/Ours/checkpoint_step_20000.pt`

Reason: the self-trained model currently has a 4K context configuration and is
still an early checkpoint. It should be added after the target context window is
trained, so comparisons are fair and meaningful.

## Benchmark v1: Lost-in-the-Middle Subset

### Purpose

Benchmark v1 is the main evidence-position benchmark. It follows the
Lost-in-the-Middle style: keep the task fixed and move the gold evidence across
the input. It has two subtests:

- v1-a: key-value retrieval;
- v1-b: multi-document QA.

The key-value task provides clean automatic scoring. The QA task provides a more
natural setting and checks whether findings survive outside a synthetic table.

### v1-a: Key-Value Retrieval

Task:

```text
Input: many key-value pairs.
Question: What is the value associated with key <K>?
Answer: the target value.
```

Controlled variables:

- number of key-value pairs: `75`, `140`, `300`, matching the official
  Lost-in-the-Middle KV files available in the repository;
- target pair position: `0%`, `10%`, `25%`, `50%`, `75%`, `90%`, `100%`;
- examples per cell: `50`.

Total planned examples:

```text
3 key-value sizes * 7 positions * 50 examples = 1050 examples
```

Primary metrics:

- `exact_match`;
- `contains_value`;
- `first_value_match`;
- `format_error_rate`.

Expected analysis:

- line plot: accuracy vs target position, one line per key-value size;
- heatmap: key-value size by target position;
- format analysis: whether wrong strict exact match is a retrieval error or an
  output-format error.

Interpretation:

- If the middle positions drop while edges stay strong, the model shows a
  Lost-in-the-Middle-style position effect in pure retrieval.
- If performance only drops when the number of key-value pairs grows, candidate
  confusion and memory load are likely major factors.

### v1-b: Multi-Document QA

Task:

```text
Input: a list of documents, one of which contains the answer.
Question: answer using the documents.
Answer: one of the known answer aliases.
```

Controlled variables:

- number of documents: `10`, `20`;
- gold document position: `0%`, `25%`, `50%`, `75%`, `100%`;
- examples per cell: `50`.

Total planned examples:

```text
2 document counts * 5 positions * 50 examples = 500 examples
```

Primary metrics:

- `answer_contains`;
- `alias_match`;
- `no_answer_rate`;
- `format_error_rate`.

Expected analysis:

- line plot: QA accuracy vs gold document position;
- comparison: v1-a retrieval curve vs v1-b QA curve.

Interpretation:

- If key-value retrieval is stable but QA drops, the bottleneck is probably not
  pure retrieval; it may be semantic matching, answer generation, or instruction
  following.
- If both retrieval and QA show a middle drop, the position effect is more
  fundamental.

## Benchmark v2: RULER Subset

### Purpose

Benchmark v2 diagnoses why simple single-needle tasks are insufficient. The
previous smoke experiments showed that Qwen3-0.6B-Base can solve simple
single-evidence numeric retrieval at 4K/8K/16K with perfect content accuracy.
RULER-style variants add controlled difficulty through multiple keys, multiple
values, and multiple queries.

### Selected Tasks

Use a focused subset of RULER retrieval tasks:

- single-needle retrieval: easy baseline;
- multi-key retrieval: multiple candidate keys, one queried key;
- multi-value retrieval: one key associated with multiple required values;
- multi-query retrieval: several queries in the same prompt.

The exact RULER task names should follow the official implementation available
in the repository. If official task names differ, map them to the four semantic
categories above and document the mapping in each run config.

### Planned Configuration

Context lengths:

```text
4K, 8K, 16K, 32K
```

Examples:

```text
50 examples per task and context length
```

Position handling:

- Prefer RULER's official generation settings for comparability.
- If the official generator exposes evidence positions, also run a
  position-stratified slice at `0%`, `25%`, `50%`, `75%`, `100%`.
- If position is not directly exposed, record the generated evidence position
  from the prompt and analyze it post hoc.

Primary metrics:

- `exact_match`;
- `partial_match`;
- `missing_value_rate`;
- `wrong_key_rate`;
- `wrong_value_rate`.

Expected analysis:

- accuracy vs context length;
- accuracy by RULER task variant;
- if position is available, accuracy vs evidence position;
- error-type breakdown by task variant.

Interpretation:

- If single-needle retrieval remains strong but multi-key retrieval drops,
  simple NIAH is overestimating model ability.
- If multi-query retrieval drops more than multi-key retrieval, capacity and
  multi-target tracking are central weaknesses.
- If errors are dominated by wrong-key predictions, post-training should include
  hard negative retrieval examples.

## Execution Order

1. Implement or adapt v1-a key-value retrieval first.
2. Run v1-a at full planned size for `Qwen3-0.6B-Base`.
3. Inspect the position curve and error formatting.
4. Implement or adapt v1-b multi-document QA.
5. Run v1-b and compare retrieval vs QA behavior.
6. Add the RULER subset and run task/context-length diagnostics.
7. Produce plots and a short result memo before designing post-training.

This order keeps the work efficient: the first result should already provide a
position curve, while later tasks add naturalness and diagnostic depth.

## Output Structure

Recommended directory layout:

```text
/home/public/bjh/dym/NLP/evaluation/
  benchmarks/
    lost_middle/
    ruler/
  configs/
    qwen3_0_6b_litm_kv.json
    qwen3_0_6b_litm_qa.json
    qwen3_0_6b_ruler_subset.json
  outputs/
    qwen3_0_6b_litm_kv/
    qwen3_0_6b_litm_qa/
    qwen3_0_6b_ruler_subset/
  docs/
    benchmark_v1_v2_design.md
```

Each run should write:

- `run_config.json`;
- `examples.jsonl`;
- `predictions.jsonl`;
- `summary.csv`;
- optional plot files under `plots/`.

## Reporting Plan

The benchmark section of the final report should emphasize curves and error
types rather than only aggregate scores.

Core figures:

1. v1-a KV accuracy by evidence position and number of key-value pairs.
2. v1-b QA accuracy by gold document position and document count.
3. KV vs QA comparison showing whether pure retrieval and natural QA fail in the
   same way.
4. RULER task difficulty curve across context lengths.
5. Error-type stacked bars for RULER variants.

Core conclusions to look for:

- whether Qwen3-0.6B-Base exhibits a middle-position weakness;
- whether the weakness appears only under candidate confusion;
- whether natural QA is harder than structured retrieval;
- what post-training data should target first.

## Implementation Constraints

- Keep the first implementation model-agnostic but only run
  `Qwen3-0.6B-Base` initially.
- Do not include the self-trained checkpoint until its context window training
  is ready.
- Preserve the existing smoke benchmark outputs.
- Use deterministic decoding for primary runs.
- Report format errors separately from retrieval errors.
