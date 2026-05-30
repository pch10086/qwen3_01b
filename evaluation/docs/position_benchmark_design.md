# Position-Controlled Needle QA Benchmark

## Goal

This benchmark studies how evidence position affects long-context question answering.
The first version is intentionally narrow: one synthetic retrieval-style task, a
small smoke configuration, and deterministic result files that can later support
deeper analysis.

## Research Question

Given the same evidence sentence and question, how does model accuracy change
when the evidence appears at different relative positions in the prompt?

The first pass answers "does the pipeline work?". Later passes will answer:

- whether the model shows a middle-position accuracy drop;
- whether longer contexts amplify this drop;
- whether targeted post-training can flatten the position-accuracy curve.

## Task

Each example contains filler text, one evidence sentence, and a question.

Example evidence:

```text
The access code for Project Aurora is 73915.
```

Example question:

```text
What is the access code for Project Aurora?
```

The expected answer is the numeric code. The benchmark reports both strict
exact match and a more robust `contains_answer` score, because base language
models may continue text instead of producing a clean answer.

## Smoke Configuration

- model: `Qwen/Qwen3-0.6B-Base`
- context length targets: 1024, 4096 tokens
- evidence positions: 0.0, 0.5, 1.0
- samples per cell: 2
- total examples: 12

## Output Files

Each run writes:

- `run_config.json`: resolved command/configuration;
- `predictions.jsonl`: one record per example, including prompt length,
  evidence position, generated text, answer, and scores;
- `summary.csv`: aggregate metrics by context length and evidence position.

## Later Extensions

After the smoke run works, extend the same runner to:

- longer contexts: 4K, 8K, 16K, 32K;
- more positions: 0%, 10%, 25%, 50%, 75%, 90%, 100%;
- distractor needles, where a similar but wrong evidence sentence appears at a
  competing position;
- answer log-probability analysis and error typing.
