# English Pretraining Corpus Download Plan

This document records the English-only corpus plan for the 0.1B Qwen3-style long-context language model. The plan is intentionally compact: use a few high-quality sources as the primary mixture, and keep extra datasets as optional replacements instead of adding every available corpus.

The goal is to build two token manifests:

- Stage 1 base pretraining: `data/processed/pretrain_en_10b_bpe64k/manifest.json`
- Stage 2 long-context continued pretraining: `data/processed/pretrain_en_longctx_500m_bpe64k/manifest.json`

All token budgets below mean tokens after encoding with this project's 64K BPE tokenizer, not words, bytes, or dataset-native token counts.

## Final Recommendation

Use this compact primary mixture first.

| Stage | Target tokens | Main sequence length | Main sources | Purpose |
|---|---:|---:|---|---|
| Stage 1: base pretraining | 10.0B | 2048 | FineWeb-Edu, FineMath/OpenWebMath, peS2o, PG-19, English Wikipedia | Learn general English language modeling, academic style, math/reasoning text, and long-document structure. |
| Stage 2: long-context continued pretraining | 0.5B | 4096 first, then 8192 if memory allows | peS2o, PG-19, FineMath/OpenWebMath, English Wikipedia, FineWeb-Edu replay | Adapt to long documents and extended RoPE positions without forgetting normal English. |
| Total | 10.5B | - | 5 primary source families | English-only long-context research baseline for the 0.1B model. |

Why this is compact:

- FineWeb-Edu is the main high-quality English web corpus.
- FineMath or OpenWebMath covers math/reasoning style without adding several overlapping math corpora.
- peS2o covers academic prose and long structured documents.
- PG-19 covers long-book continuity, but stays capped because its older book style can bias generation.
- English Wikipedia adds stable encyclopedic text without dominating the mixture.

## Stage 1 Corpus: Base Pretraining

Stage 1 should prioritize clean and broad English. Do not over-mix small sources: a 0.1B model benefits more from a stable distribution than from many tiny slices.

| Priority | Category | Dataset | Suggested source name | Target tokens | Share |
|---:|---|---|---|---:|---:|
| 1 | Educational English web | FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, prefer high-quality/sample splits such as `sample-10BT` when available | 6.5B | 65% |
| 2 | Math and reasoning web text | FineMath preferred; OpenWebMath fallback | `HuggingFaceTB/finemath` preferred, or `open-web-math/open-web-math` if FineMath is unavailable | 1.2B | 12% |
| 3 | Academic and scientific prose | peS2o | `allenai/peS2o`; prefer complete body text | 1.0B | 10% |
| 4 | Long books and narrative text | PG-19 train | `pg19` or a compatible Hugging Face mirror; use train split only | 0.8B | 8% |
| 5 | English encyclopedia | English Wikipedia | `wikimedia/wikipedia`, English config such as `20231101.en` | 0.5B | 5% |
|  |  |  | **Total** | **10.0B** | **100%** |

### Stage 1 Rationale

- Keep FineWeb-Edu as the backbone because it gives broad English coverage with educational filtering.
- Use only one main math family. FineMath is preferred if it is easy to download; OpenWebMath is an acceptable fallback because it is already a high-quality math web corpus.
- Keep peS2o at 10% because academic prose is useful for long-context research, but too much paper text can make a small model overly formal.
- Keep PG-19 below 10% in Stage 1. It is useful for long contexts, but it contains older public-domain books and can shift model style if overused.
- Keep Wikipedia at 5%. It is clean and factual, but too much encyclopedia text can make generation stiff.

## Stage 2 Corpus: Long-Context Continued Pretraining

Stage 2 should not be a random re-sample of Stage 1. It should be built from naturally long, ordered documents. The packing rule is more important than the dataset name: preserve books, papers, articles, sections, and long tutorials whenever possible.

| Priority | Category | Dataset | Sampling rule | Target tokens | Share |
|---:|---|---|---|---:|---:|
| 1 | Long academic papers | peS2o | Prefer papers with complete body text and section structure. | 175M | 35% |
| 2 | Long books | PG-19 train | Keep book/chapter order; use train split only. | 100M | 20% |
| 3 | Long math tutorials and solutions | FineMath or OpenWebMath | Prefer long tutorials, proofs, lecture notes, and detailed solutions. | 75M | 15% |
| 4 | Long encyclopedia articles | English Wikipedia | Prefer long articles and preserve title/section/paragraph order. | 50M | 10% |
| 5 | General replay data | FineWeb-Edu | High-quality ordinary English text from Stage 1 sources to reduce forgetting. | 100M | 20% |
|  |  |  | **Total** | **500M** | **100%** |

### Stage 2 Rationale

- peS2o is the main long-context source because academic papers naturally contain long sections, references, and cross-section dependencies.
- PG-19 stays useful, but it is capped at 20% to avoid overfitting to old book style.
- FineMath/OpenWebMath adds long reasoning-style documents without turning the model into a pure math model.
- Wikipedia long articles add factual sectioned documents.
- FineWeb-Edu replay is deliberately kept at 20% to protect normal English ability during long-context adaptation.

## Stage 2 Length Schedule

Use the longest schedule that fits the available GPUs. If only `seq_len=4096` is stable, put all 500M long-context tokens into the first row.

| Substage | Target tokens | Suggested setting | Data requirement |
|---|---:|---|---|
| Stage 2A | 300M | `seq_len=4096`, `context_length=8192` | Documents should usually contain at least 4K tokens after encoding. |
| Stage 2B | 150M | `seq_len=8192`, `context_length=16384` | Prefer peS2o, PG-19, and long math/tutorial documents with at least 8K tokens. |
| Stage 2C, optional | 50M | `seq_len=16384`, `context_length=32768` or `40960` | Use only very long papers, books, manuals, and tutorials. |

## Optional Sources

These sources are not part of the primary plan. Use them only as replacements when they match the evaluation goal or when a primary source is unavailable.

| Optional source | Use when | Replacement rule |
|---|---|---|
| DCLM Baseline, `mlfoundations/dclm-baseline-1.0` | FineWeb-Edu download is too slow, or the web corpus needs more general English diversity. | Replace up to 1.0B FineWeb-Edu tokens. Do not add it on top of the 10B budget. |
| The Stack Dedup, `bigcode/the-stack-dedup` | The evaluation includes code, repository reasoning, or technical documentation tasks. | Replace up to 0.5B FineWeb-Edu or math tokens. Keep code at or below 5% unless code is a stated target. |
| Proof-Pile-2 arXiv/algebraic-stack, `EleutherAI/proof-pile-2` | FineMath/OpenWebMath is unavailable, or the project needs more arXiv/formal-math text. | Replace part of FineMath/OpenWebMath or peS2o. Do not combine all math corpora at full size. |

## Download Priority

Download in this order so the project can start training before every optional source is complete.

1. FineWeb-Edu for the Stage 1 backbone.
2. FineMath if available, otherwise OpenWebMath.
3. peS2o for academic and long-document coverage.
4. PG-19 train split for long-book coverage.
5. English Wikipedia for stable encyclopedic text.
6. Optional replacement sources only if the primary sources cannot fill the planned token budget or if the benchmark goal requires them.

## Directory Layout

Recommended raw and processed layout:

```text
data/
  raw/
    fineweb_edu/
    finemath_or_openwebmath/
    pes2o/
    pg19/
    wikipedia_en/
    optional/
      dclm_baseline/
      the_stack_dedup/
      proof_pile_2/
  processed/
    pretrain_en_10b_bpe64k/
      manifest.json
      shard_00000.bin
      shard_00001.bin
      ...
    pretrain_en_longctx_500m_bpe64k/
      manifest.json
      shard_00000.bin
      shard_00001.bin
      ...
```

The final manifests should follow the format already supported by `qwen3_06b.cli_pretrain`:

```json
{
  "dtype": "uint16",
  "total_tokens": 10000000000,
  "shards": [
    {"path": "shard_00000.bin", "tokens": 2000000}
  ]
}
```

## Quality Gates Before Full Tokenization

Run these checks on a small sample before committing to a full download or tokenization job.

| Check | Minimum requirement |
|---|---|
| Language | English-only. Reject documents with poor language detection confidence or obvious multilingual contamination. |
| Text quality | Remove broken extraction, repeated boilerplate, navigation text, ads, tables of links, and extremely short fragments. |
| Document length | Stage 1 documents should generally have at least 128 tokens. Stage 2 documents should usually have at least 4K tokens before long-context packing. |
| Duplicates | Deduplicate within each source at minimum. Cross-source near-deduplication is preferred for FineWeb-Edu, Wikipedia, and web-math overlap. |
| Evaluation leakage | Remove benchmark validation/test data and obvious benchmark mirrors before training. |
| Source balance | Keep the primary mixture close to the target shares. Do not let any optional source silently push the distribution away from the table above. |

Practical sample review:

1. Download or stream a small sample from each selected source.
2. Inspect at least 50 raw documents per source.
3. Encode a small token sample with the project tokenizer.
4. Record token length distribution, rejected-document rate, and approximate bytes-per-token.
5. Only start the full manifest build after the sample looks clean.

## Filtering Rules

Apply these rules before tokenization where possible:

- Keep English-only data. Do not mix multilingual Common Crawl, mC4, OSCAR, or non-English Wikipedia unless an English split is explicitly selected.
- Remove empty, extremely short, broken, boilerplate-heavy, ad-heavy, or duplicate documents.
- Remove instruction/chat/SFT-style data from pretraining unless a later supervised fine-tuning stage is planned separately.
- Keep code data out of the primary mixture unless the project explicitly evaluates code tasks.
- Do not add every available math corpus. Use FineMath or OpenWebMath as the main math source, and treat Proof-Pile-2 as a replacement.
- Keep PG-19 validation/test untouched if PG-19 is used for evaluation.

## Packing Rules

Stage 1:

- Shuffle documents after source-level filtering.
- Pack text into 2048-token sequences.
- It is acceptable to mix documents in the same packed sequence if separators are inserted consistently.

Stage 2:

- Preserve document order.
- Prefer complete books, chapters, papers, sections, articles, tutorials, or manuals.
- Avoid creating long sequences by randomly concatenating unrelated short snippets.
- Add FineWeb-Edu replay data, but keep most Stage 2 tokens from naturally long documents.

## Train/Eval Separation

- If PG-19 is used for evaluation, train only on the train split and keep validation/test splits untouched.
- Do not train on benchmark validation/test sets such as LongBench, NarrativeQA, Qasper, GovReport, or Needle-in-a-Haystack variants.
- For any long-context benchmark built from public datasets, check whether the raw source overlaps with PG-19, peS2o, arXiv, Wikipedia, FineWeb-Edu, FineMath, or OpenWebMath before using it as a clean evaluation set.

## Practical Fallbacks

If downloading or processing all 10B tokens is too slow, use this smaller but still coherent plan:

| Stage | Target tokens | Composition |
|---|---:|---|
| Stage 1 small run | 2.0B | 1.3B FineWeb-Edu, 0.25B FineMath/OpenWebMath, 0.2B peS2o, 0.15B PG-19 train, 0.1B Wikipedia |
| Stage 2 small run | 100M | 35M peS2o, 20M PG-19 train, 15M FineMath/OpenWebMath, 10M Wikipedia, 20M FineWeb-Edu replay |

This fallback is useful for debugging the full data pipeline and producing early loss curves. The full 10B + 500M plan should remain the main experiment target.
