# Benchmark v1/v2 设计：长上下文中的证据位置

## 目标

这套基准用于研究证据位置如何影响小型预训练语言模型的长上下文行为。当前首先评测的模型是 `Qwen3-0.6B-Base`。本轮暂不纳入自训练 checkpoint，因为它的长上下文窗口仍在训练中；后续可以用同一套基准继续评测。

这套基准刻意保持较窄范围。它不追求给出宽泛的排行榜分数，而是希望为以下三个问题提供可解释证据：

1. 当相关证据在上下文中移动时，模型表现会发生什么变化？
2. 任务难度或候选项混淆是否会放大位置效应？
3. 后续 post-training 应优先针对哪些失败模式？

## 相关工作依据

这套基准采用已有 benchmark protocol，而不是完全自定义任务。

- **Lost in the Middle** 研究模型如何利用长输入中不同位置的相关信息，任务包括多文档问答和 key-value 检索。它直接启发 Benchmark v1。
- **RULER** 说明简单的 needle-in-a-haystack 任务可能高估模型的长上下文能力，并提供了可配置的合成检索任务。它启发 Benchmark v2。

这样既能让项目具备外部可信度，又能保留课程项目所需的、受位置控制的分析。

## 模型范围

初始 benchmark 模型：

- `Qwen3-0.6B-Base`
- 本地路径：`/home/public/bjh/dym/NLP/models/Qwen3-0.6B-Base`

当前暂不纳入：

- `/home/public/bjh/dym/NLP/evaluation/Ours/checkpoint_step_20000.pt`

原因：自训练模型目前仍是 4K context 配置，并且还是较早期 checkpoint。应在目标上下文窗口训练完成后再加入评测，这样比较才公平且有意义。

## Benchmark v1：Lost-in-the-Middle 子集

### 目的

Benchmark v1 是主要的证据位置 benchmark。它沿用 Lost-in-the-Middle 思路：保持任务不变，只移动 gold evidence 在输入中的位置。它包含两个子测试：

- v1-a：key-value 检索；
- v1-b：多文档问答。

key-value 任务便于进行干净的自动评分。QA 任务更接近自然场景，用来检查结论是否能从合成表格任务迁移到更自然的文本任务。

### v1-a：Key-Value 检索

任务：

```text
输入：大量 key-value pair。
问题：key <K> 对应的 value 是什么？
答案：目标 value。
```

控制变量：

- key-value pair 数量：`75`、`140`、`300`，对应仓库中已有的官方 Lost-in-the-Middle KV 文件；
- 目标 pair 位置：`0%`、`10%`、`25%`、`50%`、`75%`、`90%`、`100%`；
- 每个 cell 的样本数：`50`。

计划样本总数：

```text
3 种 key-value 规模 * 7 个位置 * 50 个样本 = 1050 个样本
```

主要指标：

- `exact_match`；
- `contains_value`；
- `first_value_match`；
- `format_error_rate`。

预期分析：

- 折线图：accuracy vs target position，每条线对应一种 key-value 规模；
- 热力图：key-value 规模 by target position；
- 格式分析：严格 exact match 错误到底是检索错误，还是输出格式错误。

解释方式：

- 如果中间位置表现下降，而两端表现较强，说明模型在纯检索任务中呈现 Lost-in-the-Middle 风格的位置效应。
- 如果只有在 key-value pair 数量增加时性能才下降，那么候选项混淆和记忆负载很可能是主要因素。

### v1-b：多文档问答

任务：

```text
输入：一个文档列表，其中一个文档包含答案。
问题：根据文档回答问题。
答案：已知 answer alias 中的一个。
```

控制变量：

- 文档数量：`10`、`20`；
- gold document 位置：`0%`、`25%`、`50%`、`75%`、`100%`；
- 每个 cell 的样本数：`50`。

计划样本总数：

```text
2 种文档数量 * 5 个位置 * 50 个样本 = 500 个样本
```

主要指标：

- `answer_contains`；
- `alias_match`；
- `no_answer_rate`；
- `format_error_rate`。

预期分析：

- 折线图：QA accuracy vs gold document position；
- 对比：v1-a 检索曲线 vs v1-b QA 曲线。

解释方式：

- 如果 key-value 检索稳定但 QA 下降，瓶颈可能不是纯检索，而是语义匹配、答案生成或指令跟随。
- 如果检索和 QA 都出现中间位置下降，说明位置效应更基础。

## Benchmark v2：RULER 子集

### 目的

Benchmark v2 用于诊断为什么简单的 single-needle 任务不够充分。之前的 smoke 实验显示，`Qwen3-0.6B-Base` 能在 4K/8K/16K 的简单单证据数字检索中达到完美的内容准确率。RULER 风格变体通过多个 key、多个 value 和多个 query 引入受控难度。

### 选定任务

使用 RULER 检索任务中的一个聚焦子集：

- single-needle retrieval：简单 baseline；
- multi-key retrieval：多个候选 key，只查询其中一个 key；
- multi-value retrieval：一个 key 对应多个需要回答的 value；
- multi-query retrieval：同一个 prompt 中包含多个 query。

具体 RULER 任务名应遵循仓库中的官方实现。如果官方任务名不同，则将其映射到以上四个语义类别，并在每次 run config 中记录映射关系。

### 计划配置

上下文长度：

```text
4K, 8K, 16K, 32K
```

样本数：

```text
每个任务、每个上下文长度 50 个样本
```

位置处理：

- 优先使用 RULER 的官方生成设置，以保证可比性。
- 如果官方 generator 暴露证据位置参数，则额外运行一个按位置分层的 slice：`0%`、`25%`、`50%`、`75%`、`100%`。
- 如果位置不能直接控制，则从生成的 prompt 中记录证据位置，并在事后进行分析。

主要指标：

- `exact_match`；
- `partial_match`；
- `missing_value_rate`；
- `wrong_key_rate`；
- `wrong_value_rate`。

预期分析：

- accuracy vs context length；
- accuracy by RULER task variant；
- 如果位置可用，则分析 accuracy vs evidence position；
- 按任务变体拆分 error type。

解释方式：

- 如果 single-needle retrieval 仍然很强，但 multi-key retrieval 下降，说明简单 NIAH 高估了模型能力。
- 如果 multi-query retrieval 比 multi-key retrieval 下降更多，说明容量和多目标追踪是核心弱点。
- 如果错误主要是 wrong-key prediction，后续 post-training 应加入 hard negative retrieval 样本。

## 执行顺序

1. 先实现或适配 v1-a key-value 检索。
2. 对 `Qwen3-0.6B-Base` 跑完 v1-a 的完整计划规模。
3. 检查位置曲线和输出格式错误。
4. 实现或适配 v1-b 多文档 QA。
5. 运行 v1-b，并比较 retrieval 与 QA 行为。
6. 加入 RULER 子集，运行任务/上下文长度诊断。
7. 在设计 post-training 前，产出图表和简短结果 memo。

这个顺序能保持工作效率：第一个结果就能提供位置曲线，后续任务再补充自然性和诊断深度。

## 输出结构

推荐目录结构：

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

每次运行应写出：

- `run_config.json`；
- `examples.jsonl`；
- `predictions.jsonl`；
- `summary.csv`；
- 可选图表文件，放在 `plots/` 下。

## 报告计划

最终报告中的 benchmark 部分应重点展示曲线和错误类型，而不是只报告 aggregate score。

核心图表：

1. v1-a KV accuracy by evidence position and number of key-value pairs。
2. v1-b QA accuracy by gold document position and document count。
3. KV vs QA 对比，用于判断纯检索和自然 QA 是否以相同方式失败。
4. RULER task difficulty curve across context lengths。
5. RULER variants 的 error-type stacked bar。

重点关注的核心结论：

- `Qwen3-0.6B-Base` 是否表现出中间位置弱点；
- 该弱点是否只在候选项混淆时出现；
- 自然 QA 是否比结构化检索更难；
- post-training 数据应优先针对什么问题。

## 实现约束

- 第一版实现保持 model-agnostic，但初始只运行 `Qwen3-0.6B-Base`。
- 在自训练 checkpoint 的上下文窗口训练完成前，不纳入该 checkpoint。
- 保留已有 smoke benchmark 输出。
- 主实验使用 deterministic decoding。
- 将格式错误和检索错误分开报告。
