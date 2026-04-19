# PRISM — Multi-Layer Retrieval for Mixed Code + Documentation Corpora

> **What this is:** An exploration of whether routing questions across multiple pre-compiled knowledge representations — code structure graph, document index, synthesised wiki pages, and keyword search — can match the accuracy of raw file retrieval at significantly lower token cost. This is a research benchmark, not a production system.

---

## The Core Idea

Most retrieval approaches treat a codebase as flat text. PRISM tests a different hypothesis:

> **Different questions need different representations of the same repository.**

A question like *"what calls `AuthService.login()`?"* is best answered by a code structure graph. A question like *"why was single-use token design chosen?"* is best answered by a pre-synthesised design rationale page. Sending all files to the AI for both wastes tokens and introduces noise.

PRISM builds four views of a repository once, caches them, then routes each question to the right view — or combination of views — with a dynamic token budget.

---

## Architecture

![PRISM Architecture](architecture.svg)

---

## The Four Layers

| Layer | What it captures | Built how | Cost |
|---|---|---|---|
| **1. Code Graph** | Every function, class, call edge, and import relationship | Tree-sitter AST parser — no AI needed | $0 |
| **2. Doc Index** | Per-document summary, key facts, and code cross-references | One AI call per document, cached by SHA-256 | ~$0.002/doc |
| **3. LLMWiki** | Pre-synthesised "entity pages" integrating code structure with doc content | One AI call per concept page, cached | ~$0.004/page |
| **4. BM25** | Keyword search index across all three layers | Pure term-frequency maths — no AI | $0 |

When a question arrives, a **router** (one AI call) classifies the intent and assigns layers and a token budget:

| Question type | Budget | Layer split | Example |
|---|---|---|---|
| structural | 1,500 tok | Graph only | "What calls `AuthService.login()`?" |
| rationale | 2,000 tok | Wiki 80% + graph 20% | "Why was single-use token design chosen?" |
| factual | 2,500 tok | Wiki 60% + doc index 40% | "What does the refresh endpoint guarantee?" |
| similarity | 3,000 tok | BM25 60% + wiki 20% + graph 20% | "Find code similar to the token expiry logic" |
| comprehensive | 5,000 tok | Wiki 40% + graph 30% + BM25 30% | "How does `AuthService` use `TokenStore`?" |

If a layer has nothing useful to contribute — returning less than 30% of its allocated budget — unused tokens carry forward to the next layer (**adaptive budget spill**).

---

## What the Benchmark Measured

Three hypotheses tested across corpora of 6, 47, and 145 Python files plus a mixed code+documentation corpus of 15 files:

### H1 — Token reduction grows with corpus size

| Corpus | Files | Naive tokens | Code graph tokens | Reduction |
|---|---|---|---|---|
| Small | 6 | 3,138 | 336 | 9.3× |
| Medium | 47 | 21,189 | 1,310 | 16.2× |
| Large | 145 | 65,854 | 1,740 | **37.8×** |

Confirmed. Compression ratio grows monotonically. The graph output is bounded by `--budget`; naive scales linearly with corpus size.

### H2 — Each additional layer adds accuracy without proportional token cost

Tested on a 15-file mixed corpus (10 Python files + 5 design docs), scored by Claude as judge out of 15, across 5 questions:

| Approach | Avg score /15 | Avg tokens | Gap to raw |
|---|---|---|---|
| Code graph alone | 10.4 | 1,697 | −4.0 pts |
| + Doc index | 12.6 | 1,873 | −1.8 pts |
| + LLMWiki | 13.6 | 2,006 | −0.8 pts |
| **PRISM routed** | **12.8** | **3,295** | **−1.6 pts** |
| Raw top-10 files | 14.4 | 8,804 | baseline |

Confirmed directionally. Each layer closes the gap. PRISM reaches within 1.6 points of raw at 2.67× fewer tokens.

### H3 — Code graph alone cannot answer documentation questions

A code structure graph extracted **0 nodes from 15 markdown files** — tree-sitter has no documentation handler. Every doc query returned an empty result. The Doc Index and LLMWiki layers resolve this by bypassing the AST entirely.

Confirmed. Doc blindspot is real and the layered fix works.

---

## Honest Limitations

This is a research exploration. Before treating these results as production evidence, these limitations matter:

**1. The evaluation set is very small.**
Accuracy results are based on 5 questions per corpus, scored by an AI judge. Five questions is a demo, not a statistically meaningful sample. Scores also shift between runs (observed variance of ±1–2 points on the same questions). Claims about accuracy should be treated as directional, not precise.

**2. Baselines are not state-of-the-art.**
The comparisons are against naive full reads, grep top-5, and code graph only. There is no comparison against a proper embedding retriever + reranker, hybrid sparse+dense retrieval, or repo-map approaches (like those used in Aider). The token efficiency claims hold against naive baselines — not necessarily against the best modern alternatives.

**3. The router is hand-tuned, not generalised.**
The routing system works for the 5 question types defined here. A disambiguation rule had to be manually added mid-experiment to correctly route "how does X use Y?" questions. A router that needs hand-authored rules per question pattern will not generalise cleanly to unseen domains without additional tuning.

**4. The corpus sizes do not represent enterprise scale.**
The largest corpus is 145 files and ~66,000 tokens. A real enterprise repository has thousands of files, multiple languages, generated code, stale documentation, and conflicting specs. Results at 145 files do not directly predict behaviour at 10,000 files.

**5. LLM-as-judge is the accuracy measurement.**
There is no human evaluation, no ground-truth answer set, and no task-specific correctness check. The judge model is the same model used for retrieval in several layers, which introduces potential bias. Accuracy figures are indicative, not rigorous.

**6. Cost figures are point-in-time.**
Dollar costs are calculated against April 2026 Claude pricing. Model pricing changes frequently. The token efficiency ratios are durable; the dollar figures are not.

**7. Adaptive spill did not fire in testing.**
The budget reallocation mechanism (carrying unused tokens forward when a layer underperforms) is implemented correctly but returned 0 spill events on the current corpus — every layer was productive for every question tested. The mechanism works as a safety net but is unproven in practice.

---

## What This Is Good For

Despite the limitations, the repo demonstrates something real:

- **The multi-representation idea is directionally correct.** Different question types benefit from different views of a repository. This is underexplored in most retrieval systems.
- **Pre-compilation is a viable strategy.** Building layers once and routing queries across them is meaningfully cheaper than retrieving at query time for repeated use cases.
- **Layered indexing compounds.** Each layer added measurably improved accuracy in testing. The progression from graph-only (10.4) → hybrid (12.6) → wiki (13.6) is consistent and interpretable.
- **Cost-awareness matters.** Most retrieval benchmarks measure accuracy only. This benchmark explicitly models token cost, build amortisation, and break-even — which is relevant for production decisions.

---

## What Would Make This Stronger

If this were to evolve from exploration to evidence:

- Expand to 50+ questions per corpus with human-verified ground truth
- Add strong baselines: embedding retrieval + reranker, repo-map + long context
- Test on real open-source repositories at 1,000+ file scale
- Replace LLM-as-judge with task-specific correctness metrics (exact match, citation grounding)
- Stress-test the router on out-of-distribution question types
- Measure latency, not just token count

---

## Running It

```bash
pip install graphifyy tiktoken anthropic python-dotenv reportlab

echo "ANTHROPIC_API_KEY=sk-..." > .env

python benchmark.py          # runs all 15 tests, builds layers on first run
python generate_pdf_report.py  # generates benchmark-results/prism-benchmark-report.pdf
```

All four layers are built on the first run and cached by SHA-256. Subsequent runs reuse the cache — 89.4% hit rate observed when 5 of 47 files changed.

---

## Project Structure

```
prism-benchmark/
├── benchmark.py              # 15-test benchmark suite
├── generate_pdf_report.py    # PDF report generator
├── architecture.svg          # Architecture diagram
├── benchmark-corpus/
│   ├── small/                # 6 Python files (~3k tokens)
│   ├── medium/               # 47 files (~21k tokens)
│   ├── large/                # 145 files (~66k tokens)
│   ├── docs/                 # 15 markdown READMEs (doc-only corpus)
│   └── mixed/                # 10 code + 5 design docs — primary test corpus
├── graphify-out/mixed/
│   ├── graph.json            # Layer 1: code AST graph
│   ├── doc_index.json        # Layer 2: per-doc LLM extractions
│   ├── bm25_index.json       # Layer 4: keyword index
│   └── wiki/                 # Layer 3: 7 entity pages
└── benchmark-results/
    ├── data.json             # Raw results (all 15 tests)
    ├── prism-benchmark-report.pdf
    └── accuracy/             # Per-question answer files
```

---

## Relation to Prior Work

| Approach | Strength | Gap this repo explores |
|---|---|---|
| Raw file reading | Perfect recall | Cost scales linearly — impractical at corpus size |
| Embedding + vector search | Works on any text | Misses code structure; no pre-synthesis |
| Code graph (graphify / tree-sitter) | Captures architecture and call edges | Cannot read documentation |
| RAG + reranking | Strong text retrieval | Retrieves at query time; no cross-modal fusion |
| LLMWiki *(Karpathy)* | Pre-synthesised compact knowledge | Code-focused; no routing or budget management |
| **PRISM** | Combines all four with question-aware routing | Small eval set; router not yet generalised |

---

*Built with [graphify](https://github.com/graphifyy/graphify) · [tiktoken](https://github.com/openai/tiktoken) · [Anthropic Claude](https://anthropic.com) · pure-Python BM25 · [reportlab](https://www.reportlab.com)*
