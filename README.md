# PRISM — Exploring Multi-Layer Retrieval for Mixed Code + Documentation Corpora

> **One sentence:** Graphify builds the map. PRISM decides which map to use.

> **What this is:** A research exploration into whether routing questions across multiple pre-compiled knowledge representations — code structure graph, document index, synthesised wiki pages, and keyword search — can match the accuracy of raw file retrieval at significantly lower token cost. This is a benchmark artifact and thinking piece, not a production system.

> **Name note:** This project is unrelated to any published PRISM papers in multimodal reasoning or speech processing. The acronym stands for *Pre-compiled Retrieval with Intelligent Strata Management* and refers only to this retrieval routing experiment.

---

## The Problem

Most retrieval approaches treat a repository as flat text. You either read everything (expensive, noisy) or retrieve by similarity (misses structure). Neither is right for all question types.

A question like *"what calls `AuthService.login()`?"* is best answered by a code structure graph. A question like *"why was single-use token design chosen?"* is best answered by a pre-synthesised design rationale page. A question like *"how does `AuthService` use `TokenStore`?"* needs both.

**The hypothesis:** different questions need different representations of the same repository. No single representation wins across all question types. The right move is to build multiple views once, then route intelligently per question.

---

## How PRISM Relates to Graphify

[Graphify](https://github.com/safishamsi/graphify) is the foundation for PRISM's Layer 1. PRISM uses it directly (`pip install graphifyy`) to parse code via tree-sitter AST, build a knowledge graph of nodes and edges, and cluster it into semantic communities.

| | Graphify | PRISM |
|---|---|---|
| **Core idea** | Build a rich knowledge graph of a codebase | Route across multiple representations per question |
| **Focus** | Representation — what does the system look like? | Retrieval strategy — which view answers this question? |
| **When it acts** | Build time (preprocessing) | Query time (routing) |
| **Output** | Graph + queryable structure | Answer + cost/accuracy measurement |
| **Maturity** | Polished tool, widely adopted | Early-stage benchmark exploration |

They are not competitors — they are complementary layers. Graphify builds the map. PRISM is the routing brain that decides which map (or combination of maps) to use for each question.

If PRISM were to mature, Graphify would remain its structural foundation. The other layers (doc index, LLMWiki, BM25) would sit alongside it — not replace it.

---

## Architecture

![PRISM Architecture](architecture.svg)

---

## The Four Layers

| Layer | What it captures | Built how | Cost |
|---|---|---|---|
| **1. Code Graph** (via Graphify) | Every function, class, call edge, and import relationship | Tree-sitter AST — no AI needed | $0 |
| **2. Doc Index** | Per-document summary, key facts, and code cross-references | One AI call per document, cached by SHA-256 | ~$0.002/doc |
| **3. LLMWiki** | Pre-synthesised entity pages integrating code structure with doc content | One AI call per concept page, cached | ~$0.004/page |
| **4. BM25** | Keyword search index across all three layers | Pure term-frequency maths — no AI | $0 |

When a question arrives, a **router** (one AI call) classifies intent and assigns layers and a token budget:

| Question type | Budget | Layer split | Example |
|---|---|---|---|
| structural | 1,500 tok | Graph only | "What calls `AuthService.login()`?" |
| rationale | 2,000 tok | Wiki 80% + graph 20% | "Why was single-use token design chosen?" |
| factual | 2,500 tok | Wiki 60% + doc index 40% | "What does the refresh endpoint guarantee?" |
| similarity | 3,000 tok | BM25 60% + wiki 20% + graph 20% | "Find code similar to the token expiry logic" |
| comprehensive | 5,000 tok | Wiki 40% + graph 30% + BM25 30% | "How does `AuthService` use `TokenStore`?" |

If a layer returns less than 30% of its allocated budget, unused tokens carry forward to the next layer automatically — **adaptive budget spill**.

---

## Benchmark Results

Three hypotheses tested across corpora of 6, 47, and 145 Python files, plus a 15-file mixed code+documentation corpus:

### H1 — Token reduction grows with corpus size ✅

| Corpus | Files | Naive tokens | PRISM tokens | Reduction |
|---|---|---|---|---|
| Small | 6 | 3,138 | 336 | 9.3× |
| Medium | 47 | 21,189 | 1,310 | 16.2× |
| Large | 145 | 65,854 | 1,740 | **37.8×** |

Confirmed. Compression ratio grows monotonically. Graph output is bounded by `--budget`; naive scales linearly with corpus size.

### H2 — Each additional layer adds accuracy without proportional token cost ✅

Tested on a 15-file mixed corpus (10 Python files + 5 design docs), scored by Claude as judge out of 15, across 5 questions:

| Approach | Avg score /15 | Avg tokens | Gap to raw |
|---|---|---|---|
| Code graph alone (Graphify) | 10.4 | 1,697 | −4.0 pts |
| + Doc index | 12.6 | 1,873 | −1.8 pts |
| + LLMWiki | 13.6 | 2,006 | −0.8 pts |
| **PRISM routed** | **12.8** | **3,295** | **−1.6 pts at 2.67× fewer tokens** |
| Raw top-10 files | 14.4 | 8,804 | baseline |

Confirmed directionally. Each layer closes the accuracy gap. PRISM reaches within 1.6 points of raw at 2.67× fewer tokens.

### H3 — Code graph alone cannot answer documentation questions ✅

Graphify's tree-sitter AST extracted **0 nodes from 15 markdown files** — it has no documentation handler by design. Every doc query returned an empty result. The Doc Index and LLMWiki layers resolve this by bypassing the AST entirely.

### Break-even

| Corpus | One-time build cost | Saving per query | Break-even |
|---|---|---|---|
| Small (6 files) | 7,800 tokens | 2,802 tokens | ~3 queries |
| Medium (47 files) | 189,800 tokens | 19,879 tokens | ~10 queries |
| Large (145 files) | 171,600 tokens | 64,114 tokens | ~3 queries |

After break-even every query is pure savings. SHA-256 caching means only changed files trigger a rebuild — 89.4% cache hit rate observed when 5 of 47 files changed.

---

## Honest Limitations

This is a research exploration. Before treating these results as production evidence:

**Small evaluation set.** Accuracy results are based on 5 questions per corpus scored by an AI judge. Five questions is a demo, not a statistically meaningful sample. Scores shift ±1–2 points between runs on identical questions. Results are directional, not precise.

**Weak baselines.** Comparisons are against naive full reads, grep top-5, and code graph only. There is no comparison against a proper embedding retriever + reranker, hybrid sparse+dense retrieval, or repo-map approaches. Token efficiency claims hold against naive baselines — not necessarily against strong modern alternatives.

**Hand-tuned router.** A disambiguation rule had to be manually added mid-experiment to correctly classify "how does X use Y?" questions. A router that requires hand-authored rules per question pattern will not generalise cleanly to unseen domains.

**Corpus sizes don't represent enterprise scale.** The largest test corpus is 145 files (~66k tokens). A real enterprise repository has thousands of files, multiple languages, generated code, stale documentation, and conflicting specs.

**LLM-as-judge is the accuracy measurement.** No human evaluation, no ground-truth answer set, no task-specific correctness check. The judge model is the same model used for several retrieval layers, introducing potential bias.

**Cost figures are point-in-time.** Dollar costs use April 2026 Claude pricing. Token efficiency ratios are durable; dollar figures are not.

**Adaptive spill did not fire in testing.** The budget reallocation mechanism is implemented and correct, but returned 0 events on the current corpus — every layer was productive for every question tested. Works as a safety net; unproven in practice.

---

## What This Is Good For

- **The multi-representation idea is directionally correct.** Different question types benefit from different views of a repository. This is underexplored in most retrieval systems.
- **Pre-compilation is viable for repeated use.** Building layers once and routing across them is meaningfully cheaper than retrieving at query time when the same corpus is queried many times.
- **Layered indexing compounds.** The progression graph-only (10.4) → hybrid (12.6) → wiki (13.6) is consistent and interpretable across runs.
- **Cost-awareness is underrepresented in retrieval benchmarks.** This benchmark explicitly models token cost, build amortisation, and break-even — relevant for real deployment decisions.

---

## What Would Make This Stronger

- Expand to 50–100 questions with human-verified ground truth
- Add strong baselines: embedding retrieval + reranker, repo-map + long context
- Test on real open-source repositories at 1,000+ file scale
- Replace LLM-as-judge with task-specific correctness metrics
- Stress-test the router on out-of-distribution question types
- Ship a usable CLI: `prism run <repo_path>`

---

## Running It on the Pre-Built Corpus

```bash
pip install graphifyy tiktoken anthropic python-dotenv reportlab

echo "ANTHROPIC_API_KEY=sk-..." > .env

python benchmark.py            # runs all 15 tests, builds all 4 layers on first run
python generate_pdf_report.py  # generates benchmark-results/prism-benchmark-report.pdf
```

All four layers are built on the first run and cached by SHA-256. Subsequent runs reuse the cache.

## Running It on Your Own Repository

The benchmark is built around the `mixed` corpus in `benchmark-corpus/mixed/`. To point it at your own code:

1. **Point the corpus path** — edit `MIXED_CORPUS_DIR` in `benchmark.py` to your repo path:
   ```python
   MIXED_CORPUS_DIR = Path("/path/to/your/repo")
   ```

2. **Define your wiki entities** — edit `WIKI_ENTITIES` in `benchmark.py` to name the key concepts in your codebase (services, models, APIs). Each entry gets its own synthesised knowledge page:
   ```python
   WIKI_ENTITIES = [
       {"name": "YourService", "keywords": ["YourService", "your_service"]},
       ...
   ]
   ```

3. **Set your questions** — replace `MIXED_QUESTIONS` with questions relevant to your repo.

4. **Run** — `python benchmark.py` will build all four layers for your repo and run the benchmark.

The first run is the expensive one (builds graph, doc index, wiki, BM25). Every subsequent run reuses the cache and costs near zero for non-LLM tests.

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
│   ├── graph.json            # Layer 1: code AST graph (via Graphify)
│   ├── doc_index.json        # Layer 2: per-doc LLM extractions
│   ├── bm25_index.json       # Layer 4: keyword index
│   └── wiki/                 # Layer 3: synthesised entity pages
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
| **Graphify** | Rich code knowledge graph, AST-based, widely adopted | Single representation; no cross-modal doc routing |
| RAG + reranking | Strong text retrieval | Retrieves at query time; no cross-modal fusion |
| [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) *(Karpathy)* | Pre-synthesised compact knowledge | Single-layer; no routing or budget management |
| **PRISM** | Routes across all of the above per question type | Small eval set; router not yet generalised; early stage |

---

---

## Credits & Attributions

| What | Credit |
|---|---|
| **Code graph (Layer 1)** | [Graphify](https://github.com/safishamsi/graphify) by Safi Shamsi — AST parsing via tree-sitter |
| **LLM Wiki concept (Layer 3)** | [Andrej Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — pre-synthesised entity pages from raw sources |
| **BM25 algorithm (Layer 4)** | Robertson & Zaragoza (2009), *The Probabilistic Relevance Framework: BM25 and Beyond* — standard Okapi BM25, k1=1.5, b=0.75 |
| **Doc corpus** | Public READMEs from [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy, MIT), [Transformers](https://github.com/huggingface/transformers), [Whisper](https://github.com/openai/whisper), [vLLM](https://github.com/vllm-project/vllm), [LlamaIndex](https://github.com/run-llama/llama_index), and others — fetched from their public repositories under their respective licences |
| **Token counting** | [tiktoken](https://github.com/openai/tiktoken) by OpenAI |
| **LLM inference & judging** | [Anthropic Claude](https://anthropic.com) (claude-sonnet-4-6) |
| **PDF generation** | [reportlab](https://www.reportlab.com) |

The mixed code+doc corpus (`benchmark-corpus/mixed/`) is original synthetic code written for this benchmark and is not derived from any external project.
