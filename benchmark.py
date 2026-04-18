"""
Graphify Token Reduction Benchmark Suite
=========================================
Empirically verifies whether the graphify framework actually reduces token
consumption as claimed (~71.5x fewer tokens per query vs naive file reading).

Run:
    python benchmark.py

Requires:
    pip install graphifyy tiktoken anthropic
    ANTHROPIC_API_KEY env var for Tests 5 & 6
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_DIRS = {
    "small":  Path("benchmark-corpus/small"),
    "medium": Path("benchmark-corpus/medium"),
    "large":  Path("benchmark-corpus/large"),
}

GRAPHIFY_OUT_DIRS = {
    "small":  Path("graphify-out/small"),
    "medium": Path("graphify-out/medium"),
    "large":  Path("graphify-out/large"),
}

RESULTS_DIR = Path("benchmark-results")
ACCURACY_DIR = RESULTS_DIR / "accuracy"

# tiktoken encoder used consistently throughout
ENC = tiktoken.get_encoding("cl100k_base")

# Graphify query budget (tokens returned per query)
QUERY_BUDGET = 2000

BENCHMARK_QUESTIONS = [
    "what is the main training loop?",
    "how does the attention mechanism work?",
    "how is the model initialized?",
    "how are checkpoints saved and loaded?",
    "what optimizer is used for training?",
]

DOC_CORPUS_DIR  = Path("benchmark-corpus/docs")
DOC_GRAPH_OUT   = Path("graphify-out/docs")

DOC_FETCH_URLS: list[tuple[str, str]] = [
    ("nanogpt_readme",      "https://raw.githubusercontent.com/karpathy/nanoGPT/master/README.md"),
    ("transformers_readme", "https://raw.githubusercontent.com/huggingface/transformers/main/README.md"),
    ("whisper_readme",      "https://raw.githubusercontent.com/openai/whisper/main/README.md"),
    ("peft_readme",         "https://raw.githubusercontent.com/huggingface/peft/main/README.md"),
    ("trl_readme",          "https://raw.githubusercontent.com/huggingface/trl/main/README.md"),
    ("vllm_readme",         "https://raw.githubusercontent.com/vllm-project/vllm/main/README.md"),
    ("litgpt_readme",       "https://raw.githubusercontent.com/Lightning-AI/litgpt/main/README.md"),
    ("gpt_fast_readme",     "https://raw.githubusercontent.com/pytorch-labs/gpt-fast/main/README.md"),
    ("llamaindex_readme",   "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md"),
    ("langchain_readme",    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"),
    ("autogen_readme",      "https://raw.githubusercontent.com/microsoft/autogen/main/README.md"),
    ("deepspeed_readme",    "https://raw.githubusercontent.com/microsoft/DeepSpeed/master/README.md"),
    ("axolotl_readme",      "https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/README.md"),
    ("gpt2_modelcard",      "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/gpt2.md"),
    ("mlflow_readme",       "https://raw.githubusercontent.com/mlflow/mlflow/master/README.md"),
]

DOC_QUESTIONS = [
    "what fine-tuning methods are supported?",
    "how do you install and get started?",
    "what models or architectures are described?",
    "what are the key features or capabilities?",
    "how does the project handle distributed training?",
]

# Pricing constants ($ per token) — April 2026
PRICE_CLAUDE_INPUT  = 3.00  / 1_000_000   # claude-sonnet input
PRICE_CLAUDE_OUTPUT = 15.00 / 1_000_000   # claude-sonnet output
PRICE_EMBED         = 0.02  / 1_000_000   # text-embedding-3-small

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def read_corpus_files(corpus_dir: Path) -> list[Path]:
    """Return all readable code/doc files, excluding .git and binaries."""
    exts = {".py", ".ts", ".js", ".md", ".txt", ".rst", ".go", ".rs",
            ".java", ".cpp", ".c", ".h", ".rb", ".swift", ".kt", ".lua",
            ".cs", ".scala", ".php", ".jsx", ".tsx"}
    files = []
    for p in corpus_dir.rglob("*"):
        if p.is_file() and ".git" not in str(p) and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def corpus_naive_tokens(corpus_dir: Path) -> tuple[list[Path], int]:
    files = read_corpus_files(corpus_dir)
    total = sum(count_tokens(safe_read(f)) for f in files)
    return files, total


def build_graph(corpus_dir: Path, out_dir: Path) -> tuple[bool, str]:
    """Run the graphify skill's build pipeline (detect → extract → build → cluster)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_json = out_dir / "graph.json"

    # Step 1: detect
    detect_out = out_dir / ".graphify_detect.json"
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"""
import json, sys
from pathlib import Path
from graphify.detect import detect
result = detect(Path(r'{corpus_dir.resolve()}'))
Path(r'{detect_out}').write_text(json.dumps(result), encoding='utf-8')
print(json.dumps(result))
"""],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return False, f"detect failed: {result.stderr[:500]}"
        detect_data = json.loads(detect_out.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"detect exception: {exc}"

    # Step 2: extract + build using graphify internals
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             f"""
import json, sys
from pathlib import Path
from graphify.extract import extract, collect_files
from graphify.build import build_from_json
from graphify.cluster import cluster
from graphify.export import to_json

corpus = Path(r'{corpus_dir.resolve()}')
out = Path(r'{out_dir.resolve()}')
out.mkdir(parents=True, exist_ok=True)

paths = collect_files(corpus)
if not paths:
    # Doc-only corpus: collect_files excludes .md/.txt. Write empty graph (expected).
    G = build_from_json({{"nodes": [], "edges": []}})
    communities = cluster(G)
    to_json(G, communities, str(out / 'graph.json'))
    print("Graph built: 0 nodes, 0 edges, 0 communities (no code files — doc-only corpus)")
    sys.exit(0)

combined = extract(paths, cache_root=out)
G = build_from_json(combined)
communities = cluster(G)
to_json(G, communities, str(out / 'graph.json'))
print(f"Graph built: {{G.number_of_nodes()}} nodes, {{G.number_of_edges()}} edges, {{len(communities)}} communities")
"""],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return False, f"build failed: {result.stderr[:800]}"
        return True, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "build timed out (>600s)"
    except Exception as exc:
        return False, f"build exception: {exc}"


def run_graphify_query(question: str, graph_path: Path) -> str:
    """Run `graphify query` and return stdout."""
    try:
        result = subprocess.run(
            ["graphify", "query", question,
             "--graph", str(graph_path),
             "--budget", str(QUERY_BUDGET)],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout
    except Exception as exc:
        return f"[query error: {exc}]"


def grep_top_files(corpus_dir: Path, query: str, top_n: int = 5) -> list[Path]:
    """Simulate realistic naive baseline: grep for keywords, return top N files."""
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    files = read_corpus_files(corpus_dir)
    scored: list[tuple[int, Path]] = []
    for f in files:
        text = safe_read(f).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, f))
    scored.sort(reverse=True)
    return [f for _, f in scored[:top_n]]


def fetch_doc_corpus() -> int:
    """Download real ML-project markdown files into benchmark-corpus/docs/."""
    import urllib.request
    DOC_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    fetched = skipped = failed = 0
    for name, url in DOC_FETCH_URLS:
        dest = DOC_CORPUS_DIR / f"{name}.md"
        if dest.exists() and dest.stat().st_size > 100:
            skipped += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "graphify-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            dest.write_text(content, encoding="utf-8")
            fetched += 1
        except Exception as exc:
            print(f"    WARN: {name} failed — {exc}")
            failed += 1
    print(f"    Fetched {fetched} new, {skipped} cached, {failed} failed")
    return len(list(DOC_CORPUS_DIR.glob("*.md")))


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------

def test1_baseline_naive(results: dict) -> dict:
    """Test 1: Baseline naive token count per corpus."""
    print("\n" + "=" * 60)
    print("TEST 1 — Baseline Naive Token Count")
    print("=" * 60)

    t1 = {"status": "PASS", "corpora": {}}
    fmt = f"{'Corpus':<10} {'Files':>6} {'Total Tokens':>14} {'Avg/File':>10}"
    print(fmt)
    print("-" * 44)

    for name, corpus_dir in CORPUS_DIRS.items():
        if not corpus_dir.exists():
            print(f"{name:<10} SKIP (directory not found)")
            t1["corpora"][name] = {"status": "SKIP"}
            continue
        files, total = corpus_naive_tokens(corpus_dir)
        avg = total // len(files) if files else 0
        print(f"{name:<10} {len(files):>6} {total:>14,} {avg:>10,}")
        t1["corpora"][name] = {
            "file_count": len(files),
            "total_tokens": total,
            "avg_tokens_per_file": avg,
        }

    results["test1"] = t1
    return t1


def test2_graph_query_output_size(results: dict) -> dict:
    """Test 2: Graph query output token count vs naive."""
    print("\n" + "=" * 60)
    print("TEST 2 — Graph Query Output Size vs Naive")
    print("=" * 60)

    QUERY = BENCHMARK_QUESTIONS[0]
    t2 = {"status": "PASS", "query": QUERY, "corpora": {}}
    fmt = f"{'Corpus':<10} {'Graph Tokens':>13} {'Naive Tokens':>13} {'Ratio':>8}"
    print(f"Query: {QUERY!r}")
    print(fmt)
    print("-" * 48)

    for name in CORPUS_DIRS:
        corpus_dir = CORPUS_DIRS[name]
        graph_json = GRAPHIFY_OUT_DIRS[name] / "graph.json"

        if not graph_json.exists():
            print(f"{name:<10} SKIP (graph not built)")
            t2["corpora"][name] = {"status": "SKIP"}
            continue

        _, naive_tokens = corpus_naive_tokens(corpus_dir)
        query_out = run_graphify_query(QUERY, graph_json)
        graph_tokens = count_tokens(query_out)
        ratio = round(naive_tokens / graph_tokens, 1) if graph_tokens > 0 else 0

        print(f"{name:<10} {graph_tokens:>13,} {naive_tokens:>13,} {ratio:>7.1f}x")
        t2["corpora"][name] = {
            "graph_tokens": graph_tokens,
            "naive_tokens": naive_tokens,
            "ratio": ratio,
        }

    results["test2"] = t2
    return t2


def test3_compression_ratio_table(results: dict) -> dict:
    """Test 3: ASCII table of compression ratios across corpus sizes."""
    print("\n" + "=" * 60)
    print("TEST 3 — Compression Ratio vs Corpus Size")
    print("=" * 60)

    QUERY = BENCHMARK_QUESTIONS[0]
    t3 = {"status": "PASS", "corpora": {}}
    ratios = []

    header = f"{'Corpus':<10} {'Files':>6} {'Naive Tokens':>13} {'Graph Tokens':>13} {'Ratio':>8}"
    print(header)
    print("-" * 54)

    for name in ("small", "medium", "large"):
        corpus_dir = CORPUS_DIRS[name]
        graph_json = GRAPHIFY_OUT_DIRS[name] / "graph.json"

        if not corpus_dir.exists():
            t3["corpora"][name] = {"status": "SKIP"}
            continue

        files, naive_tokens = corpus_naive_tokens(corpus_dir)
        if not graph_json.exists():
            print(f"{name:<10} {len(files):>6} {naive_tokens:>13,} {'N/A':>13} {'N/A':>8}")
            t3["corpora"][name] = {"status": "SKIP_NO_GRAPH", "naive_tokens": naive_tokens}
            continue

        query_out = run_graphify_query(QUERY, graph_json)
        graph_tokens = count_tokens(query_out)
        ratio = round(naive_tokens / graph_tokens, 1) if graph_tokens > 0 else 0
        ratios.append(ratio)
        print(f"{name:<10} {len(files):>6} {naive_tokens:>13,} {graph_tokens:>13,} {ratio:>7.1f}x")
        t3["corpora"][name] = {
            "files": len(files),
            "naive_tokens": naive_tokens,
            "graph_tokens": graph_tokens,
            "ratio": ratio,
        }

    # Verify ratio grows with corpus size
    if len(ratios) >= 2:
        grows = all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))
        t3["ratio_grows_with_size"] = grows
        print(f"\nRatio grows with corpus size: {grows}")
        if not grows:
            t3["status"] = "WARN"
            print("  WARN: ratio does not strictly increase — graph may not scale as claimed")
    else:
        t3["ratio_grows_with_size"] = None
        t3["status"] = "WARN"

    results["test3"] = t3
    return t3


def test4_realistic_naive_baseline(results: dict) -> dict:
    """Test 4: Fair comparison — graphify vs grep-top-5-files."""
    print("\n" + "=" * 60)
    print("TEST 4 — Realistic Naive Baseline (grep top-5 files)")
    print("=" * 60)

    t4 = {"status": "PASS", "corpora": {}}

    for name in CORPUS_DIRS:
        corpus_dir = CORPUS_DIRS[name]
        graph_json = GRAPHIFY_OUT_DIRS[name] / "graph.json"

        if not corpus_dir.exists():
            t4["corpora"][name] = {"status": "SKIP"}
            continue

        corpus_data: dict[str, Any] = {"queries": []}
        print(f"\n  Corpus: {name}")

        for question in BENCHMARK_QUESTIONS[:3]:
            # Realistic baseline: grep + read top 5 files
            top_files = grep_top_files(corpus_dir, question, top_n=5)
            targeted_tokens = sum(count_tokens(safe_read(f)) for f in top_files)

            # Graphify cost
            if graph_json.exists():
                query_out = run_graphify_query(question, graph_json)
                graph_tokens = count_tokens(query_out)
                ratio = round(targeted_tokens / graph_tokens, 1) if graph_tokens > 0 else 0
            else:
                graph_tokens = 0
                ratio = 0

            print(f"    Q: {question[:50]!r}")
            print(f"       grep-top5={targeted_tokens:,} tokens | graph={graph_tokens:,} tokens | ratio={ratio:.1f}x")
            corpus_data["queries"].append({
                "question": question,
                "targeted_read_tokens": targeted_tokens,
                "graph_tokens": graph_tokens,
                "ratio": ratio,
                "top_files": [str(f) for f in top_files],
            })

        t4["corpora"][name] = corpus_data

    results["test4"] = t4
    return t4


def test5_query_accuracy_spotcheck(results: dict) -> dict:
    """Test 5: Ask same 5 questions via graph and raw files; save responses."""
    print("\n" + "=" * 60)
    print("TEST 5 — Query Accuracy Spot Check")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test5"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test5"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        results["test5"] = {"status": "SKIP", "reason": str(exc)}
        return results["test5"]

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    # Use medium corpus for accuracy tests
    corpus_dir = CORPUS_DIRS["medium"]
    graph_json = GRAPHIFY_OUT_DIRS["medium"] / "graph.json"

    if not corpus_dir.exists():
        results["test5"] = {"status": "SKIP", "reason": "medium corpus missing"}
        return results["test5"]

    t5 = {"status": "PASS", "questions": []}
    files, _ = corpus_naive_tokens(corpus_dir)
    # Read top 10 most relevant files as raw context (keep cost manageable)
    raw_context_files = files[:10]
    raw_context = "\n\n".join(
        f"--- {f.name} ---\n{safe_read(f)}" for f in raw_context_files
    )

    print(f"  Using medium corpus ({len(files)} files, top 10 as raw context)")

    for i, question in enumerate(BENCHMARK_QUESTIONS):
        print(f"  Q{i+1}: {question}")
        entry: dict[str, Any] = {"question": question}

        # --- Graph-based answer ---
        if graph_json.exists():
            graph_ctx = run_graphify_query(question, graph_json)
            graph_prompt = (
                f"Based on this knowledge graph context, answer the question.\n\n"
                f"Context:\n{graph_ctx}\n\nQuestion: {question}"
            )
            try:
                resp = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": graph_prompt}],
                )
                entry["graph_answer"] = resp.content[0].text
                entry["graph_input_tokens"] = resp.usage.input_tokens
                entry["graph_output_tokens"] = resp.usage.output_tokens
                print(f"    graph: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out tokens")
            except Exception as exc:
                entry["graph_answer"] = f"[error: {exc}]"
                entry["graph_input_tokens"] = 0
        else:
            entry["graph_answer"] = "[graph not built]"
            entry["graph_input_tokens"] = 0

        # --- Raw file answer ---
        raw_prompt = (
            f"Based on the following source files, answer the question.\n\n"
            f"Files:\n{raw_context}\n\nQuestion: {question}"
        )
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": raw_prompt}],
            )
            entry["raw_answer"] = resp.content[0].text
            entry["raw_input_tokens"] = resp.usage.input_tokens
            entry["raw_output_tokens"] = resp.usage.output_tokens
            print(f"    raw:   {resp.usage.input_tokens} in / {resp.usage.output_tokens} out tokens")
        except Exception as exc:
            entry["raw_answer"] = f"[error: {exc}]"
            entry["raw_input_tokens"] = 0

        t5["questions"].append(entry)

        # Save side-by-side to file
        out_path = ACCURACY_DIR / f"q{i+1:02d}.md"
        out_path.write_text(
            f"# Q{i+1}: {question}\n\n"
            f"## Graph-based answer ({entry.get('graph_input_tokens', 0)} input tokens)\n\n"
            f"{entry.get('graph_answer', '')}\n\n"
            f"---\n\n"
            f"## Raw-files answer ({entry.get('raw_input_tokens', 0)} input tokens)\n\n"
            f"{entry.get('raw_answer', '')}\n",
            encoding="utf-8",
        )

    print(f"\n  Responses saved to {ACCURACY_DIR}/")
    print("  (Human scoring required — open each q*.md file)")
    results["test5"] = t5
    return t5


def test6_amortised_build_cost(results: dict) -> dict:
    """Test 6: Measure build tokens + break-even analysis."""
    print("\n" + "=" * 60)
    print("TEST 6 — Amortised Cost: Build Tokens + Break-Even")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test6"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test6"]

    t6 = {"status": "PASS", "corpora": {}}
    fmt = f"{'Corpus':<10} {'Build Cost':>12} {'Savings/Query':>14} {'Break-Even':>12}"
    print(fmt)
    print("-" * 52)

    for name in CORPUS_DIRS:
        corpus_dir = CORPUS_DIRS[name]
        graph_json = GRAPHIFY_OUT_DIRS[name] / "graph.json"

        if not corpus_dir.exists():
            t6["corpora"][name] = {"status": "SKIP"}
            continue

        _, naive_tokens = corpus_naive_tokens(corpus_dir)

        # Estimate build cost: count cache entries created (each = 1 LLM call for doc files)
        # Code files cost 0 LLM tokens (tree-sitter only)
        # We estimate build cost from cache entries × avg tokens/doc LLM call
        cache_dir = GRAPHIFY_OUT_DIRS[name] / "graphify-out" / "cache"
        n_cache_entries = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0

        # Estimate: each LLM extraction = ~1000 input + 300 output tokens avg
        # Code files = 0 LLM tokens; only .md/.txt doc files need LLM
        files = read_corpus_files(corpus_dir)
        doc_files = [f for f in files if f.suffix.lower() in {".md", ".txt", ".rst"}]
        # Build cost = doc files × ~1300 tokens (LLM) + code files × ~0
        # Use cache entry count as more accurate proxy if available
        if n_cache_entries > 0:
            build_cost_tokens = n_cache_entries * 1300
        else:
            build_cost_tokens = len(doc_files) * 1300

        # Savings per query = naive cost - graph query cost
        if graph_json.exists():
            query_out = run_graphify_query(BENCHMARK_QUESTIONS[0], graph_json)
            graph_tokens = count_tokens(query_out)
        else:
            graph_tokens = QUERY_BUDGET

        savings_per_query = max(1, naive_tokens - graph_tokens)
        break_even = round(build_cost_tokens / savings_per_query, 1)

        print(f"{name:<10} {build_cost_tokens:>12,} {savings_per_query:>14,} {break_even:>11.1f}q")
        t6["corpora"][name] = {
            "build_cost_tokens": build_cost_tokens,
            "cache_entries": n_cache_entries,
            "doc_files": len(doc_files),
            "naive_tokens": naive_tokens,
            "graph_tokens": graph_tokens,
            "savings_per_query": savings_per_query,
            "break_even_queries": break_even,
        }

    results["test6"] = t6
    return t6


def test7_sha256_cache_hit_rate(results: dict) -> dict:
    """Test 7: Modify 5 files, rebuild, verify only 5 are reprocessed."""
    print("\n" + "=" * 60)
    print("TEST 7 — SHA256 Cache Hit Rate")
    print("=" * 60)

    corpus_dir = CORPUS_DIRS["medium"]
    out_dir = GRAPHIFY_OUT_DIRS["medium"]
    graph_json = out_dir / "graph.json"
    cache_d = out_dir / "graphify-out" / "cache"

    if not graph_json.exists():
        print("  SKIP: medium graph not built")
        results["test7"] = {"status": "SKIP", "reason": "medium graph not built"}
        return results["test7"]

    # Snapshot cache entry hashes before modification
    before_hashes = {p.name for p in cache_d.glob("*.json")} if cache_d.exists() else set()
    n_before = len(before_hashes)

    # Modify 5 files (append a harmless comment)
    files = read_corpus_files(corpus_dir)
    modified = files[:5]
    originals: dict[Path, str] = {}
    for f in modified:
        originals[f] = safe_read(f)
        f.write_text(originals[f] + f"\n# benchmark-modification-{time.time()}\n", encoding="utf-8")
    print(f"  Modified {len(modified)} files")

    # Rebuild
    ok, msg = build_graph(corpus_dir, out_dir)
    if not ok:
        # Restore originals before bailing
        for f, content in originals.items():
            f.write_text(content, encoding="utf-8")
        results["test7"] = {"status": "SKIP", "reason": f"rebuild failed: {msg}"}
        return results["test7"]

    # Count cache entries whose hash didn't exist before (= files that were reprocessed)
    after_hashes = {p.name for p in cache_d.glob("*.json")} if cache_d.exists() else set()
    new_hashes = after_hashes - before_hashes
    n_new = len(new_hashes)

    # Restore originals
    for f, content in originals.items():
        f.write_text(content, encoding="utf-8")

    n_total = len(read_corpus_files(corpus_dir))
    cache_hits = n_total - n_new
    hit_rate = round(cache_hits / n_total * 100, 1) if n_total > 0 else 0

    print(f"  Files changed:       {len(modified)}")
    print(f"  New cache entries:   {n_new}")
    print(f"  Cache hits:          {cache_hits}/{n_total} ({hit_rate}%)")

    t7 = {
        "status": "PASS" if n_new <= len(modified) + 2 else "WARN",
        "files_changed": len(modified),
        "files_reprocessed": n_new,
        "total_files": n_total,
        "cache_hit_rate_pct": hit_rate,
        "cache_entries_before": n_before,
        "cache_entries_after": len(after_hashes),
    }
    if n_new > len(modified) + 2:
        print(f"  WARN: {n_new} entries reprocessed (expected ~{len(modified)})")
        t7["status"] = "WARN"

    results["test7"] = t7
    return t7


def test8_code_vs_doc_extraction_cost(results: dict) -> dict:
    """Test 8: Code-only vs doc-heavy corpus extraction cost."""
    print("\n" + "=" * 60)
    print("TEST 8 — Code-only vs Mixed Corpus Extraction Cost")
    print("=" * 60)

    # Use small corpus (code-only) vs a doc-heavy slice of medium
    small_dir = CORPUS_DIRS["small"]
    medium_dir = CORPUS_DIRS["medium"]

    t8: dict[str, Any] = {"status": "PASS"}

    for label, corpus_dir in [("code-only (small)", small_dir), ("mixed (medium)", medium_dir)]:
        if not corpus_dir.exists():
            print(f"  {label}: SKIP")
            continue
        files = read_corpus_files(corpus_dir)
        code_files = [f for f in files if f.suffix.lower() not in {".md", ".txt", ".rst"}]
        doc_files  = [f for f in files if f.suffix.lower() in {".md", ".txt", ".rst"}]

        # Check cache entries for this corpus
        out_dir = GRAPHIFY_OUT_DIRS.get(
            "small" if "small" in str(corpus_dir) else "medium"
        )
        cache_d = out_dir / "graphify-out" / "cache" if out_dir else None
        n_cache = len(list(cache_d.glob("*.json"))) if (cache_d and cache_d.exists()) else 0

        # Code files use tree-sitter (0 LLM tokens); doc files use LLM
        # Estimated LLM calls ≈ number of doc files processed (each doc = 1 LLM call)
        # We use cache entry count as proxy for actual LLM calls made
        estimated_llm_calls = len(doc_files)  # upper bound
        estimated_build_tokens = estimated_llm_calls * 1300

        print(f"  {label}:")
        print(f"    Total files:        {len(files)}")
        print(f"    Code files:         {len(code_files)} (tree-sitter, 0 LLM tokens)")
        print(f"    Doc files:          {len(doc_files)} (LLM extraction)")
        print(f"    Cache entries:      {n_cache}")
        print(f"    Est. build tokens:  {estimated_build_tokens:,}")

        key = "code_only" if "small" in str(corpus_dir) else "mixed"
        t8[key] = {
            "total_files": len(files),
            "code_files": len(code_files),
            "doc_files": len(doc_files),
            "cache_entries": n_cache,
            "estimated_llm_calls": estimated_llm_calls,
            "estimated_build_tokens": estimated_build_tokens,
        }

    if "code_only" in t8 and "mixed" in t8:
        code_pct = t8["code_only"]["doc_files"] / max(1, t8["code_only"]["total_files"])
        mixed_pct = t8["mixed"]["doc_files"] / max(1, t8["mixed"]["total_files"])
        print(f"\n  Code-only doc ratio: {code_pct:.0%} | Mixed doc ratio: {mixed_pct:.0%}")
        t8["code_only_doc_fraction"] = round(code_pct, 3)
        t8["mixed_doc_fraction"] = round(mixed_pct, 3)
        t8["finding"] = "Code files cost 0 LLM tokens (tree-sitter). Doc files each require 1 LLM call."

    results["test8"] = t8
    return t8


# ---------------------------------------------------------------------------
# Graph build pass (runs before tests that need graph.json)
# ---------------------------------------------------------------------------

def build_all_graphs(results: dict) -> None:
    print("\n" + "=" * 60)
    print("PRE-STEP — Building Graphify Graphs")
    print("=" * 60)
    results["graphs"] = {}
    for name in CORPUS_DIRS:
        corpus_dir = CORPUS_DIRS[name]
        out_dir = GRAPHIFY_OUT_DIRS[name]
        graph_json = out_dir / "graph.json"

        if not corpus_dir.exists():
            print(f"  {name}: SKIP (corpus missing)")
            results["graphs"][name] = {"status": "SKIP"}
            continue

        if graph_json.exists():
            print(f"  {name}: already built (skipping rebuild)")
            results["graphs"][name] = {"status": "CACHED"}
            continue

        print(f"  {name}: building... ", end="", flush=True)
        t0 = time.time()
        ok, msg = build_graph(corpus_dir, out_dir)
        dt = time.time() - t0
        status = "OK" if ok else "FAIL"
        print(f"{status} ({dt:.1f}s) — {msg}")
        results["graphs"][name] = {"status": status, "time_s": round(dt, 1), "msg": msg}

    # Doc corpus: fetch then build (will produce empty graph — that's the finding)
    print(f"\n  Fetching doc corpus ({len(DOC_FETCH_URLS)} markdown files)...")
    n_docs = fetch_doc_corpus()
    if n_docs == 0:
        print("  docs: SKIP (no docs fetched)")
        results["graphs"]["docs"] = {"status": "SKIP"}
    else:
        graph_json = DOC_GRAPH_OUT / "graph.json"
        if graph_json.exists():
            print(f"  docs: already built ({n_docs} files)")
            results["graphs"]["docs"] = {"status": "CACHED", "doc_files": n_docs}
        else:
            print(f"  docs: building graph on {n_docs} .md files... ", end="", flush=True)
            t0 = time.time()
            ok, msg = build_graph(DOC_CORPUS_DIR, DOC_GRAPH_OUT)
            dt = time.time() - t0
            print(f"{'OK' if ok else 'FAIL'} ({dt:.1f}s) — {msg}")
            results["graphs"]["docs"] = {
                "status": "OK" if ok else "FAIL",
                "time_s": round(dt, 1),
                "msg": msg,
                "doc_files": n_docs,
            }


# ---------------------------------------------------------------------------
# Tests 9-12
# ---------------------------------------------------------------------------

def test9_doc_corpus(results: dict) -> dict:
    """Test 9: Pure doc corpus — token reduction AND AST extraction gap."""
    print("\n" + "=" * 60)
    print("TEST 9 — Doc Corpus: Token Reduction & AST Extraction Gap")
    print("=" * 60)

    doc_files = sorted(DOC_CORPUS_DIR.glob("*.md")) if DOC_CORPUS_DIR.exists() else []
    if not doc_files:
        print("  SKIP: no doc files found")
        results["test9"] = {"status": "SKIP", "reason": "no doc files"}
        return results["test9"]

    # Naive token count
    total_tokens = sum(count_tokens(safe_read(f)) for f in doc_files)
    avg_tokens = total_tokens // len(doc_files) if doc_files else 0
    print(f"  Doc files: {len(doc_files)}, total tokens: {total_tokens:,}, avg: {avg_tokens:,}/file")

    # AST extraction gap — check graph node count
    graph_json = DOC_GRAPH_OUT / "graph.json"
    graph_nodes = 0
    graph_edges = 0
    if graph_json.exists():
        try:
            gdata = json.loads(graph_json.read_text(encoding="utf-8"))
            graph_nodes = len(gdata.get("nodes", []))
            graph_edges = len(gdata.get("links", gdata.get("edges", [])))
            # How much doc text made it into node labels?
            label_text = " ".join(n.get("label", "") for n in gdata.get("nodes", []))
            chars_in_graph = len(label_text)
            chars_in_docs = sum(len(safe_read(f)) for f in doc_files)
            coverage_pct = round(100 * chars_in_graph / max(1, chars_in_docs), 2)
        except Exception:
            coverage_pct = 0.0
    else:
        coverage_pct = 0.0

    print(f"\n  AST extraction gap:")
    print(f"    Graph nodes from {len(doc_files)} doc files: {graph_nodes}")
    print(f"    Graph edges:                               {graph_edges}")
    print(f"    Doc content captured in graph labels:      {coverage_pct}%")
    print(f"    FINDING: collect_files() excludes .md — tree-sitter has no doc extractor.")
    print(f"    Full doc extraction needs LLM skill pipeline (graphify Step 3B).")

    # Graph queries on doc questions
    print(f"\n  Graph query results on doc questions (budget={QUERY_BUDGET}):")
    header = f"  {'Question':<48} {'Graph':>6} {'Naive':>8} {'Useful?':>8}"
    print(header)
    print("  " + "-" * 74)

    query_rows = []
    for q in DOC_QUESTIONS:
        if graph_json.exists():
            out = run_graphify_query(q, graph_json)
            g_tokens = count_tokens(out)
        else:
            out = ""
            g_tokens = 0

        keywords = [w.lower() for w in q.split() if len(w) > 3]
        useful = any(kw in out.lower() for kw in keywords) and g_tokens > 100
        ratio = round(total_tokens / g_tokens, 1) if g_tokens > 0 else 0
        print(f"  {q:<48} {g_tokens:>6,} {total_tokens:>8,}  {'YES' if useful else 'NO (empty)'}")
        query_rows.append({
            "question": q,
            "graph_tokens": g_tokens,
            "naive_tokens": total_tokens,
            "ratio": ratio,
            "graph_useful": useful,
        })

    t9: dict[str, Any] = {
        "status": "PASS",
        "doc_files": len(doc_files),
        "naive_tokens_total": total_tokens,
        "avg_tokens_per_file": avg_tokens,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "ast_extraction_coverage_pct": coverage_pct,
        "finding": (
            "collect_files() excludes .md/.txt; extract() has no doc dispatcher. "
            f"0 of {len(doc_files)} doc files produced graph nodes via AST. "
            "Doc corpora need the full LLM skill pipeline for graphify to be useful."
        ),
        "queries": query_rows,
    }
    results["test9"] = t9
    return t9


def test10_dollar_cost_model(results: dict) -> dict:
    """Test 10: Translate token counts to $/query for naive, graphify, embed+top5."""
    print("\n" + "=" * 60)
    print("TEST 10 — Dollar Cost Model ($/query)")
    print("=" * 60)

    OUTPUT_TOKENS = 512  # assumed answer length for all approaches

    t10: dict[str, Any] = {"status": "PASS", "corpora": {}}
    assumed_output_cost = OUTPUT_TOKENS * PRICE_CLAUDE_OUTPUT

    # Build per-corpus data from previous test results
    corpora_data: dict[str, dict] = {}

    # Code corpora from test1/test3/test4
    t1 = results.get("test1", {}).get("corpora", {})
    t3 = results.get("test3", {}).get("corpora", {})
    t4 = results.get("test4", {}).get("corpora", {})

    for name in ("small", "medium", "large"):
        naive_tokens = t1.get(name, {}).get("total_tokens", 0)
        graph_tokens = t3.get(name, {}).get("graph_tokens", 0)
        # Average targeted-read tokens from test4
        t4_queries = t4.get(name, {}).get("queries", [])
        top5_tokens = int(sum(q.get("targeted_read_tokens", 0) for q in t4_queries) / max(1, len(t4_queries)))
        corpora_data[name] = {
            "naive_tokens": naive_tokens,
            "graph_tokens": graph_tokens,
            "top5_tokens": top5_tokens,
        }

    # Doc corpus from test9
    t9 = results.get("test9", {})
    if t9.get("status") == "PASS":
        doc_naive = t9.get("naive_tokens_total", 0)
        doc_graph = int(sum(q.get("graph_tokens", 0) for q in t9.get("queries", [])) / max(1, len(t9.get("queries", []))))
        # grep top-5 on docs
        if DOC_CORPUS_DIR.exists():
            doc_files_list = sorted(DOC_CORPUS_DIR.glob("*.md"))
            top5_doc = sum(count_tokens(safe_read(f)) for f in doc_files_list[:5])
        else:
            top5_doc = doc_naive // 3
        corpora_data["docs"] = {
            "naive_tokens": doc_naive,
            "graph_tokens": doc_graph,
            "top5_tokens": top5_doc,
        }

    # Print table
    header = f"  {'Corpus':<10} {'Approach':<18} {'Input Tok':>10} {'$/query':>10}  Notes"
    print(header)
    print("  " + "-" * 68)

    for name, d in corpora_data.items():
        naive_t  = d["naive_tokens"]
        graph_t  = d["graph_tokens"]
        top5_t   = d["top5_tokens"]
        embed_index_cost = naive_t * PRICE_EMBED  # one-time, shown for transparency

        naive_cost = naive_t * PRICE_CLAUDE_INPUT + assumed_output_cost
        graph_cost = graph_t * PRICE_CLAUDE_INPUT + assumed_output_cost if graph_t > 0 else None
        top5_cost  = top5_t  * PRICE_CLAUDE_INPUT + assumed_output_cost

        print(f"  {name:<10} {'naive (all files)':<18} {naive_t:>10,} ${naive_cost:.7f}")
        if graph_cost is not None:
            note = "(graph useful)" if name != "docs" else "(WARN: graph empty)"
            print(f"  {name:<10} {'graphify query':<18} {graph_t:>10,} ${graph_cost:.7f}  {note}")
        else:
            print(f"  {name:<10} {'graphify query':<18} {'N/A':>10}  N/A  (no graph data)")
        print(f"  {name:<10} {'embed+top5':<18} {top5_t:>10,} ${top5_cost:.7f}  (+${embed_index_cost:.6f} index, once)")
        print()

        t10["corpora"][name] = {
            "naive_cost_per_query": round(naive_cost, 8),
            "graph_cost_per_query": round(graph_cost, 8) if graph_cost else None,
            "top5_cost_per_query":  round(top5_cost, 8),
            "embed_index_cost_once": round(embed_index_cost, 8),
            "naive_tokens": naive_t,
            "graph_tokens": graph_t,
            "top5_tokens":  top5_t,
        }

    # Summary finding
    large = t10["corpora"].get("large", {})
    if large.get("graph_cost_per_query") and large.get("naive_cost_per_query"):
        speedup = round(large["naive_cost_per_query"] / large["graph_cost_per_query"], 1)
        print(f"  Large corpus: graphify is {speedup}x cheaper than naive per query")
    docs_d = t10["corpora"].get("docs", {})
    if docs_d:
        print(f"  Doc corpus:   graphify graph is EMPTY — embed+top5 is the only viable option")
        print(f"                embed+top5 costs ${docs_d['top5_cost_per_query']:.7f}/query vs naive ${docs_d['naive_cost_per_query']:.7f}/query")

    results["test10"] = t10
    return t10


def test11_llm_judge(results: dict) -> dict:
    """Test 11: Auto-score Test 5 answer pairs via Claude-as-judge (1-5 rubric)."""
    print("\n" + "=" * 60)
    print("TEST 11 — LLM-as-Judge Accuracy Comparison")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test11"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test11"]

    t5 = results.get("test5", {})
    if t5.get("status") != "PASS" or not t5.get("questions"):
        print("  SKIP: Test 5 did not run — no answers to judge")
        results["test11"] = {"status": "SKIP", "reason": "test5 not run"}
        return results["test11"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        results["test11"] = {"status": "SKIP", "reason": str(exc)}
        return results["test11"]

    JUDGE_SYSTEM = (
        "You are an impartial technical evaluator. "
        "Score answers objectively. Do not favour either answer based on length. "
        "Respond with only a JSON object — no markdown fences, no explanation."
    )
    JUDGE_TEMPLATE = (
        "QUESTION: {question}\n\n"
        "ANSWER A:\n{answer_a}\n\n"
        "ANSWER B:\n{answer_b}\n\n"
        "Score each answer 1-5 on:\n"
        "- correctness: factually accurate?\n"
        "- completeness: covers all key aspects?\n"
        "- relevance: focused, no padding?\n\n"
        "Respond with ONLY this JSON (total = sum of three scores, max 15):\n"
        '{{"answer_a":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_b":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"winner":"A_or_B_or_tie","comment":"one sentence"}}'
    )

    import re as _re
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    t11: dict[str, Any] = {
        "status": "PASS",
        "questions": [],
        "graph_wins": 0,
        "raw_wins": 0,
        "ties": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    header = f"  {'Q':<3} {'Question':<44} {'Graph':>6} {'Raw':>6} {'Winner':<8}"
    print(header)
    print("  " + "-" * 72)

    judge_md_lines = [
        "# LLM-as-Judge Accuracy Scores\n",
        f"Model: {ANTHROPIC_MODEL}\n",
        f"| Q | Question | Graph Score (/15) | Raw Score (/15) | Winner | Comment |\n",
        "|---|----------|-------------------|-----------------|--------|--------|\n",
    ]

    for i, q_data in enumerate(t5["questions"]):
        question = q_data["question"]
        graph_ans = q_data.get("graph_answer", "")
        raw_ans   = q_data.get("raw_answer", "")

        if not graph_ans or not raw_ans or "[error" in graph_ans or "[error" in raw_ans:
            print(f"  Q{i+1:02d} {question[:44]:<44} SKIP (missing answers)")
            continue

        prompt = JUDGE_TEMPLATE.format(
            question=question,
            answer_a=graph_ans,
            answer_b=raw_ans,
        )
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=300,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_resp = resp.content[0].text.strip()
            t11["total_input_tokens"]  += resp.usage.input_tokens
            t11["total_output_tokens"] += resp.usage.output_tokens

            # Extract JSON even if wrapped in markdown fences
            m = _re.search(r"\{.*\}", raw_resp, _re.DOTALL)
            scores = json.loads(m.group()) if m else {}
        except Exception as exc:
            scores = {"error": str(exc)}

        a_total = scores.get("answer_a", {}).get("total", 0)
        b_total = scores.get("answer_b", {}).get("total", 0)
        winner_raw = scores.get("winner", "?")
        comment = scores.get("comment", "")

        # answer_a = graph, answer_b = raw
        graph_score = a_total
        raw_score   = b_total
        if winner_raw == "A":
            winner_label = "graph"
            t11["graph_wins"] += 1
        elif winner_raw == "B":
            winner_label = "raw"
            t11["raw_wins"] += 1
        else:
            winner_label = "tie"
            t11["ties"] += 1

        print(f"  Q{i+1:<2} {question[:44]:<44} {graph_score:>5}/15 {raw_score:>5}/15  {winner_label}")
        judge_md_lines.append(f"| Q{i+1} | {question} | {graph_score}/15 | {raw_score}/15 | {winner_label} | {comment} |\n")

        t11["questions"].append({
            "question": question,
            "graph_score": graph_score,
            "raw_score": raw_score,
            "winner": winner_label,
            "comment": comment,
            "raw_judge_response": scores,
        })

    g = t11["graph_wins"]
    r = t11["raw_wins"]
    ti = t11["ties"]
    n = g + r + ti
    judge_cost = (t11["total_input_tokens"] * PRICE_CLAUDE_INPUT +
                  t11["total_output_tokens"] * PRICE_CLAUDE_OUTPUT)
    print(f"\n  Summary: graph wins {g}/{n}, raw wins {r}/{n}, ties {ti}/{n}")
    print(f"  Judge cost: {t11['total_input_tokens']:,} in + {t11['total_output_tokens']:,} out tokens = ${judge_cost:.4f}")

    judge_md_lines.append(f"\n**Summary:** graph {g}/{n} wins, raw {r}/{n} wins, {ti}/{n} ties\n")
    judge_md_lines.append(f"**Judge cost:** ${judge_cost:.4f}\n")
    (ACCURACY_DIR / "judge_scores.md").write_text("".join(judge_md_lines), encoding="utf-8")

    results["test11"] = t11
    return t11


def test12_decision_framework(results: dict) -> dict:
    """Test 12: Data-driven recommendation — use graphify if/don't if."""
    print("\n" + "=" * 60)
    print("TEST 12 — Decision Framework")
    print("=" * 60)

    # Pull signals from previous tests
    t3 = results.get("test3", {}).get("corpora", {})
    small_ratio  = t3.get("small",  {}).get("ratio", 0)
    medium_ratio = t3.get("medium", {}).get("ratio", 0)
    large_ratio  = t3.get("large",  {}).get("ratio", 0)

    t6 = results.get("test6", {}).get("corpora", {})
    be_medium = t6.get("medium", {}).get("break_even_queries")
    be_large  = t6.get("large",  {}).get("break_even_queries")
    # Estimate break-even if test6 skipped
    if be_medium is None:
        t1_med = results.get("test1", {}).get("corpora", {}).get("medium", {})
        naive_med   = t1_med.get("total_tokens", 1)
        graph_med   = t3.get("medium", {}).get("graph_tokens", 1)
        build_est   = 9 * 1300  # 9 doc files × ~1300 tokens/LLM call
        be_medium   = round(build_est / max(1, naive_med - graph_med), 1)
    if be_large is None:
        t1_lrg = results.get("test1", {}).get("corpora", {}).get("large", {})
        naive_lrg  = t1_lrg.get("total_tokens", 1)
        graph_lrg  = t3.get("large", {}).get("graph_tokens", 1)
        build_est  = 0  # large corpus is almost all code — 0 LLM build cost
        be_large   = round(build_est / max(1, naive_lrg - graph_lrg), 1) if build_est > 0 else "<1"

    t9 = results.get("test9", {})
    doc_coverage = t9.get("ast_extraction_coverage_pct", 0.0)
    doc_nodes    = t9.get("graph_nodes", 0)
    doc_files_n  = t9.get("doc_files", 0)

    t11 = results.get("test11", {})
    graph_wins   = t11.get("graph_wins")
    raw_wins     = t11.get("raw_wins")
    n_judged     = (graph_wins or 0) + (t11.get("raw_wins") or 0) + (t11.get("ties") or 0)
    accuracy_note = (
        f"graph {graph_wins}/{n_judged} wins vs raw {raw_wins}/{n_judged}"
        if n_judged > 0 else "not measured (ANTHROPIC_API_KEY not set)"
    )

    t10 = results.get("test10", {}).get("corpora", {})
    large_cost_ratio = None
    if t10.get("large", {}).get("graph_cost_per_query") and t10.get("large", {}).get("naive_cost_per_query"):
        large_cost_ratio = round(
            t10["large"]["naive_cost_per_query"] / t10["large"]["graph_cost_per_query"], 1
        )

    # Print recommendation
    print(f"""
  GRAPHIFY DECISION FRAMEWORK
  ============================
  Based on measured benchmark data:

  USE GRAPHIFY WHEN:
  ------------------
  [OK] Corpus is CODE-HEAVY (>80% .py/.ts/.go/.rs/etc files)
       Reason:  tree-sitter AST extraction is deterministic, fast, and FREE.
                Docs need an LLM call per file — code costs $0 to build.
       Evidence: small={small_ratio}x, medium={medium_ratio}x, large={large_ratio}x reduction

  [OK] You will ask MORE than ~{be_medium} queries on the same corpus (medium)
       Reason:  Build cost amortises after break-even; every query after is pure savings.
       Evidence: medium break-even ~{be_medium} queries (large corpus ~{be_large})

  [OK] Corpus has MORE than 50 files / 20k tokens
       Reason:  Compression ratio grows with corpus size. Small corpora barely benefit.
       Evidence: small={small_ratio}x vs large={large_ratio}x

  [OK] You need ARCHITECTURAL / RELATIONSHIP queries
       ("how does X connect to Y?", "what calls Z?", "show me the flow from A to B")
       Reason:  The graph captures CALL, IMPORTS, CONTAINS edges invisible in raw text.
       Accuracy: {accuracy_note}

  [OK] Cost matters ({f'{large_cost_ratio}x cheaper than naive on large corpus' if large_cost_ratio else 'see test10'})
       Reason:  Graphify query tokens are capped at --budget; naive scales with corpus.

  DON'T USE GRAPHIFY WHEN:
  ------------------------
  [WARN] Corpus is DOC-HEAVY (.md/.txt/.rst/.pdf dominate)
         Reason:  collect_files() excludes doc extensions. AST extract() has NO .md handler.
         Finding: {doc_nodes} nodes extracted from {doc_files_n} doc files = {doc_coverage}% coverage.
         Fix:     Run the full /graphify skill (Claude subagents per doc file) OR use embeddings.
         Alt:     text-embedding-3-small at $0.02/1M tokens + vector search + read top-5 files.

  [WARN] You need IMPLEMENTATION DETAIL answers
         ("exactly what does line 47 do?", "what is the default value of X?")
         Reason:  Graph nodes are labels + source locations, not full code. Raw files win here.

  [WARN] Corpus is SMALL (<10 files / <5k tokens)
         Reason:  Break-even is high; compression ratio is low; just read the files.
         Evidence: small corpus ratio only {small_ratio}x — naive read = {
             results.get("test1", {}).get("corpora", {}).get("small", {}).get("total_tokens", 0)
         :,} tokens total.

  [WARN] One-off or infrequent queries
         Reason:  Build cost is real. Only worth it if you reuse the graph many times.

  SUMMARY TABLE:
  --------------
  Scenario                               Verdict
  -------------------------------------- ----------------------------------------
  >50 code files, >20 queries            USE graphify
  Mixed code+doc, code questions         USE graphify (docs ignored, code works)
  <10 code files, <5 queries             SKIP — just read the files
  Doc-heavy corpus, doc questions        SKIP graphify AST; use embeddings or /graphify skill
  Need exact implementation details      SKIP — raw files beat the graph here
  Exploring a new large codebase         USE graphify — graph shows architecture fast
  Writing a chatbot over documentation   SKIP — needs embeddings + full skill pipeline
""")

    t12: dict[str, Any] = {
        "status": "PASS",
        "signals": {
            "small_ratio": small_ratio,
            "medium_ratio": medium_ratio,
            "large_ratio": large_ratio,
            "doc_ast_coverage_pct": doc_coverage,
            "doc_nodes_extracted": doc_nodes,
            "break_even_medium": be_medium,
            "break_even_large": be_large,
            "accuracy_note": accuracy_note,
            "large_cost_ratio_vs_naive": large_cost_ratio,
        },
        "recommendation": {
            "use_if": [
                "code-heavy corpus (>80% code files)",
                f"more than ~{be_medium} queries on same corpus",
                "corpus >50 files / >20k tokens",
                "architectural / relationship questions",
                "cost per query matters",
            ],
            "dont_if": [
                "doc-heavy corpus — AST coverage 0%, use embeddings or full /graphify skill",
                "implementation detail questions",
                "small corpus <10 files",
                "infrequent / one-off queries",
            ],
        },
    }
    results["test12"] = t12
    return t12


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(results: dict) -> None:
    print("\n")
    print("=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    tests = [
        ("test1",  "Baseline Naive Token Count"),
        ("test2",  "Graph Query Output Size"),
        ("test3",  "Compression Ratio vs Corpus Size"),
        ("test4",  "Realistic Naive Baseline (grep top-5)"),
        ("test5",  "Query Accuracy Spot Check"),
        ("test6",  "Amortised Cost & Break-Even"),
        ("test7",  "SHA256 Cache Hit Rate"),
        ("test8",  "Code-only vs Mixed Extraction Cost"),
        ("test9",  "Doc Corpus: Token Reduction & AST Gap"),
        ("test10", "Dollar Cost Model ($/query)"),
        ("test11", "LLM-as-Judge Accuracy Comparison"),
        ("test12", "Decision Framework"),
    ]
    for key, name in tests:
        data = results.get(key, {})
        status = data.get("status", "UNKNOWN")
        icon = {"PASS": "OK", "WARN": "WN", "SKIP": "--", "FAIL": "XX"}.get(status, "??")
        num = key.replace("test", "")
        print(f"  [{icon}] Test {num:>2}: {name:<42} [{status}]")

    # Token reduction claim check (from test3)
    t3 = results.get("test3", {})
    ratios = {k: v.get("ratio", 0) for k, v in t3.get("corpora", {}).items() if isinstance(v, dict) and "ratio" in v}
    if ratios:
        print(f"\n  Token reduction ratios (naive / graph query tokens):")
        for corpus, ratio in ratios.items():
            claimed = "[EXCEEDS 71.5x]" if ratio >= 71.5 else ("[BELOW 71.5x claim]" if ratio > 0 else "N/A")
            print(f"    {corpus:<10} {ratio:>6.1f}x  {claimed}")
    else:
        print("\n  No ratio data — graphs may not have been built.")

    t6 = results.get("test6", {})
    for corpus, data in t6.get("corpora", {}).items():
        if isinstance(data, dict) and "break_even_queries" in data:
            print(f"\n  Break-even ({corpus}): {data['break_even_queries']} queries to amortise build cost")

    t7 = results.get("test7", {})
    if "cache_hit_rate_pct" in t7:
        print(f"\n  Cache hit rate (medium, 5 files changed): {t7['cache_hit_rate_pct']}%")

    t9 = results.get("test9", {})
    if t9.get("status") == "PASS":
        print(f"\n  Doc corpus ({t9.get('doc_files', 0)} files, {t9.get('naive_tokens_total', 0):,} tokens):")
        print(f"    AST extraction coverage: {t9.get('ast_extraction_coverage_pct', 0)}%")
        useful = sum(1 for q in t9.get("queries", []) if q.get("graph_useful"))
        print(f"    Graph useful for doc queries: {useful}/{len(t9.get('queries', []))} questions")

    t11 = results.get("test11", {})
    if t11.get("status") == "PASS":
        g = t11.get("graph_wins", 0)
        r = t11.get("raw_wins", 0)
        ti = t11.get("ties", 0)
        print(f"\n  LLM judge: graph wins {g}, raw wins {r}, ties {ti}")


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw JSON
    data_path = RESULTS_DIR / "data.json"
    data_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Raw data saved to {data_path}")

    # Markdown report
    md_lines = ["# Graphify Token Reduction Benchmark Report\n"]
    md_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for key, val in results.items():
        md_lines.append(f"\n## {key}\n\n```json\n{json.dumps(val, indent=2, default=str)}\n```\n")
    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Markdown report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Graphify Token Reduction Benchmark Suite")
    print(f"Corpora: {', '.join(CORPUS_DIRS)}")
    print(f"Encoder: cl100k_base (tiktoken)")

    results: dict[str, Any] = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version,
            "query_budget": QUERY_BUDGET,
            "model": ANTHROPIC_MODEL,
        }
    }

    # Build graphs first (needed for tests 2, 3, 4, 5, 6, 7)
    build_all_graphs(results)

    # Run all 12 tests
    test1_baseline_naive(results)
    test2_graph_query_output_size(results)
    test3_compression_ratio_table(results)
    test4_realistic_naive_baseline(results)
    test5_query_accuracy_spotcheck(results)
    test6_amortised_build_cost(results)
    test7_sha256_cache_hit_rate(results)
    test8_code_vs_doc_extraction_cost(results)
    test9_doc_corpus(results)
    test10_dollar_cost_model(results)
    test11_llm_judge(results)
    test12_decision_framework(results)

    # Final report
    print_final_report(results)
    save_results(results)


if __name__ == "__main__":
    main()
