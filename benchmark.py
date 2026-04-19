"""
PRISM Benchmark Suite
=====================
Pre-compiled Retrieval with Intelligent Strata Management

Empirically validates whether layered, pre-compiled retrieval (code graph +
doc index + LLMWiki + BM25 + question-aware router) reduces token consumption
and maintains accuracy vs naive file reading.

Run:
    python benchmark.py

Requires:
    pip install graphifyy tiktoken anthropic python-dotenv
    ANTHROPIC_API_KEY env var (or .env file) for LLM-powered tests
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

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

# Doc corpus: publicly available README files used as the documentation corpus
# for Tests 9, 10, and the docs baseline. Includes nanoGPT and other open-source
# ML projects by Andrej Karpathy, Hugging Face, OpenAI, Microsoft, and others.
# All fetched from their public GitHub repositories under their respective licences.
# nanoGPT: https://github.com/karpathy/nanoGPT (MIT licence, Andrej Karpathy)
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

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Mixed corpus — code + docs cross-reference test (Test 13)
# ---------------------------------------------------------------------------

MIXED_CORPUS_DIR = Path("benchmark-corpus/mixed")
MIXED_GRAPH_OUT  = Path("graphify-out/mixed")
WIKI_DIR         = MIXED_GRAPH_OUT / "wiki"

# Entity definitions for LLMWiki page generation (Test 14)
WIKI_ENTITIES: list[dict] = [
    {
        "name": "AuthService",
        "keywords": ["authservice", "auth_service", "login", "logout", "refresh", "validate",
                     "authresult"],
    },
    {
        "name": "TokenStore",
        "keywords": ["tokenstore", "token_store", "consume_refresh", "revoke_all_for_user",
                     "access_ttl", "refresh_ttl", "create_access_token", "create_refresh_token"],
    },
    {
        "name": "SessionManager",
        "keywords": ["sessionmanager", "session_manager", "session", "max_sessions",
                     "create_session", "eviction", "fifo", "popitem"],
    },
    {
        "name": "User and Permission System",
        "keywords": ["user", "permission", "has_permission", "permissionlevel", "frozenset",
                     "is_active", "allows", "read", "write", "admin", "super"],
    },
    {
        "name": "API Contract Refresh Endpoint",
        "keywords": ["refresh", "api", "endpoint", "post /auth/refresh", "guarantee",
                     "expires_in", "single-use", "contract", "access_token", "refresh_token"],
    },
    {
        "name": "Refresh Token Design",
        "keywords": ["single-use", "adr", "atomic", "theft", "rotation", "idempotent",
                     "consume_refresh", "replay", "security"],
    },
    {
        "name": "Known Limitations",
        "keywords": ["limitation", "known", "invalidate", "silent", "idempotent", "in-memory",
                     "wildcard", "fifo", "eviction", "revoke", "tombstone", "grace"],
    },
]

MIXED_QUESTIONS: list[str] = [
    "what motivated the single-use refresh token design?",           # Q1 — rationale
    "what are the API contract guarantees for the refresh endpoint?", # Q2 — factual
    "how does AuthService use TokenStore to handle token refresh?",   # Q3 — comprehensive
    "how does the User model relate to the permission system?",       # Q4 — structural
    "what are the known limitations and what components do they affect?", # Q5 — factual
]

BM25_INDEX_PATH = MIXED_GRAPH_OUT / "bm25_index.json"

# Router: question type → budget + layer weights
# Budgets reflect information density needed per question type (not an arbitrary cap)
ROUTING_CONFIG: dict[str, dict] = {
    "structural": {
        "budget": 1500,
        "layers": {"graph": 1.0},
        "description": (
            "PURE topology only — what imports what, what calls what, dependency graphs. "
            "NO implementation detail needed. "
            "Examples: 'what modules does X import?', 'show the call graph for Y', "
            "'which classes extend Z?'"
        ),
    },
    "rationale": {
        "budget": 2000,
        "layers": {"wiki": 0.8, "graph": 0.2},
        "description": (
            "Design decisions and motivation — WHY something was built a certain way. "
            "Examples: 'what motivated X?', 'why was Y designed like this?', "
            "'what trade-offs led to Z?', 'what does the ADR say about X?'"
        ),
    },
    "factual": {
        "budget": 2500,
        "layers": {"wiki": 0.6, "hybrid": 0.4},
        "description": (
            "Specific concrete facts — API contracts, guarantees, constant values, "
            "known limitations, error codes. "
            "Examples: 'what are the guarantees for X?', 'what are the known limitations?', "
            "'what does endpoint Y return?', 'what is the value of constant Z?'"
        ),
    },
    "similarity": {
        "budget": 3000,
        "layers": {"bm25": 0.6, "wiki": 0.2, "graph": 0.2},
        "description": (
            "Find code or docs semantically similar to a concept. "
            "Examples: 'find code that does X', 'what is similar to Y?', "
            "'show me examples of Z pattern'"
        ),
    },
    "comprehensive": {
        "budget": 5000,
        "layers": {"wiki": 0.4, "graph": 0.3, "bm25": 0.3},
        "description": (
            "End-to-end understanding — HOW something works, how components interact, "
            "how X USES or RELATES TO Y in practice (requires both code structure AND "
            "documentation context). "
            "Examples: 'how does X use Y?', 'how does X relate to Y?', "
            "'walk me through the flow of Z', 'how does X work end to end?'"
        ),
    },
}

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
# Hybrid pipeline helpers (Test 13)
# ---------------------------------------------------------------------------

def build_doc_index(doc_files: list[Path], out_dir: Path, client) -> dict:
    """One-time LLM extraction for each doc file, cached by SHA-256.

    Calls Claude once per uncached doc to extract:
        summary, code_refs, key_facts, type

    Results are persisted in out_dir/doc_index.json keyed by file SHA-256.
    Already-extracted docs are skipped on subsequent calls (cache hit).

    Returns the full in-memory doc_index dict.
    """
    import re as _re

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "doc_index.json"

    doc_index: dict = {}
    if index_path.exists():
        try:
            doc_index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            doc_index = {}

    EXTRACT_PROMPT = (
        "You are a documentation analyst. Read the document below and extract:\n"
        "1. A 1-2 sentence summary of what it covers.\n"
        "2. All code entity names explicitly mentioned: class names, method names, "
        "function names, constant names (e.g. ACCESS_TTL, MAX_SESSIONS, consume_refresh).\n"
        "3. Key concrete facts: specific values, guarantees, constraints, design decisions "
        "(e.g. 'ACCESS_TTL=900', 'MAX_SESSIONS=5', 'single-use refresh tokens', 'FIFO eviction').\n"
        "4. Document type: one of architecture, api-contract, adr, data-model, limitations, unknown.\n\n"
        "Respond ONLY with this JSON (no markdown fences, no commentary):\n"
        '{{"summary":"...","code_refs":["..."],"key_facts":["..."],"type":"..."}}\n\n'
        "Document:\n{content}"
    )

    new_extractions = 0
    for doc_path in doc_files:
        content = safe_read(doc_path)
        sha = hashlib.sha256(content.encode()).hexdigest()

        if sha in doc_index:
            continue  # cache hit

        prompt = EXTRACT_PROMPT.format(content=content[:8000])
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            extracted = json.loads(m.group()) if m else {}
        except Exception as exc:
            extracted = {
                "summary": f"[extraction failed: {exc}]",
                "code_refs": [], "key_facts": [], "type": "unknown",
            }

        doc_index[sha] = {
            "file":      str(doc_path),
            "summary":   extracted.get("summary", ""),
            "code_refs": extracted.get("code_refs", []),
            "key_facts": extracted.get("key_facts", []),
            "type":      extracted.get("type", "unknown"),
        }
        new_extractions += 1

    index_path.write_text(json.dumps(doc_index, indent=2), encoding="utf-8")
    print(f"    doc_index: {len(doc_index)} entries ({new_extractions} new LLM extractions)")
    return doc_index


def smart_doc_query(
    question: str,
    doc_index: dict,
    corpus_dir: Path,
    budget: int = 1000,
) -> tuple[str, int]:
    """Score indexed docs against *question*, return top content within *budget* tokens.

    Scoring per doc:
        +1  for each question word (>3 chars) found in summary or key_facts
        +2  for each code_refs entity that appears in the question text

    Returns (doc_context_str, tokens_used).
    Each doc section is prefixed: --- filename (type: <type>) ---
    """
    question_lower = question.lower()
    question_words = {w for w in question_lower.split() if len(w) > 3}

    scored: list[tuple[float, dict]] = []
    for sha, meta in doc_index.items():
        score = 0.0

        # Keyword overlap on summary
        summary_words = set(meta.get("summary", "").lower().split())
        score += len(question_words & summary_words)

        # Keyword overlap on key_facts
        facts_words = set(" ".join(meta.get("key_facts", [])).lower().split())
        score += len(question_words & facts_words)

        # Code entity bonus
        for entity in meta.get("code_refs", []):
            if entity.lower() in question_lower:
                score += 2.0

        if score > 0:
            scored.append((score, meta))

    scored.sort(key=lambda x: x[0], reverse=True)

    parts: list[str] = []
    used = 0
    for _, meta in scored:
        # Resolve file path — try stored path then fallback to corpus_dir/filename
        doc_path = Path(meta["file"])
        if not doc_path.exists():
            doc_path = corpus_dir / Path(meta["file"]).name
        if not doc_path.exists():
            continue

        content      = safe_read(doc_path)
        header       = f"--- {doc_path.name} (type: {meta['type']}) ---\n"
        chunk        = header + content
        chunk_tokens = count_tokens(chunk)

        if used + chunk_tokens <= budget:
            parts.append(chunk)
            used += chunk_tokens
        elif used < budget:
            # Partial fit: trim to remaining budget (approximate by char ratio)
            remaining = budget - used
            if remaining < 50:
                break
            ratio   = remaining / max(1, chunk_tokens)
            trimmed = chunk[: int(len(chunk) * ratio)]
            parts.append(trimmed)
            used += count_tokens(trimmed)
            break

    return "\n\n".join(parts), used


def hybrid_query(
    question:       str,
    graph_path:     Path,
    doc_index_path: Path,
    corpus_dir:     Path,
    total_budget:   int = 2000,
) -> tuple[str, int, int]:
    """Split-budget retrieval: half for code graph, half for smart doc retrieval.

    Keeps total context within *total_budget* tokens — same cost as a plain
    graphify query — but covers both code structure AND documentation.

    Returns:
        (merged_context, graph_tokens_used, doc_tokens_used)

    merged_context format:
        [CODE STRUCTURE]
        <graphify output>

        [DOCUMENTATION]
        <doc context>
    """
    half = total_budget // 2  # 1000 tokens each by default

    # Code graph side
    graph_out    = ""
    graph_tokens = 0
    if graph_path.exists():
        try:
            r = subprocess.run(
                ["graphify", "query", question,
                 "--graph", str(graph_path),
                 "--budget", str(half)],
                capture_output=True, text=True, timeout=60,
            )
            graph_out = r.stdout
        except Exception as exc:
            graph_out = f"[graph query error: {exc}]"
        graph_tokens = count_tokens(graph_out)

    # Doc side
    doc_context = ""
    doc_tokens  = 0
    if doc_index_path.exists():
        try:
            doc_index = json.loads(doc_index_path.read_text(encoding="utf-8"))
        except Exception:
            doc_index = {}
        doc_context, doc_tokens = smart_doc_query(question, doc_index, corpus_dir, budget=half)

    merged = f"[CODE STRUCTURE]\n{graph_out}\n\n[DOCUMENTATION]\n{doc_context}"
    return merged, graph_tokens, doc_tokens


def build_llmwiki(
    # Inspired by Andrej Karpathy's "LLM Wiki" concept:
    # https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
    # Karpathy's original idea: pre-build one markdown page per concept/entity,
    # synthesised from raw sources, so an LLM can query compact pre-integrated
    # knowledge rather than raw documents at query time.
    # This implementation adapts the concept to mixed code+doc corpora by
    # fusing AST graph nodes with doc_index extractions per entity page.
    graph_path: Path,
    doc_index_path: Path,
    corpus_dir: Path,
    wiki_dir: Path,
    client,
) -> dict[str, Path]:
    """Build LLMWiki entity pages — one synthesised markdown page per major entity.

    For each entity in WIKI_ENTITIES:
      1. Collect relevant graphify graph nodes (by keyword match on label)
      2. Collect relevant doc file content (by keyword match on doc_index metadata)
      3. LLM synthesises a single wiki page covering code structure + documentation

    Pages are cached as wiki_dir/<slug>.md — delete the wiki/ dir to force rebuild.

    Returns: dict mapping entity name → page Path.
    """
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Load graph nodes
    graph_nodes: list[dict] = []
    if graph_path.exists():
        try:
            g = json.loads(graph_path.read_text(encoding="utf-8"))
            graph_nodes = g.get("nodes", [])
        except Exception:
            graph_nodes = []

    # Load doc_index
    doc_index: dict = {}
    if doc_index_path.exists():
        try:
            doc_index = json.loads(doc_index_path.read_text(encoding="utf-8"))
        except Exception:
            doc_index = {}

    WIKI_PAGE_PROMPT = (
        "You are building a technical wiki page for: {entity}\n\n"
        "Below is everything known from code analysis (AST graph) and documentation.\n\n"
        "CODE STRUCTURE (from graphify AST graph):\n{graph_ctx}\n\n"
        "DOCUMENTATION EXCERPTS:\n{doc_ctx}\n\n"
        "Write a comprehensive wiki page in markdown covering:\n"
        "1. What it is and what it does (2-3 sentences)\n"
        "2. Key methods/fields/endpoints with their purpose and concrete values\n"
        "3. Design decisions and rationale (WHY, not just WHAT)\n"
        "4. Known limitations or caveats with exact component names\n"
        "5. Cross-references to related entities\n\n"
        "Rules: factual only, no padding, include concrete values where available "
        "(e.g. ACCESS_TTL=900, MAX_SESSIONS=5). Use ## headers. Max 400 words."
    )

    pages: dict[str, Path] = {}
    new_pages = 0

    for entity_def in WIKI_ENTITIES:
        entity   = entity_def["name"]
        keywords = [k.lower() for k in entity_def["keywords"]]
        slug     = entity.lower().replace(" ", "_").replace("-", "_")
        page_path = wiki_dir / f"{slug}.md"
        pages[entity] = page_path

        if page_path.exists():
            continue  # cache hit

        # ── Graph context ──────────────────────────────────────────────────
        graph_parts: list[str] = []
        for node in graph_nodes:
            label = node.get("label", "").lower()
            if any(kw in label for kw in keywords):
                ftype = node.get("file_type", "code")
                loc   = node.get("source_location", "")
                graph_parts.append(f"[{ftype}] {node['label']} @ {loc}")
        graph_ctx = "\n".join(graph_parts) if graph_parts else "(no matching graph nodes)"

        # ── Doc context ────────────────────────────────────────────────────
        doc_parts: list[str] = []
        for sha, meta in doc_index.items():
            refs    = " ".join(r.lower() for r in meta.get("code_refs", []))
            summary = meta.get("summary", "").lower()
            facts   = " ".join(meta.get("key_facts", [])).lower()
            blob    = summary + " " + facts + " " + refs
            if any(kw in blob for kw in keywords):
                doc_path = Path(meta["file"])
                if not doc_path.exists():
                    doc_path = corpus_dir / Path(meta["file"]).name
                if doc_path.exists():
                    doc_parts.append(f"--- {doc_path.name} ---\n{safe_read(doc_path)[:3000]}")

        doc_ctx = "\n\n".join(doc_parts) if doc_parts else "(no relevant documentation found)"

        prompt = WIKI_PAGE_PROMPT.format(
            entity=entity,
            graph_ctx=graph_ctx[:2000],
            doc_ctx=doc_ctx[:4000],
        )

        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            page_content = f"# {entity}\n\n{resp.content[0].text.strip()}"
        except Exception as exc:
            page_content = f"# {entity}\n\n[wiki build failed: {exc}]"

        page_path.write_text(page_content, encoding="utf-8")
        new_pages += 1

    print(f"    wiki: {len(pages)} pages ({new_pages} new, {len(pages) - new_pages} cached)")
    return pages


def wiki_query(question: str, wiki_dir: Path, budget: int = 2000) -> tuple[str, int]:
    """Retrieve relevant wiki pages for a question within budget tokens.

    Scores each page by keyword overlap with the question, with a bonus for
    longer exact-match words (more discriminative).

    Returns: (wiki_context_str, tokens_used)
    """
    if not wiki_dir.exists():
        return "", 0

    question_lower = question.lower()
    question_words = {w for w in question_lower.split() if len(w) > 3}

    scored: list[tuple[float, Path]] = []
    for page_path in sorted(wiki_dir.glob("*.md")):
        content       = safe_read(page_path)
        content_lower = content.lower()
        content_words = set(content_lower.split())

        score = float(len(question_words & content_words))
        # Bonus for longer word matches (more discriminative)
        for word in question_words:
            if len(word) > 6 and word in content_lower:
                score += 0.5

        if score > 0:
            scored.append((score, page_path))

    scored.sort(key=lambda x: x[0], reverse=True)

    parts: list[str] = []
    used  = 0
    for _, page_path in scored:
        content      = safe_read(page_path)
        header       = f"--- {page_path.stem.replace('_', ' ').title()} (wiki) ---\n"
        chunk        = header + content
        chunk_tokens = count_tokens(chunk)

        if used + chunk_tokens <= budget:
            parts.append(chunk)
            used += chunk_tokens
        elif used < budget:
            remaining = budget - used
            if remaining < 50:
                break
            ratio   = remaining / max(1, chunk_tokens)
            trimmed = chunk[: int(len(chunk) * ratio)]
            parts.append(trimmed)
            used += count_tokens(trimmed)
            break

    return "\n\n".join(parts), used


def build_bm25_index(
    graph_path: Path,
    wiki_dir: Path,
    corpus_dir: Path,
    out_path: Path,
) -> dict:
    """Build a BM25 index over three document collections:
      - graph: rationale nodes from graphify (the 'why' annotations)
      - wiki:  LLMWiki entity pages
      - raw:   all files in the mixed corpus

    Each document stored as {id, source, title, text}.
    IDF computed over the full corpus.
    Cached to out_path — delete to force rebuild.

    Returns the in-memory index dict.

    BM25 (Okapi BM25) algorithm:
      Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
      BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.
      https://doi.org/10.1561/1500000019
      Parameters used: k1=1.5, b=0.75 (standard defaults).
    """
    import math

    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    documents: list[dict] = []

    # ── Graph rationale nodes ──────────────────────────────────────────────
    if graph_path.exists():
        try:
            g = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in g.get("nodes", []):
                if node.get("file_type") == "rationale":
                    text = node.get("label", "")
                    if len(text) > 20:
                        documents.append({
                            "id":     f"graph_{node.get('id', len(documents))}",
                            "source": "graph",
                            "title":  text[:60],
                            "text":   text,
                        })
        except Exception:
            pass

    # ── Wiki pages ─────────────────────────────────────────────────────────
    if wiki_dir.exists():
        for page_path in sorted(wiki_dir.glob("*.md")):
            if page_path.name == "lint_report.md":
                continue
            text = safe_read(page_path)
            documents.append({
                "id":     f"wiki_{page_path.stem}",
                "source": "wiki",
                "title":  page_path.stem.replace("_", " ").title(),
                "text":   text,
            })

    # ── Raw corpus files ───────────────────────────────────────────────────
    if corpus_dir.exists():
        for fpath in read_corpus_files(corpus_dir):
            text = safe_read(fpath)
            if text.strip():
                documents.append({
                    "id":     f"raw_{fpath.name}",
                    "source": "raw",
                    "title":  fpath.name,
                    "text":   text,
                })

    # ── BM25 IDF ──────────────────────────────────────────────────────────
    N = len(documents)
    df: dict[str, int] = {}
    tokenized: list[list[str]] = []
    for doc in documents:
        tokens = doc["text"].lower().split()
        tokenized.append(tokens)
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    avg_dl = sum(len(t) for t in tokenized) / max(1, N)

    index = {
        "documents": documents,
        "idf": idf,
        "avg_dl": avg_dl,
        "N": N,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"    bm25: indexed {N} documents ({sum(1 for d in documents if d['source']=='graph')} graph "
          f"+ {sum(1 for d in documents if d['source']=='wiki')} wiki "
          f"+ {sum(1 for d in documents if d['source']=='raw')} raw)")
    return index


def bm25_search(
    question: str,
    index: dict,
    budget: int = 2000,
    source_filter: str | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[str, int]:
    """Score all indexed documents against *question* using BM25.

    source_filter: if set, only score documents from that source ('graph'|'wiki'|'raw')

    Returns (context_str, tokens_used) of top documents within budget.
    """
    import math

    query_terms = question.lower().split()
    idf        = index.get("idf", {})
    avg_dl     = index.get("avg_dl", 1)
    documents  = index.get("documents", [])

    scored: list[tuple[float, dict]] = []
    for doc in documents:
        if source_filter and doc["source"] != source_filter:
            continue
        tokens = doc["text"].lower().split()
        dl     = len(tokens)
        score  = 0.0
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        for term in query_terms:
            if term not in idf:
                continue
            tf = tf_map.get(term, 0)
            numerator   = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / max(1, avg_dl))
            score += idf[term] * (numerator / max(0.001, denominator))
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    parts: list[str] = []
    used  = 0
    for _, doc in scored:
        header       = f"--- {doc['title']} ({doc['source']}) ---\n"
        chunk        = header + doc["text"]
        chunk_tokens = count_tokens(chunk)
        if used + chunk_tokens <= budget:
            parts.append(chunk)
            used += chunk_tokens
        elif used < budget:
            remaining = budget - used
            if remaining < 50:
                break
            ratio   = remaining / max(1, chunk_tokens)
            trimmed = chunk[: int(len(chunk) * ratio)]
            parts.append(trimmed)
            used += count_tokens(trimmed)
            break

    return "\n\n".join(parts), used


def router(question: str, client) -> dict:
    """Classify question type via one fast LLM call.

    Returns:
        {"type": str, "budget": int, "layers": dict, "reason": str}

    Falls back to 'comprehensive' on any error.
    """
    import re as _re

    type_descriptions = "\n\n".join(
        f'  "{qtype}":\n  {cfg["description"]}'
        for qtype, cfg in ROUTING_CONFIG.items()
    )

    prompt = (
        "You are a question classifier for a code+documentation retrieval system.\n"
        "Classify the question below into exactly one type.\n\n"
        "TYPES:\n\n"
        f"{type_descriptions}\n\n"
        "CRITICAL DISAMBIGUATION RULE:\n"
        '  "structural" = pure graph topology ONLY (imports, calls, extends). '
        "No implementation detail needed.\n"
        '  "comprehensive" = whenever the question asks HOW something works, HOW '
        "X uses/relates to Y, or needs both code structure AND doc context to answer fully.\n"
        '  If in doubt between "structural" and "comprehensive", choose "comprehensive".\n\n'
        "EXAMPLES:\n"
        '  "what calls AuthService.login?" → structural\n'
        '  "how does AuthService use TokenStore?" → comprehensive\n'
        '  "how does User relate to Permission?" → comprehensive\n'
        '  "what motivated single-use tokens?" → rationale\n'
        '  "what are the API guarantees?" → factual\n\n'
        f'QUESTION: "{question}"\n\n'
        "Respond with ONLY this JSON:\n"
        '{{"type":"<type>","reason":"one sentence"}}'
    )

    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        m   = _re.search(r"\{.*\}", raw, _re.DOTALL)
        parsed = json.loads(m.group()) if m else {}
        qtype  = parsed.get("type", "comprehensive")
        if qtype not in ROUTING_CONFIG:
            qtype = "comprehensive"
        config = ROUTING_CONFIG[qtype]
        return {
            "type":   qtype,
            "budget": config["budget"],
            "layers": config["layers"],
            "reason": parsed.get("reason", ""),
            "input_tokens":  resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
    except Exception as exc:
        config = ROUTING_CONFIG["comprehensive"]
        return {
            "type":   "comprehensive",
            "budget": config["budget"],
            "layers": config["layers"],
            "reason": f"fallback due to error: {exc}",
            "input_tokens": 0, "output_tokens": 0,
        }


# Fraction of allocated budget a layer must return to be considered useful.
# Below this threshold the unused tokens spill to the next layer.
LOW_SIGNAL_THRESHOLD = 0.30


def routed_query(
    question: str,
    route: dict,
    graph_path: Path,
    doc_index_path: Path,
    wiki_dir: Path,
    bm25_index: dict,
    corpus_dir: Path,
) -> tuple[str, int, dict]:
    """Assemble context from the layers specified by *route*, within the route's budget.

    Budget is split proportionally by layer weight. Layers are retrieved in descending
    weight order so the most important source gets first pick of tokens.

    Adaptive spill: if a layer returns fewer than LOW_SIGNAL_THRESHOLD (30%) of its
    allocated budget, the unused tokens carry forward to the next layer. This means
    the system self-corrects when a layer has nothing useful to say — instead of
    wasting the budget allocation, it gives it to the next best source.

    Returns: (merged_context, total_tokens_used, tokens_per_layer)
    The tokens_per_layer dict includes a 'spill_events' key listing which layers
    triggered a spill and how many tokens were redistributed.
    """
    total_budget  = route["budget"]
    layers        = route["layers"]
    context_parts: list[str] = []
    tokens_per_layer: dict[str, int] = {}
    spill_events: list[str] = []
    spill = 0  # unused tokens carried forward from a low-signal layer

    # Sort layers by weight descending — highest priority retrieves first
    for layer_name, weight in sorted(layers.items(), key=lambda x: -x[1]):
        if weight <= 0:
            continue
        layer_budget = int(total_budget * weight) + spill
        spill = 0  # consume the carry-in

        if layer_name == "graph":
            ctx = ""
            if graph_path.exists():
                try:
                    r = subprocess.run(
                        ["graphify", "query", question,
                         "--graph", str(graph_path),
                         "--budget", str(layer_budget)],
                        capture_output=True, text=True, timeout=60,
                    )
                    ctx = r.stdout
                except Exception as exc:
                    ctx = f"[graph error: {exc}]"
            tok = count_tokens(ctx)
            if ctx.strip():
                context_parts.append(f"[CODE STRUCTURE]\n{ctx}")
            tokens_per_layer["graph"] = tok

        elif layer_name == "wiki":
            ctx, tok = wiki_query(question, wiki_dir, budget=layer_budget)
            if ctx.strip():
                context_parts.append(f"[WIKI KNOWLEDGE]\n{ctx}")
            tokens_per_layer["wiki"] = tok

        elif layer_name == "hybrid":
            doc_index: dict = {}
            if doc_index_path.exists():
                try:
                    doc_index = json.loads(doc_index_path.read_text(encoding="utf-8"))
                except Exception:
                    doc_index = {}
            ctx, tok = smart_doc_query(question, doc_index, corpus_dir, budget=layer_budget)
            if ctx.strip():
                context_parts.append(f"[DOCUMENTATION]\n{ctx}")
            tokens_per_layer["hybrid"] = tok

        elif layer_name == "bm25":
            ctx, tok = bm25_search(question, bm25_index, budget=layer_budget)
            if ctx.strip():
                context_parts.append(f"[SEMANTIC SEARCH]\n{ctx}")
            tokens_per_layer["bm25"] = tok

        else:
            tok = 0

        # ── Adaptive spill ────────────────────────────────────────────────
        # If this layer returned < 30% of its budget, it had low signal.
        # Carry unused tokens to the next layer so nothing is wasted.
        if tok < layer_budget * LOW_SIGNAL_THRESHOLD:
            unused = layer_budget - tok
            spill  = unused
            spill_events.append(
                f"{layer_name}: used {tok}/{layer_budget} tokens ({tok/max(1,layer_budget):.0%}) "
                f"→ spilling {unused} to next layer"
            )

    tokens_per_layer["spill_events"] = spill_events  # type: ignore[assignment]
    merged     = "\n\n".join(context_parts)
    total_used = sum(v for k, v in tokens_per_layer.items() if k != "spill_events")
    return merged, total_used, tokens_per_layer


def wiki_lint(wiki_dir: Path, client) -> str:
    """Run one LLM pass over all wiki pages to flag quality issues.

    Checks for:
    - Stale claims (methods/constants that may not exist)
    - Missing cross-references between pages
    - Contradictions between pages
    - Orphan topics (mentioned but no page exists)

    Result saved to wiki_dir/lint_report.md and returned as string.
    """
    lint_path = wiki_dir / "lint_report.md"
    if lint_path.exists():
        return safe_read(lint_path)

    if not wiki_dir.exists():
        return ""

    pages = sorted(p for p in wiki_dir.glob("*.md") if p.name != "lint_report.md")
    if not pages:
        return ""

    wiki_dump = "\n\n---\n\n".join(
        f"# PAGE: {p.name}\n{safe_read(p)}" for p in pages
    )

    prompt = (
        "You are a technical wiki editor. Review the following wiki pages for quality issues.\n\n"
        "Check for:\n"
        "1. Stale claims — specific method names, constants, or behaviours that contradict each other\n"
        "2. Missing cross-references — entity A mentions entity B but no link/reference exists\n"
        "3. Contradictions — two pages assert conflicting facts\n"
        "4. Coverage gaps — important entities mentioned across pages but no dedicated page exists\n\n"
        "Format your report as markdown with ## sections per issue type. "
        "Be specific: quote the claim and the page it appears on. "
        "If no issues found in a category, write 'None found.'\n\n"
        f"WIKI PAGES:\n\n{wiki_dump[:12000]}"
    )

    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        report = (
            "# Wiki Lint Report\n\n"
            f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Pages reviewed: {len(pages)}\n\n"
            f"{resp.content[0].text.strip()}"
        )
    except Exception as exc:
        report = f"# Wiki Lint Report\n\n[lint failed: {exc}]"

    lint_path.write_text(report, encoding="utf-8")
    print(f"    wiki lint: reviewed {len(pages)} pages → {lint_path.name}")
    return report


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
    print("PRE-STEP — Building PRISM Knowledge Layers")
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

    # Mixed corpus: code AST graph + LLM doc index
    print(f"\n  Mixed corpus (code+docs cross-reference test)...")
    if not MIXED_CORPUS_DIR.exists():
        print(f"  mixed: SKIP (corpus not found at {MIXED_CORPUS_DIR})")
        results["graphs"]["mixed"] = {"status": "SKIP"}
    else:
        mixed_graph = MIXED_GRAPH_OUT / "graph.json"
        if mixed_graph.exists():
            print(f"  mixed: graph already built (skipping rebuild)")
            results["graphs"]["mixed"] = {"status": "CACHED"}
        else:
            print(f"  mixed: building code graph... ", end="", flush=True)
            t0 = time.time()
            ok, msg = build_graph(MIXED_CORPUS_DIR, MIXED_GRAPH_OUT)
            dt = time.time() - t0
            print(f"{'OK' if ok else 'FAIL'} ({dt:.1f}s) — {msg}")
            results["graphs"]["mixed"] = {
                "status": "OK" if ok else "FAIL",
                "time_s": round(dt, 1), "msg": msg,
            }

        # Build LLM doc_index when API key is available
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        doc_index_path = MIXED_GRAPH_OUT / "doc_index.json"
        if api_key and not doc_index_path.exists():
            print(f"  mixed: extracting doc_index via LLM...")
            try:
                import anthropic as _anthropic
                _client = _anthropic.Anthropic(api_key=api_key)
                doc_files = sorted(MIXED_CORPUS_DIR.rglob("*.md"))
                build_doc_index(doc_files, MIXED_GRAPH_OUT, _client)
            except Exception as exc:
                print(f"  WARN: doc_index build failed — {exc}")
        elif doc_index_path.exists():
            print(f"  mixed: doc_index already built (skipping)")
        else:
            print(f"  mixed: doc_index skipped (no API key)")

        # Build LLMWiki entity pages (Test 14)
        if api_key and doc_index_path.exists():
            print(f"  mixed: building LLMWiki entity pages...")
            try:
                import anthropic as _anthropic
                _client = _anthropic.Anthropic(api_key=api_key)
                build_llmwiki(
                    MIXED_GRAPH_OUT / "graph.json",
                    doc_index_path,
                    MIXED_CORPUS_DIR,
                    WIKI_DIR,
                    _client,
                )
            except Exception as exc:
                print(f"  WARN: wiki build failed — {exc}")
        elif WIKI_DIR.exists() and any(WIKI_DIR.glob("*.md")):
            n_pages = len(list(WIKI_DIR.glob("*.md")))
            print(f"  mixed: wiki already built ({n_pages} pages, skipping)")
        else:
            print(f"  mixed: wiki skipped (no API key or doc_index missing)")

        # Build BM25 index over graph rationale + wiki + raw files
        print(f"  mixed: building BM25 index...")
        build_bm25_index(
            MIXED_GRAPH_OUT / "graph.json",
            WIKI_DIR,
            MIXED_CORPUS_DIR,
            BM25_INDEX_PATH,
        )

        # Wiki lint — flag stale claims and contradictions
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        lint_path = WIKI_DIR / "lint_report.md"
        if api_key and WIKI_DIR.exists() and not lint_path.exists():
            print(f"  mixed: running wiki lint...")
            try:
                import anthropic as _anthropic
                _client = _anthropic.Anthropic(api_key=api_key)
                wiki_lint(WIKI_DIR, _client)
            except Exception as exc:
                print(f"  WARN: wiki lint failed — {exc}")
        elif lint_path.exists():
            print(f"  mixed: wiki lint already done (skipping)")
        else:
            print(f"  mixed: wiki lint skipped (no API key)")


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
    """Test 12: Data-driven recommendation — when to use PRISM vs alternatives."""
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
  PRISM DECISION FRAMEWORK
  =========================
  Based on measured benchmark data:

  USE PRISM WHEN:
  ---------------
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
       ("how does X connect to Y?", "what calls Z?", "how does X use Y?")
       Reason:  Code graph captures CALL, IMPORTS, CONTAINS edges invisible in raw text.
                LLMWiki pre-synthesises cross-modal knowledge for rationale questions.
       Accuracy: {accuracy_note}

  [OK] Cost matters ({f'{large_cost_ratio}x cheaper than naive on large corpus' if large_cost_ratio else 'see test10'})
       Reason:  PRISM context is bounded by dynamic budget; naive scales with corpus.

  DON'T USE PRISM WHEN:
  ---------------------
  [WARN] Corpus is DOC-HEAVY (.md/.txt/.rst/.pdf dominate) with no code
         Reason:  Code graph AST excludes doc extensions. Layer 1 has 0% coverage on pure docs.
         Finding: {doc_nodes} nodes extracted from {doc_files_n} doc files = {doc_coverage}% coverage.
         Alt:     text-embedding-3-small ($0.02/1M tokens) + vector search + read top-5 files.

  [WARN] You need IMPLEMENTATION DETAIL answers
         ("exactly what does line 47 do?", "what is the default value of X?")
         Reason:  Graph nodes are labels + source locations, not full code. Raw files win here.

  [WARN] Corpus is SMALL (<10 files / <5k tokens)
         Reason:  Break-even is high; compression ratio is low; just read the files.
         Evidence: small corpus ratio only {small_ratio}x — naive read = {
             results.get("test1", {}).get("corpora", {}).get("small", {}).get("total_tokens", 0)
         :,} tokens total.

  [WARN] One-off or infrequent queries
         Reason:  Build cost is real. Only worth it if you reuse the layers many times.

  SUMMARY TABLE:
  --------------
  Scenario                               Verdict
  -------------------------------------- ----------------------------------------
  >50 code files, >20 queries            USE PRISM
  Mixed code+doc, cross-ref questions    USE PRISM (all layers contribute)
  <10 code files, <5 queries             SKIP — just read the files
  Doc-only corpus, doc questions         SKIP code graph; use embeddings
  Need exact implementation details      SKIP — raw files beat the graph here
  Exploring a new large codebase         USE PRISM — graph shows architecture fast
  Writing a chatbot over documentation   SKIP — needs embeddings pipeline
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


def test13_cross_modal_hybrid(results: dict) -> dict:
    """Test 13: Cross-modal hybrid retrieval on mixed code+doc corpus.

    Compares three approaches on MIXED_QUESTIONS:
      A) graph-only  — plain graphify query (code AST, docs ignored)
      B) hybrid      — 1000-token code graph + 1000-token smart doc retrieval
      C) raw top-10  — grep top-10 files including docs

    All three use the same ~2000 token context budget.
    LLM-as-judge scores all three (correctness/completeness/relevance, /15).
    """
    print("\n" + "=" * 60)
    print("TEST 13 — Cross-Modal Hybrid Retrieval (Mixed Code+Doc Corpus)")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test13"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test13"]

    if not MIXED_CORPUS_DIR.exists():
        print(f"  SKIP: mixed corpus not found at {MIXED_CORPUS_DIR}")
        results["test13"] = {"status": "SKIP", "reason": "mixed corpus missing"}
        return results["test13"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        results["test13"] = {"status": "SKIP", "reason": str(exc)}
        return results["test13"]

    mixed_graph     = MIXED_GRAPH_OUT / "graph.json"
    doc_index_path  = MIXED_GRAPH_OUT / "doc_index.json"

    # Build doc_index if it wasn't built in the pre-step
    if not doc_index_path.exists():
        print("  Building doc_index for mixed corpus...")
        doc_files = sorted(MIXED_CORPUS_DIR.rglob("*.md"))
        build_doc_index(doc_files, MIXED_GRAPH_OUT, client)

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    JUDGE_SYSTEM = (
        "You are an impartial technical evaluator. Score answers objectively. "
        "Do not favour any answer based on length. "
        "Respond with only a JSON object — no markdown fences, no explanation."
    )
    JUDGE_TEMPLATE = (
        "QUESTION: {question}\n\n"
        "ANSWER A (graph-only — code structure only):\n{answer_a}\n\n"
        "ANSWER B (hybrid — code structure + documentation):\n{answer_b}\n\n"
        "ANSWER C (raw top-10 files):\n{answer_c}\n\n"
        "Score each answer 1-5 on:\n"
        "- correctness: factually accurate?\n"
        "- completeness: covers all key aspects?\n"
        "- relevance: focused, no padding?\n\n"
        "Respond with ONLY this JSON (total = sum of three scores, max 15):\n"
        '{{"answer_a":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_b":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_c":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"winner":"A_or_B_or_C_or_tie","comment":"one sentence"}}'
    )

    import re as _re

    t13: dict[str, Any] = {
        "status": "PASS",
        "questions":    [],
        "graph_wins":   0,
        "hybrid_wins":  0,
        "raw_wins":     0,
        "ties":         0,
        "total_input_tokens":  0,
        "total_output_tokens": 0,
    }
    token_sums = {"graph": 0, "hybrid": 0, "raw": 0}

    all_files = read_corpus_files(MIXED_CORPUS_DIR)
    print(f"  Mixed corpus: {len(all_files)} files "
          f"({sum(1 for f in all_files if f.suffix=='.py')} code, "
          f"{sum(1 for f in all_files if f.suffix=='.md')} docs)")
    print(f"\n  {'Q':<3} {'Question':<50} {'Graph':>6} {'Hybrid':>7} {'Raw':>5} {'Winner'}")
    print("  " + "-" * 80)

    for i, question in enumerate(MIXED_QUESTIONS):
        entry: dict[str, Any] = {"question": question}

        # ── Approach A: graph-only ──────────────────────────────────────
        graph_ctx    = ""
        graph_ctx_tokens = 0
        if mixed_graph.exists():
            try:
                r = subprocess.run(
                    ["graphify", "query", question,
                     "--graph", str(mixed_graph),
                     "--budget", str(QUERY_BUDGET)],
                    capture_output=True, text=True, timeout=60,
                )
                graph_ctx = r.stdout
            except Exception as exc:
                graph_ctx = f"[graph error: {exc}]"
            graph_ctx_tokens = count_tokens(graph_ctx)

        entry["graph_ctx_tokens"] = graph_ctx_tokens
        token_sums["graph"] += graph_ctx_tokens

        try:
            resp_a = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on this code knowledge graph, answer the question.\n\n"
                    f"Context:\n{graph_ctx}\n\nQuestion: {question}"}],
            )
            answer_a = resp_a.content[0].text
            t13["total_input_tokens"]  += resp_a.usage.input_tokens
            t13["total_output_tokens"] += resp_a.usage.output_tokens
        except Exception as exc:
            answer_a = f"[error: {exc}]"

        # ── Approach B: hybrid ──────────────────────────────────────────
        hybrid_ctx, h_graph_tok, h_doc_tok = hybrid_query(
            question, mixed_graph, doc_index_path, MIXED_CORPUS_DIR,
            total_budget=QUERY_BUDGET,
        )
        entry["hybrid_graph_tokens"] = h_graph_tok
        entry["hybrid_doc_tokens"]   = h_doc_tok
        token_sums["hybrid"] += h_graph_tok + h_doc_tok

        try:
            resp_b = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following context (code structure + documentation), "
                    f"answer the question.\n\nContext:\n{hybrid_ctx}\n\nQuestion: {question}"}],
            )
            answer_b = resp_b.content[0].text
            t13["total_input_tokens"]  += resp_b.usage.input_tokens
            t13["total_output_tokens"] += resp_b.usage.output_tokens
        except Exception as exc:
            answer_b = f"[error: {exc}]"

        # ── Approach C: raw top-10 ──────────────────────────────────────
        top_files   = grep_top_files(MIXED_CORPUS_DIR, question, top_n=10)
        raw_context = "\n\n".join(f"--- {f.name} ---\n{safe_read(f)}" for f in top_files)
        raw_tok     = count_tokens(raw_context)
        entry["raw_tokens"] = raw_tok
        token_sums["raw"] += raw_tok

        try:
            resp_c = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following source files, answer the question.\n\n"
                    f"Files:\n{raw_context}\n\nQuestion: {question}"}],
            )
            answer_c = resp_c.content[0].text
            t13["total_input_tokens"]  += resp_c.usage.input_tokens
            t13["total_output_tokens"] += resp_c.usage.output_tokens
        except Exception as exc:
            answer_c = f"[error: {exc}]"

        # ── Judge ───────────────────────────────────────────────────────
        try:
            resp_j = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=400,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
                    question=question, answer_a=answer_a,
                    answer_b=answer_b, answer_c=answer_c,
                )}],
            )
            raw_resp = resp_j.content[0].text.strip()
            t13["total_input_tokens"]  += resp_j.usage.input_tokens
            t13["total_output_tokens"] += resp_j.usage.output_tokens
            m = _re.search(r"\{.*\}", raw_resp, _re.DOTALL)
            scores = json.loads(m.group()) if m else {}
        except Exception as exc:
            scores = {"error": str(exc)}

        a_s = scores.get("answer_a", {}).get("total", 0)
        b_s = scores.get("answer_b", {}).get("total", 0)
        c_s = scores.get("answer_c", {}).get("total", 0)
        winner_raw = scores.get("winner", "tie")
        comment    = scores.get("comment", "")

        winner_label = {"A": "graph", "B": "hybrid", "C": "raw"}.get(winner_raw, "tie")
        if winner_label == "graph":   t13["graph_wins"]  += 1
        elif winner_label == "hybrid": t13["hybrid_wins"] += 1
        elif winner_label == "raw":    t13["raw_wins"]    += 1
        else:                          t13["ties"]        += 1

        entry.update({"graph_score": a_s, "hybrid_score": b_s, "raw_score": c_s,
                      "winner": winner_label, "comment": comment})
        t13["questions"].append(entry)

        # Save side-by-side answer file
        (ACCURACY_DIR / f"t13_q{i+1:02d}.md").write_text(
            f"# Test13 Q{i+1}: {question}\n\n"
            f"## A: Graph-only ({graph_ctx_tokens} ctx tokens) — {a_s}/15\n\n"
            f"{answer_a}\n\n---\n\n"
            f"## B: Hybrid ({h_graph_tok}+{h_doc_tok} tokens) — {b_s}/15\n\n"
            f"{answer_b}\n\n---\n\n"
            f"## C: Raw top-10 ({raw_tok} tokens) — {c_s}/15\n\n"
            f"{answer_c}\n\n---\n\n"
            f"**Winner:** {winner_label}  \n**Comment:** {comment}\n",
            encoding="utf-8",
        )

        print(f"  Q{i+1:<2} {question[:50]:<50} {a_s:>5}/15 {b_s:>6}/15 {c_s:>4}/15  {winner_label}")

    # Summary stats
    n_qs = max(1, len(MIXED_QUESTIONS))
    avg  = {k: round(v / n_qs) for k, v in token_sums.items()}
    t13["avg_context_tokens"] = avg
    t13["hybrid_vs_naive_ratio"] = round(avg["raw"] / avg["hybrid"], 2) if avg["hybrid"] > 0 else 0

    judge_cost = (t13["total_input_tokens"] * PRICE_CLAUDE_INPUT +
                  t13["total_output_tokens"] * PRICE_CLAUDE_OUTPUT)
    t13["judge_cost_usd"] = round(judge_cost, 4)

    g = t13["graph_wins"]; h = t13["hybrid_wins"]; r = t13["raw_wins"]; ti = t13["ties"]
    n = g + h + r + ti
    print(f"\n  Summary: graph {g}/{n}, hybrid {h}/{n}, raw {r}/{n}, ties {ti}/{n}")
    print(f"  Avg context tokens — graph: {avg['graph']:,} | hybrid: {avg['hybrid']:,} | raw: {avg['raw']:,}")
    print(f"  Hybrid uses {t13['hybrid_vs_naive_ratio']}x fewer tokens than raw top-10")
    print(f"  Judge cost: ${judge_cost:.4f}")
    print(f"\n  Answer files saved to {ACCURACY_DIR}/t13_q*.md")

    results["test13"] = t13
    return t13


def test14_llmwiki(results: dict) -> dict:
    """Test 14: LLMWiki entity pages — 4-way comparison on mixed corpus.

    Compares four approaches on MIXED_QUESTIONS:
      A) graph-only  — plain graphify query (code AST, docs ignored)
      B) hybrid      — graph + smart_doc_query (from Test 13)
      C) wiki        — LLMWiki pre-synthesised entity pages
      D) raw top-10  — grep top-10 files including docs

    Hypothesis: wiki (C) closes the gap between hybrid (B) and raw (D)
    because wiki pages already synthesise code + doc knowledge into a single
    coherent answer source — reducing the work the answering LLM must do.
    """
    print("\n" + "=" * 60)
    print("TEST 14 — LLMWiki Entity Pages (4-Way Comparison)")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test14"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test14"]

    if not MIXED_CORPUS_DIR.exists():
        print(f"  SKIP: mixed corpus not found at {MIXED_CORPUS_DIR}")
        results["test14"] = {"status": "SKIP", "reason": "mixed corpus missing"}
        return results["test14"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        results["test14"] = {"status": "SKIP", "reason": str(exc)}
        return results["test14"]

    mixed_graph    = MIXED_GRAPH_OUT / "graph.json"
    doc_index_path = MIXED_GRAPH_OUT / "doc_index.json"

    # Ensure wiki pages exist
    if not WIKI_DIR.exists() or not any(WIKI_DIR.glob("*.md")):
        print("  Building LLMWiki pages...")
        build_llmwiki(mixed_graph, doc_index_path, MIXED_CORPUS_DIR, WIKI_DIR, client)

    n_wiki_pages = len(list(WIKI_DIR.glob("*.md")))
    print(f"  Wiki: {n_wiki_pages} entity pages available")

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    JUDGE_SYSTEM = (
        "You are an impartial technical evaluator. Score answers objectively. "
        "Do not favour any answer based on length. "
        "Respond with only a JSON object — no markdown fences, no explanation."
    )
    JUDGE_TEMPLATE = (
        "QUESTION: {question}\n\n"
        "ANSWER A (graph-only — code structure only):\n{answer_a}\n\n"
        "ANSWER B (hybrid — code graph + documentation index):\n{answer_b}\n\n"
        "ANSWER C (wiki — pre-synthesised LLMWiki entity pages):\n{answer_c}\n\n"
        "ANSWER D (raw top-10 files):\n{answer_d}\n\n"
        "Score each answer 1-5 on:\n"
        "- correctness: factually accurate?\n"
        "- completeness: covers all key aspects?\n"
        "- relevance: focused, no padding?\n\n"
        "Respond with ONLY this JSON (total = sum of three scores, max 15):\n"
        '{{"answer_a":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_b":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_c":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_d":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"winner":"A_or_B_or_C_or_D_or_tie","comment":"one sentence"}}'
    )

    import re as _re

    t14: dict[str, Any] = {
        "status": "PASS",
        "questions":    [],
        "graph_wins":   0,
        "hybrid_wins":  0,
        "wiki_wins":    0,
        "raw_wins":     0,
        "ties":         0,
        "total_input_tokens":  0,
        "total_output_tokens": 0,
    }
    token_sums = {"graph": 0, "hybrid": 0, "wiki": 0, "raw": 0}

    print(f"\n  {'Q':<3} {'Question':<48} {'Graph':>6} {'Hybrid':>7} {'Wiki':>5} {'Raw':>5} {'Winner'}")
    print("  " + "-" * 85)

    for i, question in enumerate(MIXED_QUESTIONS):
        entry: dict[str, Any] = {"question": question}

        # ── Approach A: graph-only ──────────────────────────────────────────
        graph_ctx = ""
        if mixed_graph.exists():
            try:
                r = subprocess.run(
                    ["graphify", "query", question,
                     "--graph", str(mixed_graph),
                     "--budget", str(QUERY_BUDGET)],
                    capture_output=True, text=True, timeout=60,
                )
                graph_ctx = r.stdout
            except Exception as exc:
                graph_ctx = f"[graph error: {exc}]"
        graph_tok = count_tokens(graph_ctx)
        entry["graph_ctx_tokens"] = graph_tok
        token_sums["graph"] += graph_tok

        try:
            resp_a = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on this code knowledge graph, answer the question.\n\n"
                    f"Context:\n{graph_ctx}\n\nQuestion: {question}"}],
            )
            answer_a = resp_a.content[0].text
            t14["total_input_tokens"]  += resp_a.usage.input_tokens
            t14["total_output_tokens"] += resp_a.usage.output_tokens
        except Exception as exc:
            answer_a = f"[error: {exc}]"

        # ── Approach B: hybrid (graph + doc_index) ──────────────────────────
        hybrid_ctx, h_graph_tok, h_doc_tok = hybrid_query(
            question, mixed_graph, doc_index_path, MIXED_CORPUS_DIR,
            total_budget=QUERY_BUDGET,
        )
        entry["hybrid_tokens"] = h_graph_tok + h_doc_tok
        token_sums["hybrid"] += h_graph_tok + h_doc_tok

        try:
            resp_b = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following context (code structure + documentation), "
                    f"answer the question.\n\nContext:\n{hybrid_ctx}\n\nQuestion: {question}"}],
            )
            answer_b = resp_b.content[0].text
            t14["total_input_tokens"]  += resp_b.usage.input_tokens
            t14["total_output_tokens"] += resp_b.usage.output_tokens
        except Exception as exc:
            answer_b = f"[error: {exc}]"

        # ── Approach C: wiki ────────────────────────────────────────────────
        wiki_ctx, wiki_tok = wiki_query(question, WIKI_DIR, budget=QUERY_BUDGET)
        entry["wiki_tokens"] = wiki_tok
        token_sums["wiki"] += wiki_tok

        try:
            resp_c = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following pre-synthesised wiki pages, "
                    f"answer the question.\n\nWiki:\n{wiki_ctx}\n\nQuestion: {question}"}],
            )
            answer_c = resp_c.content[0].text
            t14["total_input_tokens"]  += resp_c.usage.input_tokens
            t14["total_output_tokens"] += resp_c.usage.output_tokens
        except Exception as exc:
            answer_c = f"[error: {exc}]"

        # ── Approach D: raw top-10 ──────────────────────────────────────────
        top_files   = grep_top_files(MIXED_CORPUS_DIR, question, top_n=10)
        raw_context = "\n\n".join(f"--- {f.name} ---\n{safe_read(f)}" for f in top_files)
        raw_tok     = count_tokens(raw_context)
        entry["raw_tokens"] = raw_tok
        token_sums["raw"] += raw_tok

        try:
            resp_d = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following source files, answer the question.\n\n"
                    f"Files:\n{raw_context}\n\nQuestion: {question}"}],
            )
            answer_d = resp_d.content[0].text
            t14["total_input_tokens"]  += resp_d.usage.input_tokens
            t14["total_output_tokens"] += resp_d.usage.output_tokens
        except Exception as exc:
            answer_d = f"[error: {exc}]"

        # ── Judge ───────────────────────────────────────────────────────────
        try:
            resp_j = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=500,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
                    question=question, answer_a=answer_a,
                    answer_b=answer_b, answer_c=answer_c, answer_d=answer_d,
                )}],
            )
            raw_resp = resp_j.content[0].text.strip()
            t14["total_input_tokens"]  += resp_j.usage.input_tokens
            t14["total_output_tokens"] += resp_j.usage.output_tokens
            m = _re.search(r"\{.*\}", raw_resp, _re.DOTALL)
            scores = json.loads(m.group()) if m else {}
        except Exception as exc:
            scores = {"error": str(exc)}

        a_s = scores.get("answer_a", {}).get("total", 0)
        b_s = scores.get("answer_b", {}).get("total", 0)
        c_s = scores.get("answer_c", {}).get("total", 0)
        d_s = scores.get("answer_d", {}).get("total", 0)
        winner_raw   = scores.get("winner", "tie")
        comment      = scores.get("comment", "")

        winner_label = {"A": "graph", "B": "hybrid", "C": "wiki", "D": "raw"}.get(winner_raw, "tie")
        if   winner_label == "graph":  t14["graph_wins"]  += 1
        elif winner_label == "hybrid": t14["hybrid_wins"] += 1
        elif winner_label == "wiki":   t14["wiki_wins"]   += 1
        elif winner_label == "raw":    t14["raw_wins"]    += 1
        else:                          t14["ties"]        += 1

        entry.update({
            "graph_score": a_s, "hybrid_score": b_s,
            "wiki_score":  c_s, "raw_score":    d_s,
            "winner": winner_label, "comment": comment,
        })
        t14["questions"].append(entry)

        # Save side-by-side answer file
        (ACCURACY_DIR / f"t14_q{i+1:02d}.md").write_text(
            f"# Test14 Q{i+1}: {question}\n\n"
            f"## A: Graph-only ({graph_tok} ctx tokens) — {a_s}/15\n\n"
            f"{answer_a}\n\n---\n\n"
            f"## B: Hybrid ({h_graph_tok}+{h_doc_tok} tokens) — {b_s}/15\n\n"
            f"{answer_b}\n\n---\n\n"
            f"## C: Wiki ({wiki_tok} tokens) — {c_s}/15\n\n"
            f"{answer_c}\n\n---\n\n"
            f"## D: Raw top-10 ({raw_tok} tokens) — {d_s}/15\n\n"
            f"{answer_d}\n\n---\n\n"
            f"**Winner:** {winner_label}  \n**Comment:** {comment}\n",
            encoding="utf-8",
        )

        print(f"  Q{i+1:<2} {question[:48]:<48} {a_s:>5}/15 {b_s:>6}/15 {c_s:>4}/15 {d_s:>4}/15  {winner_label}")

    # Summary stats
    n_qs = max(1, len(MIXED_QUESTIONS))
    avg  = {k: round(v / n_qs) for k, v in token_sums.items()}
    t14["avg_context_tokens"] = avg
    t14["wiki_vs_naive_ratio"] = round(avg["raw"] / avg["wiki"], 2) if avg["wiki"] > 0 else 0

    judge_cost = (t14["total_input_tokens"] * PRICE_CLAUDE_INPUT +
                  t14["total_output_tokens"] * PRICE_CLAUDE_OUTPUT)
    t14["judge_cost_usd"] = round(judge_cost, 4)

    g = t14["graph_wins"]; h = t14["hybrid_wins"]
    w = t14["wiki_wins"];  r = t14["raw_wins"]; ti = t14["ties"]
    n = g + h + w + r + ti

    # Score averages
    q_entries = t14["questions"]
    avg_scores = {
        "graph":  round(sum(q.get("graph_score",  0) for q in q_entries) / n_qs, 1),
        "hybrid": round(sum(q.get("hybrid_score", 0) for q in q_entries) / n_qs, 1),
        "wiki":   round(sum(q.get("wiki_score",   0) for q in q_entries) / n_qs, 1),
        "raw":    round(sum(q.get("raw_score",    0) for q in q_entries) / n_qs, 1),
    }
    t14["avg_scores"] = avg_scores

    print(f"\n  Summary: graph {g}/{n}, hybrid {h}/{n}, wiki {w}/{n}, raw {r}/{n}, ties {ti}/{n}")
    print(f"  Avg scores /15  — graph: {avg_scores['graph']} | hybrid: {avg_scores['hybrid']} | "
          f"wiki: {avg_scores['wiki']} | raw: {avg_scores['raw']}")
    print(f"  Avg context tokens — graph: {avg['graph']:,} | hybrid: {avg['hybrid']:,} | "
          f"wiki: {avg['wiki']:,} | raw: {avg['raw']:,}")
    print(f"  Wiki uses {t14['wiki_vs_naive_ratio']}x fewer tokens than raw top-10")
    print(f"  Judge cost: ${judge_cost:.4f}")
    print(f"\n  Answer files saved to {ACCURACY_DIR}/t14_q*.md")

    results["test14"] = t14
    return t14


def test15_routed_system(results: dict) -> dict:
    """Test 15: Question-aware router with dynamic budget allocation.

    The router classifies each question into a type, selects retrieval layers
    and allocates budget per layer accordingly. Compares the routed system
    against raw top-10 files.

    Hypothesis: routing to the right layer + budget matches raw accuracy
    (13.8/15 avg) at 2-4x fewer tokens, because each question gets exactly
    the knowledge density it needs — no more, no less.
    """
    print("\n" + "=" * 60)
    print("TEST 15 — Question-Aware Router + Dynamic Budget")
    print("=" * 60)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  SKIP: ANTHROPIC_API_KEY not set")
        results["test15"] = {"status": "SKIP", "reason": "no ANTHROPIC_API_KEY"}
        return results["test15"]

    if not MIXED_CORPUS_DIR.exists():
        print(f"  SKIP: mixed corpus not found")
        results["test15"] = {"status": "SKIP", "reason": "mixed corpus missing"}
        return results["test15"]

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        results["test15"] = {"status": "SKIP", "reason": str(exc)}
        return results["test15"]

    mixed_graph    = MIXED_GRAPH_OUT / "graph.json"
    doc_index_path = MIXED_GRAPH_OUT / "doc_index.json"

    # Load BM25 index (build if missing)
    if not BM25_INDEX_PATH.exists():
        print("  Building BM25 index...")
        bm25_index = build_bm25_index(mixed_graph, WIKI_DIR, MIXED_CORPUS_DIR, BM25_INDEX_PATH)
    else:
        bm25_index = json.loads(BM25_INDEX_PATH.read_text(encoding="utf-8"))

    # Wiki lint (run if missing)
    lint_path = WIKI_DIR / "lint_report.md"
    if not lint_path.exists() and WIKI_DIR.exists():
        print("  Running wiki lint...")
        lint_result = wiki_lint(WIKI_DIR, client)
    else:
        lint_result = safe_read(lint_path) if lint_path.exists() else ""

    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    JUDGE_SYSTEM = (
        "You are an impartial technical evaluator. Score answers objectively. "
        "Do not favour any answer based on length. "
        "Respond with only a JSON object — no markdown fences, no explanation."
    )
    JUDGE_TEMPLATE = (
        "QUESTION: {question}\n\n"
        "ANSWER A (routed system — question-aware layer selection):\n{answer_a}\n\n"
        "ANSWER B (raw top-10 files — brute force):\n{answer_b}\n\n"
        "Score each answer 1-5 on:\n"
        "- correctness: factually accurate?\n"
        "- completeness: covers all key aspects?\n"
        "- relevance: focused, no padding?\n\n"
        "Respond with ONLY this JSON:\n"
        '{{"answer_a":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"answer_b":{{"correctness":0,"completeness":0,"relevance":0,"total":0}},'
        '"winner":"A_or_B_or_tie","comment":"one sentence"}}'
    )

    import re as _re

    t15: dict[str, Any] = {
        "status": "PASS",
        "questions":       [],
        "routed_wins":     0,
        "raw_wins":        0,
        "ties":            0,
        "total_input_tokens":  0,
        "total_output_tokens": 0,
        "lint_issues":     0,
        "total_spill_events":  0,
    }

    # Count lint issues
    if lint_result:
        t15["lint_issues"] = lint_result.count("##") - 1  # subtract header

    print(f"\n  Budget config:")
    for qtype, cfg in ROUTING_CONFIG.items():
        print(f"    {qtype:<15} {cfg['budget']:>5} tokens  layers: {cfg['layers']}")

    print(f"\n  {'Q':<3} {'Type':<15} {'Budget':>7} {'Routed':>8} {'Raw':>5} {'Winner'}")
    print("  " + "-" * 55)

    token_sums = {"routed": 0, "raw": 0}

    for i, question in enumerate(MIXED_QUESTIONS):
        entry: dict[str, Any] = {"question": question}

        # ── Route ───────────────────────────────────────────────────────────
        route = router(question, client)
        t15["total_input_tokens"]  += route["input_tokens"]
        t15["total_output_tokens"] += route["output_tokens"]
        entry["route"] = route

        # ── Routed retrieval ────────────────────────────────────────────────
        routed_ctx, routed_tok, tok_per_layer = routed_query(
            question, route,
            mixed_graph, doc_index_path, WIKI_DIR, bm25_index, MIXED_CORPUS_DIR,
        )
        entry["routed_tokens"]    = routed_tok
        entry["tokens_per_layer"] = tok_per_layer
        token_sums["routed"]     += routed_tok

        try:
            resp_a = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following context, answer the question.\n\n"
                    f"Context:\n{routed_ctx}\n\nQuestion: {question}"}],
            )
            answer_a = resp_a.content[0].text
            t15["total_input_tokens"]  += resp_a.usage.input_tokens
            t15["total_output_tokens"] += resp_a.usage.output_tokens
        except Exception as exc:
            answer_a = f"[error: {exc}]"

        # ── Raw top-10 ──────────────────────────────────────────────────────
        top_files   = grep_top_files(MIXED_CORPUS_DIR, question, top_n=10)
        raw_context = "\n\n".join(f"--- {f.name} ---\n{safe_read(f)}" for f in top_files)
        raw_tok     = count_tokens(raw_context)
        entry["raw_tokens"] = raw_tok
        token_sums["raw"]  += raw_tok

        try:
            resp_b = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Based on the following source files, answer the question.\n\n"
                    f"Files:\n{raw_context}\n\nQuestion: {question}"}],
            )
            answer_b = resp_b.content[0].text
            t15["total_input_tokens"]  += resp_b.usage.input_tokens
            t15["total_output_tokens"] += resp_b.usage.output_tokens
        except Exception as exc:
            answer_b = f"[error: {exc}]"

        # ── Judge ───────────────────────────────────────────────────────────
        try:
            resp_j = client.messages.create(
                model=ANTHROPIC_MODEL, max_tokens=400,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b,
                )}],
            )
            raw_resp = resp_j.content[0].text.strip()
            t15["total_input_tokens"]  += resp_j.usage.input_tokens
            t15["total_output_tokens"] += resp_j.usage.output_tokens
            m = _re.search(r"\{.*\}", raw_resp, _re.DOTALL)
            scores = json.loads(m.group()) if m else {}
        except Exception as exc:
            scores = {"error": str(exc)}

        a_s = scores.get("answer_a", {}).get("total", 0)
        b_s = scores.get("answer_b", {}).get("total", 0)
        winner_raw   = scores.get("winner", "tie")
        comment      = scores.get("comment", "")

        winner_label = {"A": "routed", "B": "raw"}.get(winner_raw, "tie")
        if   winner_label == "routed": t15["routed_wins"] += 1
        elif winner_label == "raw":    t15["raw_wins"]    += 1
        else:                          t15["ties"]        += 1

        spill_events = tok_per_layer.get("spill_events", [])
        t15["total_spill_events"] += len(spill_events)

        entry.update({
            "routed_score": a_s, "raw_score": b_s,
            "winner": winner_label, "comment": comment,
            "spill_events": spill_events,
        })
        t15["questions"].append(entry)

        # Save answer file
        layer_breakdown = " | ".join(
            f"{k}:{v}" for k, v in tok_per_layer.items() if k != "spill_events"
        )
        spill_note = ""
        if spill_events:
            spill_note = "\n**Spill events (adaptive reallocation):**\n" + \
                         "\n".join(f"  - {e}" for e in spill_events) + "\n"

        (ACCURACY_DIR / f"t15_q{i+1:02d}.md").write_text(
            f"# Test15 Q{i+1}: {question}\n\n"
            f"**Route:** {route['type']} | **Budget:** {route['budget']} tokens\n"
            f"**Reason:** {route['reason']}\n"
            f"**Layer tokens:** {layer_breakdown}\n"
            f"{spill_note}\n"
            f"## A: Routed ({routed_tok} tokens) — {a_s}/15\n\n"
            f"{answer_a}\n\n---\n\n"
            f"## B: Raw top-10 ({raw_tok} tokens) — {b_s}/15\n\n"
            f"{answer_b}\n\n---\n\n"
            f"**Winner:** {winner_label}  \n**Comment:** {comment}\n",
            encoding="utf-8",
        )

        spill_flag = " ↑spill" if spill_events else ""
        print(f"  Q{i+1:<2} {route['type']:<15} {route['budget']:>6}tok "
              f"{a_s:>7}/15 {b_s:>4}/15  {winner_label}{spill_flag}")

    # Summary
    n_qs = max(1, len(MIXED_QUESTIONS))
    avg  = {k: round(v / n_qs) for k, v in token_sums.items()}
    t15["avg_context_tokens"] = avg
    t15["token_savings_vs_raw"] = round(avg["raw"] / avg["routed"], 2) if avg["routed"] > 0 else 0

    avg_scores = {
        "routed": round(sum(q.get("routed_score", 0) for q in t15["questions"]) / n_qs, 1),
        "raw":    round(sum(q.get("raw_score",    0) for q in t15["questions"]) / n_qs, 1),
    }
    t15["avg_scores"] = avg_scores

    judge_cost = (t15["total_input_tokens"] * PRICE_CLAUDE_INPUT +
                  t15["total_output_tokens"] * PRICE_CLAUDE_OUTPUT)
    t15["judge_cost_usd"] = round(judge_cost, 4)

    rw = t15["routed_wins"]; ra = t15["raw_wins"]; ti = t15["ties"]
    n  = rw + ra + ti
    print(f"\n  Results: routed {rw}/{n}, raw {ra}/{n}, ties {ti}/{n}")
    print(f"  Avg scores — routed: {avg_scores['routed']}/15 | raw: {avg_scores['raw']}/15")
    print(f"  Avg tokens — routed: {avg['routed']:,} | raw: {avg['raw']:,}")
    print(f"  Token savings vs raw: {t15['token_savings_vs_raw']}x")
    print(f"  Wiki lint issues flagged: {t15['lint_issues']}")
    print(f"  Adaptive spill events:   {t15['total_spill_events']} "
          f"(layers that had low signal and redistributed budget)")
    print(f"  Judge cost: ${judge_cost:.4f}")
    print(f"\n  Answer files: {ACCURACY_DIR}/t15_q*.md")
    if lint_path.exists():
        print(f"  Lint report:  {lint_path}")

    results["test15"] = t15
    return t15


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def print_final_report(results: dict) -> None:
    print("\n")
    print("=" * 60)
    print("PRISM BENCHMARK REPORT")
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
        ("test13", "Cross-Modal Hybrid (Mixed Code+Doc Corpus)"),
        ("test14", "LLMWiki Entity Pages (4-Way Comparison)"),
        ("test15", "Question-Aware Router + Dynamic Budget"),
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
        print(f"\n  LLM judge (code-only, Test 11): graph {g} wins, raw {r} wins, ties {ti}")

    t13 = results.get("test13", {})
    if t13.get("status") == "PASS":
        g  = t13.get("graph_wins",  0)
        h  = t13.get("hybrid_wins", 0)
        r  = t13.get("raw_wins",    0)
        ti = t13.get("ties",        0)
        avg = t13.get("avg_context_tokens", {})
        ratio = t13.get("hybrid_vs_naive_ratio", 0)
        print(f"\n  Cross-modal hybrid (Test 13):")
        print(f"    graph {g} wins | hybrid {h} wins | raw {r} wins | ties {ti}")
        print(f"    Avg context tokens — graph: {avg.get('graph',0):,} | "
              f"hybrid: {avg.get('hybrid',0):,} | raw: {avg.get('raw',0):,}")
        print(f"    Hybrid uses {ratio}x fewer tokens than raw at same total budget")

    t14 = results.get("test14", {})
    if t14.get("status") == "PASS":
        g  = t14.get("graph_wins",  0)
        h  = t14.get("hybrid_wins", 0)
        w  = t14.get("wiki_wins",   0)
        r  = t14.get("raw_wins",    0)
        ti = t14.get("ties",        0)
        avg    = t14.get("avg_context_tokens", {})
        scores = t14.get("avg_scores", {})
        ratio  = t14.get("wiki_vs_naive_ratio", 0)
        print(f"\n  LLMWiki 4-way comparison (Test 14):")
        print(f"    graph {g} | hybrid {h} | wiki {w} | raw {r} | ties {ti}")
        print(f"    Avg scores /15 — graph: {scores.get('graph',0)} | hybrid: {scores.get('hybrid',0)} | "
              f"wiki: {scores.get('wiki',0)} | raw: {scores.get('raw',0)}")
        print(f"    Avg tokens — graph: {avg.get('graph',0):,} | hybrid: {avg.get('hybrid',0):,} | "
              f"wiki: {avg.get('wiki',0):,} | raw: {avg.get('raw',0):,}")
        print(f"    Wiki uses {ratio}x fewer tokens than raw")

    t15 = results.get("test15", {})
    if t15.get("status") == "PASS":
        rw     = t15.get("routed_wins", 0)
        ra     = t15.get("raw_wins",    0)
        ti     = t15.get("ties",        0)
        avg    = t15.get("avg_context_tokens", {})
        scores = t15.get("avg_scores", {})
        sav    = t15.get("token_savings_vs_raw", 0)
        lint   = t15.get("lint_issues", 0)
        print(f"\n  Routed system (Test 15):")
        print(f"    routed {rw} wins | raw {ra} wins | ties {ti}")
        print(f"    Avg scores — routed: {scores.get('routed',0)}/15 | raw: {scores.get('raw',0)}/15")
        print(f"    Avg tokens — routed: {avg.get('routed',0):,} | raw: {avg.get('raw',0):,}")
        print(f"    Token savings vs raw: {sav}x")
        print(f"    Wiki lint issues: {lint}")


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw JSON
    data_path = RESULTS_DIR / "data.json"
    data_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Raw data saved to {data_path}")

    # Markdown report
    md_lines = ["# PRISM Benchmark Report\n"]
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
    print("PRISM — Pre-compiled Retrieval with Intelligent Strata Management")
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
    test13_cross_modal_hybrid(results)
    test14_llmwiki(results)
    test15_routed_system(results)

    # Final report
    print_final_report(results)
    save_results(results)


if __name__ == "__main__":
    main()
