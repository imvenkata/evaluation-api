## Synthetic Ground Truth Architecture

This document explains the end-to-end design for generating synthetic ground-truth data for search/retrieval evaluation, why specific features exist (distractors, dedup, multi-golden), and how to control behavior via `evaluation_api/configs/generation_config.py`.

### Goals
- Build realistic, hard, and diverse query→relevance pairs to evaluate ranking/retrieval.
- Support multiple correct documents per query (multi-golden).
- Provide configurable query-type variation and scalable performance (caching, concurrency).

### Pipeline
```text
[configs/generation_config.py]
        │
        ▼
[generation/cli.py]
  1) data_loader.load_data → ChunkData[]
  2) chunk_validator.validate_chunks → (valid[], rejected[])
  3) chunk_selector.ContextSelector(valid).select_contexts() → SelectionBundle[]
  4) query_generator.QueryGenerator.generate_queries(bundles) → GeneratedQuery[]
  5) evaluation_layer.evaluate_queries(queries, mode) → ValidatedGroundTruth[]
  6) Save JSONL → output/synthetic_ground_truth.jsonl
```

### Key concepts
- `ChunkData` (from your precomputed corpus): `{ doc_id, chunk_id, chunk_text, embedding }`
- `SelectionBundle`: `{ golden_chunks: ChunkData[], distractor_chunks: ChunkData[] }`
- `GeneratedQuery`: `{ query: str, query_type: str, golden_chunks: [...] }`
- `ValidatedGroundTruth`: `{ query, expected_doc_ids, context_chunks, validation }`

## Why these features

### Distractors (hard negatives)
- Force queries to target the correct content, not just any loosely related snippet.
- Increase difficulty and diagnostic power of ranking metrics by requiring the system to rank goldens above near misses.

### Deduplication
- Prevents redundant or trivially similar chunks from skewing selection and evaluation.
- Maintains a compact, meaningful set of goldens and distractors.

### Multi-golden (multi-document relevance)
- Real queries often have correct answers in multiple documents.
- Cluster mode selects additional goldens from different `doc_id`s above a similarity threshold so `expected_doc_ids` naturally includes all independently sufficient documents.

### Query-type variation
- Covers diverse user behaviors and system sensitivities: factual, keyword, exact snippet, web-like, misspellings, long/short lengths, concept-seeking, low-overlap paraphrases, comparison.
- Lets you bias toward types that stress BM25 or semantic retrieval.

### Multiple evaluation modes
- nonLLM: BM25 rank/margin + coverage gating; deterministic and fast.
- llm: answer using golden context + Ragas; semantic gating.
- hybrid: try nonLLM first, escalate a sample to LLM by `EVAL_LLM_SAMPLE_RATE`.

### Caching
- Large-scale generation can be slow/expensive; caching for data loading, selection, LLM calls, and evaluation improves rerun speed and reproducibility.

## Selection: single vs multi-golden

- Single-golden mode: one golden chunk per bundle + hard negative distractors.
- Multi-golden “cluster” mode:
  - For each seed chunk, fetch a larger neighbor pool via the search backend (FAISS/Azure Search).
  - Select additional goldens from different `doc_id`s whose similarity ≥ `GOLDEN_SIM_THRESHOLD` (e.g., cosine ≥ 0.92), up to `MAX_GOLDEN_DOCS`.
  - Distractors exclude those golden doc_ids and are drawn from the remaining neighbor pool.
  - If fewer than `GOLDEN_MIN_DOCS` equivalents are found, fall back to a single-golden bundle.

Rationale: A query should count as correct if any equivalent-answer document is retrieved. This mirrors real evaluation where multiple documents can independently satisfy the query.

## Query generation

- Azure OpenAI with type-specific prompts; cache-aware for speed.
- Types (controllable in config):
  - concept_seeking (abstract “why/how”)
  - exact_snippet (verbatim substring)
  - web_search_like (short search-engine style)
  - low_overlap (paraphrase with low lexical overlap)
  - fact_seeking (single clear answer)
  - keyword (identifier-only)
  - misspellings (typos/transpositions)
  - long / medium / short (token length control)
  - comparison (when multiple goldens exist)
- Per-type token caps and post-processing enforce output length.

## Evaluation

- Non-LLM: BM25 rank, margin, and coverage thresholds ensure goldens beat distractors lexically.
- LLM/Hybrid: answer using only goldens and score with Ragas; accept if above threshold (or based on config). Hybrid escalates by sample rate.

## Configuration guide (`evaluation_api/configs/generation_config.py`)

### Data
- `INPUT_TYPE`: "chunks"
- `INPUT_PATHS`: list of input files/dirs
- `EMBED_DIM`: embedding dimension check

### Selection and diversity
- `SELECTION_SAMPLE_MODE`: "chunks" | "documents"
- `SELECTION_SAMPLE_RATE`: sampling fraction when not targeting totals
- `SELECTION_TARGET_QUERIES`: desired total queries; selector estimates bundle count from query sampling settings
- `SELECTION_NUM_BUNDLES`: direct control of bundle count (overrides when >0)
- `SELECTOR_DEDUP_DOCS`: spread bundles across distinct documents
- `NUM_DISTRACTORS`: hard negatives per bundle

### Multi-golden (multi-document)
- `MULTI_GOLDEN_MODE`: "off" | "cluster"
- `GOLDEN_SIM_THRESHOLD`: similarity to accept as equivalent answer (e.g., 0.92)
- `MAX_GOLDEN_DOCS`: cap on goldens per bundle (e.g., 3–5)
- `GOLDEN_MIN_DOCS`: minimum to keep multi-golden, else fallback to single

### Query variation
- `QUERY_TYPES`: list of enabled types
- `QUERY_SAMPLING_MODE`: "all_per_bundle" | "sample_per_bundle"
- `MIN_QUERY_TYPES_PER_BUNDLE` / `MAX_QUERY_TYPES_PER_BUNDLE`: number of types sampled per bundle (when using sampling)
- `QUERY_TYPE_WEIGHTS`: weighting (e.g., {"keyword": 2.0, "web_search_like": 1.5})
- `QUERY_TYPE_MAX_TOKENS`: per-type caps (e.g., long=64, short=8)
- `QUERY_LENGTH_TARGETS`: token ranges for long/medium/short

### LLM and performance
- Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- Generation: `TEMPERATURE`, `MAX_TOKENS`, `LLM_TIMEOUT_SECONDS`
- Concurrency/rate: `LLM_MAX_WORKERS`, `LLM_BURST_CAPACITY`, `LLM_REFILL_RATE`

### Evaluation thresholds
- Non-LLM: `BM25_MIN_MARGIN`, `COVERAGE_MIN_GOLDEN`, `COVERAGE_GAP_MIN`
- LLM/Hybrid: `RAGAS_CONTEXT_RELEVANCE_THRESHOLD`, `EVAL_LLM_SAMPLE_RATE`

### Caching
- Master: `ENABLE_CACHING`, `CACHE_DIR`
- Component toggles: `CACHE_DATA_LOADING`, `CACHE_SELECTION`, `CACHE_LLM_QUERIES`, `CACHE_EVALUATION`

## Recipes

### Balanced, realistic dataset
- Enable multi-golden clustering:
```python
MULTI_GOLDEN_MODE = "cluster"
GOLDEN_SIM_THRESHOLD = 0.92
MAX_GOLDEN_DOCS = 3
GOLDEN_MIN_DOCS = 2
SELECTOR_DEDUP_DOCS = True
```
- Vary query styles per bundle:
```python
QUERY_SAMPLING_MODE = "sample_per_bundle"
MIN_QUERY_TYPES_PER_BUNDLE = 2
MAX_QUERY_TYPES_PER_BUNDLE = 3
QUERY_TYPES = [
  "concept_seeking", "exact_snippet", "web_search_like", "low_overlap",
  "fact_seeking", "keyword", "misspellings", "long", "medium", "short"
]
```
- Cost-aware validation:
```python
# hybrid mixes speed and quality; set sample rate as budget allows
EVAL_LLM_SAMPLE_RATE = 0.25
```

### BM25-leaning evaluation
```python
QUERY_TYPE_WEIGHTS = {"keyword": 2.0, "web_search_like": 1.5, "exact_snippet": 1.5, "low_overlap": 0.5}
BM25_MIN_MARGIN = 0.1
COVERAGE_MIN_GOLDEN = 0.4
COVERAGE_GAP_MIN = 0.05
```

### Semantic-retrieval stress test
```python
QUERY_TYPE_WEIGHTS = {"low_overlap": 2.0, "concept_seeking": 1.5, "long": 1.5}
EVAL_LLM_SAMPLE_RATE = 1.0
RAGAS_CONTEXT_RELEVANCE_THRESHOLD = 0.85
```

## Output format
Each line in `output/synthetic_ground_truth.jsonl` is a `ValidatedGroundTruth`:
```json
{
  "query": "why use semantic search for ranking?",
  "expected_doc_ids": ["doc-001", "doc-003"],
  "context_chunks": [
    {"doc_id": "doc-001", "chunk_id": "chunk-001", "chunk": "..."},
    {"doc_id": "doc-003", "chunk_id": "chunk-017", "chunk": "..."}
  ],
  "validation": {
    "query_type": "concept_seeking",
    "evaluation": "hybrid_llm",
    "bm25_margin": 0.72,
    "ragas_context_relevance": 0.91
  }
}
```

## Running
```bash
python -m evaluation_api.generation.cli --config /Users/venkata/ai/evaluation-api/evaluation_api/configs/generation_config.py --evaluation-mode hybrid
```

If you share your target dataset size, preferred query mix, and cost constraints, we can suggest concrete config values for a high-signal generation run.
