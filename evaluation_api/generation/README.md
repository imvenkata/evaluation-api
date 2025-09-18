### Synthetic Ground Truth Generation for Search Evaluation

This module builds high-quality synthetic ground-truth data to evaluate Azure AI Search ranking and retrieval. It generates a dataset of query → expected_doc_ids pairs (plus context and validation metadata) using a validation-first, distractor-aware pipeline.

---

## Why this exists

- **Problem**: Real evaluation sets are scarce, skew simple, and inflate metrics. Most generators produce easy queries that any vector index can match via keyword overlap.
- **Solution**: A two-gate pipeline that:
  - Pre-validates chunks (length/duplicates) to ensure quality inputs.
  - Selects “golden” chunks with hard negative “distractors” using FAISS.
  - Generates queries that are answerable only by the golden chunk(s), not by distractors.
  - Post-validates generated pairs with RAG evaluation metrics before acceptance.

This yields harder, more discriminative evaluation data that better measures Azure AI Search performance (e.g., NDCG, MRR).

---

## End-to-end pipeline

```text
[configs/generation_config.py]
        │
        ▼
[generation/cli.py]
  1) data_loader.load_data → ChunkData[]
  2) chunk_validator.validate_chunks → (valid[], rejected[])
  3) chunk_selector.ContextSelector(valid).select_contexts() → SelectionBundle[]
  4) query_generator.QueryGenerator.generate_queries(bundles) → GeneratedQuery[] (stub)
  5) evaluation_layer.evaluate_queries(queries) → ValidatedGroundTruth[] (stub)
  6) Save JSONL to output/synthetic_ground_truth.jsonl
```

---

## Technologies and rationale

- **FAISS (faiss-cpu)**: High-performance nearest neighbor search to find hard negatives; we use `IndexFlatIP` on L2-normalized vectors so inner-product ≈ cosine similarity.
- **NumPy / scikit-learn**: Array ops and cosine similarity for duplicate detection.
- **OpenAI (Azure OpenAI)**: LLM for query generation (stubbed now). Will use `AZURE_OPENAI_KEY`, endpoint, and deployment from config.
- **Ragas / DeepEval**: Post-generation evaluation to gate outputs by faithfulness and context relevance (stubbed now).
- **LlamaIndex**: Planned for document ingestion/chunking when `INPUT_TYPE="documents"` path is implemented.
- **argparse / tqdm / logging**: CLI orchestration, progress bars, and structured logs.

Design choices:
- Validation-first to avoid wasting compute on junk inputs.
- Distractor-aware selection to stress ranking precision.
- Config-driven for reproducibility and portability across repos.

---

## Data model (contracts)

- `ChunkData`: `{ doc_id: str, chunk_id: str, chunk_text: str, embedding: float[] }`
- `SelectionBundle`: `{ golden_chunks: ChunkData[], distractor_chunks: ChunkData[] }`
- `GeneratedQuery`: `{ query: str, golden_chunks: ChunkData[], query_type: str }`
- `ValidatedGroundTruth`: `{ query: str, expected_doc_ids: str[], context_chunks: {doc_id,chunk_id,chunk}[], validation: {…} }`

Input formats supported by `data_loader`:
- JSONL file(s) with keys: `doc_id`, `chunk_id`, `chunk` (or `chunk_text`), `embedding` (or `content_vector`).
- Directory of `.json` or `.jsonl` files with the same schema. `content_vector` is auto-mapped to `embedding`.

Output format (`output/synthetic_ground_truth.jsonl`): one `ValidatedGroundTruth` per line.

---

## Core methods

### 1) Loading (`data_loader.py`)
- Accepts `INPUT_PATHS` of files or directories.
- Reads `.jsonl` line-by-line or `.json` whole-file.
- Maps `content_vector` → `embedding` when needed.
- Verifies consistent embedding dimensionality; warns if it diverges from `EMBED_DIM`.

### 2) Validation (`chunk_validator.py`)
- Heuristics: token length bounds (`MIN_TOKEN_LENGTH`, `MAX_TOKEN_LENGTH`).
- Duplicate detection: O(n^2) cosine similarity with threshold `DUPLICATE_COSINE_SIM` (good for small/medium sets). Future enhancement: approximate NN for scalability.

### 3) Selection (`chunk_selector.py`)
- L2-normalizes all embeddings and builds `IndexFlatIP` (cosine-equivalent over normalized vectors).
- For each sampled golden chunk (controlled by `SELECTION_SAMPLE_RATE`), searches top-k neighbors and picks `NUM_DISTRACTORS` non-self distractors.
- Deterministic when `SEED` is provided.

### 4) Query generation (`query_generator.py`)
- Current: STUB. Produces mock queries by inspecting golden chunk text.
- Planned: Azure OpenAI call with a “distractor-aware” prompt so queries are only answerable by golden context.
  - Reads secrets from env: `AZURE_OPENAI_KEY`.
  - Reads endpoint/model from config: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`.

### 5) Evaluation (`evaluation_layer.py`)
- Current: STUB. Assigns random scores and filters by thresholds.
- Planned: Use Ragas (Context Relevance, Faithfulness) and DeepEval checks; persist per-metric scores in `validation`.

---

## Configuration (`evaluation_api/configs/generation_config.py`)

Required/commonly used keys:
- Data: `INPUT_TYPE` ("chunks"), `INPUT_PATHS`, `EMBED_DIM`
- Validation: `MIN_TOKEN_LENGTH`, `MAX_TOKEN_LENGTH`, `DUPLICATE_COSINE_SIM`
- Selection: `SELECTION_SAMPLE_RATE`, `NUM_DISTRACTORS`, `SEED`
- Generation: `QUERY_TYPES`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- Evaluation: `RAGAS_CONTEXT_RELEVANCE_THRESHOLD`, `DEEPEVAL_FAITHFULNESS_THRESHOLD`
- Output: `OUTPUT_PATH`, `REJECTED_CHUNKS_PATH`

Example (tailored to sample data):

```python
INPUT_TYPE = "chunks"
INPUT_PATHS = ["/Users/venkata/ai/evaluation-api/evaluation_api/sample_data"]

OUTPUT_PATH = "./output/synthetic_ground_truth.jsonl"
REJECTED_CHUNKS_PATH = "./output/rejected_chunks.jsonl"

MIN_TOKEN_LENGTH = 5
MAX_TOKEN_LENGTH = 350
DUPLICATE_COSINE_SIM = 0.98

SELECTION_SAMPLE_RATE = 1.0
NUM_DISTRACTORS = 3
QUERY_TYPES = ["factual", "keyword"]

AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "<your-deployment>"

EMBED_DIM = 512
SEED = 42

RAGAS_CONTEXT_RELEVANCE_THRESHOLD = 0.8
DEEPEVAL_FAITHFULNESS_THRESHOLD = 0.85
```

Secrets:
- Provide your Azure key via environment variable: `AZURE_OPENAI_KEY`.

```bash
export AZURE_OPENAI_KEY="<your-key>"
```

---

## Usage

Run the pipeline via CLI:

```bash
python -m evaluation_api.generation.cli --config /Users/venkata/ai/evaluation-api/evaluation_api/configs/generation_config.py
```

Artifacts:
- Final dataset: `./output/synthetic_ground_truth.jsonl`
- Rejected chunks log: `./output/rejected_chunks.jsonl`

---

## Azure AI Search alignment

- The output `expected_doc_ids` are derived from the golden chunks’ `doc_id`s and intended to be used as the target set for query relevance evaluation (e.g., NDCG@k, MRR@k) against Azure AI Search.
- The “distractor-aware” generation makes queries that require the index to rank the correct document(s) above semantically similar near-misses, highlighting ranking nuance.
- Ensure your Azure Search indexing uses the same embedding model/preprocessing as the chunks used here.

---

## Performance & scaling notes

- Duplicate detection is O(n^2); for large corpora, replace with ANN-based dedup or block by locality with FAISS.
- FAISS can be swapped to GPU or IVF/HNSW for speed/recall tradeoffs.
- Memory: `IndexFlatIP` stores all vectors; approximate indices reduce memory.

---

## Troubleshooting

- FAISS not installed: `pip install faiss-cpu`.
- Embedding dim mismatch: Check `EMBED_DIM` and source embeddings.
- No bundles selected: Lower `SELECTION_SAMPLE_RATE` bounds checks or confirm there are valid chunks.
- Few validated pairs: Relax Ragas/DeepEval thresholds or adjust prompts.

---

## Roadmap

- Replace stubs with:
  - Azure OpenAI generation with temperature/top_p sweeps and diversity checks.
  - Ragas/DeepEval integration and per-metric logging.
- Advanced validation: stopword ratio, language ID, readability checks.
- Scalable dedup (ANN), multi-golden-chunk tasks, and richer query types (comparison, temporal, multi-hop).

---

## Licenses & credits

This module builds upon open-source packages cited above. Respect respective licenses when moving this into your production repo.


