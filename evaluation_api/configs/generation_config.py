# This file stores all the parameters for the generation pipeline.

import os
from dotenv import load_dotenv

# Load .env from repo root (two levels up) and current working directory, if present
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))
load_dotenv(os.path.join(os.getcwd(), ".env"))

# --- Data Input Configuration ---
# "chunks" or "documents"
# We will only implement the "chunks" path for now.
INPUT_TYPE = "chunks"

# Paths to your pre-computed chunks: can be JSONL files or directories of JSON/JSONL
# For local testing, point to the sample_data directory with 5 JSON files
INPUT_PATHS = ["/Users/venkata/ai/evaluation-api/evaluation_api/sample_data"]

# --- Data Output Configuration ---
OUTPUT_PATH = "./output/synthetic_ground_truth.jsonl"
REJECTED_CHUNKS_PATH = "./output/rejected_chunks.jsonl"
CACHE_PATH = ".cache/generation/"

# --- Chunk Validation Thresholds ---
MIN_TOKEN_LENGTH = 5
MAX_TOKEN_LENGTH = 5000
# Cosine similarity threshold for flagging duplicates
DUPLICATE_COSINE_SIM = 0.98

# --- Context Selection ---
# How many queries to attempt to generate (rate-based)
# 0.1 = attempt to generate queries for 10% of the selected unit (see SELECTION_SAMPLE_MODE)
SELECTION_SAMPLE_RATE = 1

# Choose sampling unit: "chunks" or "documents"
SELECTION_SAMPLE_MODE = os.getenv("SELECTION_SAMPLE_MODE", "chunks")

# Alternatively, request a fixed number of queries (overrides rate if > 0).
# When > 0, the selector targets ceil(SELECTION_TARGET_QUERIES / est_queries_per_bundle) bundles.
SELECTION_TARGET_QUERIES = int(os.getenv("SELECTION_TARGET_QUERIES", "20"))

# Or specify fixed number of bundles directly (overrides target-queries logic when > 0)
SELECTION_NUM_BUNDLES = int(os.getenv("SELECTION_NUM_BUNDLES", "0"))

# Prefer sampling bundles across distinct documents to increase diversity
SELECTOR_DEDUP_DOCS = bool(os.getenv("SELECTOR_DEDUP_DOCS", "True").lower() in ("true", "1", "yes"))

# --- Multi-golden selection (multi-document ground truth) ---
# Modes: off | cluster
MULTI_GOLDEN_MODE = os.getenv("MULTI_GOLDEN_MODE", "cluster")
# Cosine similarity threshold for considering another chunk as equivalent-answer
GOLDEN_SIM_THRESHOLD = float(os.getenv("GOLDEN_SIM_THRESHOLD", "0.92"))
# Max number of distinct documents to include as goldens
MAX_GOLDEN_DOCS = int(os.getenv("MAX_GOLDEN_DOCS", "5"))
# Minimum number of goldens (docs) to form a multi-golden bundle; otherwise fall back to single
GOLDEN_MIN_DOCS = int(os.getenv("GOLDEN_MIN_DOCS", "2"))

# Number of "hard negative" distractors to find
NUM_DISTRACTORS = 3

# --- Query Generation ---
# Types of queries to generate in the corpus. Supported values:
#   "concept_seeking", "exact_snippet", "web_search_like", "low_overlap",
#   "fact_seeking", "keyword", "misspellings", "long", "medium", "short",
#   "comparison"
# Users can override via environment variable QUERY_TYPES (comma-separated)
_DEFAULT_QUERY_TYPES = [
    "concept_seeking",
    "exact_snippet",
    "web_search_like",
    "low_overlap",
    "fact_seeking",
    "keyword",
    "misspellings",
    "long",
    "medium",
    "short",
]
_env_query_types = os.getenv("QUERY_TYPES", None)
if _env_query_types:
    QUERY_TYPES = [t.strip() for t in _env_query_types.split(",") if t.strip()]
else:
    QUERY_TYPES = _DEFAULT_QUERY_TYPES

# Query type sampling strategy per bundle:
#   - "all_per_bundle": generate every type in QUERY_TYPES per bundle
#   - "sample_per_bundle": sample a subset per bundle using weights/ranges (recommended)
QUERY_SAMPLING_MODE = os.getenv("QUERY_SAMPLING_MODE", "sample_per_bundle")

# For sample_per_bundle: range of query types per bundle
MIN_QUERY_TYPES_PER_BUNDLE = int(os.getenv("MIN_QUERY_TYPES_PER_BUNDLE", "2"))
MAX_QUERY_TYPES_PER_BUNDLE = int(os.getenv("MAX_QUERY_TYPES_PER_BUNDLE", "3"))

# Optional weight map for types (edit in code). If empty, uniform weights are used.
# Example: {"keyword": 2.0, "web_search_like": 1.5}
QUERY_TYPE_WEIGHTS = {}

# Per-type max tokens for LLM output (query length control)
# Users can override via env, e.g., QUERY_MAX_TOKENS_long=64
QUERY_TYPE_MAX_TOKENS = {
    "concept_seeking": 48,
    "exact_snippet": 96,
    "web_search_like": 16,
    "low_overlap": 48,
    "fact_seeking": 24,
    "keyword": 8,
    "misspellings": 24,
    "long": 64,
    "medium": 32,
    "short": 8,
    "comparison": 32,
}

# Length targets (token ranges) to guide prompts
QUERY_LENGTH_TARGETS = {
    "long": (21, 128),
    "medium": (5, 20),
    "short": (1, 4),
}

# Apply environment overrides for per-type max tokens, accepting both
# QUERY_MAX_TOKENS_<type> and QUERY_TYPE_MAX_TOKENS_<type> (case-sensitive)
for _qt, _default in list(QUERY_TYPE_MAX_TOKENS.items()):
    for _prefix in ("QUERY_MAX_TOKENS_", "QUERY_TYPE_MAX_TOKENS_"):
        _env_key_mixed = f"{_prefix}{_qt}"
        _env_key_upper = f"{_prefix}{_qt.upper()}"
        _val = os.getenv(_env_key_mixed, os.getenv(_env_key_upper, None))
        if _val is not None:
            try:
                QUERY_TYPE_MAX_TOKENS[_qt] = int(_val)
            except ValueError:
                pass

# Azure OpenAI config (for real implementation)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# --- Embeddings / Reproducibility ---
# Set to match input embedding dimension
EMBED_DIM = 512
SEED = 42

# --- Azure Blob (optional path) ---
# When INPUT_TYPE == "azure_blob_chunks", these are used
BLOB_ACCOUNT_URL = os.getenv("AZURE_BLOB_ACCOUNT_URL", "")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "")
BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "")
BLOB_MAX_WORKERS = int(os.getenv("AZURE_BLOB_MAX_WORKERS", "64"))

# --- Search Backend Selection ---
# Options: "faiss", "azure_search", "hybrid_search"
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "faiss")

# --- FAISS Backend Configuration ---
# Leave USE_IVF_SELECTION unset to allow auto-enable for large datasets (>=20k)
USE_IVF_SELECTION = None
IVF_NLIST = int(os.getenv("IVF_NLIST", "1024"))
IVF_NPROBE = int(os.getenv("IVF_NPROBE", "64"))

# --- Dedup tuning ---
DEDUP_MAX_NEIGHBORS = int(os.getenv("DEDUP_MAX_NEIGHBORS", "20"))
# Guard to avoid O(n^2) cosine fallback on large datasets if FAISS is missing
DEDUP_LARGE_GUARD_N = int(os.getenv("DEDUP_LARGE_GUARD_N", "50000"))

# --- Azure AI Search Backend Configuration ---
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")

# Azure Search field mappings
AZURE_SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "content_vector")
AZURE_SEARCH_CONTENT_FIELD = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "chunk_text")
AZURE_SEARCH_DOC_ID_FIELD = os.getenv("AZURE_SEARCH_DOC_ID_FIELD", "doc_id")
AZURE_SEARCH_CHUNK_ID_FIELD = os.getenv("AZURE_SEARCH_CHUNK_ID_FIELD", "chunk_id")

# Azure Search performance tuning
AZURE_SEARCH_BATCH_SIZE = int(os.getenv("AZURE_SEARCH_BATCH_SIZE", "100"))
AZURE_SEARCH_RETRY_ATTEMPTS = int(os.getenv("AZURE_SEARCH_RETRY_ATTEMPTS", "3"))
AZURE_SEARCH_TIMEOUT_SECONDS = int(os.getenv("AZURE_SEARCH_TIMEOUT_SECONDS", "30"))

# --- Evaluation Layer ---
# Minimum Ragas/DeepEval scores to accept a synthetic query
RAGAS_CONTEXT_RELEVANCE_THRESHOLD = 0.8
DEEPEVAL_FAITHFULNESS_THRESHOLD = 0.85

BM25_MIN_MARGIN = 0.5
COVERAGE_MIN_GOLDEN = 0.6
COVERAGE_GAP_MIN = 0.2
EVAL_LLM_SAMPLE_RATE = 0.1

# --- LLM Concurrency / Rate Limiting ---
# Max concurrent LLM calls; tune to your Azure OpenAI limits
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "16"))
# Approx QPS cap across workers - increase based on your Azure OpenAI quota
LLM_MAX_QPS = float(os.getenv("LLM_MAX_QPS", "10.0"))

# --- LLM Request Optimization ---
# Optimize for faster query generation
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))  # Lower = more consistent, faster
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32"))       # Shorter queries = faster generation
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "15"))  # Prevent hanging requests

# --- Advanced Rate Limiting ---
# Burst capacity for better throughput
LLM_BURST_CAPACITY = int(os.getenv("LLM_BURST_CAPACITY", "50"))  # Allow bursts up to 50 requests
LLM_REFILL_RATE = float(os.getenv("LLM_REFILL_RATE", "10.0"))    # Refill at 10 tokens/second

# --- Caching Configuration ---
# Enable caching for massive performance improvements
ENABLE_CACHING = bool(os.getenv("ENABLE_CACHING", "True").lower() in ("true", "1", "yes"))
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")

# Cache settings for different components
CACHE_DATA_LOADING = bool(os.getenv("CACHE_DATA_LOADING", "True").lower() in ("true", "1", "yes"))
CACHE_VALIDATION = bool(os.getenv("CACHE_VALIDATION", "True").lower() in ("true", "1", "yes"))
CACHE_SELECTION = bool(os.getenv("CACHE_SELECTION", "True").lower() in ("true", "1", "yes"))
CACHE_LLM_QUERIES = bool(os.getenv("CACHE_LLM_QUERIES", "True").lower() in ("true", "1", "yes"))
CACHE_EVALUATION = bool(os.getenv("CACHE_EVALUATION", "True").lower() in ("true", "1", "yes"))

# Cache management
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "168"))  # 1 week default
CACHE_MAX_SIZE_MB = int(os.getenv("CACHE_MAX_SIZE_MB", "1024"))  # 1GB default


