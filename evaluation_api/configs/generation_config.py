# --- File: search-evaluation-api/configs/generation_config.py ---
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
MAX_TOKEN_LENGTH = 350
# Cosine similarity threshold for flagging duplicates
DUPLICATE_COSINE_SIM = 0.98

# --- Context Selection ---
# How many queries to attempt to generate
# 1.0 = attempt to generate queries for 100% of valid chunks
SELECTION_SAMPLE_RATE = 1.0
# Number of "hard negative" distractors to find
NUM_DISTRACTORS = 3

# --- Query Generation ---
# Types of queries to generate per chunk
QUERY_TYPES = ["factual", "keyword"]
# Azure OpenAI config (for real implementation)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# --- Embeddings / Reproducibility ---
# Set to 1024 to match input embedding dimension
EMBED_DIM = 1024
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
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "8"))
# Approx QPS cap across workers
LLM_MAX_QPS = float(os.getenv("LLM_MAX_QPS", "2.0"))