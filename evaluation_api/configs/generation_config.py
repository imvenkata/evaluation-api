# --- File: search-evaluation-api/configs/generation_config.py ---
# This file stores all the parameters for the generation pipeline.

# --- Data Input Configuration ---
# "chunks" or "documents"
# We will only implement the "chunks" path for now.
INPUT_TYPE = "chunks"

# Path to your pre-computed chunks (JSONL format)
# Assumes each line is: {"doc_id": "...", "chunk_id": "...", "chunk": "...", "embedding": [...]}
INPUT_PATHS = ["./data/precomputed_chunks.jsonl"]

# --- Data Output Configuration ---
OUTPUT_PATH = "./output/synthetic_ground_truth.jsonl"
REJECTED_CHUNKS_PATH = "./output/rejected_chunks.jsonl"
CACHE_PATH = ".cache/generation/"

# --- Chunk Validation Thresholds ---
MIN_TOKEN_LENGTH = 20
MAX_TOKEN_LENGTH = 1024
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
AZURE_OPENAI_ENDPOINT = "YOUR_AOAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"

# --- Evaluation Layer ---
# Minimum Ragas/DeepEval scores to accept a synthetic query
RAGAS_CONTEXT_RELEVANCE_THRESHOLD = 0.8
DEEPEVAL_FAITHFULNESS_THRESHOLD = 0.85

# --- End File: search-evaluation-api/configs/generation_config.py ---
