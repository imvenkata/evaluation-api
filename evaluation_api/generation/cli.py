# --- File: evaluation_api/generation/cli.py ---
# This is the main orchestrator you will run from your terminal.

import argparse
import logging
import importlib.util
import sys
import os
import json
from typing import List

# Import all the modules in the pipeline
from . import data_loader
from . import chunk_validator
from . import chunk_selector
from . import query_generator
from . import evaluation_layer
from .models import ValidatedGroundTruth, ChunkData

# Module-level logger
logger = logging.getLogger("generation.cli")

# --- Configuration ---
def load_config(config_path):
    """Loads a Python config file from a path."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location("generation_config", config_path)
    if spec is None:
        logger.error("Could not load config spec from %s", config_path)
        sys.exit(1)
        
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# --- Utility ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def save_dataset(dataset: List[ValidatedGroundTruth], path: str):
    """Saves the final dataset to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item.to_dict()) + "\n")
    logger.info("Final dataset saved to %s", path)

def save_rejected(chunks: List[ChunkData], path: str):
    """Saves the rejected chunks for auditing."""
    if not path or not chunks:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "reject_reason": chunk.validation_meta.get('reject_reason', 'Unknown')
            }) + "\n")
    logger.info("Rejected chunks log saved to %s", path)


# --- Main Orchestration ---
def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Ground Truth Generation Pipeline for Search Evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the generation_config.py file"
    )
    parser.add_argument(
        "--evaluation-mode",
        type=str,
        default="llm",
        choices=["none", "nonllm", "llm", "hybrid"],
        help="Evaluation strategy: none|nonllm|llm|hybrid"
    )
    args = parser.parse_args()

    setup_logging()

    logger.info("--- Starting Synthetic Ground Truth Pipeline ---")
    
    # 1. Load Config
    logger.info("Loading configuration from %s...", args.config)
    config = load_config(args.config)

    # 2. Load Data
    all_chunks = data_loader.load_data(config)
    if not all_chunks:
        logger.error("No data loaded. Exiting.")
        return

    # 3. Validate Chunks (and build/reuse search backend)
    valid_chunks, rejected_chunks, backend = chunk_validator.validate_chunks(all_chunks, config)
    save_rejected(rejected_chunks, config.REJECTED_CHUNKS_PATH)
    if not valid_chunks:
        logger.error("No valid chunks after validation. Exiting.")
        return

    # Optional: Drop per-chunk embeddings to reduce memory when using FAISS backend
    try:
        be_info = backend.get_backend_info() if backend else {}
        if be_info.get("backend") == "FAISS":
            for c in valid_chunks:
                # Remove external embedding copy; FAISS backend holds normalized matrix
                c.embedding = []  # type: ignore[assignment]
            logger.info("Dropped per-chunk embeddings to reduce memory (FAISS in use).")
    except Exception:
        # Best-effort; continue if anything goes wrong
        pass

    # 4. Select Contexts (reuse backend if available)
    selector = chunk_selector.ContextSelector(valid_chunks, config, backend=backend)
    logger.info("Using search backend: %s", selector.get_backend_info())
    bundles = selector.select_contexts()
    if not bundles:
        logger.error("No context bundles selected. Exiting.")
        return

    # 5. Generate Queries (with caching)
    q_generator = query_generator.QueryGenerator(config)
    generated_queries = q_generator.generate_queries(bundles)
    if not generated_queries:
        logger.error("No queries were generated. Exiting.")
        return
    
    # Log cache statistics
    cache_stats = q_generator.get_cache_stats()
    if cache_stats.get("caching") != "disabled":
        logger.info("LLM Cache Stats: %d files, %.2f MB", 
                   cache_stats.get("file_count", 0), 
                   cache_stats.get("total_size_mb", 0))

    # 6. Evaluate Queries (configurable)
    final_dataset = evaluation_layer.evaluate_queries(generated_queries, config, args.evaluation_mode)
    if not final_dataset:
        logger.error("No queries passed final evaluation. No dataset will be saved.")
        return

    # 7. Save Final Dataset
    save_dataset(final_dataset, config.OUTPUT_PATH)
    
    logger.info("--- Pipeline Completed Successfully ---")
    logger.info("Generated %s high-quality QA pairs.", len(final_dataset))


if __name__ == "__main__":
# This allows you to run:
# python -m search_evaluation_api.generation.cli --config ./search-evaluation-api/configs/generation_config.py
    main()

# --- End File: search-evaluation-api/generation/cli.py ---
