# --- File: search-evaluation-api/generation/data_loader.py ---
# This module implements the logic for loading data based on the config.

import json
import logging
from typing import List
from .models import ChunkData

logger = logging.getLogger(__name__)

def load_data(config) -> List[ChunkData]:
    """
    Loads data based on the INPUT_TYPE specified in the config.
    """
    if config.INPUT_TYPE == "chunks":
        return _load_from_chunks(config.INPUT_PATHS)
    elif config.INPUT_TYPE == "documents":
        raise NotImplementedError(
            "Document processing is not yet implemented. "
            "Please set INPUT_TYPE to 'chunks' in your config."
        )
    else:
        raise ValueError(f"Unknown INPUT_TYPE: {config.INPUT_TYPE}")

def _load_from_chunks(input_paths: List[str]) -> List[ChunkData]:
    """Loads pre-computed chunks from a list of JSONL files."""
    chunks = []
    for path in input_paths:
        logger.info(f"Loading pre-computed chunks from {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        chunk = ChunkData(
                            doc_id=data['doc_id'],
                            chunk_id=data['chunk_id'],
                            chunk_text=data['chunk'],
                            embedding=data['embedding']
                        )
                        chunks.append(chunk)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed line in {path}: {e}")
        except FileNotFoundError:
            logger.error(f"Input file not found: {path}")
            raise
    
    if not chunks:
        logger.error("No chunks were loaded. Please check INPUT_PATHS.")
        
    return chunks

# --- End File: search-evaluation-api/generation/data_loader.py ---
