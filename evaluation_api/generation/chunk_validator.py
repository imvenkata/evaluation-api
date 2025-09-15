# --- File: search-evaluation-api/generation/chunk_validator.py ---
# This module filters out low-quality or duplicate chunks.

import logging
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .models import ChunkData

logger = logging.getLogger(__name__)

def validate_chunks(chunks: List[ChunkData], config) -> Tuple[List[ChunkData], List[ChunkData]]:
    """
    Applies heuristic and similarity-based validation to filter chunks.
    """
    logger.info(f"Starting validation for {len(chunks)} chunks...")
    
    # 1. Heuristic Validation
    valid_chunks = []
    rejected_chunks = []
    for chunk in tqdm(chunks, desc="Applying heuristics"):
        len_tokens = len(chunk.chunk_text.split())
        if not (config.MIN_TOKEN_LENGTH <= len_tokens <= config.MAX_TOKEN_LENGTH):
            chunk.validation_meta['reject_reason'] = f"Token length ({len_tokens}) out of bounds."
            rejected_chunks.append(chunk)
            continue
        
        valid_chunks.append(chunk)

    logger.info(f"Heuristics passed: {len(valid_chunks)}, rejected: {len(rejected_chunks)}")

    # 2. Duplicate Validation
    # TODO: This O(n^2) check is slow for >10k chunks. 
    # Replace with an approximate nearest neighbor (ANN) index if performance is an issue.
    if not valid_chunks:
        return [], rejected_chunks
        
    logger.info("Checking for near-duplicates (this may take a while)...")
    embeddings = np.array([c.embedding for c in valid_chunks])
    
    # Handle case where embeddings might be empty or malformed
    if embeddings.size == 0:
        logger.warning("No embeddings found in valid chunks. Skipping duplicate check.")
        return valid_chunks, rejected_chunks
        
    try:
        sim_matrix = cosine_similarity(embeddings)
    except Exception as e:
        logger.error(f"Error computing cosine similarity: {e}. Skipping duplicate check.")
        return valid_chunks, rejected_chunks

    
    duplicate_indices = set()
    for i in tqdm(range(len(valid_chunks)), desc="Finding duplicates"):
        if i in duplicate_indices:
            continue
        # Find items with similarity > threshold (sim_matrix[i, i] is 1.0)
        duplicate_matches = np.where(sim_matrix[i] > config.DUPLICATE_COSINE_SIM)[0]
        
        for j in duplicate_matches:
            if i != j:
                duplicate_indices.add(j)

    final_valid_chunks = []
    for i, chunk in enumerate(valid_chunks):
        if i in duplicate_indices:
            chunk.validation_meta['reject_reason'] = "Near-duplicate chunk."
            rejected_chunks.append(chunk)
        else:
            chunk.validation_meta['status'] = "Validated"
            final_valid_chunks.append(chunk)

    logger.info(f"Duplicate check complete. Final valid chunks: {len(final_valid_chunks)}")
    return final_valid_chunks, rejected_chunks

# --- End File: search-evaluation-api/generation/chunk_validator.py ---
