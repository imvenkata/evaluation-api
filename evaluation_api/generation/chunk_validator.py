# --- File: search-evaluation-api/generation/chunk_validator.py ---
# This module filters out low-quality or duplicate chunks.

import logging
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .models import ChunkData

logger = logging.getLogger(__name__)

def validate_chunks(
    chunks: List[ChunkData],
    config,
    backend: Optional[Any] = None,
) -> Tuple[List[ChunkData], List[ChunkData], Optional[Any]]:
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

    # 2. Duplicate Validation using search backend
    if not valid_chunks:
        return [], rejected_chunks, backend

    # Check if we should use search backend for duplicate detection
    use_backend_dedup = getattr(config, 'USE_BACKEND_DUPLICATE_DETECTION', True)
    if use_backend_dedup:
        final_valid, final_rejected, be = validate_duplicates_with_backend(valid_chunks, rejected_chunks, config, backend)
        return final_valid, final_rejected, be
    else:
        v, r = validate_duplicates_legacy(valid_chunks, rejected_chunks, config)
        return v, r, backend


def validate_duplicates_with_backend(
    valid_chunks: List[ChunkData],
    rejected_chunks: List[ChunkData],
    config,
    backend: Optional[Any] = None,
) -> Tuple[List[ChunkData], List[ChunkData], Optional[Any]]:
    """Use search backend for duplicate detection"""
    from .search_backends import create_search_backend
    
    logger.info("Checking for near-duplicates using search backend...")
    
    # Create backend for duplicate detection if not provided
    be = backend or create_search_backend(valid_chunks, config)
    if be is None:
        logger.warning("Failed to create search backend for duplicate detection, falling back to legacy method")
        v, r = validate_duplicates_legacy(valid_chunks, rejected_chunks, config)
        return v, r, None
    
    try:
        duplicate_pairs = be.find_duplicates(config.DUPLICATE_COSINE_SIM)
        
        # Mark duplicates for removal (keep first occurrence)
        duplicate_indices = set()
        for i, j in duplicate_pairs:
            duplicate_indices.add(j)  # Remove the second occurrence
        
        final_valid_chunks = []
        for i, chunk in enumerate(valid_chunks):
            if i in duplicate_indices:
                chunk.validation_meta['reject_reason'] = f"Near-duplicate chunk (via {be.get_backend_info().get('backend', 'search backend')})"
                rejected_chunks.append(chunk)
            else:
                chunk.validation_meta['status'] = "Validated"
                final_valid_chunks.append(chunk)
        
        logger.info("Duplicate check complete via %s. Final valid chunks: %s", 
                   be.get_backend_info().get('backend', 'search backend'), len(final_valid_chunks))
        return final_valid_chunks, rejected_chunks, be
        
    except Exception as e:
        logger.error("Error in backend duplicate detection: %s. Falling back to legacy method.", e)
        v, r = validate_duplicates_legacy(valid_chunks, rejected_chunks, config)
        return v, r, be


def validate_duplicates_legacy(valid_chunks: List[ChunkData], rejected_chunks: List[ChunkData], config) -> Tuple[List[ChunkData], List[ChunkData]]:
    """Legacy FAISS/cosine-based duplicate detection"""
    logger.info("Checking for near-duplicates using legacy method...")
    embeddings = np.array([c.embedding for c in valid_chunks], dtype=np.float32)
    if embeddings.size == 0:
        logger.warning("No embeddings found in valid chunks. Skipping duplicate check.")
        return valid_chunks, rejected_chunks

    # Normalize for cosine via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings_norm = embeddings / norms

    duplicate_indices = set()
    use_faiss = True
    try:
        import importlib
        faiss = importlib.import_module("faiss")  # type: ignore

        num = embeddings_norm.shape[0]
        dim = embeddings_norm.shape[1]

        # Choose index type based on dataset size
        if num >= 20000:
            # IVF for large datasets
            nlist = min(4096, max(64, int(np.sqrt(num))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            try:
                index.train(embeddings_norm)
                index.add(embeddings_norm)
                index.nprobe = min(64, nlist)
            except Exception:  # noqa: BLE001
                logger.warning("IVF training failed or insufficient data; falling back to Flat index.")
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings_norm)
        else:
            # Flat for small/medium datasets
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings_norm)

        # Search top-k neighbors for each vector
        k_neighbors = min(getattr(config, 'DEDUP_MAX_NEIGHBORS', 20) + 1, num)
        sims, inds = index.search(embeddings_norm, k_neighbors)

        # Keep-first policy: for each i, mark j>i with sim>threshold as duplicate
        thr = float(getattr(config, 'DUPLICATE_COSINE_SIM', 0.98))
        for i in tqdm(range(num), desc="Finding duplicates (ANN)"):
            if i in duplicate_indices:
                continue
            for j_idx, j in enumerate(inds[i]):
                if j == i:
                    continue
                if j < 0:
                    continue
                if sims[i, j_idx] >= thr:
                    if j > i:
                        duplicate_indices.add(int(j))
    except Exception:  # noqa: BLE001
        # Guard: on large datasets, abort instead of O(n^2) fallback
        num = embeddings_norm.shape[0]
        large_guard = int(getattr(config, 'DEDUP_LARGE_GUARD_N', 50000))
        if num >= large_guard:
            logger.error("FAISS unavailable for dedup and dataset is large (n=%s). Aborting to avoid O(n^2) computation.", num)
            raise RuntimeError("Dedup requires FAISS for large datasets; install faiss-cpu or lower dataset size.")
        use_faiss = False
        logger.warning("FAISS unavailable; falling back to O(n^2) cosine duplicate check for n=%s (below guard=%s).", num, large_guard)
        try:
            sim_matrix = cosine_similarity(embeddings_norm)
            for i in tqdm(range(len(valid_chunks)), desc="Finding duplicates (cosine)"):
                if i in duplicate_indices:
                    continue
                duplicate_matches = np.where(sim_matrix[i] >= getattr(config, 'DUPLICATE_COSINE_SIM', 0.98))[0]
                for j in duplicate_matches:
                    if i != j and j > i:
                        duplicate_indices.add(j)
        except Exception as e:  # noqa: BLE001
            logger.error("Error computing cosine similarity: %s. Skipping duplicate check.", e)
            return valid_chunks, rejected_chunks

    final_valid_chunks = []
    for i, chunk in enumerate(valid_chunks):
        if i in duplicate_indices:
            chunk.validation_meta['reject_reason'] = "Near-duplicate chunk."
            rejected_chunks.append(chunk)
        else:
            chunk.validation_meta['status'] = "Validated"
            final_valid_chunks.append(chunk)

    logger.info("Duplicate check complete (%s). Final valid chunks: %s", "FAISS" if use_faiss else "cosine", len(final_valid_chunks))
    return final_valid_chunks, rejected_chunks

# --- End File: search-evaluation-api/generation/chunk_validator.py ---
