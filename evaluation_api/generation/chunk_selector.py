# --- File: search-evaluation-api/generation/chunk_selector.py ---
# This module builds a vector index to find "golden" chunks 
# and their "distractor" (hard negative) neighbors.

import logging
import random
from typing import List
from tqdm import tqdm

from .models import ChunkData, SelectionBundle
from .search_backends import create_search_backend

logger = logging.getLogger(__name__)

class ContextSelector:
    def __init__(self, chunks: List[ChunkData], config, backend=None):
        self.chunks = chunks
        self.config = config
        # Set deterministic seeds if provided
        seed = getattr(self.config, "SEED", None)
        if seed is not None:
            random.seed(seed)
            logger.info("ContextSelector seeded with SEED=%s", seed)
        
        # Initialize or reuse search backend
        self.backend = backend if backend is not None else create_search_backend(chunks, config)
        if self.backend is None:
            raise RuntimeError("Failed to initialize search backend")

    def get_backend_info(self):
        """Get information about the current search backend"""
        return self.backend.get_backend_info() if self.backend else {"backend": "None"}

    def select_contexts(self) -> List[SelectionBundle]:
        """
        Selects golden chunks and finds their distractors using the configured backend.
        """
        if not self.backend:
            logger.error("Search backend not available. Cannot select contexts.")
            return []
            
        # Determine target number of bundles
        direct_bundles = int(getattr(self.config, "SELECTION_NUM_BUNDLES", 0) or 0)
        if direct_bundles > 0:
            target_bundles = direct_bundles
        else:
            # Estimate bundles from target queries and per-bundle query sampling
            target_queries = int(getattr(self.config, "SELECTION_TARGET_QUERIES", 0) or 0)
            if target_queries > 0:
                sampling_mode = str(getattr(self.config, "QUERY_SAMPLING_MODE", "all_per_bundle")).lower()
                if sampling_mode == "sample_per_bundle":
                    min_q = int(getattr(self.config, "MIN_QUERY_TYPES_PER_BUNDLE", 1))
                    max_q = int(getattr(self.config, "MAX_QUERY_TYPES_PER_BUNDLE", max(1, len(getattr(self.config, "QUERY_TYPES", ["keyword"])))))
                    avg_q = max(1, int(round((min_q + max_q) / 2)))
                    queries_per_bundle = avg_q
                else:
                    queries_per_bundle = max(1, len(getattr(self.config, "QUERY_TYPES", ["keyword"])) )
                target_bundles = max(1, (target_queries + queries_per_bundle - 1) // queries_per_bundle)
            else:
                target_bundles = None

        # Determine sampling unit
        sample_mode = getattr(self.config, "SELECTION_SAMPLE_MODE", "chunks").lower()
        sample_rate = float(getattr(self.config, "SELECTION_SAMPLE_RATE", 0.1))
        dedup_docs = bool(getattr(self.config, "SELECTOR_DEDUP_DOCS", True))

        if sample_mode == "documents" or dedup_docs:
            # Group chunks by doc_id, sample documents, then pick one representative chunk per doc
            from collections import defaultdict
            doc_to_chunks = defaultdict(list)
            for c in self.chunks:
                doc_to_chunks[c.doc_id].append(c)
            doc_ids = list(doc_to_chunks.keys())
            if target_bundles is not None:
                num_docs = min(target_bundles, len(doc_ids))
            else:
                num_docs = max(1, min(int(len(doc_ids) * sample_rate), len(doc_ids)))
            if num_docs > len(doc_ids):
                num_docs = len(doc_ids)
            sampled_docs = random.sample(doc_ids, num_docs) if doc_ids else []
            sampled_chunks = [random.choice(doc_to_chunks[d]) for d in sampled_docs]
        else:
            # Chunk-based sampling
            total = len(self.chunks)
            if target_bundles is not None:
                num_to_sample = min(target_bundles, total)
            else:
                num_to_sample = max(1, min(int(total * sample_rate), total))
            sampled_chunks = random.sample(self.chunks, num_to_sample) if total else []

        logger.info("Attempting to select %s contexts (mode=%s) using %s backend...",
                   len(sampled_chunks), sample_mode, self.backend.get_backend_info().get('backend', 'Unknown'))
        
        bundles = []
        mode = str(getattr(self.config, "MULTI_GOLDEN_MODE", "off")).lower()
        sim_thr = float(getattr(self.config, "GOLDEN_SIM_THRESHOLD", 0.92))
        max_golden_docs = int(getattr(self.config, "MAX_GOLDEN_DOCS", 3))
        min_golden_docs = int(getattr(self.config, "GOLDEN_MIN_DOCS", 2))

        for golden_chunk in tqdm(sampled_chunks, desc="Selecting contexts"):
            try:
                if mode == "cluster":
                    # Get a larger pool to choose both multi-goldens and distractors
                    k_pool = max(20, self.config.NUM_DISTRACTORS * 4)
                    # Try to use scores if backend supports it
                    try:
                        pairs = self.backend.find_similar_chunks_with_scores(golden_chunk, k_pool)  # type: ignore[attr-defined]
                    except AttributeError:
                        neighbors = self.backend.find_similar_chunks(golden_chunk, k_pool)
                        pairs = [(c, 0.0) for c in neighbors]

                    # Select additional goldens from different docs above similarity threshold
                    golden_docs = {golden_chunk.doc_id}
                    golden_list = [golden_chunk]
                    distractor_pool: List[ChunkData] = []
                    for candidate, sim in pairs:
                        if candidate.doc_id in golden_docs:
                            continue
                        if sim >= sim_thr and len(golden_docs) < max_golden_docs:
                            golden_docs.add(candidate.doc_id)
                            golden_list.append(candidate)
                        else:
                            distractor_pool.append(candidate)
                    # If we didn't reach minimum golden docs, fall back to single-golden
                    if len(golden_docs) < min_golden_docs:
                        golden_list = [golden_chunk]
                        # Recompute distractors from top-k pool
                        distractor_pool = [c for c, _ in pairs]

                    # Now pick distractors excluding golden docs
                    distractors: List[ChunkData] = []
                    for cand in distractor_pool:
                        if cand.doc_id not in golden_docs:
                            distractors.append(cand)
                        if len(distractors) >= self.config.NUM_DISTRACTORS:
                            break
                    if len(distractors) < self.config.NUM_DISTRACTORS:
                        logger.debug("Chunk %s found only %s/%s distractors (cluster mode).",
                                     golden_chunk.chunk_id, len(distractors), self.config.NUM_DISTRACTORS)

                    bundles.append(SelectionBundle(golden_chunks=golden_list, distractor_chunks=distractors))
                else:
                    # Single-golden mode (existing behavior)
                    distractor_chunks = self.backend.find_similar_chunks(
                        golden_chunk,
                        self.config.NUM_DISTRACTORS
                    )
                    if len(distractor_chunks) < self.config.NUM_DISTRACTORS:
                        logger.debug("Chunk %s found only %s/%s distractors.",
                                     golden_chunk.chunk_id, len(distractor_chunks), self.config.NUM_DISTRACTORS)
                    bundles.append(SelectionBundle(golden_chunks=[golden_chunk], distractor_chunks=distractor_chunks))
            except (ValueError, RuntimeError) as e:
                logger.warning("Error finding distractors for chunk %s: %s", golden_chunk.chunk_id, e)
                
        logger.info("Created %s selection bundles.", len(bundles))
        return bundles

# --- End File: search-evaluation-api/generation/chunk_selector.py ---
