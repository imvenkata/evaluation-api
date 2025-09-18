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
        target_queries = int(getattr(self.config, "SELECTION_TARGET_QUERIES", 0) or 0)
        queries_per_bundle = max(1, len(getattr(self.config, "QUERY_TYPES", ["factual"])) )
        if target_queries > 0:
            target_bundles = max(1, (target_queries + queries_per_bundle - 1) // queries_per_bundle)
        else:
            target_bundles = None

        # Determine sampling unit
        sample_mode = getattr(self.config, "SELECTION_SAMPLE_MODE", "chunks").lower()
        sample_rate = float(getattr(self.config, "SELECTION_SAMPLE_RATE", 0.1))

        if sample_mode == "documents":
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
            sampled_docs = random.sample(doc_ids, num_docs)
            sampled_chunks = [random.choice(doc_to_chunks[d]) for d in sampled_docs]
        else:
            # Chunk-based sampling
            if target_bundles is not None:
                num_to_sample = min(target_bundles, len(self.chunks))
            else:
                num_to_sample = max(1, min(int(len(self.chunks) * sample_rate), len(self.chunks)))
            sampled_chunks = random.sample(self.chunks, num_to_sample)

        logger.info("Attempting to select %s contexts (mode=%s) using %s backend...",
                   len(sampled_chunks), sample_mode, self.backend.get_backend_info().get('backend', 'Unknown'))
        
        bundles = []
        for golden_chunk in tqdm(sampled_chunks, desc="Selecting contexts"):
            try:
                # Use backend to find similar chunks (distractors)
                distractor_chunks = self.backend.find_similar_chunks(
                    golden_chunk, 
                    self.config.NUM_DISTRACTORS
                )
                
                if len(distractor_chunks) < self.config.NUM_DISTRACTORS:
                    logger.debug("Chunk %s found only %s/%s distractors.", 
                               golden_chunk.chunk_id, len(distractor_chunks), self.config.NUM_DISTRACTORS)

                bundles.append(
                    SelectionBundle(
                        golden_chunks=[golden_chunk],
                        distractor_chunks=distractor_chunks
                    )
                )
            except (ValueError, RuntimeError) as e:
                logger.warning("Error finding distractors for chunk %s: %s", golden_chunk.chunk_id, e)
                
        logger.info("Created %s selection bundles.", len(bundles))
        return bundles

# --- End File: search-evaluation-api/generation/chunk_selector.py ---
