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
    def __init__(self, chunks: List[ChunkData], config):
        self.chunks = chunks
        self.config = config
        # Set deterministic seeds if provided
        seed = getattr(self.config, "SEED", None)
        if seed is not None:
            random.seed(seed)
            logger.info("ContextSelector seeded with SEED=%s", seed)
        
        # Initialize search backend
        self.backend = create_search_backend(chunks, config)
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
            
        num_to_sample = int(len(self.chunks) * self.config.SELECTION_SAMPLE_RATE)
        num_to_sample = max(1, min(num_to_sample, len(self.chunks)))
        logger.info("Attempting to select %s contexts for query generation using %s backend...", 
                   num_to_sample, self.backend.get_backend_info().get('backend', 'Unknown'))
        
        # Sample chunks for processing
        sampled_chunks = random.sample(self.chunks, num_to_sample)
        
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
