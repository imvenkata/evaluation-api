# --- File: search-evaluation-api/generation/chunk_selector.py ---
# This module builds a vector index to find "golden" chunks 
# and their "distractor" (hard negative) neighbors.

import logging
import numpy as np
import faiss
import random
from typing import List
from tqdm import tqdm

from .models import ChunkData, SelectionBundle

logger = logging.getLogger(__name__)

class ContextSelector:
    def __init__(self, chunks: List[ChunkData], config):
        self.chunks = chunks
        self.config = config
        self.index = self._build_index()

    def _build_index(self):
        """Builds a FAISS index for fast nearest neighbor search."""
        if not self.chunks:
            logger.warning("No chunks to index.")
            return None
            
        try:
            embeddings = np.array([c.embedding for c in self.chunks]).astype('float32')
            dimension = embeddings.shape[1]
        except Exception as e:
            logger.error(f"Error creating embedding matrix: {e}. Chunks might have inconsistent embedding dimensions.")
            return None
            
        # Using IndexFlatIP (Inner Product) which is equivalent to Cosine Similarity for normalized vectors
        logger.info(f"Building FAISS IndexFlatIP for {len(self.chunks)} vectors of dim {dimension}...")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        logger.info("FAISS index built.")
        return index

    def select_contexts(self) -> List[SelectionBundle]:
        """
        Selects golden chunks and finds their distractors.
        For now, we implement single-golden-chunk selection.
        """
        if not self.index:
            logger.error("FAISS index not available. Cannot select contexts.")
            return []
            
        num_to_sample = int(len(self.chunks) * self.config.SELECTION_SAMPLE_RATE)
        logger.info(f"Attempting to select {num_to_sample} contexts for query generation...")
        
        # Sample unique chunk indices
        if num_to_sample > len(self.chunks):
            num_to_sample = len(self.chunks)
            
        sampled_indices = random.sample(range(len(self.chunks)), num_to_sample)
        
        bundles = []
        for i in tqdm(sampled_indices, desc="Selecting contexts"):
            golden_chunk = self.chunks[i]
            
            # Search for k + 1 neighbors (k distractors + the item itself)
            k_neighbors = self.config.NUM_DISTRACTORS + 1
            if k_neighbors > len(self.chunks):
                k_neighbors = len(self.chunks) # Cannot request more neighbors than chunks

            query_vector = np.array([golden_chunk.embedding]).astype('float32')
            
            try:
                # D = distances (cosine similarities), I = indices
                D, I = self.index.search(query_vector, k_neighbors)
                
                distractor_chunks = []
                for j in I[0]:
                    # The first item (j=i) is the golden chunk itself. Skip it.
                    if j == i:
                        continue
                    distractor_chunks.append(self.chunks[j])
                
                # Ensure we only have the number of distractors requested
                distractor_chunks = distractor_chunks[:self.config.NUM_DISTRACTORS]
                
                if len(distractor_chunks) < self.config.NUM_DISTRACTORS:
                    logger.debug(f"Chunk {golden_chunk.chunk_id} found < {self.config.NUM_DISTRACTORS} distractors.")

                bundles.append(
                    SelectionBundle(
                        golden_chunks=[golden_chunk],
                        distractor_chunks=distractor_chunks
                    )
                )
            except Exception as e:
                logger.warning(f"Error during FAISS search for chunk {i}: {e}")
                
        logger.info(f"Created {len(bundles)} selection bundles.")
        return bundles

# --- End File: search-evaluation-api/generation/chunk_selector.py ---
