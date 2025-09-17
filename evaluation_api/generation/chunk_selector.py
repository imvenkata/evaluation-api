# --- File: search-evaluation-api/generation/chunk_selector.py ---
# This module builds a vector index to find "golden" chunks 
# and their "distractor" (hard negative) neighbors.

import logging
import numpy as np
import random
from typing import List
from tqdm import tqdm

from .models import ChunkData, SelectionBundle

logger = logging.getLogger(__name__)

class ContextSelector:
    def __init__(self, chunks: List[ChunkData], config):
        self.chunks = chunks
        self.config = config
        # Set deterministic seeds if provided
        seed = getattr(self.config, "SEED", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            logger.info("ContextSelector seeded with SEED=%s", seed)
        self.index = self._build_index()

    def _build_index(self):
        """Builds a FAISS index for fast nearest neighbor search."""
        if not self.chunks:
            logger.warning("No chunks to index.")
            return None
        try:
            # Local import to avoid linter error when faiss isn't installed in dev env
            import faiss  # type: ignore
        except ImportError:
            logger.error("faiss is not installed. Please install faiss-cpu to use ContextSelector.")
            return None
            
        # Build embeddings matrix and validate dimensions
        embeddings_list = [c.embedding for c in self.chunks]
        if not embeddings_list:
            logger.error("No embeddings available to index.")
            return None
        dim0 = len(embeddings_list[0])
        for idx, emb in enumerate(embeddings_list):
            if len(emb) != dim0:
                logger.error("Inconsistent embedding dimension at index %s: %s vs %s", idx, len(emb), dim0)
                return None

        embeddings = np.array(embeddings_list, dtype=np.float32)
        # L2-normalize embeddings so Inner Product ≈ Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
        dimension = embeddings.shape[1]
            
        # Using IndexFlatIP (Inner Product) which is equivalent to Cosine Similarity for normalized vectors
        use_ivf = bool(getattr(self.config, 'USE_IVF_SELECTION', False))
        if use_ivf and len(self.chunks) >= 20000:
            nlist = int(getattr(self.config, 'IVF_NLIST', min(4096, max(256, int(np.sqrt(len(self.chunks)))))))
            nprobe = int(getattr(self.config, 'IVF_NPROBE', min(64, max(8, nlist // 16))))
            logger.info("Building FAISS IVF index for %s vectors dim %s (nlist=%s, nprobe=%s)...", len(self.chunks), dimension, nlist, nprobe)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = nprobe
        else:
            logger.info("Building FAISS IndexFlatIP for %s vectors of dim %s...", len(self.chunks), dimension)
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
        num_to_sample = max(1, num_to_sample)
        logger.info("Attempting to select %s contexts for query generation...", num_to_sample)
        
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
            # Normalize query vector to match index normalization
            qn = np.linalg.norm(query_vector, axis=1, keepdims=True)
            qn = np.maximum(qn, 1e-12)
            query_vector = query_vector / qn
            
            try:
                # distances (cosine sims), indices
                _, I = self.index.search(query_vector, k_neighbors)
                
                distractor_chunks = []
                for j in I[0]:
                    # The first item (j=i) is the golden chunk itself. Skip it.
                    if j == i:
                        continue
                    distractor_chunks.append(self.chunks[j])
                
                # Ensure we only have the number of distractors requested
                distractor_chunks = distractor_chunks[:self.config.NUM_DISTRACTORS]
                
                if len(distractor_chunks) < self.config.NUM_DISTRACTORS:
                    logger.debug("Chunk %s found < %s distractors.", golden_chunk.chunk_id, self.config.NUM_DISTRACTORS)

                bundles.append(
                    SelectionBundle(
                        golden_chunks=[golden_chunk],
                        distractor_chunks=distractor_chunks
                    )
                )
            except RuntimeError as e:
                logger.warning("Error during FAISS search for chunk %s: %s", i, e)
                
        logger.info("Created %s selection bundles.", len(bundles))
        return bundles

# --- End File: search-evaluation-api/generation/chunk_selector.py ---
