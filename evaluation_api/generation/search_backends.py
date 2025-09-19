# --- File: evaluation_api/generation/search_backends.py ---
# Abstract backend interface and implementations for similarity search

from abc import ABC, abstractmethod
import logging
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from .models import ChunkData

logger = logging.getLogger(__name__)


class SearchBackend(ABC):
    """Abstract base class for similarity search backends"""
    
    def __init__(self, chunks: List[ChunkData], config):
        self.chunks = chunks
        self.config = config
        self.chunk_lookup = {c.chunk_id: c for c in chunks}
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        raise NotImplementedError
    
    @abstractmethod
    def find_similar_chunks(self, chunk: ChunkData, k: int) -> List[ChunkData]:
        """Find k most similar chunks to the given chunk (excluding itself)"""
        raise NotImplementedError

    def find_similar_chunks_with_scores(self, chunk: ChunkData, k: int) -> List[Tuple[ChunkData, float]]:
        """Optional: Return (chunk, similarity) pairs; default uses find_similar_chunks without scores."""
        sims = self.find_similar_chunks(chunk, k)
        return [(c, 0.0) for c in sims]
    
    @abstractmethod
    def find_duplicates(self, similarity_threshold: float) -> List[Tuple[int, int]]:
        """Find pairs of chunk indices that are duplicates above threshold"""
        raise NotImplementedError
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend-specific information for logging"""
        raise NotImplementedError


class FAISSBackend(SearchBackend):
    """FAISS-based similarity search backend"""
    
    def __init__(self, chunks: List[ChunkData], config):
        super().__init__(chunks, config)
        self.index = None
        self.embeddings = None
        self.chunk_id_to_index = {}
        
    def initialize(self) -> bool:
        """Build FAISS index"""
        if not self.chunks:
            logger.warning("No chunks to index.")
            return False
            
        try:
            import faiss  # type: ignore
        except ImportError:
            logger.error("faiss not installed. Please install faiss-cpu")
            return False
        
        # Build embeddings matrix
        embeddings_list = [c.embedding for c in self.chunks]
        if not embeddings_list:
            logger.error("No embeddings available to index.")
            return False
            
        # Validate dimensions
        dim0 = len(embeddings_list[0])
        for idx, emb in enumerate(embeddings_list):
            if len(emb) != dim0:
                logger.error("Inconsistent embedding dimension at index %s: %s vs %s", idx, len(emb), dim0)
                return False

        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        # Enforce configured embedding dimension (fail fast on mismatch)
        expected_dim = getattr(self.config, 'EMBED_DIM', None)
        if expected_dim is not None and int(expected_dim) != self.embeddings.shape[1]:
            logger.error("Embedding dimension mismatch: data=%s, config.EMBED_DIM=%s", self.embeddings.shape[1], expected_dim)
            return False
        # Build O(1) lookup for chunk_id -> index
        self.chunk_id_to_index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        
        # L2-normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.embeddings = self.embeddings / norms
        dimension = self.embeddings.shape[1]
        
        # Choose index type based on configuration and dataset size
        # Auto-enable IVF for large datasets unless explicitly disabled
        use_ivf_cfg = getattr(self.config, 'USE_IVF_SELECTION', None)
        use_ivf = (len(self.chunks) >= 20000) if use_ivf_cfg is None else bool(use_ivf_cfg)
        if use_ivf and len(self.chunks) >= 20000:
            nlist = int(getattr(self.config, 'IVF_NLIST', min(4096, max(256, int(np.sqrt(len(self.chunks)))))))
            nprobe = int(getattr(self.config, 'IVF_NPROBE', min(64, max(8, nlist // 16))))
            logger.info("Building FAISS IVF index for %s vectors dim %s (nlist=%s, nprobe=%s)...", len(self.chunks), dimension, nlist, nprobe)
            
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = nprobe
        else:
            logger.info("Building FAISS IndexFlatIP for %s vectors of dim %s...", len(self.chunks), dimension)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)
            
        logger.info("FAISS index built successfully.")
        return True
    
    def find_similar_chunks(self, chunk: ChunkData, k: int) -> List[ChunkData]:
        """Find k most similar chunks using FAISS"""
        if self.index is None:
            logger.error("FAISS index not initialized")
            return []
            
        # Find chunk index via O(1) lookup
        chunk_idx = self.chunk_id_to_index.get(chunk.chunk_id)
                
        if chunk_idx is None:
            logger.warning("Chunk %s not found in index", chunk.chunk_id)
            return []
        
        # Prepare query vector from internal normalized embeddings to avoid relying on external chunk.embedding
        if self.embeddings is None:
            logger.error("FAISS embeddings matrix is not available")
            return []
        query_vector = self.embeddings[chunk_idx:chunk_idx+1]
        
        # Search for k+1 neighbors (including self)
        k_search = min(k + 1, len(self.chunks))
        try:
            _, indices = self.index.search(query_vector, k_search)
            
            similar_chunks = []
            for idx in indices[0]:
                if idx != chunk_idx and idx < len(self.chunks):  # Exclude self
                    similar_chunks.append(self.chunks[idx])
                if len(similar_chunks) >= k:
                    break
                    
            return similar_chunks
        except RuntimeError as e:
            logger.error("Error during FAISS search: %s", e)
            return []

    def find_similar_chunks_with_scores(self, chunk: ChunkData, k: int) -> List[Tuple[ChunkData, float]]:
        if self.index is None:
            logger.error("FAISS index not initialized")
            return []
        chunk_idx = self.chunk_id_to_index.get(chunk.chunk_id)
        if chunk_idx is None or self.embeddings is None:
            return []
        query_vector = self.embeddings[chunk_idx:chunk_idx+1]
        k_search = min(k + 1, len(self.chunks))
        try:
            sims, indices = self.index.search(query_vector, k_search)
            pairs: List[Tuple[ChunkData, float]] = []
            for sim, idx in zip(sims[0], indices[0]):
                if idx != chunk_idx and idx < len(self.chunks):
                    pairs.append((self.chunks[idx], float(sim)))
                if len(pairs) >= k:
                    break
            return pairs
        except RuntimeError as e:
            logger.error("Error during FAISS search: %s", e)
            return []
    
    def find_duplicates(self, similarity_threshold: float) -> List[Tuple[int, int]]:
        """Find duplicate pairs using FAISS"""
        if self.index is None or self.embeddings is None:
            logger.error("FAISS index not initialized")
            return []
            
        duplicates = []
        k_neighbors = min(getattr(self.config, 'DEDUP_MAX_NEIGHBORS', 20) + 1, len(self.chunks))
        
        try:
            similarities, indices = self.index.search(self.embeddings, k_neighbors)
            
            for i in range(len(self.chunks)):
                for j_idx, j in enumerate(indices[i]):
                    if j > i and similarities[i, j_idx] >= similarity_threshold:
                        duplicates.append((i, int(j)))
                        
        except RuntimeError as e:
            logger.error("Error finding duplicates with FAISS: %s", e)
            
        return duplicates
    
    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "FAISS",
            "index_type": type(self.index).__name__ if self.index else "None",
            "num_vectors": len(self.chunks),
            "vector_dim": self.embeddings.shape[1] if self.embeddings is not None else None
        }


class AzureSearchBackend(SearchBackend):
    """Azure AI Search-based similarity search backend"""
    
    def __init__(self, chunks: List[ChunkData], config):
        super().__init__(chunks, config)
        self.search_client = None
        
    def initialize(self) -> bool:
        """Initialize Azure Search client"""
        try:
            from azure.search.documents import SearchClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError:
            logger.error("Azure Search SDK not installed. Please install azure-search-documents")
            return False
            
        # Validate configuration
        endpoint = getattr(self.config, 'AZURE_SEARCH_ENDPOINT', None)
        index_name = getattr(self.config, 'AZURE_SEARCH_INDEX_NAME', None)
        key = getattr(self.config, 'AZURE_SEARCH_KEY', None)
        
        if not all([endpoint, index_name, key]):
            logger.error("Azure Search configuration incomplete. Need AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, AZURE_SEARCH_KEY")
            return False
        
        try:
            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(key)
            )
            
            # Test connection with a simple search
            list(self.search_client.search("", top=1))
            logger.info("Azure Search client initialized successfully")
            return True
            
        except (ImportError, ValueError, RuntimeError) as e:
            logger.error("Failed to initialize Azure Search client: %s", e)
            return False
    
    def find_similar_chunks(self, chunk: ChunkData, k: int) -> List[ChunkData]:
        """Find similar chunks using Azure AI Search vector search"""
        if self.search_client is None:
            logger.error("Azure Search client not initialized")
            return []
        
        vector_field = getattr(self.config, 'AZURE_SEARCH_VECTOR_FIELD', 'content_vector')
        chunk_id_field = getattr(self.config, 'AZURE_SEARCH_CHUNK_ID_FIELD', 'chunk_id')
        
        try:
            vector_query = {
                "value": chunk.embedding,
                "fields": vector_field,
                "k": k + 5  # Get extra for filtering
            }
            
            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                top=k + 5,
                select=[chunk_id_field, "@search.score"]
            )
            
            similar_chunks = []
            for result in results:
                result_chunk_id = result[chunk_id_field]
                # Skip if it's the same chunk
                if result_chunk_id == chunk.chunk_id:
                    continue
                # Look up in our chunk collection
                if result_chunk_id in self.chunk_lookup:
                    similar_chunk = self.chunk_lookup[result_chunk_id]
                    # Add search score as metadata
                    similar_chunk.validation_meta['search_score'] = result.get("@search.score", 0.0)
                    similar_chunks.append(similar_chunk)
                    
                if len(similar_chunks) >= k:
                    break
                    
            return similar_chunks
            
        except (ValueError, RuntimeError) as e:
            logger.error("Error during Azure Search vector search: %s", e)
            return []

    def find_similar_chunks_with_scores(self, chunk: ChunkData, k: int) -> List[Tuple[ChunkData, float]]:
        if self.search_client is None:
            logger.error("Azure Search client not initialized")
            return []
        vector_field = getattr(self.config, 'AZURE_SEARCH_VECTOR_FIELD', 'content_vector')
        chunk_id_field = getattr(self.config, 'AZURE_SEARCH_CHUNK_ID_FIELD', 'chunk_id')
        try:
            vector_query = {
                "value": chunk.embedding,
                "fields": vector_field,
                "k": k + 5
            }
            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                top=k + 5,
                select=[chunk_id_field, "@search.score"]
            )
            pairs: List[Tuple[ChunkData, float]] = []
            for result in results:
                result_chunk_id = result[chunk_id_field]
                if result_chunk_id == chunk.chunk_id:
                    continue
                if result_chunk_id in self.chunk_lookup:
                    similar_chunk = self.chunk_lookup[result_chunk_id]
                    score = float(result.get("@search.score", 0.0))
                    pairs.append((similar_chunk, score))
                if len(pairs) >= k:
                    break
            return pairs
        except (ValueError, RuntimeError) as e:
            logger.error("Error during Azure Search vector search: %s", e)
            return []
    
    def find_duplicates(self, similarity_threshold: float) -> List[Tuple[int, int]]:
        """Find duplicates using Azure AI Search (more complex implementation)"""
        if self.search_client is None:
            logger.error("Azure Search client not initialized")
            return []
        
        # Note: This is a simplified implementation
        # For large datasets, consider batch processing or different strategies
        duplicates = []
        vector_field = getattr(self.config, 'AZURE_SEARCH_VECTOR_FIELD', 'content_vector')
        chunk_id_field = getattr(self.config, 'AZURE_SEARCH_CHUNK_ID_FIELD', 'chunk_id')
        
        try:
            # Sample chunks for duplicate detection to avoid too many API calls
            sample_size = min(1000, len(self.chunks))
            sample_chunks = random.sample(self.chunks, sample_size)
            
            for chunk in tqdm(sample_chunks, desc="Checking duplicates via Azure Search"):
                vector_query = {
                    "value": chunk.embedding,
                    "fields": vector_field,
                    "k": 10
                }
                
                results = self.search_client.search(
                    search_text="",
                    vector_queries=[vector_query],
                    top=10,
                    select=[chunk_id_field, "@search.score"]
                )
                
                for result in results:
                    result_chunk_id = result[chunk_id_field]
                    if result_chunk_id != chunk.chunk_id:
                        # Convert Azure Search score to similarity
                        # Note: This conversion may need adjustment based on your search configuration
                        similarity = result.get("@search.score", 0.0)
                        if similarity >= similarity_threshold:
                            # Find indices in original chunk list
                            chunk_idx = next((idx for idx, c in enumerate(self.chunks) if c.chunk_id == chunk.chunk_id), None)
                            duplicate_idx = next((idx for idx, c in enumerate(self.chunks) if c.chunk_id == result_chunk_id), None)
                            if chunk_idx is not None and duplicate_idx is not None and chunk_idx != duplicate_idx:
                                duplicates.append((min(chunk_idx, duplicate_idx), max(chunk_idx, duplicate_idx)))
                
        except (ValueError, RuntimeError) as e:
            logger.error("Error finding duplicates with Azure Search: %s", e)
            
        # Remove duplicate pairs
        return list(set(duplicates))
    
    def get_backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "Azure AI Search",
            "endpoint": getattr(self.config, 'AZURE_SEARCH_ENDPOINT', 'Not configured'),
            "index_name": getattr(self.config, 'AZURE_SEARCH_INDEX_NAME', 'Not configured'),
            "num_chunks": len(self.chunks)
        }


class HybridSearchBackend(AzureSearchBackend):
    """Enhanced Azure Search backend with hybrid capabilities"""
    
    def find_similar_chunks(self, chunk: ChunkData, k: int) -> List[ChunkData]:
        """Find similar chunks using hybrid search (vector + text)"""
        if self.search_client is None:
            logger.error("Azure Search client not initialized")
            return []
        
        vector_field = getattr(self.config, 'AZURE_SEARCH_VECTOR_FIELD', 'content_vector')
        chunk_id_field = getattr(self.config, 'AZURE_SEARCH_CHUNK_ID_FIELD', 'chunk_id')
        
        # Extract key terms for text component
        key_terms = self._extract_key_terms(chunk.chunk_text)
        
        try:
            vector_query = {
                "value": chunk.embedding,
                "fields": vector_field,
                "k": k + 5
            }
            
            results = self.search_client.search(
                search_text=key_terms,  # Text component
                vector_queries=[vector_query],
                top=k + 10,
                search_mode="all",  # Hybrid mode
                select=[chunk_id_field, "@search.score"]
            )
            
            similar_chunks = []
            for result in results:
                result_chunk_id = result[chunk_id_field]
                if result_chunk_id == chunk.chunk_id:
                    continue
                    
                if result_chunk_id in self.chunk_lookup:
                    similar_chunk = self.chunk_lookup[result_chunk_id]
                    similar_chunk.validation_meta['hybrid_score'] = result.get("@search.score", 0.0)
                    similar_chunks.append(similar_chunk)
                    
                if len(similar_chunks) >= k:
                    break
                    
            return similar_chunks
            
        except (ValueError, RuntimeError) as e:
            logger.error("Error during hybrid search: %s", e)
            return []
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> str:
        """Extract key terms from text for hybrid search"""
        # Simple implementation - can be enhanced with NLP techniques
        import re
        from collections import Counter
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most common terms
        common_terms = Counter(words).most_common(max_terms)
        return ' '.join([term for term, _ in common_terms])
    
    def get_backend_info(self) -> Dict[str, Any]:
        info = super().get_backend_info()
        info["backend"] = "Hybrid Azure AI Search"
        return info


def create_search_backend(chunks: List[ChunkData], config) -> Optional[SearchBackend]:
    """Factory function to create appropriate search backend"""
    backend_type = getattr(config, 'SEARCH_BACKEND', 'faiss').lower()
    
    backend_map = {
        'faiss': FAISSBackend,
        'azure_search': AzureSearchBackend,
        'hybrid_search': HybridSearchBackend
    }
    
    if backend_type not in backend_map:
        logger.error("Unknown search backend: %s. Available: %s", backend_type, list(backend_map.keys()))
        return None
    
    logger.info("Creating %s search backend", backend_type)
    backend = backend_map[backend_type](chunks, config)
    
    if backend.initialize():
        logger.info("Search backend initialized: %s", backend.get_backend_info())
        return backend
    else:
        logger.error("Failed to initialize %s backend", backend_type)
        return None
