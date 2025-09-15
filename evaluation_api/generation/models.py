# --- File: search-evaluation-api/generation/models.py ---
# We need standardized dataclasses to pass data between modules.

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ChunkData:
    """Represents a single, validated chunk of data with its embedding."""
    doc_id: str
    chunk_id: str
    chunk_text: str
    embedding: List[float]
    # Store validation results
    validation_meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SelectionBundle:
    """A bundle of chunks selected for generating a single query."""
    golden_chunks: List[ChunkData]
    distractor_chunks: List[ChunkData]

@dataclass
class GeneratedQuery:
    """A query generated from a bundle, before RAG evaluation."""
    query: str
    golden_chunks: List[ChunkData]
    query_type: str # e.g., "factual", "keyword"

@dataclass
class ValidatedGroundTruth:
    """The final, validated data object, ready to be written to JSONL."""
    query: str
    expected_doc_ids: List[str]
    context_chunks: List[Dict[str, str]]
    validation: Dict[str, Any]

    # Add __dict__ for easy JSON serialization in the CLI
    def __dict__(self):
        return {
            "query": self.query,
            "expected_doc_ids": self.expected_doc_ids,
            "context_chunks": self.context_chunks,
            "validation": self.validation,
        }

# --- End File: search-evaluation-api/generation/models.py ---
