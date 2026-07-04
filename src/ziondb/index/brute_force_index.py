from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from ziondb.index.vector_index import VectorIndex
from ziondb.index.models import IndexSearchResult
from ziondb.index.similarity import SimilarityMetric, CosineSimilarity
from ziondb.storage.record import ChunkRecord
from ziondb.storage.storage import RecordProvider
from ziondb.storage.exceptions import RecordNotFoundError

if TYPE_CHECKING:
    from ziondb.retrieval.search_context import SearchContext

logger = logging.getLogger(__name__)

class BruteForceIndex(VectorIndex):
    """A baseline, brute-force vector search index implementing the VectorIndex interface."""

    def __init__(
        self, 
        record_provider: RecordProvider, 
        metric: Optional[SimilarityMetric] = None
    ) -> None:
        """
        Initialize the BruteForceIndex.

        Args:
            record_provider: A read-only provider to access stored records.
            metric: An optional similarity calculator. Defaults to CosineSimilarity.
        """
        self.record_provider = record_provider
        self.metric = metric if metric is not None else CosineSimilarity()
        
        # Primary in-memory index mapping record ID to its vector embedding
        self._id_to_vector: Dict[str, np.ndarray] = {}
        
        # In-memory cache mapping record ID to its metadata dictionary for fast filtering
        self._id_to_metadata: Dict[str, Dict[str, Any]] = {}

    def insert(self, record: ChunkRecord) -> None:
        """
        Inserts a single record's vector representation and metadata into the index.

        Args:
            record: The ChunkRecord to index.
        """
        # Note: If the record is already stored, we overwrite the vector in the index.
        self._id_to_vector[record.id] = record.embedding
        self._id_to_metadata[record.id] = record.metadata or {}
        logger.debug(f"Indexed vector and metadata for record '{record.id}' in BruteForceIndex.")

    def remove(self, record_id: str) -> None:
        """
        Removes a record's vector representation and metadata from the index.

        Args:
            record_id: The ID of the record to remove.

        Raises:
            RecordNotFoundError: If the ID does not exist in the index.
        """
        if record_id not in self._id_to_vector:
            raise RecordNotFoundError(f"Record with ID '{record_id}' not found in the index.")
        
        del self._id_to_vector[record_id]
        if record_id in self._id_to_metadata:
            del self._id_to_metadata[record_id]
        logger.debug(f"Removed record '{record_id}' from BruteForceIndex.")

    def update(self, record: ChunkRecord) -> None:
        """
        Updates an existing record's vector representation and metadata in the index.

        Args:
            record: The updated ChunkRecord.

        Raises:
            RecordNotFoundError: If the record ID does not exist in the index.
        """
        if record.id not in self._id_to_vector:
            raise RecordNotFoundError(f"Record with ID '{record.id}' not found in the index.")

        self._id_to_vector[record.id] = record.embedding
        self._id_to_metadata[record.id] = record.metadata or {}
        logger.debug(f"Updated vector and metadata for record '{record.id}' in BruteForceIndex.")

    def search(self, context: SearchContext) -> List[IndexSearchResult]:
        """
        Searches the index for similar vectors, applying metadata filtering before similarity computation.

        Args:
            context: The SearchContext wrapping query parameters.

        Returns:
            List[IndexSearchResult]: A sorted list of the closest matches, ordered from best to worst.
        """
        if context.top_k <= 0 or not self._id_to_vector:
            return []

        # Use overridden similarity metric if provided in the context, otherwise index default
        metric = context.similarity_metric if context.similarity_metric is not None else self.metric
        filter_expr = context.filter_expression

        results: List[IndexSearchResult] = []

        # Iterate over every indexed vector and compute similarity score if filter passes
        for record_id, vector in self._id_to_vector.items():
            # Evaluate metadata filter expression before similarity calculation
            if filter_expr is not None:
                record_metadata = self._id_to_metadata.get(record_id, {})
                if not filter_expr.evaluate(record_metadata):
                    continue

            score = metric.calculate(context.query_vector, vector)
            results.append(IndexSearchResult(record_id=record_id, score=score))

        # Sort based on metric direction (highest score first for cosine/dot product, lowest first for distance)
        sorted_results = sorted(
            results,
            key=lambda x: x.score,
            reverse=metric.greater_is_better
        )

        return sorted_results[:context.top_k]

    def rebuild(self) -> None:
        """Completely rebuilds the index from the underlying storage provider."""
        logger.info("Rebuilding BruteForceIndex from the underlying RecordProvider...")
        self._id_to_vector.clear()
        self._id_to_metadata.clear()
        
        for record in self.record_provider.iterate():
            self._id_to_vector[record.id] = record.embedding
            self._id_to_metadata[record.id] = record.metadata or {}

        logger.info(f"Rebuild finished. Indexed {len(self._id_to_vector)} vectors.")

    def size(self) -> int:
        """
        Returns the total number of items currently indexed.

        Returns:
            int: The index size.
        """
        return len(self._id_to_vector)
