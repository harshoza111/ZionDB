import logging
from typing import Dict, List, Optional
import numpy as np

from ziondb.index.vector_index import VectorIndex
from ziondb.index.models import SearchResult
from ziondb.index.similarity import SimilarityMetric, CosineSimilarity
from ziondb.storage.record import ChunkRecord
from ziondb.storage.storage import RecordProvider
from ziondb.storage.exceptions import RecordNotFoundError

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

    def insert(self, record: ChunkRecord) -> None:
        """
        Inserts a single record's vector representation into the index.

        Args:
            record: The ChunkRecord to index.
        """
        # Note: If the record is already stored, we overwrite the vector in the index.
        self._id_to_vector[record.id] = record.embedding
        logger.debug(f"Indexed vector for record '{record.id}' in BruteForceIndex.")

    def remove(self, record_id: str) -> None:
        """
        Removes a record's vector representation from the index.

        Args:
            record_id: The ID of the record to remove.

        Raises:
            RecordNotFoundError: If the ID does not exist in the index.
        """
        if record_id not in self._id_to_vector:
            raise RecordNotFoundError(f"Record with ID '{record_id}' not found in the index.")
        
        del self._id_to_vector[record_id]
        logger.debug(f"Removed record '{record_id}' vector from BruteForceIndex.")

    def update(self, record: ChunkRecord) -> None:
        """
        Updates an existing record's vector representation in the index.

        Args:
            record: The updated ChunkRecord.

        Raises:
            RecordNotFoundError: If the record ID does not exist in the index.
        """
        if record.id not in self._id_to_vector:
            raise RecordNotFoundError(f"Record with ID '{record.id}' not found in the index.")

        self._id_to_vector[record.id] = record.embedding
        logger.debug(f"Updated vector for record '{record.id}' in BruteForceIndex.")

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        """
        Searches the index for the top-k most similar records relative to the query vector.

        Args:
            query_vector: The query embedding vector (1D numpy array).
            top_k: The maximum number of search results to return.

        Returns:
            List[SearchResult]: A sorted list of the closest matches, ordered from best to worst.
        """
        if top_k <= 0 or not self._id_to_vector:
            return []

        results: List[SearchResult] = []

        # Iterate over every indexed vector and compute similarity score
        for record_id, vector in self._id_to_vector.items():
            score = self.metric.calculate(query_vector, vector)
            results.append(SearchResult(record_id=record_id, score=score))

        # Sort based on metric direction (highest score first for cosine/dot product, lowest first for distance)
        sorted_results = sorted(
            results,
            key=lambda x: x.score,
            reverse=self.metric.greater_is_better
        )

        return sorted_results[:top_k]

    def rebuild(self) -> None:
        """Completely rebuilds the index from the underlying storage provider."""
        logger.info("Rebuilding BruteForceIndex from the underlying RecordProvider...")
        self._id_to_vector.clear()
        
        for record in self.record_provider.iterate():
            self._id_to_vector[record.id] = record.embedding

        logger.info(f"Rebuild finished. Indexed {len(self._id_to_vector)} vectors.")

    def size(self) -> int:
        """
        Returns the total number of items currently indexed.

        Returns:
            int: The index size.
        """
        return len(self._id_to_vector)
