from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ziondb.storage.record import ChunkRecord
from ziondb.index.models import SearchResult

class VectorIndex(ABC):
    """Abstract interface defining the operations contract for a vector index search engine."""

    @abstractmethod
    def insert(self, record: ChunkRecord) -> None:
        """
        Inserts a single record's vector representation into the index.

        Args:
            record: The ChunkRecord containing the ID and vector to index.
        """
        pass

    @abstractmethod
    def remove(self, record_id: str) -> None:
        """
        Removes a record's vector representation from the index.

        Args:
            record_id: The ID of the record to remove.
        """
        pass

    @abstractmethod
    def update(self, record: ChunkRecord) -> None:
        """
        Updates an existing record's vector representation in the index.

        Args:
            record: The updated ChunkRecord.
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        """
        Searches the index for the top-k most similar records relative to the query vector.

        Args:
            query_vector: The query embedding vector (1D numpy array).
            top_k: The maximum number of search results to return.

        Returns:
            List[SearchResult]: A sorted list of the closest matches.
        """
        pass

    @abstractmethod
    def rebuild(self) -> None:
        """Completely rebuilds the index from the underlying storage provider."""
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Returns the total number of items currently indexed.

        Returns:
            int: The index size.
        """
        pass
