from abc import ABC, abstractmethod
from typing import List
from ziondb.retrieval.search_result import SearchResult


class PostProcessor(ABC):
    """Abstract base class for all post-processing stages of retrieved search results."""

    @abstractmethod
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Processes a list of hydrated SearchResult objects.

        Args:
            results: The list of retrieved and hydrated SearchResult matches.

        Returns:
            List[SearchResult]: The post-processed list of matches.
        """
        pass
