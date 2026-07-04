from abc import ABC, abstractmethod
from typing import List
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult


class SearchExecutor(ABC):
    """Abstract base class representing a search execution strategy."""

    @abstractmethod
    def execute(self, request: SearchRequest) -> List[SearchResult]:
        """
        Executes the search strategy for a SearchRequest.

        Args:
            request: The SearchRequest containing query parameters.

        Returns:
            List[SearchResult]: The final post-processed matches.
        """
        pass
