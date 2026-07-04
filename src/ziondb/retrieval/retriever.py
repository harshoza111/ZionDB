from typing import List
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult
from ziondb.retrieval.executors.search_executor import SearchExecutor


class Retriever:
    """Lightweight coordinator that delegates similarity search to a SearchExecutor strategy."""

    def __init__(self, search_executor: SearchExecutor) -> None:
        """
        Initialize the Retriever with a search execution strategy.

        Args:
            search_executor: The execution strategy (e.g. DenseSearchExecutor).
        """
        self.search_executor = search_executor

    def retrieve(self, request: SearchRequest) -> List[SearchResult]:
        """
        Retrieves search results by delegating to the search executor.

        Args:
            request: The SearchRequest containing query parameters.

        Returns:
            List[SearchResult]: The final Matches.
        """
        return self.search_executor.execute(request)
