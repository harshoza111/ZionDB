from typing import List
from ziondb.retrieval.postprocessing.post_processor import PostProcessor
from ziondb.retrieval.search_result import SearchResult


class IdentityPostProcessor(PostProcessor):
    """A baseline post-processor that returns search results completely unmodified."""

    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Returns the input results directly without performing any changes.

        Args:
            results: The input SearchResult matches.

        Returns:
            List[SearchResult]: The same SearchResult list unmodified.
        """
        return results
