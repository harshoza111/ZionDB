from typing import List
from ziondb.storage.storage import RecordProvider
from ziondb.index.models import IndexSearchResult
from ziondb.retrieval.search_result import SearchResult


class HydrationService:
    """Service responsible for loading detailed ChunkRecords from storage for search matches."""

    def __init__(self, record_provider: RecordProvider) -> None:
        """
        Initialize the HydrationService.

        Args:
            record_provider: The read-only storage provider to retrieve records.
        """
        self.record_provider = record_provider

    def hydrate(self, index_results: List[IndexSearchResult]) -> List[SearchResult]:
        """
        Retrieves full ChunkRecords for each index match and constructs public SearchResults.

        Args:
            index_results: The list of raw ID matches returned by the index.

        Returns:
            List[SearchResult]: The list of hydrated search results with full record details.
        """
        results: List[SearchResult] = []
        for index_res in index_results:
            try:
                record = self.record_provider.get(index_res.record_id)
                results.append(
                    SearchResult(
                        record=record,
                        score=index_res.score
                    )
                )
            except Exception:
                # Handle gracefully if storage and index are temporarily out-of-sync
                continue
        return results
