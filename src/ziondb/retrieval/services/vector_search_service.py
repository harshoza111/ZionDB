from __future__ import annotations
from typing import List, TYPE_CHECKING
from ziondb.index.vector_index import VectorIndex
from ziondb.index.models import IndexSearchResult

if TYPE_CHECKING:
    from ziondb.retrieval.search_context import SearchContext


class VectorSearchService:
    """Service responsible for executing the similarity search against the VectorIndex."""

    def __init__(self, vector_index: VectorIndex) -> None:
        """
        Initialize the VectorSearchService.

        Args:
            vector_index: The VectorIndex instance to query.
        """
        self.vector_index = vector_index

    def search(self, context: SearchContext) -> List[IndexSearchResult]:
        """
        Searches the vector index using the provided SearchContext.

        Args:
            context: The SearchContext wrapping query vector and filters.

        Returns:
            List[IndexSearchResult]: The list of matched record IDs and similarity scores.
        """
        return self.vector_index.search(context)
