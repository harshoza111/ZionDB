from typing import List
from ziondb.core.interfaces import EmbeddingProvider
from ziondb.index.vector_index import VectorIndex
from ziondb.storage.storage import RecordProvider
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult

class Retriever:
    """Orchestrates similarity search by embedding queries, searching the index, and fetching records."""

    def __init__(
        self,
        vector_index: VectorIndex,
        record_provider: RecordProvider,
        embedding_provider: EmbeddingProvider
    ) -> None:
        """
        Initialize the Retriever.

        Args:
            vector_index: The vector index to search.
            record_provider: The storage record provider to fetch complete ChunkRecords.
            embedding_provider: The embedding provider to embed text queries.
        """
        self.vector_index = vector_index
        self.record_provider = record_provider
        self.embedding_provider = embedding_provider

    def retrieve(self, request: SearchRequest) -> List[SearchResult]:
        """
        Retrieves search results for a SearchRequest.

        Args:
            request: The SearchRequest containing the query and settings.

        Returns:
            List[SearchResult]: A list of matches with ChunkRecords and similarity scores.
        """
        if not request.query.strip():
            return []

        # 1. Generate the query embedding
        query_embeddings = self.embedding_provider.embed([request.query])
        if not query_embeddings:
            return []
        query_vector = query_embeddings[0]

        # 2. Call VectorIndex.search()
        index_results = self.vector_index.search(query_vector, request.top_k)

        # 3. Retrieve ChunkRecords from RecordProvider and assemble results
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
