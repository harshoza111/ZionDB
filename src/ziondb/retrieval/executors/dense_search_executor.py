from typing import List
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult
from ziondb.retrieval.executors.search_executor import SearchExecutor
from ziondb.retrieval.services.query_embedding_service import QueryEmbeddingService
from ziondb.retrieval.services.vector_search_service import VectorSearchService
from ziondb.retrieval.services.hydration_service import HydrationService
from ziondb.retrieval.postprocessing.post_processor import PostProcessor


class DenseSearchExecutor(SearchExecutor):
    """Executes the standard dense vector similarity search pipeline."""

    def __init__(
        self,
        query_embedding_service: QueryEmbeddingService,
        vector_search_service: VectorSearchService,
        hydration_service: HydrationService,
        post_processor: PostProcessor
    ) -> None:
        """
        Initialize the DenseSearchExecutor with its required retrieval services.

        Args:
            query_embedding_service: Service to embed text query and contextualize.
            vector_search_service: Service to query the vector index.
            hydration_service: Service to fetch ChunkRecord objects.
            post_processor: Processor to adjust, sort, or filter hydrated results.
        """
        self.query_embedding_service = query_embedding_service
        self.vector_search_service = vector_search_service
        self.hydration_service = hydration_service
        self.post_processor = post_processor

    def execute(self, request: SearchRequest) -> List[SearchResult]:
        """
        Executes similarity search for a SearchRequest using standard services.

        Args:
            request: The SearchRequest containing query parameters.

        Returns:
            List[SearchResult]: The final post-processed matches.
        """
        if not request.query.strip():
            return []

        # 1. Embed query text and construct search context
        context = self.query_embedding_service.embed_and_contextualize(request)

        # 2. Search index
        index_results = self.vector_search_service.search(context)

        # 3. Hydrate matching index IDs into records
        hydrated_results = self.hydration_service.hydrate(index_results)

        # 4. Post-process hydrated matches
        final_results = self.post_processor.process(hydrated_results)

        return final_results
