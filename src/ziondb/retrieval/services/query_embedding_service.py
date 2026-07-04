from typing import Optional
from ziondb.core.interfaces import EmbeddingProvider
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_context import SearchContext
from ziondb.retrieval.filter import parse_filter
from ziondb.index.similarity import CosineSimilarity


class QueryEmbeddingService:
    """Service responsible for generating query embeddings and constructing SearchContext."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        """
        Initialize the QueryEmbeddingService.

        Args:
            embedding_provider: Provider to generate vector embeddings from queries.
        """
        self.embedding_provider = embedding_provider

    def embed_and_contextualize(self, request: SearchRequest) -> SearchContext:
        """
        Embeds the query text and packages search arguments into a SearchContext.

        Args:
            request: The user SearchRequest.

        Returns:
            SearchContext: The context needed to execute the search in the index.
        """
        # Generate the query embedding vector
        query_embeddings = self.embedding_provider.embed([request.query])
        if not query_embeddings:
            raise ValueError("Embedding provider failed to generate query vector.")
        query_vector = query_embeddings[0]

        # Parse filter dictionary into FilterExpression AST
        filter_expr = parse_filter(request.metadata_filter)

        # Resolve similarity metric overrides if any
        similarity_metric = None
        if request.similarity_metric:
            if request.similarity_metric.lower() == "cosine":
                similarity_metric = CosineSimilarity()
            else:
                raise ValueError(f"Unsupported similarity metric override: '{request.similarity_metric}'")

        return SearchContext(
            query_vector=query_vector,
            top_k=request.top_k,
            filter_expression=filter_expr,
            similarity_metric=similarity_metric
        )
