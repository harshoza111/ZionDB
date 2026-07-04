from datetime import datetime, timezone
import numpy as np
import pytest

from ziondb import ZionDB, CollectionConfig
from ziondb.retrieval.filter import (
    EqualFilter,
    AndFilter,
    OrFilter,
    NotFilter,
    parse_filter,
)
from ziondb.retrieval.search_request import SearchRequest
from ziondb.retrieval.search_result import SearchResult
from ziondb.index.models import IndexSearchResult
from ziondb.retrieval.services.query_embedding_service import QueryEmbeddingService
from ziondb.retrieval.services.vector_search_service import VectorSearchService
from ziondb.retrieval.services.hydration_service import HydrationService
from ziondb.retrieval.postprocessing.identity_post_processor import IdentityPostProcessor
from ziondb.retrieval.retriever import Retriever
from ziondb.retrieval.executors.search_executor import SearchExecutor
from ziondb.retrieval.executors.dense_search_executor import DenseSearchExecutor



def test_filter_expression_ast():
    metadata = {
        "author": "Alice",
        "year": 2024,
        "category": "AI",
        "published": True,
    }

    # EqualFilter
    assert EqualFilter("author", "Alice").evaluate(metadata) is True
    assert EqualFilter("author", "Bob").evaluate(metadata) is False
    assert EqualFilter("year", 2024).evaluate(metadata) is True
    assert EqualFilter("missing", "val").evaluate(metadata) is False
    assert EqualFilter("author", "Alice").evaluate(None) is False

    # AndFilter
    and_true = AndFilter([EqualFilter("author", "Alice"), EqualFilter("year", 2024)])
    assert and_true.evaluate(metadata) is True
    
    and_false = AndFilter([EqualFilter("author", "Alice"), EqualFilter("year", 2023)])
    assert and_false.evaluate(metadata) is False
    
    assert AndFilter([]).evaluate(metadata) is True

    # OrFilter
    or_true = OrFilter([EqualFilter("author", "Bob"), EqualFilter("year", 2024)])
    assert or_true.evaluate(metadata) is True
    
    or_false = OrFilter([EqualFilter("author", "Bob"), EqualFilter("year", 2023)])
    assert or_false.evaluate(metadata) is False
    
    assert OrFilter([]).evaluate(metadata) is False

    # NotFilter
    not_true = NotFilter(EqualFilter("author", "Bob"))
    assert not_true.evaluate(metadata) is True
    
    not_false = NotFilter(EqualFilter("author", "Alice"))
    assert not_false.evaluate(metadata) is False


def test_filter_parser():
    # None / Empty handling
    assert parse_filter(None) is None
    assert parse_filter({}) is None

    # Simple EqualFilter
    expr = parse_filter({"category": "AI"})
    assert isinstance(expr, EqualFilter)
    assert expr.key == "category"
    assert expr.value == "AI"

    # Implicit AND
    expr = parse_filter({"author": "Alice", "year": 2024})
    assert isinstance(expr, AndFilter)
    assert len(expr.expressions) == 2
    assert any(isinstance(e, EqualFilter) and e.key == "author" and e.value == "Alice" for e in expr.expressions)
    assert any(isinstance(e, EqualFilter) and e.key == "year" and e.value == 2024 for e in expr.expressions)

    # Explicit $and
    expr = parse_filter({"$and": [{"category": "AI"}, {"published": True}]})
    assert isinstance(expr, AndFilter)
    assert len(expr.expressions) == 2

    # Explicit $or
    expr = parse_filter({"$or": [{"category": "AI"}, {"category": "DB"}]})
    assert isinstance(expr, OrFilter)
    assert len(expr.expressions) == 2

    # Explicit $not
    expr = parse_filter({"$not": {"category": "DB"}})
    assert isinstance(expr, NotFilter)
    assert isinstance(expr.expression, EqualFilter)
    assert expr.expression.key == "category"
    assert expr.expression.value == "DB"

    # Explicit $eq comparison
    expr = parse_filter({"author": {"$eq": "Alice"}})
    assert isinstance(expr, EqualFilter)
    assert expr.key == "author"
    assert expr.value == "Alice"

    # Explicit $ne comparison (translates to NotFilter(EqualFilter))
    expr = parse_filter({"author": {"$ne": "Bob"}})
    assert isinstance(expr, NotFilter)
    assert isinstance(expr.expression, EqualFilter)
    assert expr.expression.key == "author"
    assert expr.expression.value == "Bob"

    # Parser validation errors
    with pytest.raises(ValueError):
        parse_filter({"$and": "not-a-list"})

    with pytest.raises(ValueError):
        parse_filter({"$or": "not-a-list"})

    with pytest.raises(ValueError):
        parse_filter({"$not": "not-a-dict"})

    with pytest.raises(ValueError):
        parse_filter({"$not": {}})

    with pytest.raises(ValueError):
        parse_filter({"category": {"$unsupported_op": "val"}})


def test_metadata_filtering_integration(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("filtered_collection")

    # Add documents with distinct metadata
    coll.add_document("Machine learning is cool.", metadata={"category": "AI", "year": 2024}, id="doc_ai_1")
    coll.add_document("Deep learning models are large.", metadata={"category": "AI", "year": 2023}, id="doc_ai_2")
    coll.add_document("Relational databases use SQL.", metadata={"category": "DB", "year": 2024}, id="doc_db_1")
    coll.add_document("NoSQL stores JSON documents.", metadata={"category": "DB", "year": 2023}, id="doc_db_2")

    # 1. Search without filters returns all relevant items
    results = coll.search(query="learning", top_k=5)
    assert len(results) == 4

    # 2. Simple equality filter: category == AI
    results = coll.search(query="learning", top_k=5, metadata_filter={"category": "AI"})
    assert len(results) == 2
    assert all(r.record.metadata.get("category") == "AI" for r in results)
    assert {r.record.system_metadata.document_id for r in results} == {"doc_ai_1", "doc_ai_2"}

    # 3. Explicit $and filter: category == AI AND year == 2024
    results = coll.search(
        query="learning",
        top_k=5,
        metadata_filter={"$and": [{"category": "AI"}, {"year": 2024}]}
    )
    assert len(results) == 1
    assert results[0].record.system_metadata.document_id == "doc_ai_1"

    # 4. Explicit $or filter: category == DB OR year == 2024
    # Matches:
    # - doc_ai_1 (year == 2024)
    # - doc_db_1 (category == DB AND year == 2024)
    # - doc_db_2 (category == DB)
    results = coll.search(
        query="learning",
        top_k=5,
        metadata_filter={"$or": [{"category": "DB"}, {"year": 2024}]}
    )
    assert len(results) == 3
    matched_ids = {r.record.system_metadata.document_id for r in results}
    assert matched_ids == {"doc_ai_1", "doc_db_1", "doc_db_2"}

    # 5. Explicit $not/$ne filter: category != AI (which is category == DB)
    results = coll.search(
        query="learning",
        top_k=5,
        metadata_filter={"category": {"$ne": "AI"}}
    )
    assert len(results) == 2
    matched_ids = {r.record.system_metadata.document_id for r in results}
    assert matched_ids == {"doc_db_1", "doc_db_2"}


def test_query_embedding_service(mock_embedder):
    service = QueryEmbeddingService(embedding_provider=mock_embedder)
    
    # Check default request processing
    request = SearchRequest(query="deep learning", top_k=3, metadata_filter={"category": "AI"})
    context = service.embed_and_contextualize(request)
    
    assert context.top_k == 3
    assert context.query_vector.shape == (384,)
    assert isinstance(context.filter_expression, EqualFilter)
    assert context.filter_expression.key == "category"
    assert context.filter_expression.value == "AI"
    assert context.similarity_metric is None

    # Check metric overrides resolution
    request_with_metric = SearchRequest(query="test", similarity_metric="cosine")
    context_with_metric = service.embed_and_contextualize(request_with_metric)
    assert context_with_metric.similarity_metric is not None
    assert context_with_metric.similarity_metric.greater_is_better is True

    # Check unsupported metric error
    request_bad_metric = SearchRequest(query="test", similarity_metric="euclidean")
    with pytest.raises(ValueError, match="Unsupported similarity metric override"):
        service.embed_and_contextualize(request_bad_metric)


def test_vector_search_service(mock_embedder):
    from ziondb.storage.in_memory_storage import InMemoryStorage
    from ziondb.index.brute_force_index import BruteForceIndex
    from ziondb.storage.record import ChunkRecord, SystemMetadata

    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    sys_meta = SystemMetadata(
        document_id="doc_1",
        embedding_model="all-MiniLM-L6-v2",
        chunking_method="kamradt",
        token_count=10,
        created_at=datetime.now(timezone.utc)
    )
    
    # Store and index a record
    record = ChunkRecord(
        id="chunk_1",
        text="Hello world",
        embedding=np.zeros(384),
        metadata={"category": "test"},
        system_metadata=sys_meta
    )
    index.insert(record)
    
    # Wrap in VectorSearchService
    service = VectorSearchService(vector_index=index)
    
    # Search
    from ziondb.retrieval.search_context import SearchContext
    context = SearchContext(query_vector=np.zeros(384), top_k=10)
    index_results = service.search(context)
    
    assert len(index_results) == 1
    assert isinstance(index_results[0], IndexSearchResult)
    assert index_results[0].record_id == "chunk_1"


def test_hydration_service():
    from ziondb.storage.in_memory_storage import InMemoryStorage
    from ziondb.storage.record import ChunkRecord, SystemMetadata

    storage = InMemoryStorage()
    
    sys_meta = SystemMetadata(
        document_id="doc_1",
        embedding_model="all-MiniLM-L6-v2",
        chunking_method="kamradt",
        token_count=10,
        created_at=datetime.now(timezone.utc)
    )
    
    # Insert record in storage
    record = ChunkRecord(
        id="chunk_1",
        text="Hydrated content",
        embedding=np.zeros(384),
        metadata={},
        system_metadata=sys_meta
    )
    storage.insert(record)
    
    service = HydrationService(record_provider=storage)
    
    # Hydrate index results
    index_results = [IndexSearchResult(record_id="chunk_1", score=0.85)]
    hydrated = service.hydrate(index_results)
    
    assert len(hydrated) == 1
    assert isinstance(hydrated[0], SearchResult)
    assert hydrated[0].record.id == "chunk_1"
    assert hydrated[0].record.text == "Hydrated content"
    assert hydrated[0].score == 0.85

    # Check graceful sync error handling (e.g. non-existent record ID)
    missing_index_results = [IndexSearchResult(record_id="missing_chunk", score=0.5)]
    hydrated_missing = service.hydrate(missing_index_results)
    assert len(hydrated_missing) == 0


def test_identity_post_processor():
    from ziondb.storage.record import ChunkRecord, SystemMetadata

    sys_meta = SystemMetadata(
        document_id="doc_1",
        embedding_model="all-MiniLM-L6-v2",
        chunking_method="kamradt",
        token_count=10,
        created_at=datetime.now(timezone.utc)
    )
    
    record = ChunkRecord(
        id="chunk_1",
        text="content",
        embedding=np.zeros(384),
        metadata={},
        system_metadata=sys_meta
    )
    
    results = [SearchResult(record=record, score=0.9)]
    
    processor = IdentityPostProcessor()
    processed_results = processor.process(results)
    
    assert processed_results is results


def test_dense_search_executor_empty_query(mock_embedder):
    from ziondb.storage.in_memory_storage import InMemoryStorage
    from ziondb.index.brute_force_index import BruteForceIndex

    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    q_service = QueryEmbeddingService(embedding_provider=mock_embedder)
    v_service = VectorSearchService(vector_index=index)
    h_service = HydrationService(record_provider=storage)
    post_proc = IdentityPostProcessor()
    
    executor = DenseSearchExecutor(
        query_embedding_service=q_service,
        vector_search_service=v_service,
        hydration_service=h_service,
        post_processor=post_proc
    )
    
    # Empty query should return empty results
    request = SearchRequest(query="   ", top_k=5)
    results = executor.execute(request)
    assert results == []


def test_retriever_delegation():
    class DummySearchExecutor(SearchExecutor):
        def __init__(self):
            self.called = False
            self.passed_request = None

        def execute(self, request: SearchRequest) -> List[SearchResult]:
            self.called = True
            self.passed_request = request
            return []

    dummy_executor = DummySearchExecutor()
    retriever = Retriever(search_executor=dummy_executor)
    
    request = SearchRequest(query="learning")
    results = retriever.retrieve(request)
    
    assert results == []
    assert dummy_executor.called is True
    assert dummy_executor.passed_request is request


