import pytest
import numpy as np
from ziondb import ZionDB, CollectionConfig, SearchRequest
from ziondb.collection.exceptions import CollectionAlreadyExistsError, CollectionNotFoundError
from ziondb.core.models import TextDocument


def test_collection_lifecycle(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)

    # 1. List is empty initially
    assert db.list_collections() == []
    assert db.exists("test_coll") is False

    # 2. Create collection
    coll = db.create_collection("test_coll")
    assert coll.name == "test_coll"
    assert db.exists("test_coll") is True
    assert db.list_collections() == ["test_coll"]

    # 3. Duplicate creation raises error
    with pytest.raises(CollectionAlreadyExistsError):
        db.create_collection("test_coll")

    # 4. Lookup collection
    retrieved = db.get_collection("test_coll")
    assert retrieved is coll

    # 5. Lookup missing raises error
    with pytest.raises(CollectionNotFoundError):
        db.get_collection("missing_coll")

    # 6. Delete collection
    db.delete_collection("test_coll")
    assert db.exists("test_coll") is False
    assert db.list_collections() == []

    # 7. Delete missing raises error
    with pytest.raises(CollectionNotFoundError):
        db.delete_collection("test_coll")


def test_collection_isolation(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)

    coll_a = db.create_collection("coll_a")
    coll_b = db.create_collection("coll_b")

    # Add document to A
    coll_a.add_document(text="Python is a programming language.", id="doc_a")

    # Add document to B
    coll_b.add_document(text="Spam is processed meat in a can.", id="doc_b")

    assert coll_a.count() > 0
    assert coll_b.count() > 0

    # Search in A should not find meat
    results_a = coll_a.search(query="processed meat", top_k=2)
    assert len(results_a) > 0
    # The result in A should be the python text, even if the query is closer to meat,
    # because meat text is not in collection A.
    assert "programming language" in results_a[0].record.text

    # Search in B should find meat
    results_b = coll_b.search(query="processed meat", top_k=2)
    assert len(results_b) > 0
    assert "processed meat" in results_b[0].record.text


def test_add_and_search_documents(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    config = CollectionConfig(
        splitter_type="regex",
        regex_pattern=r"(?<=[.!?]) +",
        buffer_size=1,
        threshold_value=50.0
    )
    coll = db.create_collection("docs_coll", config=config)

    # 1. Add single document
    doc_id = coll.add_document(
        text="The quick brown fox jumps over the lazy dog. A fast red cat leaps under the active bird.",
        metadata={"category": "animals"},
        id="animal_doc"
    )
    assert doc_id == "animal_doc"
    assert coll.count() > 0

    # 2. Add multiple documents
    docs = [
        TextDocument(text="Deep learning is part of machine learning.", metadata={"topic": "AI"}, id="ai_1"),
        TextDocument(text="Reinforcement learning matches rewards to actions.", metadata={"topic": "AI"}, id="ai_2")
    ]
    coll.add_documents(docs)

    # Total chunks count
    assert coll.count() > 1

    # 3. Search
    results = coll.search(query="neural networks deep learning", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0].score, float)
    assert results[0].record.system_metadata.document_id in ["ai_1", "ai_2", "animal_doc"]
    assert results[0].record.system_metadata.embedding_model == config.embedding_model
    assert results[0].record.system_metadata.chunking_method == config.chunking_method


def test_document_upsert(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("upsert_coll")

    # Add version 1 of document
    coll.add_document(text="The sky is blue. The grass is green.", id="doc_1")
    count_v1 = coll.count()
    assert count_v1 > 0

    # Add version 2 of document (different text, same ID)
    coll.add_document(text="Mars is red. Jupiter is gaseous.", id="doc_1")
    count_v2 = coll.count()

    # Chunks from v1 should be replaced/removed and replaced with v2
    # Verify by searching. Blue sky should not match high, Mars should match
    results_mars = coll.search(query="Mars planet", top_k=5)
    assert any("Mars is red" in r.record.text for r in results_mars)
    assert not any("The sky is blue" in r.record.text for r in results_mars)


def test_clear_collection(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("clear_coll")

    coll.add_document(text="This is test document number one.", id="d1")
    assert coll.count() > 0

    coll.clear()
    assert coll.count() == 0

    # Search on empty should return nothing
    assert coll.search(query="test", top_k=2) == []


def test_empty_search_and_edge_cases(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)
    coll = db.create_collection("edge_coll")

    # 1. Search in empty collection
    assert coll.search(query="hello", top_k=5) == []

    # 2. Empty query search
    coll.add_document(text="Valid document content.")
    assert coll.search(query="   ", top_k=5) == []


def test_invalid_configurations(mock_embedder):
    db = ZionDB(default_embedding_provider=mock_embedder)

    # 1. Unsupported similarity metric
    config_metric = CollectionConfig(similarity_metric="euclidean")
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        db.create_collection("bad_metric", config=config_metric)

    # 2. Unsupported storage type
    config_storage = CollectionConfig(storage_type="sqlite")
    with pytest.raises(ValueError, match="Unsupported storage type"):
        db.create_collection("bad_storage", config=config_storage)

    # 3. Unsupported index type
    config_index = CollectionConfig(index_type="hnsw")
    with pytest.raises(ValueError, match="Unsupported index type"):
        db.create_collection("bad_index", config=config_index)

    # 4. Unsupported splitter type
    config_splitter = CollectionConfig(splitter_type="nltk")
    with pytest.raises(ValueError, match="Unsupported splitter type"):
        db.create_collection("bad_splitter", config=config_splitter)
