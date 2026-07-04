from datetime import datetime, timezone
import numpy as np
import pytest

from ziondb.storage.exceptions import RecordNotFoundError
from ziondb.storage.record import ChunkRecord, SystemMetadata
from ziondb.storage.in_memory_storage import InMemoryStorage
from ziondb.index.models import IndexSearchResult
from ziondb.index.similarity import SimilarityMetric, CosineSimilarity
from ziondb.index.brute_force_index import BruteForceIndex
from ziondb.retrieval.search_context import SearchContext

@pytest.fixture
def sample_metadata() -> SystemMetadata:
    return SystemMetadata(
        document_id="doc_idx_1",
        embedding_model="all-MiniLM-L6-v2",
        chunking_method="kamradt",
        token_count=50,
        created_at=datetime.now(timezone.utc)
    )

@pytest.fixture
def make_record(sample_metadata):
    def _make(record_id: str, vector: list[float]) -> ChunkRecord:
        return ChunkRecord(
            id=record_id,
            text=f"Text for {record_id}",
            embedding=np.array(vector),
            metadata={},
            system_metadata=sample_metadata
        )
    return _make

def test_index_initial_state(make_record):
    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    assert index.size() == 0
    # Search empty index should return empty
    results = index.search(SearchContext(np.array([1.0, 0.0]), top_k=5))
    assert len(results) == 0

def test_index_insert_and_size(make_record):
    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    r1 = make_record("rec_1", [1.0, 0.0])
    index.insert(r1)
    
    assert index.size() == 1
    
    # Duplicate insert updates vector in the index
    r1_alt = make_record("rec_1", [0.0, 1.0])
    index.insert(r1_alt)
    assert index.size() == 1

def test_index_remove_and_update(make_record):
    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    r1 = make_record("rec_1", [1.0, 0.0])
    index.insert(r1)
    
    # Update
    r1_updated = make_record("rec_1", [0.0, 1.0])
    index.update(r1_updated)
    
    # Remove
    index.remove("rec_1")
    assert index.size() == 0
    
    # Check that removing non-existent raises RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        index.remove("rec_1")
        
    # Check that updating non-existent raises RecordNotFoundError
    with pytest.raises(RecordNotFoundError):
        index.update(r1)

def test_index_rebuild_from_provider(make_record):
    storage = InMemoryStorage()
    r1 = make_record("rec_1", [1.0, 0.0])
    r2 = make_record("rec_2", [0.0, 1.0])
    storage.insert(r1)
    storage.insert(r2)
    
    index = BruteForceIndex(storage)
    assert index.size() == 0  # Starts empty
    
    index.rebuild()
    assert index.size() == 2
    
    # Verify both records are indexed
    results = index.search(SearchContext(np.array([1.0, 0.0]), top_k=5))
    assert len(results) == 2
    assert results[0].record_id == "rec_1"
    assert results[1].record_id == "rec_2"

def test_index_cosine_similarity_search(make_record):
    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    # Standard 2D orthogonal and intermediate vectors
    r1 = make_record("rec_1", [1.0, 0.0])   # Angle: 0 deg
    r2 = make_record("rec_2", [0.707, 0.707]) # Angle: 45 deg
    r3 = make_record("rec_3", [0.0, 1.0])   # Angle: 90 deg
    
    index.insert(r1)
    index.insert(r2)
    index.insert(r3)
    
    # Query vector is along r1 [1.0, 0.0]
    query = np.array([1.0, 0.0])
    
    # Search for top-2
    results = index.search(SearchContext(query, top_k=2))
    
    assert len(results) == 2
    # Verify ordering (highest cosine similarity first)
    assert results[0].record_id == "rec_1"
    assert results[0].score == pytest.approx(1.0)
    
    assert results[1].record_id == "rec_2"
    assert results[1].score == pytest.approx(0.707, abs=1e-3)

def test_index_search_top_k_bounds(make_record):
    storage = InMemoryStorage()
    index = BruteForceIndex(storage)
    
    for i in range(10):
        index.insert(make_record(f"rec_{i}", [1.0, float(i)]))
        
    query = np.array([1.0, 1.0])
    
    # Test top_k bounds
    assert len(index.search(SearchContext(query, top_k=5))) == 5
    assert len(index.search(SearchContext(query, top_k=20))) == 10
    assert len(index.search(SearchContext(query, top_k=0))) == 0
    assert len(index.search(SearchContext(query, top_k=-5))) == 0

def test_index_custom_similarity_metric(make_record):
    """
    Test that the index handles alternative similarity metrics, e.g. Euclidean Distance
    where smaller distance values are better (greater_is_better = False).
    """
    class EuclideanDistance(SimilarityMetric):
        @property
        def greater_is_better(self) -> bool:
            return False
            
        def calculate(self, v1: np.ndarray, v2: np.ndarray) -> float:
            return float(np.linalg.norm(v1 - v2))
            
    storage = InMemoryStorage()
    index = BruteForceIndex(storage, metric=EuclideanDistance())
    
    r1 = make_record("rec_1", [1.0, 0.0])  # Dist: 0
    r2 = make_record("rec_2", [2.0, 0.0])  # Dist: 1
    r3 = make_record("rec_3", [5.0, 0.0])  # Dist: 4
    
    index.insert(r1)
    index.insert(r2)
    index.insert(r3)
    
    query = np.array([1.0, 0.0])
    
    results = index.search(SearchContext(query, top_k=3))
    
    # Sorting direction should be ascending (smallest distance first)
    assert len(results) == 3
    assert results[0].record_id == "rec_1"
    assert results[0].score == pytest.approx(0.0)
    
    assert results[1].record_id == "rec_2"
    assert results[1].score == pytest.approx(1.0)
    
    assert results[2].record_id == "rec_3"
    assert results[2].score == pytest.approx(4.0)
