from datetime import datetime, timezone
import numpy as np
import pytest

from ziondb.storage.exceptions import RecordAlreadyExistsError, RecordNotFoundError
from ziondb.storage.record import ChunkRecord, SystemMetadata
from ziondb.storage.in_memory_storage import InMemoryStorage

@pytest.fixture
def sample_metadata() -> SystemMetadata:
    return SystemMetadata(
        document_id="doc_123",
        embedding_model="all-MiniLM-L6-v2",
        chunking_method="kamradt",
        token_count=100,
        created_at=datetime.now(timezone.utc)
    )

@pytest.fixture
def make_record(sample_metadata):
    def _make(record_id: str, text: str = "some text") -> ChunkRecord:
        return ChunkRecord(
            id=record_id,
            text=text,
            embedding=np.array([1.0, 2.0, 3.0]),
            metadata={"source": "test"},
            system_metadata=sample_metadata
        )
    return _make

def test_storage_insert_and_get(make_record):
    storage = InMemoryStorage()
    record = make_record("rec_1")
    
    assert storage.size() == 0
    assert not storage.exists("rec_1")
    
    storage.insert(record)
    
    assert storage.size() == 1
    assert storage.exists("rec_1")
    
    fetched = storage.get("rec_1")
    assert fetched.id == "rec_1"
    assert fetched.text == "some text"
    assert np.array_equal(fetched.embedding, record.embedding)

def test_storage_insert_duplicate_raises_exception(make_record):
    storage = InMemoryStorage()
    r1 = make_record("rec_1")
    r2 = make_record("rec_1", text="different text")
    
    storage.insert(r1)
    with pytest.raises(RecordAlreadyExistsError):
        storage.insert(r2)

def test_storage_get_not_found_raises_exception():
    storage = InMemoryStorage()
    with pytest.raises(RecordNotFoundError):
        storage.get("non_existent")

def test_storage_update(make_record):
    storage = InMemoryStorage()
    r1 = make_record("rec_1")
    storage.insert(r1)
    
    # Create updated record
    r1_updated = make_record("rec_1", text="updated text")
    r1_updated.metadata = {"updated": True}
    
    storage.update(r1_updated)
    
    fetched = storage.get("rec_1")
    assert fetched.text == "updated text"
    assert fetched.metadata == {"updated": True}

def test_storage_update_not_found_raises_exception(make_record):
    storage = InMemoryStorage()
    r = make_record("rec_1")
    with pytest.raises(RecordNotFoundError):
        storage.update(r)

def test_storage_delete_single_record(make_record):
    storage = InMemoryStorage()
    r = make_record("rec_1")
    storage.insert(r)
    
    assert storage.size() == 1
    storage.delete("rec_1")
    
    assert storage.size() == 0
    assert not storage.exists("rec_1")
    with pytest.raises(RecordNotFoundError):
        storage.get("rec_1")

def test_storage_delete_swap_and_pop_integrity(make_record):
    """
    Verifies that the O(1) swap-and-pop list deletion preserves storage state
    and correctly updates index dictionaries for swapped elements.
    """
    storage = InMemoryStorage()
    r1 = make_record("rec_1", text="text1")
    r2 = make_record("rec_2", text="text2")
    r3 = make_record("rec_3", text="text3")
    
    storage.insert(r1)
    storage.insert(r2)
    storage.insert(r3)
    
    # Internal state: list order is [rec_1, rec_2, rec_3]
    # Delete rec_2 (middle element). It should swap with rec_3.
    # List order should become [rec_1, rec_3], and rec_3's index in dict should update to 1.
    storage.delete("rec_2")
    
    assert storage.size() == 2
    assert not storage.exists("rec_2")
    assert storage.exists("rec_1")
    assert storage.exists("rec_3")
    
    # Verify index updates and fetches still work perfectly
    fetched_1 = storage.get("rec_1")
    fetched_3 = storage.get("rec_3")
    assert fetched_1.text == "text1"
    assert fetched_3.text == "text3"
    
    # Ensure internal mapping is correct (should retrieve elements correctly)
    records = list(storage.iterate())
    assert len(records) == 2
    assert records[0].id == "rec_1"
    assert records[1].id == "rec_3"

def test_storage_delete_not_found_raises_exception():
    storage = InMemoryStorage()
    with pytest.raises(RecordNotFoundError):
        storage.delete("non_existent")

def test_storage_clear(make_record):
    storage = InMemoryStorage()
    storage.insert(make_record("rec_1"))
    storage.insert(make_record("rec_2"))
    
    assert storage.size() == 2
    storage.clear()
    
    assert storage.size() == 0
    assert not storage.exists("rec_1")
    assert not storage.exists("rec_2")

def test_storage_iterate(make_record):
    storage = InMemoryStorage()
    r1 = make_record("rec_1")
    r2 = make_record("rec_2")
    
    storage.insert(r1)
    storage.insert(r2)
    
    records = list(storage.iterate())
    assert len(records) == 2
    assert {r.id for r in records} == {"rec_1", "rec_2"}
