from dataclasses import dataclass
from ziondb.storage.record import ChunkRecord

@dataclass(slots=True)
class SearchResult:
    """Represents a public vector similarity search result returned to the user."""
    record: ChunkRecord
    score: float
