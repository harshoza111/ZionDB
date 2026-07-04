from dataclasses import dataclass

@dataclass(slots=True)
class IndexSearchResult:
    """Represents a vector similarity search result match from the index."""
    record_id: str
    score: float
