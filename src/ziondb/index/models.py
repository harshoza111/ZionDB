from dataclasses import dataclass

@dataclass(slots=True)
class SearchResult:
    """Represents a vector similarity search result match."""
    record_id: str
    score: float
