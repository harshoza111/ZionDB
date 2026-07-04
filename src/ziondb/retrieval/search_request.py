from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(slots=True)
class SearchRequest:
    """Represents a public vector similarity search request."""
    query: str
    top_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None
    similarity_metric: Optional[str] = None
