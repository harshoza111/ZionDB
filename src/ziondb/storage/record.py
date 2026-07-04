from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
import numpy as np

@dataclass(slots=True)
class SystemMetadata:
    """Represents system-generated metadata details for a stored chunk."""
    document_id: str
    embedding_model: str
    chunking_method: str
    token_count: int
    created_at: datetime

@dataclass(slots=True)
class ChunkRecord:
    """Represents a single vector-searchable chunk stored inside ZionDB."""
    id: str
    text: str
    embedding: np.ndarray  # 1D vector of embeddings
    metadata: Dict[str, Any]
    system_metadata: SystemMetadata
