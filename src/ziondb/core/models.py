from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List, Optional

@dataclass(slots=True)
class TextDocument:
    """Represents an input text document to be indexed."""
    text: str
    metadata: Dict[str, Any]
    id: Optional[str] = None

@dataclass(slots=True)
class Sentence:
    """Represents a single sentence extracted from a document."""
    text: str
    index: int
    start_char: int
    end_char: int
    document_id: str

@dataclass(slots=True)
class SentenceEmbedding:
    """Represents the embedding vector for a sentence."""
    sentence_index: int
    embedding: np.ndarray  # 1D float array

@dataclass(slots=True)
class BoundaryDecision:
    """Represents the decision on whether a semantic boundary exists after a sentence."""
    index: int  # Index of the sentence after which the boundary is evaluated
    distance: float  # Cosine distance between neighboring sentence groups
    is_boundary: bool  # True if a boundary is detected
    distance_to_threshold: float  # Difference between the distance and the boundary threshold

@dataclass(slots=True)
class Chunk:
    """Represents a semantically cohesive chunk composed of one or more sentences."""
    text: str
    index: int
    sentence_indices: List[int]
    document_id: str
    metadata: Dict[str, Any]

@dataclass(slots=True)
class ChunkEmbedding:
    """Represents the embedding vector for a final chunk."""
    chunk_index: int
    embedding: np.ndarray  # 1D float array


@dataclass(slots=True)
class ModelConfig:
    """Settings for the embedding model and ONNX runtime."""
    name: str
    cache_dir: str
    max_length: int


@dataclass(slots=True)
class SplitterConfig:
    """Settings for the sentence splitter."""
    type: str  # "spacy" or "regex"
    model_name: str
    regex_pattern: str


@dataclass(slots=True)
class BoundaryDetectorConfig:
    """Settings for the Kamradt semantic boundary detector."""
    buffer_size: int
    threshold_type: str
    threshold_value: float
    use_text_embedding: bool


@dataclass(slots=True)
class PipelineConfig:
    """Aggregated settings for the entire document indexing pipeline."""
    model: ModelConfig
    splitter: SplitterConfig
    boundary_detector: BoundaryDetectorConfig

