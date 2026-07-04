from dataclasses import dataclass
from typing import Optional

@dataclass(slots=True)
class CollectionConfig:
    """Configuration settings for a single ZionDB Collection."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunking_method: str = "kamradt"
    similarity_metric: str = "cosine"
    index_type: str = "brute_force"
    storage_type: str = "in_memory"
    
    # Embedding settings
    cache_dir: str = "models"
    max_length: int = 256
    
    # Splitter settings
    splitter_type: str = "regex"  # "regex" or "spacy"
    regex_pattern: Optional[str] = None
    spacy_model: str = "en_sentencizer"
    
    # Boundary detector settings
    buffer_size: int = 1
    threshold_type: str = "percentile"  # "percentile" or "standard_deviation"
    threshold_value: float = 50.0
    use_text_embedding: bool = False
