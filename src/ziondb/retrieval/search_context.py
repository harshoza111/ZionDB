from dataclasses import dataclass
from typing import Optional
import numpy as np

from ziondb.retrieval.filter import FilterExpression
from ziondb.index.similarity import SimilarityMetric


@dataclass(slots=True)
class SearchContext:
    """Represents the internal execution model for a search operation."""
    query_vector: np.ndarray
    top_k: int
    filter_expression: Optional[FilterExpression] = None
    similarity_metric: Optional[SimilarityMetric] = None
